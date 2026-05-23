from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QToolButton

import gui.steps.step3_mask as step3_mask_module
from core.mask_source_scope import source_scope_key
from core.scene_layout import selected_frames_path
from core.scene_project import write_mask_item
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
from gui.steps.step3_mask import MaskStep


def _app():
    return QApplication.instance() or QApplication([])


def _process_events_until(predicate, timeout_s: float = 2.0) -> None:
    app = _app()
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    app.processEvents()


def _write_scene(tmp_path: Path, drop_exists: bool = True) -> Path:
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"keep")
    if drop_exists:
        (images / "frame_0002.jpg").write_bytes(b"drop")

    rows = [
        {"seq": "1", "output_file": "images/frame_0001.jpg", "decision": "keep", "status": "ok"},
        {"seq": "2", "output_file": "images/frame_0002.jpg", "decision": "drop", "status": "redundant_drop"},
    ]
    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return scene


def _manifest_images(command: list[str]) -> list[str]:
    manifest = Path(command[command.index("--image-list") + 1])
    return [json.loads(line)["image"] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_mask_step_rejects_generation_when_drop_images_remain(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=True)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    with pytest.raises(ValueError, match="Step 2"):
        step.build_commands()


def test_mask_step_uses_standard_scene_folders_without_browse_inputs(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())

    step.set_scene_dir(str(scene))

    assert not step.findChildren(BrowseWidget)
    labels = [label.text() for label in step.findChildren(QLabel)]
    assert str(scene / "images") in labels
    assert str(scene / "masks") in labels
    assert step.primary_action_enabled()
    assert step.primary_action_tooltip() == i18n.tip("RUN_MASKS")
    assert step.mask_source_combo.currentData() == "all"
    assert step.mask_source_combo.itemText(0) == i18n.t("MASK_SOURCE_ALL")
    assert step.mask_scope_combo.currentData() == "missing"
    assert step.mask_scope_combo.itemText(0) == i18n.t("MASK_SCOPE_MISSING")
    assert step.mask_scope_combo.itemText(1) == i18n.t("MASK_SCOPE_STALE")
    assert step.mask_scope_combo.itemText(2) == i18n.t("MASK_SCOPE_ALL")


def test_mask_step_missing_scope_processes_only_unmasked_images(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"one")
    (images / "frame_0002.jpg").write_bytes(b"two")
    (masks / "frame_0001.png").write_bytes(b"existing")
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo"]
    assert _manifest_images(commands[0][1]) == ["images/frame_0002.jpg"]
    assert step._mask_batch_targets == [images / "frame_0002.jpg"]


def test_mask_step_source_filter_limits_missing_scope_to_selected_source(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "video_a_0001.jpg").write_bytes(b"a")
    (images / "video_b_0001.jpg").write_bytes(b"b")
    (masks / "video_a_0001.png").write_bytes(b"existing")
    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["output_file", "source_session", "source_video", "decision", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "output_file": "images/video_a_0001.jpg",
                "source_session": "video_a",
                "source_video": "D:/source/video_a.mp4",
                "decision": "keep",
                "status": "ok",
            }
        )
        writer.writerow(
            {
                "output_file": "images/video_b_0001.jpg",
                "source_session": "video_b",
                "source_video": "D:/source/video_b.mp4",
                "decision": "keep",
                "status": "ok",
            }
        )
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    source_index = step.mask_source_combo.findData(source_scope_key("video_extract", "video_b"))
    assert source_index >= 0
    step.mask_source_combo.setCurrentIndex(source_index)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo"]
    assert _manifest_images(commands[0][1]) == ["images/video_b_0001.jpg"]
    assert step._mask_batch_targets == [images / "video_b_0001.jpg"]


def test_mask_step_source_filter_writes_manifest_even_when_selected_source_all_targets(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "video_a_0001.jpg").write_bytes(b"a")
    (images / "video_b_0001.jpg").write_bytes(b"b")
    (masks / "video_a_0001.png").write_bytes(b"existing a")
    (masks / "video_b_0001.png").write_bytes(b"existing b")
    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["output_file", "source_session", "source_video", "decision", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "output_file": "images/video_a_0001.jpg",
                "source_session": "video_a",
                "source_video": "D:/source/video_a.mp4",
                "decision": "keep",
                "status": "ok",
            }
        )
        writer.writerow(
            {
                "output_file": "images/video_b_0001.jpg",
                "source_session": "video_b",
                "source_video": "D:/source/video_b.mp4",
                "decision": "keep",
                "status": "ok",
            }
        )
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    source_index = step.mask_source_combo.findData(source_scope_key("video_extract", "video_a"))
    assert source_index >= 0
    step.mask_source_combo.setCurrentIndex(source_index)
    step.mask_scope_combo.setCurrentIndex(2)

    commands = step.build_commands()

    assert _manifest_images(commands[0][1]) == ["images/video_a_0001.jpg"]


def test_mask_step_missing_scope_reports_no_targets_when_all_masks_exist(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"one")
    (masks / "frame_0001.png").write_bytes(b"existing")
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    with pytest.raises(ValueError, match=i18n.t("MASK_TARGETS_EMPTY")):
        step.build_commands()


def test_mask_step_stale_scope_updates_old_generated_masks_but_protects_unknown_masks(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    image_old = images / "frame_0001.jpg"
    image_unknown = images / "frame_0002.jpg"
    image_current = images / "frame_0003.jpg"
    for image in (image_old, image_unknown, image_current):
        image.write_bytes(b"image")
        (masks / f"{image.stem}.png").write_bytes(b"mask")
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    current_settings = step._mask_settings_snapshot()
    write_mask_item(
        scene,
        image_path=image_old,
        mask_path=masks / "frame_0001.png",
        settings={**current_settings, "quality": "old"},
        run_id="old",
        stats={},
    )
    write_mask_item(
        scene,
        image_path=image_current,
        mask_path=masks / "frame_0003.png",
        settings=current_settings,
        run_id="current",
        stats={},
    )
    step.mask_scope_combo.setCurrentIndex(1)

    commands = step.build_commands()

    assert _manifest_images(commands[0][1]) == ["images/frame_0001.jpg"]
    assert step._mask_batch_targets == [image_old]


def test_mask_step_yolo_class_presets_and_class_label_are_removed() -> None:
    _app()
    step = MaskStep(Path.cwd())

    labels = {label.text() for label in step.findChildren(QLabel)}
    buttons = {button.text().strip() for button in step.findChildren(QPushButton)}
    tool_buttons = {button.text().strip() for button in step.findChildren(QToolButton)}

    assert i18n.YOLO_CLASSES not in labels
    assert i18n.CLASS_PRESET_PERSON not in buttons
    assert i18n.CLASS_PRESET_VEHICLES not in buttons
    assert i18n.CLASS_PRESET_ALL not in buttons
    assert i18n.CLASS_PRESET_CLEAR not in buttons
    assert i18n.t("YOLO_CLASS_LIST_SECTION") in tool_buttons
    assert step.yolo_class_list_section.toggle_button.text().strip() == i18n.t("DETECTION_TARGET_SECTION")
    assert step.ade_class_list_section.toggle_button.text().strip() == i18n.t("DETECTION_TARGET_SECTION")
    assert step.sam_prompt_section.toggle_button.text().strip() == i18n.t("DETECTION_TARGET_SECTION")
    assert step.yolo_class_list_section.content_widget.isHidden()
    assert step.ade_class_list_section.content_widget.isHidden()
    assert step.sam_prompt_section.content_widget.isHidden()
    assert step.yolo_class_list_section.toggle_button.toolTip() == i18n.tip("YOLO_CLASS_LIST_SECTION")
    assert step.ade_class_list_section.toggle_button.toolTip() == i18n.tip("ADE20K_CLASS_LIST_SECTION")
    assert step.sam_prompt_section.toggle_button.toolTip() == i18n.tip("SAM31_PROMPT_SECTION")


def test_mask_step_yolo_level_and_expand_share_compact_row() -> None:
    _app()
    step = MaskStep(Path.cwd())

    content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
    assert step.person_backend_row.sizeHint().width() <= content_width
    assert step.yolo_settings_row.sizeHint().width() <= content_width
    assert step.yolo_level_combo.currentIndex() == 1
    assert step.person_backend_label.toolTip() == i18n.tip("PERSON_MODEL")
    assert step.person_backend_combo.itemText(0) == i18n.t("PERSON_MODEL_YOLO_SAM")
    assert step.person_backend_combo.itemText(1) == i18n.t("SKY_MODEL_MASK2FORMER")
    assert step.person_backend_combo.itemText(2) == i18n.t("PERSON_MODEL_SAM31")
    assert step.yolo_level_label.toolTip() == i18n.tip("MASK_QUALITY")
    assert step.yolo_level_combo.itemText(0) == i18n.t("MASK_QUALITY_STANDARD")
    assert step.yolo_level_combo.itemText(1) == i18n.t("MASK_QUALITY_HIGH")
    assert step.yolo_level_combo.itemText(2) == i18n.t("MASK_QUALITY_BEST")
    assert step.yolo_expand_label.toolTip() == i18n.tip("YOLO_EXPAND")
    assert step.yolo_expand_edit.value() == 0
    assert step.yolo_bottom_settings_row.isHidden()
    assert not hasattr(step, "projection_label")


def test_mask_step_sam31_apply_mode_shares_compact_settings_row(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    checkpoint = scene / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(step, "_sam31_checkpoint_path", lambda: checkpoint)
    step._update_person_backend_availability()

    step.person_backend_combo.setCurrentIndex(2)

    content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
    assert step.yolo_settings_row.sizeHint().width() <= content_width
    assert not step.sam_apply_mode_label.isHidden()
    assert step.sam_apply_mode_label.toolTip() == i18n.tip("SAM31_APPLY_MODE")
    assert step.sam_apply_mode_combo.itemData(0) == "replace"
    assert step.sam_apply_mode_combo.itemData(1) == "add"
    assert step.sam_apply_mode_combo.itemData(2) == "subtract"
    assert step.sam_custom_prompt_icon.pixmap() is not None
    assert step.sam_custom_prompt_icon.toolTip() == i18n.tip("SAM31_CUSTOM_PROMPT")
    assert step.sam_subtract_prompt_icon.pixmap() is not None
    assert step.sam_subtract_prompt_icon.toolTip() == i18n.tip("SAM31_SUBTRACT_PROMPT")
    assert step.sam_custom_prompt_edit.placeholderText() == i18n.t("SAM31_CUSTOM_PROMPT_PLACEHOLDER")
    assert step.sam_subtract_prompt_edit.placeholderText() == i18n.t("SAM31_SUBTRACT_PROMPT_PLACEHOLDER")
    assert [prompt for prompt, _cb in step.sam_prompt_cbs] == [
        "person",
        "sky",
        "tripod",
        "hand",
        "camera",
        "selfie stick",
        "cell phone",
        "car",
    ]
    prompt_layout = step.sam_prompt_section.content_layout
    assert prompt_layout.itemAt(0).layout() == step.sam_custom_prompt_row
    assert prompt_layout.itemAt(1).layout() == step.sam_subtract_prompt_row
    assert prompt_layout.itemAt(2).widget() == step.sam_prompt_grid_widget


def test_mask_step_metashape_notice_is_in_left_pane() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.metashape_notice.text() == i18n.METASHAPE_NOTICE
    assert not step.preview_pane.isAncestorOf(step.metashape_notice)


def test_mask_step_preview_mode_toolbar_is_in_preview_header() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.preview_pane.isAncestorOf(step.mask_preview.mode_toolbar)
    assert step.preview_pane.isAncestorOf(step.mask_preview.projection_toggle_btn)
    assert step.mask_preview.mode_toolbar.isAncestorOf(step.mask_preview.projection_toggle_btn)
    assert not step.mask_preview.isAncestorOf(step.mask_preview.mode_toolbar)
    assert not step.mask_preview.isAncestorOf(step.mask_preview.projection_toggle_btn)


def test_mask_step_refreshes_preview_when_activated_after_extraction(tmp_path: Path) -> None:
    _app()
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.mask_preview.current_image_path() is None

    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((32, 64, 3), 180, dtype=np.uint8))
    rows = [{"seq": "1", "output_file": "images/frame_0001.jpg", "decision": "keep", "status": "ok"}]
    csv_path = selected_frames_path(tmp_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    step.on_activated()

    assert step.mask_preview.current_image_path() == image_path
    assert step.mask_preview.image_label._source_pixmap is not None
    assert step.primary_action_enabled()


def test_mask_step_refreshes_preview_after_successful_mask_generation(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "frame_0001.jpg"
    temp_mask_path = scene / "temp_preview.png"
    existing_mask_path = masks / "frame_0001.png"
    cv2.imwrite(str(image_path), np.full((32, 64, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(temp_mask_path), np.zeros((32, 64), dtype=np.uint8))

    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    assert step.mask_preview.current_image_path() == image_path
    assert step.mask_preview.set_temporary_preview_mask(
        image_path,
        temp_mask_path,
        step._mask_preview_config_from_controls(),
    )
    step._render_mask_preview()
    assert i18n.t("MASK_PREVIEW_TEMP") in step.mask_preview.status_label.text()

    existing_mask = np.full((32, 64), 255, dtype=np.uint8)
    existing_mask[:, :32] = 0
    cv2.imwrite(str(existing_mask_path), existing_mask)

    step.on_queue_finished(True)

    assert i18n.t("MASK_PREVIEW_YOLO_EXISTING") in step.mask_preview.status_label.text()
    assert i18n.t("MASK_PREVIEW_TEMP") not in step.mask_preview.status_label.text()
    assert step.mask_preview.image_label._source_pixmap is not None


def test_mask_step_mask_preview_from_thumbnails_switches_to_single(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    image_path = images / "frame_0001.png"
    image = np.full((24, 32, 3), 120, dtype=np.uint8)
    image[4:8, 6:10] = 255
    cv2.imwrite(str(image_path), image)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_overexp_cb.setChecked(True)
    step.mask_preview.set_preview_mode("thumbnails")
    monkeypatch.setattr(step, "_confirm_yolo_sam_license_notice", lambda: True)
    monkeypatch.setattr(step, "_build_image_external_commands", lambda *_args, **_kwargs: [])

    step._run_mask_preview()

    assert step.mask_preview.preview_mode() == "single"
    assert i18n.t("MASK_PREVIEW_TEMP") in step.mask_preview.status_label.text()
    assert step.mask_preview.yolo_preview_btn.text() == i18n.t("MASK_PREVIEW_BUTTON")
    assert step.mask_preview.preview_visibility_btn.isEnabled()
    assert not (scene / "masks" / "frame_0001.png").exists()


def test_mask_step_mask_preview_visibility_toggles_active_temporary_preview(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    image_path = images / "frame_0001.png"
    image = np.full((24, 32, 3), 120, dtype=np.uint8)
    image[4:8, 6:10] = 255
    cv2.imwrite(str(image_path), image)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_overexp_cb.setChecked(True)
    monkeypatch.setattr(step, "_confirm_yolo_sam_license_notice", lambda: True)
    monkeypatch.setattr(step, "_build_image_external_commands", lambda *_args, **_kwargs: [])

    step._run_mask_preview()
    assert step.mask_preview.yolo_preview_btn.text() == i18n.t("MASK_PREVIEW_BUTTON")
    assert step.mask_preview.preview_visibility_btn.isChecked()
    step.mask_preview.preview_visibility_btn.click()
    step.mask_preview.render(step._mask_preview_config_from_controls())

    assert not step.mask_preview.preview_visibility_btn.isChecked()
    assert i18n.t("MASK_PREVIEW_TEMP") not in step.mask_preview.status_label.text()
    assert step.mask_preview.has_available_temporary_preview(step._mask_preview_config_from_controls())


def test_mask_step_preview_render_scheduling_is_debounced() -> None:
    _app()
    step = MaskStep(Path.cwd())
    calls = 0

    def fake_render() -> None:
        nonlocal calls
        calls += 1

    step._render_mask_preview = fake_render  # type: ignore[method-assign]

    step._schedule_render_mask_preview()
    step._schedule_render_mask_preview()
    step._mask_preview_render_timer.stop()

    assert calls == 0
    assert step._mask_preview_render_pending


def test_mask_step_disables_generation_without_images_dir(tmp_path: Path) -> None:
    _app()
    csv_path = selected_frames_path(tmp_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("seq,output_file,decision,status\n", encoding="utf-8")
    step = MaskStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))

    assert not step.primary_action_enabled()
    assert step.primary_action_tooltip() == i18n.t("MASK_READY_NO_IMAGES_DIR")
    with pytest.raises(ValueError, match="images/"):
        step.build_commands()


def test_mask_step_allows_external_images_without_selected_frames_csv(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"keep")
    step = MaskStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))

    assert step.primary_action_enabled()
    commands = step.build_commands()
    assert commands[0][0] == "yolo"
    assert commands[0][1][3] == str(images)


def test_mask_step_confirms_yolo_commands(monkeypatch) -> None:
    _app()
    step = MaskStep(Path.cwd())
    calls = 0

    def fake_confirm() -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(step, "_confirm_yolo_sam_license_notice", fake_confirm)

    assert not step.confirm_commands([("yolo", ["cmd"])])
    assert calls == 1
    assert step.confirm_commands([("stitch", ["cmd"])])
    assert calls == 1


def test_mask_step_confirms_semantic_primary_commands(monkeypatch) -> None:
    _app()
    step = MaskStep(Path.cwd())
    calls = 0

    def fake_confirm() -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(step, "_confirm_sky_license_notice", fake_confirm)
    step.person_backend_combo.setCurrentIndex(1)

    assert not step.confirm_commands([("yolo", ["cmd"])])
    assert calls == 1
    assert step.confirm_commands([("stitch", ["cmd"])])
    assert calls == 1


def test_mask_step_sam31_options_stay_enabled_without_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _app()
    step = MaskStep(Path.cwd())
    missing = tmp_path / "models" / "sam3.1" / "sam3.1_multiplex.pt"
    monkeypatch.setattr(step, "_sam31_checkpoint_path", lambda: missing)

    step._update_person_backend_availability()
    step._update_sky_backend_availability()

    person_item = step.person_backend_combo.model().item(2)
    sky_item = step.sky_backend_combo.model().item(1)
    assert person_item.isEnabled()
    assert sky_item.isEnabled()
    assert person_item.toolTip() == i18n.tip("SAM31_CHECKPOINT_DOWNLOAD")
    assert sky_item.toolTip() == i18n.tip("SAM31_CHECKPOINT_DOWNLOAD")


def test_mask_step_confirms_sam31_download_when_checkpoint_missing(monkeypatch) -> None:
    _app()
    step = MaskStep(Path.cwd())
    calls: list[str] = []
    step.person_backend_combo.setCurrentIndex(2)

    monkeypatch.setattr(step, "_confirm_sky_license_notice", lambda: True)

    def fake_ensure() -> bool:
        calls.append("download")
        return True

    monkeypatch.setattr(step, "_ensure_sam31_checkpoint_available", fake_ensure)

    assert step.confirm_commands([("yolo", ["cmd"])])
    assert calls == ["download"]


def test_mask_step_primary_mask_generation_is_always_selected(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)

    assert step.primary_action_enabled()
    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["yolo"]


def test_mask_step_custom_checkbox_opens_picker_and_cancels_cleanly(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    monkeypatch.setattr(step3_mask_module.QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: ("", ""))

    step.run_custom_cb.setChecked(True)

    assert not step.run_custom_cb.isChecked()
    assert step.custom_mask_path_label.text() == i18n.t("CUSTOM_MASK_NOT_SELECTED")


def test_mask_step_custom_mask_builds_final_command(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    custom_path = scene / "custom.png"
    cv2.imwrite(str(custom_path), np.full((8, 8), 255, dtype=np.uint8))
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)

    step._set_custom_mask_path(custom_path)
    commands = step.build_commands()

    assert step.run_custom_cb.isChecked()
    assert [phase for phase, _cmd in commands] == ["yolo", "custom"]
    assert commands[1] == (
        "custom",
        [
            sys.executable,
            "-u",
            str(Path.cwd() / "custom_mask.py"),
            str(scene / "images"),
            str(scene / "masks"),
            str(custom_path),
        ],
    )


def test_mask_step_mask2former_primary_builds_final_command(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)
    step.person_backend_combo.setCurrentIndex(1)

    commands = step.build_commands()

    assert commands[0][0] == "yolo"
    cmd = commands[0][1]
    assert cmd[:5] == [
        sys.executable,
        "-u",
        str(Path.cwd() / "sky_mask.py"),
        str(scene / "images"),
        str(scene / "masks"),
    ]
    assert cmd[cmd.index("--backend") + 1] == "mask2former"
    assert cmd[cmd.index("--projection") + 1] == "equirect"
    assert cmd[cmd.index("--quality") + 1] == "high"
    assert cmd[cmd.index("--inference-size") + 1] == "768"
    assert cmd[cmd.index("--labels") + 1] == "sky,person"
    assert "--replace" in cmd


def test_mask_step_sam31_primary_builds_prompt_command(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    checkpoint = scene / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(step, "_sam31_checkpoint_path", lambda: checkpoint)
    step._update_person_backend_availability()
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)
    step.person_backend_combo.setCurrentIndex(2)

    commands = step.build_commands()

    cmd = commands[0][1]
    assert cmd[cmd.index("--backend") + 1] == "sam31"
    assert cmd[cmd.index("--quality") + 1] == "high"
    assert cmd[cmd.index("--inference-size") + 1] == "1008"
    assert cmd[cmd.index("--merge-mode") + 1] == "replace"
    assert "--safe-batch" in cmd
    prompt_args = [cmd[idx + 1] for idx, value in enumerate(cmd) if value == "--sam-prompt"]
    assert prompt_args == ["person", "sky"]


def test_mask_step_sam31_prompts_trim_separator_spaces() -> None:
    assert MaskStep._split_sam_prompt_text(" tripod, hand; selfie stick ") == [
        "tripod",
        "hand",
        "selfie stick",
    ]


def test_mask_step_sam31_add_mode_builds_subtract_prompt_command(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    checkpoint = scene / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(step, "_sam31_checkpoint_path", lambda: checkpoint)
    step._update_person_backend_availability()
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)
    step.person_backend_combo.setCurrentIndex(2)
    step.sam_apply_mode_combo.setCurrentIndex(1)
    step.sam_subtract_prompt_edit.setText("male icon, female icon")

    commands = step.build_commands()

    cmd = commands[0][1]
    assert cmd[cmd.index("--merge-mode") + 1] == "add"
    assert "--replace" not in cmd
    subtract_args = [cmd[idx + 1] for idx, value in enumerate(cmd) if value == "--subtract-sam-prompt"]
    assert subtract_args == ["male icon", "female icon"]


def test_mask_step_sam31_preview_add_mode_seeds_existing_mask(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    existing = np.zeros((16, 32), dtype=np.uint8)
    cv2.imwrite(str(masks / "frame_0001.png"), existing)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    checkpoint = scene / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(step, "_sam31_checkpoint_path", lambda: checkpoint)
    step._update_person_backend_availability()
    step.person_backend_combo.setCurrentIndex(2)
    step.sam_apply_mode_combo.setCurrentIndex(1)
    preview_mask = tmp_path / "preview" / "frame_0001.png"

    step._seed_sam31_preview_base_mask(image_path, preview_mask)

    assert preview_mask.is_file()


def test_mask_step_sky_top_connected_is_opt_in(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)
    step.person_backend_combo.setCurrentIndex(1)
    step.sky_top_connected_cb.setChecked(True)

    commands = step.build_commands()

    cmd = commands[0][1]
    assert "--top-connected" in cmd
    assert "--no-top-connected" not in cmd


def test_mask_step_person_mask_can_select_sam31_backend(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    checkpoint = scene / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(step, "_sam31_checkpoint_path", lambda: checkpoint)
    step._update_person_backend_availability()

    step.person_backend_combo.setCurrentIndex(2)
    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo"]
    cmd = commands[0][1]
    assert cmd[:5] == [
        sys.executable,
        "-u",
        str(Path.cwd() / "sky_mask.py"),
        str(scene / "images"),
        str(scene / "masks"),
    ]
    assert cmd[cmd.index("--backend") + 1] == "sam31"
    assert cmd[cmd.index("--quality") + 1] == "high"
    assert cmd[cmd.index("--inference-size") + 1] == "1008"
    prompt_args = [cmd[idx + 1] for idx, value in enumerate(cmd) if value == "--sam-prompt"]
    assert prompt_args == ["person", "sky"]
    assert cmd[cmd.index("--min-score") + 1] == "0.5"
    assert "--top-connected" not in cmd
    assert "--replace" in cmd
    assert "--safe-batch" in cmd
    assert step.yolo_level_combo.isEnabled()
    assert not step.yolo_bottom_enhance_combo.isEnabled()
    assert not step.yolo_class_list_section.isEnabled()
    assert step.yolo_expand_edit.isEnabled()
    assert not step.sam_prompt_section.isHidden()


def test_mask_step_sam31_preview_does_not_use_safe_batch(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    checkpoint = scene / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(step, "_sam31_checkpoint_path", lambda: checkpoint)
    step._update_person_backend_availability()
    step.person_backend_combo.setCurrentIndex(2)

    cmd = step._build_yolo_preview_cmd(image_path, masks)

    assert cmd[cmd.index("--backend") + 1] == "sam31"
    assert "--safe-batch" not in cmd


def test_mask_step_stitch_runs_after_primary_mask(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(True)
    step.run_overexp_cb.setChecked(False)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo", "stitch"]
    assert commands[1][1][:4] == [
        sys.executable,
        "-u",
        str(Path.cwd() / "stitch_mask.py"),
        str(scene / "masks"),
    ]


def test_mask_step_overexposure_runs_after_primary_mask(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(True)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo", "overexposure"]
    assert "--replace" not in commands[1][1]


def test_mask_step_allows_generation_when_drop_images_are_removed(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    commands = step.build_commands()

    assert commands
    assert commands[0][0] == "yolo"
    assert commands[0][1][3] == str(scene / "images")
    assert commands[0][1][4] == str(scene / "masks")
    assert commands[0][1][commands[0][1].index("--quality") + 1] == "high"
    assert commands[0][1][commands[0][1].index("--projection") + 1] == "equirect"
    assert "--add-ext" not in commands[0][1]


def test_mask_step_current_reprocess_command_targets_preview_image_subfolder(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images" / "extra"
    images.mkdir(parents=True)
    image_path = images / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    cmd = step._build_yolo_current_cmd(image_path)

    assert cmd[3] == str(image_path)
    assert cmd[4] == str(scene / "masks" / "extra")
    assert step._mask_output_path_for_image(image_path) == scene / "masks" / "extra" / "frame_0001.png"
    assert "--add-ext" not in cmd


def test_mask_step_current_reprocess_external_commands_target_preview_image_subfolder(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images" / "extra"
    images.mkdir(parents=True)
    image_path = images / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    commands = step._build_current_reprocess_external_commands(image_path)

    assert [phase for phase, _cmd in commands] == ["yolo"]
    assert commands[0][1][3] == str(image_path)
    assert commands[0][1][4] == str(scene / "masks" / "extra")


def test_mask_step_current_reprocess_mask2former_uses_primary_replace(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    image_path = images / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.person_backend_combo.setCurrentIndex(1)

    commands = step._build_current_reprocess_external_commands(image_path)

    assert [phase for phase, _cmd in commands] == ["yolo"]
    assert commands[0][1][commands[0][1].index("--backend") + 1] == "mask2former"
    assert "--replace" in commands[0][1]


def test_mask_step_current_reprocess_can_apply_overexposure_only(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    image_path = images / "frame_0001.png"
    image = np.full((24, 32, 3), 120, dtype=np.uint8)
    image[4:8, 6:10] = 255
    cv2.imwrite(str(image_path), image)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(True)

    mask_path = step._mask_output_path_for_image(image_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), np.zeros((24, 32), dtype=np.uint8))
    step._apply_current_image_postprocess(image_path, mask_path, replace=True)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask[5, 7] == 0
    assert mask[12, 16] == 255


def test_mask_step_current_reprocess_can_apply_custom_only(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    image_path = images / "frame_0001.png"
    cv2.imwrite(str(image_path), np.full((24, 32, 3), 120, dtype=np.uint8))
    custom_path = scene / "custom.png"
    custom = np.full((24, 32), 255, dtype=np.uint8)
    custom[5:10, 6:12] = 0
    cv2.imwrite(str(custom_path), custom)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)
    step._set_custom_mask_path(custom_path)

    mask_path = step._mask_output_path_for_image(image_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), np.zeros((24, 32), dtype=np.uint8))
    step._apply_current_image_postprocess(image_path, mask_path, replace=True)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask[7, 8] == 0
    assert mask[12, 16] == 255


def test_mask_step_reprocesses_selected_thumbnail_images_only(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    image_paths: list[Path] = []
    for idx in range(3):
        image_path = images / f"frame_{idx:04d}.png"
        image = np.full((24, 32, 3), 120, dtype=np.uint8)
        image[4 + idx : 8 + idx, 6:10] = 255
        cv2.imwrite(str(image_path), image)
        image_paths.append(image_path)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(True)
    step.mask_preview.set_preview_mode("thumbnails")
    monkeypatch.setattr(step, "_confirm_yolo_sam_license_notice", lambda: True)
    monkeypatch.setattr(step, "_build_current_reprocess_external_commands", lambda _image_path: [])

    selection = step.mask_preview.thumbnail_view.selectionModel()
    selection.select(step.mask_preview.thumbnail_model.index(0, 0), QItemSelectionModel.ClearAndSelect)
    selection.select(step.mask_preview.thumbnail_model.index(2, 0), QItemSelectionModel.Select)

    step._run_current_image_reprocess()
    _process_events_until(lambda: not step._current_reprocess_active)
    step.mask_preview.thumbnail_model._pool.waitForDone(5000)

    first_mask = masks / "frame_0000.png"
    middle_mask = masks / "frame_0001.png"
    last_mask = masks / "frame_0002.png"
    assert first_mask.is_file()
    assert not middle_mask.exists()
    assert last_mask.is_file()
    assert cv2.imread(str(first_mask), cv2.IMREAD_GRAYSCALE) is not None
    assert step.mask_preview.status_label.text() == i18n.t("MASK_REPROCESS_SELECTED_DONE").format(
        done=2,
        total=2,
    )


def test_mask_step_normal_image_type_disables_stitch_and_uses_normal_yolo_projection(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"keep")
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    step._set_projection("normal")
    step.run_stitch_cb.setChecked(True)

    assert step._projection() == "normal"
    assert not step.run_stitch_cb.isChecked()
    assert not step.run_stitch_cb.isEnabled()
    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo"]
    yolo_cmd = commands[0][1]
    assert yolo_cmd[yolo_cmd.index("--projection") + 1] == "normal"
    assert yolo_cmd[yolo_cmd.index("--quality") + 1] == "standard"
    assert "--bottom-conf" not in yolo_cmd
    assert not step.yolo_bottom_enhance_combo.isEnabled()


def test_mask_step_mixed_image_type_splits_commands_by_manifest(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    equirect = images / "pano.jpg"
    normal = images / "normal.jpg"
    cv2.imwrite(str(equirect), np.full((32, 64, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(normal), np.full((32, 32, 3), 120, dtype=np.uint8))
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step.run_stitch_cb.setChecked(True)
    commands = step.build_commands()

    assert step._projection_mixed
    assert [phase for phase, _cmd in commands] == ["yolo_equirect", "yolo_normal", "stitch_equirect"]
    yolo_equirect = commands[0][1]
    yolo_normal = commands[1][1]
    stitch = commands[2][1]
    assert yolo_equirect[yolo_equirect.index("--projection") + 1] == "equirect"
    assert yolo_normal[yolo_normal.index("--projection") + 1] == "normal"
    equirect_manifest = Path(yolo_equirect[yolo_equirect.index("--image-list") + 1])
    normal_manifest = Path(yolo_normal[yolo_normal.index("--image-list") + 1])
    stitch_manifest = Path(stitch[stitch.index("--image-list") + 1])
    assert stitch_manifest == equirect_manifest
    assert "images/pano.jpg" in equirect_manifest.read_text(encoding="utf-8")
    assert "images/normal.jpg" not in equirect_manifest.read_text(encoding="utf-8")
    assert "images/normal.jpg" in normal_manifest.read_text(encoding="utf-8")
    assert "images/pano.jpg" not in normal_manifest.read_text(encoding="utf-8")


def test_mask_step_mixed_custom_mask_uses_all_target_manifest(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    equirect = images / "pano.jpg"
    normal = images / "normal.jpg"
    custom = scene / "custom.png"
    cv2.imwrite(str(equirect), np.full((32, 64, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(normal), np.full((32, 32, 3), 120, dtype=np.uint8))
    cv2.imwrite(str(custom), np.full((32, 64), 255, dtype=np.uint8))
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step._set_custom_mask_path(custom)
    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo_equirect", "yolo_normal", "custom"]
    custom_cmd = commands[2][1]
    custom_manifest = Path(custom_cmd[custom_cmd.index("--image-list") + 1])
    text = custom_manifest.read_text(encoding="utf-8")
    assert "images/pano.jpg" in text
    assert "images/normal.jpg" in text


def test_mask_step_quality_best_is_forwarded_to_primary_command(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step.yolo_level_combo.setCurrentIndex(2)

    yolo_cmd = step.build_commands()[0][1]

    assert yolo_cmd[yolo_cmd.index("--quality") + 1] == "best"
    assert "--bottom-conf" not in yolo_cmd
    assert "--bottom-tta-rotations" not in yolo_cmd
    assert "--bottom-filter" not in yolo_cmd


def test_mask_step_quality_is_shared_by_mask2former(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.person_backend_combo.setCurrentIndex(1)
    step.yolo_level_combo.setCurrentIndex(0)

    yolo_cmd = step.build_commands()[0][1]

    assert yolo_cmd[yolo_cmd.index("--backend") + 1] == "mask2former"
    assert yolo_cmd[yolo_cmd.index("--quality") + 1] == "standard"


def test_mask_step_image_folder_row_has_no_registration_controls() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert not hasattr(step, "projection_label")
    assert not hasattr(step, "add_external_images_btn")
    assert not hasattr(step, "open_images_dir_btn")


def test_mask_step_equirect_image_type_can_use_stitch(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step.run_stitch_cb.setChecked(True)
    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo", "stitch"]


def test_mask_step_progress_uses_completed_file_counts() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.on_line("[progress] 0/3") == (0, 3)
    assert step.on_line("Processing: frame_0001.jpg") is None
    assert step.on_line("Processed: frame_0001.jpg") == (1, 3)
    assert step.on_line("[progress] 2/3") == (2, 3)
    assert step.on_line("[progress] 3/3") == (3, 3)


def test_mask_step_rejects_generation_when_untracked_images_remain(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    (scene / "images" / "old_frame.jpg").write_bytes(b"stale")
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    with pytest.raises(ValueError, match="selected_frames.csv"):
        step.build_commands()
