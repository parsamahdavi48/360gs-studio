from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import cv2
import numpy as np
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QToolButton

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
import gui.steps.step3_mask as step3_mask_module
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
        {"seq": "2", "output_file": "images/frame_0002.jpg", "decision": "drop", "status": "thinned"},
    ]
    with (scene / "selected_frames.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return scene


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


def test_mask_step_yolo_level_and_expand_share_compact_row() -> None:
    _app()
    step = MaskStep(Path.cwd())

    content_width = SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2]
    assert step.yolo_settings_row.sizeHint().width() <= content_width
    assert step.yolo_level_combo.currentIndex() == 2
    assert step.yolo_level_label.toolTip() == i18n.tip("YOLO_LEVEL")
    assert step.yolo_expand_label.toolTip() == i18n.tip("YOLO_EXPAND")
    assert step.yolo_bottom_enhance_label.toolTip() == i18n.tip("YOLO_BOTTOM_ENHANCE")
    assert step.projection_buttons["equirect"].text() == "360°"
    assert step.yolo_bottom_enhance_combo.itemText(1) == i18n.t("YOLO_BOTTOM_STRONG")
    assert step.yolo_bottom_enhance_combo.itemText(2) == i18n.t("YOLO_BOTTOM_MAX")


def test_mask_step_metashape_notice_is_in_left_pane() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.metashape_notice.text() == i18n.METASHAPE_NOTICE
    assert not step.preview_pane.isAncestorOf(step.metashape_notice)


def test_mask_step_preview_mode_toolbar_is_in_preview_header() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.preview_pane.isAncestorOf(step.mask_preview.mode_toolbar)
    assert not step.mask_preview.isAncestorOf(step.mask_preview.mode_toolbar)


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
    with (tmp_path / "selected_frames.csv").open("w", encoding="utf-8", newline="") as f:
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
    assert step.mask_preview.set_yolo_preview_mask(image_path, temp_mask_path)
    step._render_mask_preview()
    assert i18n.t("MASK_PREVIEW_YOLO_TEMP") in step.mask_preview.status_label.text()

    existing_mask = np.full((32, 64), 255, dtype=np.uint8)
    existing_mask[:, :32] = 0
    cv2.imwrite(str(existing_mask_path), existing_mask)

    step.on_queue_finished(True)

    assert i18n.t("MASK_PREVIEW_YOLO_EXISTING") in step.mask_preview.status_label.text()
    assert i18n.t("MASK_PREVIEW_YOLO_TEMP") not in step.mask_preview.status_label.text()
    assert step.mask_preview.image_label._source_pixmap is not None


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
    (tmp_path / "selected_frames.csv").write_text("seq,output_file,decision,status\n", encoding="utf-8")
    step = MaskStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))

    assert not step.primary_action_enabled()
    assert step.primary_action_tooltip() == i18n.t("MASK_READY_NO_IMAGES_DIR")
    assert step.ready_status_label.text() == i18n.t("MASK_READY_NO_IMAGES_DIR")
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
    assert step.ready_status_label.text() == i18n.t("MASK_READY_EXTERNAL_IMAGES")
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


def test_mask_step_disables_generation_when_no_mask_task_selected(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step.run_yolo_cb.setChecked(False)
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)

    assert not step.primary_action_enabled()
    assert step.primary_action_tooltip() == i18n.t("MASK_TASK_REQUIRED")


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
    step.run_yolo_cb.setChecked(False)
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)

    step._set_custom_mask_path(custom_path)
    commands = step.build_commands()

    assert step.run_custom_cb.isChecked()
    assert commands == [
        (
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
    ]


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
    assert commands[0][1][commands[0][1].index("--level") + 1] == "2"
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
    step.run_yolo_cb.setChecked(False)
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(True)

    mask_path = step._mask_output_path_for_image(image_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    step._apply_current_image_postprocess(image_path, mask_path)

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
    step.run_yolo_cb.setChecked(False)
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(False)
    step._set_custom_mask_path(custom_path)

    mask_path = step._mask_output_path_for_image(image_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    step._apply_current_image_postprocess(image_path, mask_path)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask[7, 8] == 0
    assert mask[12, 16] == 255


def test_mask_step_reprocesses_selected_thumbnail_images_only(tmp_path: Path) -> None:
    _app()
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    image_paths: list[Path] = []
    for idx in range(3):
        image_path = images / f"frame_{idx:04d}.png"
        image = np.full((24, 32, 3), 120, dtype=np.uint8)
        image[4 + idx:8 + idx, 6:10] = 255
        cv2.imwrite(str(image_path), image)
        image_paths.append(image_path)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))
    step.run_yolo_cb.setChecked(False)
    step.run_stitch_cb.setChecked(False)
    step.run_overexp_cb.setChecked(True)
    step.mask_preview.set_preview_mode("thumbnails")

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

    assert step.projection_buttons["normal"].isChecked()
    assert not step.run_stitch_cb.isChecked()
    assert not step.run_stitch_cb.isEnabled()
    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["yolo"]
    yolo_cmd = commands[0][1]
    assert yolo_cmd[yolo_cmd.index("--projection") + 1] == "normal"
    assert yolo_cmd[yolo_cmd.index("--level") + 1] == "1"
    assert "--bottom-conf" not in yolo_cmd
    assert not step.yolo_bottom_enhance_combo.isEnabled()


def test_mask_step_bottom_enhance_strong_adds_bottom_only_yolo_args(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step.yolo_bottom_enhance_combo.setCurrentIndex(1)

    yolo_cmd = step.build_commands()[0][1]

    assert yolo_cmd[yolo_cmd.index("--bottom-conf") + 1] == "0.15"
    assert yolo_cmd[yolo_cmd.index("--bottom-tta-rotations") + 1] == "4"
    assert "--bottom-filter" in yolo_cmd
    assert "--bottom-temporal-window" not in yolo_cmd
    assert "--bottom-temporal-min-votes" not in yolo_cmd
    assert "--bottom-model" not in yolo_cmd


def test_mask_step_bottom_enhance_max_uses_bottom_x_model(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    step.yolo_bottom_enhance_combo.setCurrentIndex(2)

    yolo_cmd = step.build_commands()[0][1]

    assert yolo_cmd[yolo_cmd.index("--bottom-conf") + 1] == "0.10"
    assert yolo_cmd[yolo_cmd.index("--bottom-tta-rotations") + 1] == "4"
    assert yolo_cmd[yolo_cmd.index("--bottom-model") + 1] == "x"
    assert "--bottom-filter" in yolo_cmd
    assert "--bottom-temporal-window" not in yolo_cmd
    assert "--bottom-temporal-min-votes" not in yolo_cmd


def test_mask_step_external_image_controls_only_show_for_normal_type() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.external_images_panel.isHidden()

    step._set_projection("normal")
    assert not step.external_images_panel.isHidden()

    step._set_projection("equirect")
    assert step.external_images_panel.isHidden()


def test_mask_step_imports_external_images_into_scene_images(tmp_path: Path) -> None:
    _app()
    source = tmp_path / "source"
    source.mkdir()
    cv2.imwrite(str(source / "a.JPG"), np.full((8, 8, 3), 64, dtype=np.uint8))
    cv2.imwrite(str(source / "b.png"), np.full((8, 8, 3), 128, dtype=np.uint8))
    (source / "ignore.txt").write_text("not an image", encoding="utf-8")
    scene = tmp_path / "scene"
    scene.mkdir()
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    added, skipped = step._import_external_images_from_dir(source)

    assert added == 2
    assert skipped == 0
    assert (scene / "images" / "a.JPG").is_file()
    assert (scene / "images" / "b.png").is_file()
    assert not (scene / "images" / "ignore.txt").exists()
    assert step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("MASK_READY_EXTERNAL_IMAGES")

    added_again, skipped_again = step._import_external_images_from_dir(source)

    assert added_again == 0
    assert skipped_again == 2


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
