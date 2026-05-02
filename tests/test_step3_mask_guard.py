from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QToolButton

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.steps.base_step import SETTINGS_PANE_MARGINS, SETTINGS_PANE_WIDTH
from gui.steps.step3_mask import MaskStep


def _app():
    return QApplication.instance() or QApplication([])


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
    assert step.yolo_level_label.toolTip() == i18n.tip("YOLO_LEVEL")
    assert step.yolo_expand_label.toolTip() == i18n.tip("YOLO_EXPAND")


def test_mask_step_metashape_notice_is_in_left_pane() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step.metashape_notice.text() == i18n.METASHAPE_NOTICE
    assert not step.preview_pane.isAncestorOf(step.metashape_notice)


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


def test_mask_step_disables_generation_without_selected_frames_csv(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"keep")
    step = MaskStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))

    assert not step.primary_action_enabled()
    assert step.primary_action_tooltip() == i18n.t("MASK_READY_NO_CSV")


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
    assert "--add-ext" not in commands[0][1]


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
