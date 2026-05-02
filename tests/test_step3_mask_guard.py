from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from gui import i18n
from gui.common.browse_widget import BrowseWidget
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


def test_mask_step_rejects_generation_when_untracked_images_remain(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    (scene / "images" / "old_frame.jpg").write_bytes(b"stale")
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    with pytest.raises(ValueError, match="selected_frames.csv"):
        step.build_commands()
