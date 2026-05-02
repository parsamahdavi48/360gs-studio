from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

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


def test_mask_step_allows_generation_when_drop_images_are_removed(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    commands = step.build_commands()

    assert commands
    assert commands[0][0] == "yolo"


def test_mask_step_rejects_generation_when_untracked_images_remain(tmp_path: Path) -> None:
    _app()
    scene = _write_scene(tmp_path, drop_exists=False)
    (scene / "images" / "old_frame.jpg").write_bytes(b"stale")
    step = MaskStep(Path.cwd())
    step.set_scene_dir(str(scene))

    with pytest.raises(ValueError, match="selected_frames.csv"):
        step.build_commands()
