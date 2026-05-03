import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from cubemap_transforms_json import count_planned_outputs
from gui.steps.step4_cubemap import CubemapStep


def _app():
    return QApplication.instance() or QApplication([])


def test_cubemap_progress_parses_explicit_file_counts(tmp_path: Path) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.on_line("Converting 6 files...") == (0, 6)
    assert step.on_line("[progress] 0/6") == (0, 6)
    assert step.on_line("Processing: frame_0001.png") is None
    assert step.on_line("[progress] 4/6") == (4, 6)
    assert step.on_line("[progress] 6/6") == (6, 6)


def test_cubemap_progress_keeps_legacy_processing_fallback(tmp_path: Path) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.on_line("Converting 2 images...") == (0, 2)
    assert step.on_line("Processing: frame_0001.png") == (1, 2)
    assert step.on_line("Processing: frame_0002.png") == (2, 2)


def test_cubemap_progress_total_includes_images_and_masks(tmp_path: Path) -> None:
    image_dir = tmp_path
    images = image_dir / "images"
    masks = image_dir / "masks"
    images.mkdir()
    masks.mkdir()

    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((4, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(images / "frame_0002.png"), np.zeros((4, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(masks / "frame_0001.png"), np.zeros((4, 8), dtype=np.uint8))
    views = [
        {"name": "front", "yaw": 0.0, "pitch": 0.0},
        {"name": "right", "yaw": 90.0, "pitch": 0.0},
    ]

    total = count_planned_outputs(
        image_files=["images/frame_0001.png", "images/frame_0002.png"],
        views=views,
        image_dir=str(image_dir),
        mask_dir=str(masks),
        mask_from_alpha=False,
    )

    assert total == 6


def test_cubemap_progress_total_can_count_masks_only(tmp_path: Path) -> None:
    image_dir = tmp_path
    images = image_dir / "images"
    masks = image_dir / "masks"
    images.mkdir()
    masks.mkdir()

    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((4, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(masks / "frame_0001.png"), np.zeros((4, 8), dtype=np.uint8))
    views = [
        {"name": "front", "yaw": 0.0, "pitch": 0.0},
        {"name": "right", "yaw": 90.0, "pitch": 0.0},
    ]

    total = count_planned_outputs(
        image_files=["images/frame_0001.png"],
        views=views,
        image_dir=str(image_dir),
        mask_dir=str(masks),
        mask_from_alpha=False,
        export_images=False,
        export_masks=True,
    )

    assert total == 2


def test_cubemap_progress_total_includes_alpha_masks(tmp_path: Path) -> None:
    image_dir = tmp_path
    images = image_dir / "images"
    images.mkdir()

    rgba = np.zeros((4, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    cv2.imwrite(str(images / "frame_0001.png"), rgba)
    views = [
        {"name": "front", "yaw": 0.0, "pitch": 0.0},
        {"name": "right", "yaw": 90.0, "pitch": 0.0},
    ]

    total = count_planned_outputs(
        image_files=["images/frame_0001.png"],
        views=views,
        image_dir=str(image_dir),
        mask_dir=str(image_dir / "masks"),
        mask_from_alpha=True,
    )

    assert total == 4
