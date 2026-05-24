import os
from concurrent.futures import Future
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from core.cubemap_transforms_json import _run_bounded_conversion_jobs, count_planned_outputs
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


def test_colmap_progress_initializes_feature_phase_from_rig_images(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "output" / "colmap_rig" / "images"
    images.mkdir(parents=True)
    (images / "frame_0001_front.jpg").write_bytes(b"dummy")
    (images / "frame_0001_right.png").write_bytes(b"dummy")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.on_phase_started("colmap_feature") == (0, 2)
    assert step.on_phase_started("colmap_match") == (0, 0)


def test_colmap_progress_parses_feature_matching_and_global_mapper_logs(tmp_path: Path) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.on_line("I feature_extraction.cc:258] Processed file [2/11]") == (2, 11)
    assert step.on_line("I feature_matching.cc:231] Matching image [3/16] in 0.1s") == (3, 16)
    assert step.on_line("I pairing.cc:201] Matching block [1/2, 2/2]") == (2, 4)
    assert step.on_line(
        "I global_mapper.cc:325] Global bundle adjustment iteration 1 / 3, fixed-rotation stage finished"
    ) == (1, 8)
    assert step.on_line("I global_mapper.cc:335] Global bundle adjustment iteration 1 / 3 finished") == (2, 8)
    assert step.on_line("I global_mapper.cc:335] Global bundle adjustment iteration 3 / 3 finished") == (6, 8)
    assert step.on_line(
        "I global_mapper.cc:519] === Running iterative retriangulation and refinement ==="
    ) == (7, 8)
    assert step.on_line(
        "I global_mapper.cc:528] Iterative retriangulation and refinement done in 155.894 seconds"
    ) == (8, 8)
    assert step.on_line("I global_pipeline.cc:110] Reconstruction done in 502.287 seconds") == (8, 8)


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


def test_bounded_conversion_jobs_consumes_results_before_submitting_all(capsys) -> None:
    submitted: list[int] = []
    result_submit_counts: list[int] = []

    class CountingFuture(Future):
        def result(self, timeout=None):
            result_submit_counts.append(len(submitted))
            return super().result(timeout)

    def submit_job(job: int):
        submitted.append(job)
        future = CountingFuture()
        future.set_result(1)
        return future

    _run_bounded_conversion_jobs(
        range(7),
        submit_job,
        lambda job: str(job),
        total_outputs=7,
        max_workers=1,
        failure_context="test conversion failed",
    )

    assert submitted == list(range(7))
    assert result_submit_counts[0] == 2
    assert "7/7" in capsys.readouterr().out
