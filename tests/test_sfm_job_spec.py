from __future__ import annotations

from pathlib import Path

import pytest

from core.sfm_job_spec import (
    JOB_KIND_COLMAP_MIXED_PROJECT,
    colmap_mixed_project_job,
    load_sfm_job,
    write_sfm_job,
)


def test_colmap_mixed_project_job_round_trips(tmp_path: Path) -> None:
    payload = colmap_mixed_project_job(
        scene_dir=tmp_path / "scene",
        output_dir=tmp_path / "scene" / "output",
        views=[{"name": "front", "yaw": 0.0, "pitch": 0.0, "enabled": True}],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="8",
        jpg_quality=95,
        write_images=True,
        write_masks=True,
        invert_masks=False,
        workers="auto",
        remap_cache_limit="auto",
    )

    path = write_sfm_job(tmp_path / "job.json", payload)
    loaded = load_sfm_job(path, expected_kind=JOB_KIND_COLMAP_MIXED_PROJECT)

    assert loaded["kind"] == JOB_KIND_COLMAP_MIXED_PROJECT
    assert loaded["views"][0]["name"] == "front"
    assert loaded["output_scale"] == 0.5


def test_sfm_job_rejects_wrong_kind(tmp_path: Path) -> None:
    payload = colmap_mixed_project_job(
        scene_dir=tmp_path / "scene",
        output_dir=tmp_path / "scene" / "output",
        views=[],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="8",
        jpg_quality=95,
        write_images=True,
        write_masks=True,
        invert_masks=False,
        workers="auto",
        remap_cache_limit="auto",
    )
    payload["kind"] = "other"

    with pytest.raises(ValueError, match="Unsupported SfM job kind"):
        write_sfm_job(tmp_path / "job.json", payload)


def test_sfm_job_rejects_no_enabled_views(tmp_path: Path) -> None:
    payload = colmap_mixed_project_job(
        scene_dir=tmp_path / "scene",
        output_dir=tmp_path / "scene" / "output",
        views=[{"name": "front", "yaw": 0.0, "pitch": 0.0, "enabled": False}],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="8",
        jpg_quality=95,
        write_images=True,
        write_masks=True,
        invert_masks=False,
        workers="auto",
        remap_cache_limit="auto",
    )

    with pytest.raises(ValueError, match="enabled view"):
        write_sfm_job(tmp_path / "job.json", payload)


def test_sfm_job_rejects_invalid_quality(tmp_path: Path) -> None:
    payload = colmap_mixed_project_job(
        scene_dir=tmp_path / "scene",
        output_dir=tmp_path / "scene" / "output",
        views=[{"name": "front", "yaw": 0.0, "pitch": 0.0, "enabled": True}],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="8",
        jpg_quality=0,
        write_images=True,
        write_masks=True,
        invert_masks=False,
        workers="auto",
        remap_cache_limit="auto",
    )

    with pytest.raises(ValueError, match="jpg_quality"):
        write_sfm_job(tmp_path / "job.json", payload)
