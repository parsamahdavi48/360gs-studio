from __future__ import annotations

from pathlib import Path

import pytest

from core.dataset_job_spec import (
    JOB_KIND_METASHAPE_COLMAP,
    JOB_KIND_METASHAPE_NERF,
    JOB_KIND_REALITYSCAN_LFS_COLMAP,
    load_dataset_job,
    metashape_colmap_job,
    metashape_nerf_job,
    realityscan_lfs_colmap_job,
    write_dataset_job,
)


def test_metashape_dataset_job_round_trips(tmp_path: Path) -> None:
    job = metashape_colmap_job(
        scene_dir=tmp_path / "scene",
        images_dir=tmp_path / "scene" / "images",
        masks_dir=None,
        xml_path=tmp_path / "scene" / "cameras.xml",
        ply_path=None,
        output_dir=tmp_path / "scene" / "output" / "metashape_colmap",
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0, "enabled": True}],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="source",
        jpg_quality=88,
        undistort_alpha=1.0,
    )

    path = write_dataset_job(tmp_path / "job.json", job)
    loaded = load_dataset_job(path, expected_kind=JOB_KIND_METASHAPE_COLMAP)

    assert loaded["kind"] == JOB_KIND_METASHAPE_COLMAP
    assert loaded["views"] == [{"name": "pz", "yaw": 0.0, "pitch": 0.0, "enabled": True}]
    assert loaded["output_bit_depth"] == "source"
    assert loaded["jpg_quality"] == 88
    assert loaded["axis_transform"] == "none"
    assert loaded["final_orientation"] == "none"


def test_realityscan_dataset_job_rejects_wrong_kind(tmp_path: Path) -> None:
    job = realityscan_lfs_colmap_job(
        csv_path=tmp_path / "rs.csv",
        output_dir=tmp_path / "lfs_colmap",
        images_dir=tmp_path / "images",
        masks_dir=None,
        ply_path=None,
        skip_missing_images=False,
        pre_undistort_distorted_images=True,
        undistort_alpha=1.0,
    )
    path = write_dataset_job(tmp_path / "job.json", job)

    with pytest.raises(ValueError):
        load_dataset_job(path, expected_kind=JOB_KIND_METASHAPE_COLMAP)

    assert load_dataset_job(path, expected_kind=JOB_KIND_REALITYSCAN_LFS_COLMAP)["kind"] == JOB_KIND_REALITYSCAN_LFS_COLMAP


def test_metashape_nerf_dataset_job_round_trips(tmp_path: Path) -> None:
    job = metashape_nerf_job(
        scene_dir=tmp_path / "scene",
        images_dir=tmp_path / "scene" / "images",
        masks_dir=tmp_path / "scene" / "masks",
        xml_path=tmp_path / "scene" / "cameras.xml",
        ply_path=tmp_path / "scene" / "points.ply",
        output_dir=tmp_path / "scene" / "output" / "metashape_cubemap",
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0, "enabled": True}],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="8",
        jpg_quality=95,
        undistort_alpha=1.0,
        axis_transform="none",
        final_orientation="lichtfeld",
        write_images=False,
        write_masks=False,
    )

    path = write_dataset_job(tmp_path / "nerf_job.json", job)
    loaded = load_dataset_job(path, expected_kind=JOB_KIND_METASHAPE_NERF)

    assert loaded["kind"] == JOB_KIND_METASHAPE_NERF
    assert loaded["final_orientation"] == "lichtfeld"
    assert loaded["write_images"] is False
    assert loaded["write_masks"] is False


def test_dataset_job_rejects_view_without_orientation(tmp_path: Path) -> None:
    job = metashape_colmap_job(
        scene_dir=tmp_path / "scene",
        images_dir=tmp_path / "scene" / "images",
        masks_dir=None,
        xml_path=tmp_path / "scene" / "cameras.xml",
        ply_path=None,
        output_dir=tmp_path / "scene" / "output" / "metashape_colmap",
        views=[{"name": "pz", "enabled": True}],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="source",
        jpg_quality=88,
        undistort_alpha=1.0,
    )

    with pytest.raises(ValueError, match=r"views\[0\]\.yaw"):
        write_dataset_job(tmp_path / "job.json", job)


def test_dataset_job_rejects_invalid_numeric_range(tmp_path: Path) -> None:
    job = realityscan_lfs_colmap_job(
        csv_path=tmp_path / "rs.csv",
        output_dir=tmp_path / "lfs_colmap",
        images_dir=tmp_path / "images",
        masks_dir=None,
        ply_path=None,
        skip_missing_images=False,
        pre_undistort_distorted_images=True,
        undistort_alpha=1.5,
    )

    with pytest.raises(ValueError, match="undistort_alpha"):
        write_dataset_job(tmp_path / "job.json", job)
