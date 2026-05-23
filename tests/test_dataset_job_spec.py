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
        views=[{"name": "pz", "enabled": True}],
        output_scale=0.5,
        output_format="jpg",
        undistort_alpha=1.0,
    )

    path = write_dataset_job(tmp_path / "job.json", job)
    loaded = load_dataset_job(path, expected_kind=JOB_KIND_METASHAPE_COLMAP)

    assert loaded["kind"] == JOB_KIND_METASHAPE_COLMAP
    assert loaded["views"] == [{"name": "pz", "enabled": True}]


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
        views=[{"name": "pz", "enabled": True}],
        output_scale=0.5,
        output_format="jpg",
        output_bit_depth="8",
        jpg_quality=95,
        undistort_alpha=1.0,
        axis_transform="none",
        final_orientation="lichtfeld",
    )

    path = write_dataset_job(tmp_path / "nerf_job.json", job)
    loaded = load_dataset_job(path, expected_kind=JOB_KIND_METASHAPE_NERF)

    assert loaded["kind"] == JOB_KIND_METASHAPE_NERF
    assert loaded["final_orientation"] == "lichtfeld"
