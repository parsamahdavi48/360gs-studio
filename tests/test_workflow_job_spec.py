from __future__ import annotations

from pathlib import Path

import pytest

from core.workflow_job_spec import (
    JOB_KIND_CUBEMAP_CONVERSION,
    JOB_KIND_METASHAPE_PREPROCESS,
    JOB_KIND_SPHERESFM_TRANSFORMS,
    JOB_KIND_TRANSFORMS_TO_COLMAP,
    cubemap_conversion_job,
    load_workflow_job,
    metashape_preprocess_job,
    spheresfm_transforms_job,
    transforms_to_colmap_job,
    write_workflow_job,
)
from gui.steps.workflow_job_commands import build_workflow_job_cmd


def test_workflow_job_builders_round_trip_core_conversion_jobs(tmp_path: Path) -> None:
    metashape = metashape_preprocess_job(
        images_dir=tmp_path / "images",
        xml_path=tmp_path / "cameras.xml",
        output_dir=tmp_path / "work",
        scale=1.25,
        use_ply=True,
        ply_path=tmp_path / "points.ply",
        no_fix_rotation=False,
    )
    cubemap = cubemap_conversion_job(
        input_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        views=[{"name": "pz", "enabled": True, "yaw": 0.0, "pitch": 0.0}],
        fov=90.0,
        output_scale=0.5,
        axis_mode="none",
        image_only=False,
        colmap_rig=False,
        invert_masks=True,
        write_images=True,
        write_masks=False,
        yaw_offset_per_frame=12.5,
        output_format="png",
        output_bit_depth="16",
        jpg_quality=90,
        image_dir=tmp_path / "images",
        mask_dir=None,
        final_orientation="lichtfeld",
    )
    colmap = transforms_to_colmap_job(
        input_dir=tmp_path / "output",
        output_dir=tmp_path / "output" / "sparse" / "0",
        ply_path=tmp_path / "output" / "pointcloud.ply",
    )

    assert load_workflow_job(write_workflow_job(tmp_path / "metashape.json", metashape))["kind"] == (
        JOB_KIND_METASHAPE_PREPROCESS
    )
    loaded_cubemap = load_workflow_job(
        write_workflow_job(tmp_path / "cubemap.json", cubemap),
        expected_kind=JOB_KIND_CUBEMAP_CONVERSION,
    )
    assert loaded_cubemap["views"] == [{"name": "pz", "enabled": True, "yaw": 0.0, "pitch": 0.0}]
    assert loaded_cubemap["invert_masks"] is True
    assert loaded_cubemap["write_masks"] is False
    assert loaded_cubemap["final_orientation"] == "lichtfeld"
    assert load_workflow_job(write_workflow_job(tmp_path / "colmap.json", colmap))["kind"] == (
        JOB_KIND_TRANSFORMS_TO_COLMAP
    )


def test_workflow_job_rejects_wrong_kind(tmp_path: Path) -> None:
    path = write_workflow_job(
        tmp_path / "spheresfm.json",
        spheresfm_transforms_job(
            sparse_dir=tmp_path / "sparse" / "0",
            output_dir=tmp_path / "output",
            images_dir=tmp_path / "images",
            image_path_mode="relative",
        ),
    )

    assert load_workflow_job(path, expected_kind=JOB_KIND_SPHERESFM_TRANSFORMS)["kind"] == JOB_KIND_SPHERESFM_TRANSFORMS
    with pytest.raises(ValueError, match=JOB_KIND_CUBEMAP_CONVERSION):
        load_workflow_job(path, expected_kind=JOB_KIND_CUBEMAP_CONVERSION)


def test_workflow_job_command_targets_generic_worker(tmp_path: Path) -> None:
    job = tmp_path / "job.json"

    cmd = build_workflow_job_cmd(Path.cwd(), job, python_executable="python.exe")

    assert cmd[0] == "python.exe"
    assert cmd[2].endswith("scripts\\run_workflow_job.py")
    assert cmd[3:] == ["--job", str(job)]
