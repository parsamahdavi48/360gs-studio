from __future__ import annotations

from pathlib import Path

import pytest

from core.apriltag_scale_job_spec import (
    apriltag_scale_estimate_job,
    apriltag_scale_job_to_command,
    validate_apriltag_scale_job_payload,
)


def test_apriltag_scale_job_renders_core_module_command(tmp_path: Path) -> None:
    payload = apriltag_scale_estimate_job(
        dataset=tmp_path / "transforms.json",
        image_root=tmp_path / "images",
        report_json=tmp_path / "report.json",
        tag_size_m=0.16,
        family="tag36h11",
        tag_ids=(7, 8),
        workers="2",
        cubemap_pose_preset="lichtfeld",
    )

    cmd = apriltag_scale_job_to_command("python", payload)

    assert cmd[:4] == ["python", "-u", "-m", "core.apriltag_scale_estimate"]
    assert str(tmp_path / "transforms.json") in cmd
    assert cmd[cmd.index("--image-root") + 1] == str(tmp_path / "images")
    assert cmd[cmd.index("--tag-size-m") + 1] == "0.16"
    assert cmd[cmd.index("--workers") + 1] == "2"
    assert cmd.count("--tag-id") == 2
    assert cmd[cmd.index("--cubemap-pose-preset") + 1] == "lichtfeld"


def test_apriltag_scale_job_rejects_invalid_values(tmp_path: Path) -> None:
    payload = apriltag_scale_estimate_job(
        dataset=tmp_path / "transforms.json",
        report_json=tmp_path / "report.json",
        tag_size_m=-1.0,
        family="tag36h11",
    )

    with pytest.raises(ValueError, match="tag_size_m"):
        validate_apriltag_scale_job_payload(payload)


def test_apriltag_scale_job_rejects_duplicate_tag_ids(tmp_path: Path) -> None:
    payload = apriltag_scale_estimate_job(
        dataset=tmp_path / "transforms.json",
        report_json=tmp_path / "report.json",
        tag_size_m=0.16,
        family="tag36h11",
        tag_ids=(7, 7),
    )

    with pytest.raises(ValueError, match="duplicates"):
        validate_apriltag_scale_job_payload(payload)
