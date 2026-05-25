from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import core.metashape_dataset_cli as cli
from core.dataset_job_spec import (
    JOB_KIND_METASHAPE_COLMAP,
    JOB_KIND_METASHAPE_NERF,
    metashape_nerf_job,
    write_dataset_job,
)


def test_metashape_colmap_cli_builds_dataset_job_from_direct_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    views_json = tmp_path / "views.json"
    views_json.write_text(json.dumps({"views": [{"name": "front", "yaw": 0.0, "pitch": 0.0}]}), encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run(payload: dict[str, Any]) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(cli, "run_dataset_job_payload", fake_run)

    result = cli.main_colmap(
        [
            "--scene",
            str(tmp_path / "scene"),
            "--images",
            str(tmp_path / "scene" / "images"),
            "--masks",
            str(tmp_path / "scene" / "masks"),
            "--xml",
            str(tmp_path / "cameras.xml"),
            "--ply",
            str(tmp_path / "points.ply"),
            "--output",
            str(tmp_path / "output" / "colmap"),
            "--views-json",
            str(views_json),
            "--scale",
            "0.5",
            "--final-orientation",
            "lichtfeld",
        ]
    )

    assert result == 0
    payload = captured["payload"]
    assert payload["kind"] == JOB_KIND_METASHAPE_COLMAP
    assert payload["scene_dir"] == str(tmp_path / "scene")
    assert payload["views"] == [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    assert payload["output_scale"] == 0.5
    assert payload["final_orientation"] == "lichtfeld"


def test_metashape_nerf_cli_loads_expected_job_kind(tmp_path: Path, monkeypatch) -> None:
    job_path = write_dataset_job(
        tmp_path / "job.json",
        metashape_nerf_job(
            scene_dir=tmp_path / "scene",
            images_dir=tmp_path / "scene" / "images",
            masks_dir=None,
            xml_path=tmp_path / "cameras.xml",
            ply_path=None,
            output_dir=tmp_path / "output" / "nerf",
            views=[{"name": "front", "yaw": 0.0, "pitch": 0.0}],
            output_scale=0.5,
            output_format="jpg",
            output_bit_depth="8",
            jpg_quality=95,
            undistort_alpha=1.0,
            axis_transform="none",
            final_orientation="none",
        ),
    )
    captured: dict[str, Any] = {}

    def fake_run(payload: dict[str, Any]) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(cli, "run_dataset_job_payload", fake_run)

    result = cli.main_nerf(["--job", str(job_path)])

    assert result == 0
    assert captured["payload"]["kind"] == JOB_KIND_METASHAPE_NERF
    assert captured["payload"]["output_dir"] == str(tmp_path / "output" / "nerf")
