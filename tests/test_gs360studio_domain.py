from __future__ import annotations

import json
from pathlib import Path

import pytest

from gs360studio.domain.models import JobSpec, ProjectManifest, ViewSpec, cubemap_view_specs
from gs360studio.platform.job_store import list_jobs, recover_interrupted_jobs, save_job
from gs360studio.platform.project_store import load_project, migrate_legacy_project, save_project


def test_view_spec_normalizes_angles_and_derives_vertical_fov() -> None:
    view = ViewSpec(id="front", name="Front", yaw_deg=540, roll_deg=-540, hfov_deg=90, width=1920, height=1080)
    assert view.yaw_deg == -180
    assert view.roll_deg == -180
    assert view.effective_vfov_deg == pytest.approx(58.7155, abs=0.001)
    assert ViewSpec.from_dict(view.to_dict()) == view


def test_legacy_view_defaults_to_equal_explicit_fov_and_zero_roll() -> None:
    view = ViewSpec.from_dict({"name": "legacy", "yaw": 45, "pitch": -10, "fov": 80, "width": 1280, "height": 640})
    assert view.hfov_deg == 80
    assert view.effective_vfov_deg == 80
    assert view.roll_deg == 0


def test_cubemap_contract_has_stable_unique_ids() -> None:
    views = cubemap_view_specs(512)
    assert len(views) == 6
    assert len({view.id for view in views}) == 6
    assert all(view.width == view.height == 512 for view in views)


def test_migration_preserves_legacy_metadata_byte_for_byte(tmp_path: Path) -> None:
    legacy = tmp_path / "_stechdrive" / "project.json"
    legacy.parent.mkdir()
    original = b'{"version":1,"sources":[{"path":"video.mp4"}],"custom":true}\n'
    legacy.write_bytes(original)

    manifest, report = migrate_legacy_project(tmp_path)

    assert report.migrated and report.created
    assert legacy.read_bytes() == original
    assert manifest.stage_configuration["custom"] is True
    assert (tmp_path / "_360gs" / "project.json").is_file()
    assert (tmp_path / "_360gs" / "migration_report.json").is_file()

    loaded, second_report = load_project(tmp_path)
    assert second_report is None
    assert loaded.project_id == manifest.project_id


def test_project_save_and_interrupted_job_recovery(tmp_path: Path) -> None:
    manifest = ProjectManifest(name="Scene")
    save_project(tmp_path, manifest)
    running = JobSpec(job_type="perspective-export", status="running")
    save_job(tmp_path, running)

    recovered = recover_interrupted_jobs(tmp_path)

    assert [job.job_id for job in recovered] == [running.job_id]
    jobs = list_jobs(tmp_path)
    assert jobs[0].status == "interrupted"
    payload = json.loads((tmp_path / "_360gs" / "jobs" / f"{running.job_id}.json").read_text(encoding="utf-8"))
    assert payload["configuration_hash"] == running.configuration_hash
