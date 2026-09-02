"""Persistent job records with interrupted-job recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gs360studio.domain.models import JobSpec, atomic_write_json, utc_now
from gs360studio.platform.project_store import APP_DIR_NAME


def jobs_dir(scene_dir: str | Path) -> Path:
    return Path(scene_dir) / APP_DIR_NAME / "jobs"


def job_path(scene_dir: str | Path, job_id: str) -> Path:
    return jobs_dir(scene_dir) / f"{job_id}.json"


def save_job(scene_dir: str | Path, job: JobSpec) -> Path:
    target = job_path(scene_dir, job.job_id)
    atomic_write_json(target, job.to_dict())
    return target


def load_job(path: str | Path) -> JobSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != JobSpec.SCHEMA_VERSION:
        raise ValueError(f"unsupported job manifest: {path}")
    ignored = {"schema_version", "configuration_hash"}
    fields = {name for name in JobSpec.__dataclass_fields__}
    return JobSpec(**{key: value for key, value in payload.items() if key in fields and key not in ignored})


def list_jobs(scene_dir: str | Path) -> list[JobSpec]:
    root = jobs_dir(scene_dir)
    if not root.exists():
        return []
    result: list[JobSpec] = []
    for path in sorted(root.glob("*.json")):
        try:
            result.append(load_job(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return result


def recover_interrupted_jobs(scene_dir: str | Path) -> list[JobSpec]:
    recovered: list[JobSpec] = []
    for job in list_jobs(scene_dir):
        if job.status != "running":
            continue
        job.status = "interrupted"
        job.finished_at = utc_now()
        job.diagnostics.append({"level": "warning", "message": "Job was running when the previous session ended."})
        save_job(scene_dir, job)
        recovered.append(job)
    return recovered


def update_job(scene_dir: str | Path, job_id: str, **changes: Any) -> JobSpec:
    job = load_job(job_path(scene_dir, job_id))
    for key, value in changes.items():
        if key not in JobSpec.__dataclass_fields__:
            raise ValueError(f"unknown job field: {key}")
        setattr(job, key, value)
    save_job(scene_dir, job)
    return job
