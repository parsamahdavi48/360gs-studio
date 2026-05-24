from __future__ import annotations

from pathlib import Path

from core.app_job import AppJob, workflow_app_job
from core.workflow_job_spec import load_workflow_job


def build_workflow_job_cmd(base_dir: Path, job_path: Path, *, python_executable: str | None = None) -> AppJob:
    _ = base_dir, python_executable
    return workflow_app_job(load_workflow_job(job_path), job_path)
