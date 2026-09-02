from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.cancellation import CancellationToken

APP_JOB_DATASET = "dataset"
APP_JOB_FRAME = "frame"
APP_JOB_SFM = "sfm"
APP_JOB_WORKFLOW = "workflow"
APP_JOB_APRILTAG = "apriltag"
APP_JOB_PERSPECTIVE = "perspective"


@dataclass(frozen=True, slots=True)
class AppJob:
    """Typed in-app job descriptor used by the GUI runner."""

    job_type: str
    payload: dict[str, Any]
    job_path: Path | None = None

    @property
    def kind(self) -> str:
        return str(self.payload.get("kind") or "")

    def display_command(self) -> list[str]:
        cmd = ["app-job", self.job_type, self.kind]
        if self.job_path is not None:
            cmd.extend(["--job", str(self.job_path)])
        return cmd


def workflow_app_job(payload: dict[str, Any], job_path: str | Path | None = None) -> AppJob:
    return AppJob(APP_JOB_WORKFLOW, dict(payload), Path(job_path) if job_path else None)


def dataset_app_job(payload: dict[str, Any], job_path: str | Path | None = None) -> AppJob:
    return AppJob(APP_JOB_DATASET, dict(payload), Path(job_path) if job_path else None)


def frame_app_job(payload: dict[str, Any], job_path: str | Path | None = None) -> AppJob:
    return AppJob(APP_JOB_FRAME, dict(payload), Path(job_path) if job_path else None)


def sfm_app_job(payload: dict[str, Any], job_path: str | Path | None = None) -> AppJob:
    return AppJob(APP_JOB_SFM, dict(payload), Path(job_path) if job_path else None)


def apriltag_app_job(payload: dict[str, Any], job_path: str | Path | None = None) -> AppJob:
    return AppJob(APP_JOB_APRILTAG, dict(payload), Path(job_path) if job_path else None)


def perspective_app_job(payload: dict[str, Any], job_path: str | Path | None = None) -> AppJob:
    return AppJob(APP_JOB_PERSPECTIVE, dict(payload), Path(job_path) if job_path else None)


def run_app_job(job: AppJob, *, cancel_event: CancellationToken | None = None) -> None:
    if job.job_type == APP_JOB_WORKFLOW:
        from core.workflow_job_runner import run_workflow_job_payload

        run_workflow_job_payload(job.payload, cancel_event=cancel_event)
        return
    if job.job_type == APP_JOB_DATASET:
        from core.dataset_job_runner import run_dataset_job_payload

        run_dataset_job_payload(job.payload, cancel_event=cancel_event)
        return
    if job.job_type == APP_JOB_FRAME:
        from core.frame_job_runner import run_frame_job_payload

        run_frame_job_payload(job.payload, cancel_event=cancel_event)
        return
    if job.job_type == APP_JOB_SFM:
        from core.sfm_job_runner import run_sfm_job_payload

        run_sfm_job_payload(job.payload, cancel_event=cancel_event)
        return
    if job.job_type == APP_JOB_APRILTAG:
        from core.apriltag_scale_job_runner import run_apriltag_scale_job_payload

        run_apriltag_scale_job_payload(job.payload, cancel_event=cancel_event)
        return
    if job.job_type == APP_JOB_PERSPECTIVE:
        from gs360studio.engine.perspective_job import run_perspective_job_payload

        run_perspective_job_payload(job.payload, cancel_event=cancel_event)
        return
    raise ValueError(f"Unsupported app job type: {job.job_type}")
