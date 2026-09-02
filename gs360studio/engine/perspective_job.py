"""Typed in-process perspective export job entrypoint."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from core.cancellation import AppJobCancelled, CancellationToken, is_cancelled
from gs360studio.domain.models import JobSpec, ViewSpec, utc_now
from gs360studio.engine.perspective_export import (
    ExportRequest,
    export_image_views,
    export_video_views,
    source_signature,
)
from gs360studio.platform.job_store import save_job

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})


def request_from_payload(payload: dict[str, Any]) -> ExportRequest:
    if payload.get("kind") != "perspective_export":
        raise ValueError("unsupported perspective job kind")
    views = payload.get("views")
    if not isinstance(views, list):
        raise ValueError("perspective job requires a views list")
    return ExportRequest(
        input_path=Path(str(payload["input_path"])),
        output_dir=Path(str(payload["output_dir"])),
        views=tuple(ViewSpec.from_dict(item, index=index) for index, item in enumerate(views)),
        output_format=str(payload.get("output_format") or "png"),
        frame_interval_sec=float(payload.get("frame_interval_sec") or 1.0),
        jpeg_quality=int(payload.get("jpeg_quality") or 95),
        video_quality=int(payload.get("video_quality") or 18),
        video_preset=str(payload.get("video_preset") or "p4"),
        use_nvenc=bool(payload.get("use_nvenc", False)),
        batch_size=int(payload.get("batch_size") or 0),
        ffmpeg_path=str(payload.get("ffmpeg_path") or "ffmpeg"),
        colmap_rig=bool(payload.get("colmap_rig", False)),
        overwrite=bool(payload.get("overwrite", False)),
    )


def run_perspective_job_payload(payload: dict[str, Any], *, cancel_event: CancellationToken | None = None) -> None:
    request = request_from_payload(payload)
    scene_value = str(payload.get("scene_dir") or "").strip()
    scene_dir = Path(scene_value) if scene_value else None
    job = JobSpec(
        job_type="perspective-export",
        dependency_ids=[str(payload["resume_job_id"])] if payload.get("resume_job_id") else [],
        input_signatures={"source": source_signature(request.input_path)} if request.input_path.exists() else {},
        configuration={key: value for key, value in payload.items() if key != "scene_dir"},
        outputs=[str(request.output_dir)],
        status="running",
        started_at=utc_now(),
    )
    if scene_dir is not None:
        save_job(scene_dir, job)
    last_progress_save = 0.0

    def canceled() -> bool:
        return is_cancelled(cancel_event)

    def progress(current: int, total: int, message: str) -> None:
        nonlocal last_progress_save
        job.progress_current = max(0, int(current))
        job.progress_total = max(0, int(total))
        now = time.monotonic()
        if scene_dir is not None and (now - last_progress_save >= 0.25 or current >= total):
            save_job(scene_dir, job)
            last_progress_save = now
        print(f"PROGRESS:{current}/{total}", flush=True)
        print(message, flush=True)

    try:
        if request.input_path.is_dir():
            files = sorted(path for path in request.input_path.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
            if not files:
                raise ValueError(f"no supported images found in {request.input_path}")
            export_image_views(files, request, progress=progress, canceled=canceled)
        elif request.input_path.suffix.lower() in _IMAGE_SUFFIXES:
            export_image_views([request.input_path], request, progress=progress, canceled=canceled)
        else:
            export_video_views(request, progress=progress, canceled=canceled)
    except InterruptedError as exc:
        if scene_dir is not None:
            job.status = "canceled"
            job.finished_at = utc_now()
            job.diagnostics.append({"level": "warning", "message": str(exc)})
            save_job(scene_dir, job)
        raise AppJobCancelled(str(exc)) from exc
    except Exception as exc:
        if scene_dir is not None:
            job.status = "failed"
            job.finished_at = utc_now()
            job.diagnostics.append({"level": "error", "message": str(exc)})
            save_job(scene_dir, job)
        raise
    if scene_dir is not None:
        job.status = "completed"
        if job.progress_total <= 0:
            job.progress_total = 1
        job.progress_current = job.progress_total
        job.finished_at = utc_now()
        save_job(scene_dir, job)
    print(f"OUTPUT:{request.output_dir.resolve()}", flush=True)
