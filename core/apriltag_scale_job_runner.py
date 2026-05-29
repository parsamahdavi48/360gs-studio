from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from core.apriltag_cubemap import cubemap_view_metadata_for_pose_preset
from core.apriltag_pipeline import run_apriltag_scale_estimation
from core.apriltag_scale_estimate import build_report, resolve_estimation_input
from core.apriltag_scale_job_spec import (
    JOB_KIND_APRILTAG_SCALE_ESTIMATE,
    load_apriltag_scale_job,
    validate_apriltag_scale_job_payload,
)
from core.cancellation import CancellationToken, raise_if_cancelled


def run_apriltag_scale_job_file(path: str | Path, *, cancel_event: CancellationToken | None = None) -> None:
    run_apriltag_scale_job_payload(load_apriltag_scale_job(path), cancel_event=cancel_event)


def run_apriltag_scale_job_payload(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    validate_apriltag_scale_job_payload(job)
    raise_if_cancelled(cancel_event)
    kind = str(job["kind"])
    if kind == JOB_KIND_APRILTAG_SCALE_ESTIMATE:
        _run_estimate(job, cancel_event=cancel_event)
        return
    raise ValueError(f"Unsupported AprilTag scale job kind: {kind}")


def _run_estimate(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    args = _namespace_from_job(job)
    print("[apriltag] validating input dataset", flush=True)
    estimation_input = resolve_estimation_input(args)
    cubemap_view_params = cubemap_view_metadata_for_pose_preset(args.cubemap_pose_preset)
    run = run_apriltag_scale_estimation(
        estimation_input,
        image_root=args.image_root,
        tag_size_m=args.tag_size_m,
        family=args.family,
        tag_ids=args.tag_ids,
        min_score=args.min_score,
        min_baseline_sfm=args.min_baseline_sfm,
        cubemap_view_params=cubemap_view_params,
        workers=args.workers,
        progress_callback=lambda done, total: _progress(done, total, cancel_event=cancel_event),
        log_callback=lambda message: _log(message, cancel_event=cancel_event),
    )
    raise_if_cancelled(cancel_event)
    report = build_report(run, args)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    estimate = report["estimate"]
    print(
        "AprilTag scale estimate: "
        f"scale={estimate['scale']:.9g}, "
        f"observations={estimate['observation_count']}, "
        f"pairs={estimate['pair_count']}, "
        f"inliers={estimate['inlier_count']}, "
        f"rms={estimate['rms_residual_m']:.6g} m",
        flush=True,
    )


def _namespace_from_job(job: dict) -> Namespace:
    image_root_text = str(job.get("image_root") or "").strip()
    tag_ids = job.get("tag_ids") or []
    return Namespace(
        dataset=Path(str(job["dataset"])),
        image_root=Path(image_root_text) if image_root_text else None,
        report_json=Path(str(job["report_json"])),
        tag_size_m=float(job["tag_size_m"]),
        family=str(job["family"]),
        tag_ids=set(int(tag_id) for tag_id in tag_ids) if tag_ids else None,
        min_score=float(job.get("min_score", 0.0)),
        min_baseline_sfm=float(job.get("min_baseline_sfm", 1e-6)),
        workers=str(job.get("workers") or "auto"),
        cubemap_pose_preset=str(job.get("cubemap_pose_preset") or "auto"),
    )


def _log(message: str, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    print(f"[apriltag] {message}", flush=True)


def _progress(done: int, total: int, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    print(f"[progress] {done}/{total}", flush=True)
