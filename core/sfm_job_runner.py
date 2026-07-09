from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.cancellation import CancellationToken, raise_if_cancelled
from core.colmap_mixed_project import prepare_colmap_mixed_project
from core.sfm_job_spec import JOB_KIND_COLMAP_MIXED_PROJECT, load_sfm_job, validate_sfm_job_payload


def _progress_log_callback(cancel_event: CancellationToken | None = None) -> Callable[[int, int], None]:
    last_bucket = -1
    last_pair: tuple[int, int] | None = None

    def callback(done: int, total: int) -> None:
        nonlocal last_bucket, last_pair
        raise_if_cancelled(cancel_event)
        done = max(0, int(done))
        total = max(0, int(total))
        if total <= 0:
            return
        pair = (min(done, total), total)
        if pair == last_pair:
            return
        bucket = int((pair[0] / float(total)) * 100.0)
        if pair[0] == 0 or pair[0] >= total or bucket != last_bucket:
            print(f"[progress] {pair[0]}/{total}", flush=True)
            last_bucket = bucket
            last_pair = pair

    return callback


def run_sfm_job_file(path: str | Path, *, cancel_event: CancellationToken | None = None) -> None:
    run_sfm_job_payload(load_sfm_job(path), cancel_event=cancel_event)


def run_sfm_job_payload(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    validate_sfm_job_payload(job)
    raise_if_cancelled(cancel_event)
    kind = str(job["kind"])
    if kind == JOB_KIND_COLMAP_MIXED_PROJECT:
        _run_colmap_mixed_project(job, cancel_event=cancel_event)
        return
    raise ValueError(f"Unsupported SfM job kind: {kind}")


def _run_colmap_mixed_project(job: dict, *, cancel_event: CancellationToken | None = None) -> None:
    raise_if_cancelled(cancel_event)
    result = prepare_colmap_mixed_project(
        job["scene_dir"],
        job["output_dir"],
        views=job["views"],
        output_scale=float(job["output_scale"]),
        output_format=str(job["output_format"]),
        output_bit_depth=str(job["output_bit_depth"]),
        jpg_quality=int(job["jpg_quality"]),
        write_images=bool(job["write_images"]),
        write_masks=bool(job["write_masks"]),
        invert_masks=bool(job["invert_masks"]),
        workers=str(job["workers"]),
        remap_cache_limit=str(job["remap_cache_limit"]),
        rig_name=str(job["rig_name"]),
        source_masks_dir=Path(str(job["source_masks_dir"])) if str(job.get("source_masks_dir") or "") else None,
        cancel_event=cancel_event,
        progress_callback=_progress_log_callback(cancel_event),
    )
    for warning in result.warnings:
        print(f"Warning: {warning}", flush=True)
    print(f"Saved COLMAP mixed project metadata: {result.manifest_path}", flush=True)
    raise_if_cancelled(cancel_event)
