from __future__ import annotations

from pathlib import Path

from core.colmap_mixed_project import prepare_colmap_mixed_project
from core.sfm_job_spec import JOB_KIND_COLMAP_MIXED_PROJECT, load_sfm_job, validate_sfm_job_payload


def run_sfm_job_file(path: str | Path) -> None:
    run_sfm_job_payload(load_sfm_job(path))


def run_sfm_job_payload(job: dict) -> None:
    validate_sfm_job_payload(job)
    kind = str(job["kind"])
    if kind == JOB_KIND_COLMAP_MIXED_PROJECT:
        _run_colmap_mixed_project(job)
        return
    raise ValueError(f"Unsupported SfM job kind: {kind}")


def _run_colmap_mixed_project(job: dict) -> None:
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
    )
    for warning in result.warnings:
        print(f"Warning: {warning}", flush=True)
    print(f"Saved COLMAP mixed project manifest: {result.manifest_path}", flush=True)
