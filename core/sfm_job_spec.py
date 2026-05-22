from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SFM_JOB_SCHEMA_VERSION = 1
JOB_KIND_COLMAP_MIXED_PROJECT = "colmap_mixed_project"


def colmap_mixed_project_job(
    *,
    scene_dir: str | Path,
    output_dir: str | Path,
    views: list[dict[str, Any]],
    output_scale: float,
    output_format: str,
    output_bit_depth: str,
    jpg_quality: int,
    write_images: bool,
    write_masks: bool,
    invert_masks: bool,
    workers: str,
    remap_cache_limit: str,
    rig_name: str = "rig1",
) -> dict[str, Any]:
    return {
        "schema_version": SFM_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_COLMAP_MIXED_PROJECT,
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "views": [dict(view) for view in views],
        "output_scale": float(output_scale),
        "output_format": str(output_format),
        "output_bit_depth": str(output_bit_depth),
        "jpg_quality": int(jpg_quality),
        "write_images": bool(write_images),
        "write_masks": bool(write_masks),
        "invert_masks": bool(invert_masks),
        "workers": str(workers),
        "remap_cache_limit": str(remap_cache_limit),
        "rig_name": str(rig_name),
    }


def write_sfm_job(path: str | Path, payload: dict[str, Any]) -> Path:
    job_path = Path(path)
    _validate_payload(payload)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return job_path


def load_sfm_job(path: str | Path, *, expected_kind: str = "") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"SfM job must be a JSON object: {path}")
    _validate_payload(payload)
    if expected_kind and payload["kind"] != expected_kind:
        raise ValueError(f"SfM job kind must be {expected_kind}: {payload['kind']}")
    return payload


def _validate_payload(payload: dict[str, Any]) -> None:
    version = int(payload.get("schema_version") or 0)
    if version != SFM_JOB_SCHEMA_VERSION:
        raise ValueError(f"Unsupported SfM job schema version: {version}")
    kind = str(payload.get("kind") or "")
    if kind not in {JOB_KIND_COLMAP_MIXED_PROJECT}:
        raise ValueError(f"Unsupported SfM job kind: {kind}")
