from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.frame_pair_analysis import PAIR_THRESHOLD_PROFILE_CHOICES
from core.job_payload_validation import (
    require_bool,
    require_finite_float,
    require_int_range,
    require_kind,
    require_mapping,
    require_schema_version,
    require_str,
)

FRAME_JOB_SCHEMA_VERSION = 1
JOB_KIND_APPLY_FRAME_DECISIONS = "apply_frame_decisions"
JOB_KIND_EXTRACT_VIDEO = "extract_video"
JOB_KIND_IMPORT_IMAGE_SEQUENCE = "import_image_sequence"
JOB_KIND_IMPORT_SCENE = "import_scene"
JOB_KIND_REFRESH_SCENE_ASSETS = "refresh_scene_assets"

FRAME_JOB_KINDS = {
    JOB_KIND_APPLY_FRAME_DECISIONS,
    JOB_KIND_EXTRACT_VIDEO,
    JOB_KIND_IMPORT_IMAGE_SEQUENCE,
    JOB_KIND_IMPORT_SCENE,
    JOB_KIND_REFRESH_SCENE_ASSETS,
}


def extract_video_job(
    *,
    input_video: str | Path,
    scene_dir: str | Path,
    image_ext: str,
    jpg_quality: int,
    ffmpeg: str,
    ffprobe: str,
    output_mode: str,
    filename_prefix: str,
    interval_sec: float,
    quick_extract: bool,
    pair_motion_profile: str,
    analysis_width: int,
    fixed_smart: bool,
    min_gap_sec: float,
    max_gap_sec: float,
    allow_duplicate_video: bool = False,
    estimate_only: bool = False,
    print_summary_json: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": FRAME_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_EXTRACT_VIDEO,
        "input_video": str(input_video),
        "scene_dir": str(scene_dir),
        "image_ext": str(image_ext),
        "jpg_quality": int(jpg_quality),
        "ffmpeg": str(ffmpeg),
        "ffprobe": str(ffprobe),
        "output_mode": str(output_mode),
        "filename_prefix": str(filename_prefix),
        "interval_sec": float(interval_sec),
        "quick_extract": bool(quick_extract),
        "pair_motion_profile": str(pair_motion_profile),
        "analysis_width": int(analysis_width),
        "fixed_smart": bool(fixed_smart),
        "min_gap_sec": float(min_gap_sec),
        "max_gap_sec": float(max_gap_sec),
        "allow_duplicate_video": bool(allow_duplicate_video),
        "estimate_only": bool(estimate_only),
        "print_summary_json": bool(print_summary_json),
    }


def import_image_sequence_job(
    *,
    source_dir: str | Path,
    scene_dir: str | Path,
    prefix: str,
    recursive: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": FRAME_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_IMPORT_IMAGE_SEQUENCE,
        "source_dir": str(source_dir),
        "scene_dir": str(scene_dir),
        "prefix": str(prefix),
        "recursive": bool(recursive),
    }


def import_scene_job(
    *,
    scene_dir: str | Path,
) -> dict[str, Any]:
    return {
        "schema_version": FRAME_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_IMPORT_SCENE,
        "scene_dir": str(scene_dir),
    }


def apply_frame_decisions_job(
    *,
    scene_dir: str | Path,
    csv: str = "selected_frames.csv",
    output: str = "metashape_images",
    clean_output: bool = False,
    finalize_in_place: bool = True,
    backup_dir: str | Path | None = None,
    renumber_kept_images: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": FRAME_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_APPLY_FRAME_DECISIONS,
        "scene_dir": str(scene_dir),
        "csv": str(csv),
        "output": str(output),
        "clean_output": bool(clean_output),
        "finalize_in_place": bool(finalize_in_place),
        "backup_dir": str(backup_dir) if backup_dir else "",
        "renumber_kept_images": bool(renumber_kept_images),
    }


def refresh_scene_assets_job(*, scene_dir: str | Path) -> dict[str, Any]:
    return {
        "schema_version": FRAME_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_REFRESH_SCENE_ASSETS,
        "scene_dir": str(scene_dir),
    }


def write_frame_job(path: str | Path, payload: dict[str, Any]) -> Path:
    job_path = Path(path)
    validate_frame_job_payload(payload)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return job_path


def load_frame_job(path: str | Path, *, expected_kind: str = "") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frame job must be a JSON object: {path}")
    validate_frame_job_payload(payload)
    if expected_kind and payload["kind"] != expected_kind:
        raise ValueError(f"Frame job kind must be {expected_kind}: {payload['kind']}")
    return payload


def validate_frame_job_payload(payload: dict[str, Any]) -> None:
    data = require_mapping(payload, label="frame")
    require_schema_version(data, expected=FRAME_JOB_SCHEMA_VERSION, label="frame")
    kind = require_kind(data, allowed=FRAME_JOB_KINDS, label="frame")
    if kind == JOB_KIND_EXTRACT_VIDEO:
        _validate_extract_video_job(data)
    elif kind == JOB_KIND_IMPORT_IMAGE_SEQUENCE:
        _validate_import_image_sequence_job(data)
    elif kind == JOB_KIND_IMPORT_SCENE:
        _validate_import_scene_job(data)
    elif kind == JOB_KIND_APPLY_FRAME_DECISIONS:
        _validate_apply_frame_decisions_job(data)
    elif kind == JOB_KIND_REFRESH_SCENE_ASSETS:
        _validate_refresh_scene_assets_job(data)


def _validate_extract_video_job(payload: Mapping[str, Any]) -> None:
    for key in ("input_video", "scene_dir", "ffmpeg", "ffprobe", "output_mode", "filename_prefix"):
        require_str(payload, key, label="frame", allow_empty=key == "filename_prefix")
    image_ext = require_str(payload, "image_ext", label="frame").lower()
    if image_ext not in {"jpg", "png"}:
        raise ValueError(f"frame job field 'image_ext' must be jpg or png: {image_ext}")
    output_mode = require_str(payload, "output_mode", label="frame")
    if output_mode not in {"overwrite", "append", "replace-video"}:
        raise ValueError(f"frame job field 'output_mode' is invalid: {output_mode}")
    pair_profile = require_str(payload, "pair_motion_profile", label="frame")
    if pair_profile not in PAIR_THRESHOLD_PROFILE_CHOICES:
        raise ValueError(f"frame job field 'pair_motion_profile' is invalid: {pair_profile}")
    require_int_range(payload, "jpg_quality", label="frame", min_value=1, max_value=31)
    require_int_range(payload, "analysis_width", label="frame", min_value=0)
    require_finite_float(payload, "interval_sec", label="frame", min_value=0.0, min_inclusive=False)
    min_gap = require_finite_float(payload, "min_gap_sec", label="frame", min_value=0.0, min_inclusive=False)
    max_gap = require_finite_float(payload, "max_gap_sec", label="frame", min_value=0.0, min_inclusive=False)
    if max_gap < min_gap:
        raise ValueError("frame job field 'max_gap_sec' must be >= min_gap_sec")
    quick_extract = require_bool(payload, "quick_extract", label="frame")
    fixed_smart = require_bool(payload, "fixed_smart", label="frame")
    if quick_extract and fixed_smart:
        raise ValueError("frame job cannot combine quick_extract and fixed_smart")
    for key in ("allow_duplicate_video", "estimate_only", "print_summary_json"):
        require_bool(payload, key, label="frame")


def _validate_import_image_sequence_job(payload: Mapping[str, Any]) -> None:
    for key in ("source_dir", "scene_dir"):
        require_str(payload, key, label="frame")
    require_str(payload, "prefix", label="frame", allow_empty=True)
    require_bool(payload, "recursive", label="frame")


def _validate_import_scene_job(payload: Mapping[str, Any]) -> None:
    require_str(payload, "scene_dir", label="frame")


def _validate_apply_frame_decisions_job(payload: Mapping[str, Any]) -> None:
    for key in ("scene_dir", "csv", "output"):
        require_str(payload, key, label="frame")
    require_str(payload, "backup_dir", label="frame", allow_empty=True)
    require_bool(payload, "clean_output", label="frame")
    finalize_in_place = require_bool(payload, "finalize_in_place", label="frame")
    renumber_kept_images = require_bool(payload, "renumber_kept_images", label="frame")
    if renumber_kept_images and not finalize_in_place:
        raise ValueError("frame job field 'renumber_kept_images' requires finalize_in_place")


def _validate_refresh_scene_assets_job(payload: Mapping[str, Any]) -> None:
    require_str(payload, "scene_dir", label="frame")
