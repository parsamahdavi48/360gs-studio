from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.apriltag_cubemap import CUBEMAP_POSE_PRESET_AUTO, CUBEMAP_POSE_PRESETS
from core.apriltag_markers import MAX_APRILTAG_IDS_PER_RUN, available_families
from core.job_payload_validation import (
    require_finite_float,
    require_kind,
    require_mapping,
    require_schema_version,
    require_str,
)

APRILTAG_SCALE_JOB_SCHEMA_VERSION = 1
JOB_KIND_APRILTAG_SCALE_ESTIMATE = "apriltag_scale_estimate"


def apriltag_scale_estimate_job(
    *,
    dataset: str | Path,
    report_json: str | Path,
    tag_size_m: float,
    family: str,
    tag_ids: Sequence[int] = (),
    image_root: str | Path | None = None,
    min_score: float = 0.0,
    min_baseline_sfm: float = 1e-6,
    workers: str = "auto",
    cubemap_pose_preset: str = CUBEMAP_POSE_PRESET_AUTO,
) -> dict[str, Any]:
    return {
        "schema_version": APRILTAG_SCALE_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_APRILTAG_SCALE_ESTIMATE,
        "dataset": str(dataset),
        "image_root": str(image_root) if image_root else "",
        "report_json": str(report_json),
        "tag_size_m": float(tag_size_m),
        "family": str(family),
        "tag_ids": [int(tag_id) for tag_id in tag_ids],
        "min_score": float(min_score),
        "min_baseline_sfm": float(min_baseline_sfm),
        "workers": str(workers),
        "cubemap_pose_preset": str(cubemap_pose_preset),
    }


def validate_apriltag_scale_job_payload(payload: dict[str, Any]) -> None:
    data = require_mapping(payload, label="apriltag scale")
    require_schema_version(data, expected=APRILTAG_SCALE_JOB_SCHEMA_VERSION, label="apriltag scale")
    kind = require_kind(data, allowed={JOB_KIND_APRILTAG_SCALE_ESTIMATE}, label="apriltag scale")
    if kind == JOB_KIND_APRILTAG_SCALE_ESTIMATE:
        _validate_estimate(data)


def apriltag_scale_job_to_command(python_executable: str, payload: dict[str, Any]) -> list[str]:
    validate_apriltag_scale_job_payload(payload)
    cmd = [
        str(python_executable),
        "-u",
        "-m",
        "core.apriltag_scale_estimate",
        str(payload["dataset"]),
        "--tag-size-m",
        f"{float(payload['tag_size_m']):.9g}",
        "--family",
        str(payload["family"]),
        "--report-json",
        str(payload["report_json"]),
        "--workers",
        str(payload["workers"]),
        "--min-score",
        f"{float(payload['min_score']):.9g}",
        "--min-baseline-sfm",
        f"{float(payload['min_baseline_sfm']):.9g}",
    ]
    image_root = str(payload.get("image_root") or "").strip()
    if image_root:
        cmd.extend(["--image-root", image_root])
    for tag_id in payload.get("tag_ids") or []:
        cmd.extend(["--tag-id", str(int(tag_id))])
    preset = str(payload.get("cubemap_pose_preset") or CUBEMAP_POSE_PRESET_AUTO)
    if preset != CUBEMAP_POSE_PRESET_AUTO:
        cmd.extend(["--cubemap-pose-preset", preset])
    return cmd


def _validate_estimate(payload: Mapping[str, Any]) -> None:
    for key in ("dataset", "report_json", "family", "workers", "cubemap_pose_preset"):
        require_str(payload, key, label="apriltag scale")
    require_str(payload, "image_root", label="apriltag scale", allow_empty=True)
    require_finite_float(payload, "tag_size_m", label="apriltag scale", min_value=0.0, min_inclusive=False)
    require_finite_float(payload, "min_score", label="apriltag scale", min_value=0.0)
    require_finite_float(payload, "min_baseline_sfm", label="apriltag scale", min_value=0.0)
    family = str(payload["family"])
    if family not in available_families():
        raise ValueError(f"apriltag scale job field 'family' is invalid: {family}")
    preset = str(payload["cubemap_pose_preset"])
    if preset not in CUBEMAP_POSE_PRESETS:
        raise ValueError(f"apriltag scale job field 'cubemap_pose_preset' is invalid: {preset}")
    _validate_workers(str(payload["workers"]))
    _validate_tag_ids(payload.get("tag_ids"))


def _validate_workers(value: str) -> None:
    text = value.strip().lower()
    if text == "auto":
        return
    try:
        parsed = int(text)
    except ValueError as e:
        raise ValueError("apriltag scale job field 'workers' must be 'auto' or a positive integer") from e
    if parsed <= 0:
        raise ValueError("apriltag scale job field 'workers' must be 'auto' or a positive integer")


def _validate_tag_ids(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("apriltag scale job field 'tag_ids' must be a list")
    if len(value) > MAX_APRILTAG_IDS_PER_RUN:
        raise ValueError(f"apriltag scale job field 'tag_ids' must contain at most {MAX_APRILTAG_IDS_PER_RUN} ids")
    seen: set[int] = set()
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"apriltag scale job field 'tag_ids[{index}]' must be an integer")
        if item < 0:
            raise ValueError(f"apriltag scale job field 'tag_ids[{index}]' must be >= 0")
        if item in seen:
            raise ValueError("apriltag scale job field 'tag_ids' must not contain duplicates")
        seen.add(item)
