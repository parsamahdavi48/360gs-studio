from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.job_payload_validation import (
    require_bool,
    require_finite_float,
    require_int_range,
    require_kind,
    require_mapping,
    require_schema_version,
    require_str,
)
from core.mask_merge import (
    MASK_MERGE_ADD,
    MASK_MERGE_REPLACE,
    SUPPORTED_MERGE_MODES,
)
from core.mask_view_recipes import PROJECTION_EQUIRECT, PROJECTION_NORMAL, QUALITY_CHOICES

MASK_JOB_SCHEMA_VERSION = 1
MASK_JOB_YOLO_SAM = "yolo_sam"
MASK_JOB_SKY = "sky_mask"
MASK_JOB_INIT = "init_masks"
MASK_JOB_STITCH = "stitch_mask"
MASK_JOB_OVEREXPOSURE = "overexposure_mask"
MASK_JOB_CUSTOM = "custom_mask"
BACKEND_YOLO26_SEM = "yolo26_sem"
BACKEND_SAM31 = "sam31"
SUPPORTED_BACKENDS = (BACKEND_YOLO26_SEM, BACKEND_SAM31)

MASK_JOB_KINDS = {
    MASK_JOB_YOLO_SAM,
    MASK_JOB_SKY,
    MASK_JOB_INIT,
    MASK_JOB_STITCH,
    MASK_JOB_OVEREXPOSURE,
    MASK_JOB_CUSTOM,
}


def yolo_sam_mask_job(
    *,
    images: str | Path,
    masks: str | Path,
    quality: str,
    expand: int,
    projection: str,
    classes: Sequence[int],
    extra_args: Sequence[str] = (),
    merge_mode: str = MASK_MERGE_REPLACE,
    image_list: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MASK_JOB_SCHEMA_VERSION,
        "kind": MASK_JOB_YOLO_SAM,
        "images": str(images),
        "masks": str(masks),
        "quality": str(quality),
        "expand": int(expand),
        "projection": str(projection),
        "classes": [int(item) for item in classes],
        "extra_args": [str(item) for item in extra_args],
        "merge_mode": str(merge_mode),
        "image_list": str(image_list) if image_list else "",
    }


def sky_mask_job(
    *,
    images: str | Path,
    masks: str | Path,
    backend: str,
    projection: str,
    quality: str,
    inference_size: int,
    expand: int,
    min_score: float,
    min_area_ratio: float,
    top_connected: bool,
    labels: Sequence[str] = (),
    sam_prompts: Sequence[str] = (),
    sam_subtract_prompts: Sequence[str] = (),
    merge_mode: str = MASK_MERGE_ADD,
    replace: bool = False,
    safe_batch: bool = False,
    image_list: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MASK_JOB_SCHEMA_VERSION,
        "kind": MASK_JOB_SKY,
        "images": str(images),
        "masks": str(masks),
        "backend": str(backend),
        "projection": str(projection),
        "quality": str(quality),
        "inference_size": int(inference_size),
        "expand": int(expand),
        "min_score": float(min_score),
        "min_area_ratio": float(min_area_ratio),
        "top_connected": bool(top_connected),
        "labels": [str(item) for item in labels],
        "sam_prompts": [str(item) for item in sam_prompts],
        "sam_subtract_prompts": [str(item) for item in sam_subtract_prompts],
        "merge_mode": str(merge_mode),
        "replace": bool(replace),
        "safe_batch": bool(safe_batch),
        "image_list": str(image_list) if image_list else "",
    }


def init_masks_job(
    *,
    images: str | Path,
    masks: str | Path,
    image_list: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MASK_JOB_SCHEMA_VERSION,
        "kind": MASK_JOB_INIT,
        "images": str(images),
        "masks": str(masks),
        "image_list": str(image_list) if image_list else "",
    }


def stitch_mask_job(
    *,
    masks: str | Path,
    boundary_width: float,
    workers: int,
    image_list: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MASK_JOB_SCHEMA_VERSION,
        "kind": MASK_JOB_STITCH,
        "masks": str(masks),
        "boundary_width": float(boundary_width),
        "workers": int(workers),
        "image_list": str(image_list) if image_list else "",
    }


def overexposure_mask_job(
    *,
    images: str | Path,
    masks: str | Path,
    threshold: int,
    dilate: int,
    workers: int,
    merge_mode: str = MASK_MERGE_ADD,
    replace: bool = False,
    image_list: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MASK_JOB_SCHEMA_VERSION,
        "kind": MASK_JOB_OVEREXPOSURE,
        "images": str(images),
        "masks": str(masks),
        "threshold": int(threshold),
        "dilate": int(dilate),
        "workers": int(workers),
        "merge_mode": str(merge_mode),
        "replace": bool(replace),
        "image_list": str(image_list) if image_list else "",
    }


def custom_mask_job(
    *,
    images: str | Path,
    masks: str | Path,
    custom_mask: str | Path,
    merge_mode: str = MASK_MERGE_ADD,
    replace: bool = False,
    image_list: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MASK_JOB_SCHEMA_VERSION,
        "kind": MASK_JOB_CUSTOM,
        "images": str(images),
        "masks": str(masks),
        "custom_mask": str(custom_mask),
        "merge_mode": str(merge_mode),
        "replace": bool(replace),
        "image_list": str(image_list) if image_list else "",
    }


def validate_mask_job_payload(payload: dict[str, Any]) -> None:
    data = require_mapping(payload, label="mask")
    require_schema_version(data, expected=MASK_JOB_SCHEMA_VERSION, label="mask")
    kind = require_kind(data, allowed=MASK_JOB_KINDS, label="mask")
    if kind == MASK_JOB_YOLO_SAM:
        _validate_yolo_sam(data)
    elif kind == MASK_JOB_SKY:
        _validate_sky(data)
    elif kind == MASK_JOB_INIT:
        _validate_init(data)
    elif kind == MASK_JOB_STITCH:
        _validate_stitch(data)
    elif kind == MASK_JOB_OVEREXPOSURE:
        _validate_overexposure(data)
    elif kind == MASK_JOB_CUSTOM:
        _validate_custom(data)


def mask_job_to_command(python_executable: str, payload: dict[str, Any]) -> list[str]:
    validate_mask_job_payload(payload)
    kind = str(payload["kind"])
    if kind == MASK_JOB_YOLO_SAM:
        return _yolo_sam_command(python_executable, payload)
    if kind == MASK_JOB_SKY:
        return _sky_command(python_executable, payload)
    if kind == MASK_JOB_INIT:
        return _init_command(python_executable, payload)
    if kind == MASK_JOB_STITCH:
        return _stitch_command(python_executable, payload)
    if kind == MASK_JOB_OVEREXPOSURE:
        return _overexposure_command(python_executable, payload)
    if kind == MASK_JOB_CUSTOM:
        return _custom_command(python_executable, payload)
    raise ValueError(f"Unsupported mask job kind: {kind}")


def _validate_yolo_sam(payload: Mapping[str, Any]) -> None:
    _require_images_masks(payload)
    _require_quality_projection(payload)
    require_int_range(payload, "expand", label="mask", min_value=0)
    classes = _require_int_list(payload, "classes", allow_empty=False)
    if any(item < 0 for item in classes):
        raise ValueError("mask job field 'classes' must contain non-negative integers")
    _require_str_list(payload, "extra_args", allow_empty=True)
    merge_mode = require_str(payload, "merge_mode", label="mask")
    if merge_mode not in SUPPORTED_MERGE_MODES:
        raise ValueError(f"mask job field 'merge_mode' is invalid: {merge_mode}")
    require_str(payload, "image_list", label="mask", allow_empty=True)


def _validate_sky(payload: Mapping[str, Any]) -> None:
    _require_images_masks(payload)
    _require_quality_projection(payload)
    backend = require_str(payload, "backend", label="mask")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"mask job field 'backend' is invalid: {backend}")
    require_int_range(payload, "inference_size", label="mask", min_value=1)
    require_int_range(payload, "expand", label="mask", min_value=0)
    require_finite_float(payload, "min_score", label="mask", min_value=0.0, max_value=1.0)
    require_finite_float(payload, "min_area_ratio", label="mask", min_value=0.0, max_value=1.0)
    require_bool(payload, "top_connected", label="mask")
    require_bool(payload, "replace", label="mask")
    require_bool(payload, "safe_batch", label="mask")
    merge_mode = require_str(payload, "merge_mode", label="mask")
    if merge_mode not in SUPPORTED_MERGE_MODES:
        raise ValueError(f"mask job field 'merge_mode' is invalid: {merge_mode}")
    labels = _require_str_list(payload, "labels", allow_empty=backend != BACKEND_YOLO26_SEM)
    sam_prompts = _require_str_list(payload, "sam_prompts", allow_empty=backend != BACKEND_SAM31)
    _require_str_list(payload, "sam_subtract_prompts", allow_empty=True)
    if backend == BACKEND_YOLO26_SEM and not labels:
        raise ValueError("mask job field 'labels' is required for yolo26_sem")
    if backend == BACKEND_SAM31 and not sam_prompts:
        raise ValueError("mask job field 'sam_prompts' is required for sam31")
    require_str(payload, "image_list", label="mask", allow_empty=True)


def _validate_init(payload: Mapping[str, Any]) -> None:
    _require_images_masks(payload)
    require_str(payload, "image_list", label="mask", allow_empty=True)


def _validate_stitch(payload: Mapping[str, Any]) -> None:
    require_str(payload, "masks", label="mask")
    require_finite_float(payload, "boundary_width", label="mask", min_value=0.0, max_value=180.0, max_inclusive=False)
    require_int_range(payload, "workers", label="mask", min_value=1)
    require_str(payload, "image_list", label="mask", allow_empty=True)


def _validate_overexposure(payload: Mapping[str, Any]) -> None:
    _require_images_masks(payload)
    require_int_range(payload, "threshold", label="mask", min_value=1, max_value=254)
    require_int_range(payload, "dilate", label="mask", min_value=0)
    require_int_range(payload, "workers", label="mask", min_value=1)
    require_bool(payload, "replace", label="mask")
    merge_mode = require_str(payload, "merge_mode", label="mask")
    if merge_mode not in SUPPORTED_MERGE_MODES:
        raise ValueError(f"mask job field 'merge_mode' is invalid: {merge_mode}")
    require_str(payload, "image_list", label="mask", allow_empty=True)


def _validate_custom(payload: Mapping[str, Any]) -> None:
    _require_images_masks(payload)
    require_str(payload, "custom_mask", label="mask")
    require_bool(payload, "replace", label="mask")
    merge_mode = require_str(payload, "merge_mode", label="mask")
    if merge_mode not in SUPPORTED_MERGE_MODES:
        raise ValueError(f"mask job field 'merge_mode' is invalid: {merge_mode}")
    require_str(payload, "image_list", label="mask", allow_empty=True)


def _require_images_masks(payload: Mapping[str, Any]) -> None:
    require_str(payload, "images", label="mask")
    require_str(payload, "masks", label="mask")


def _require_quality_projection(payload: Mapping[str, Any]) -> None:
    quality = require_str(payload, "quality", label="mask")
    if quality not in QUALITY_CHOICES:
        raise ValueError(f"mask job field 'quality' is invalid: {quality}")
    projection = require_str(payload, "projection", label="mask")
    if projection not in {PROJECTION_EQUIRECT, PROJECTION_NORMAL}:
        raise ValueError(f"mask job field 'projection' is invalid: {projection}")


def _require_str_list(payload: Mapping[str, Any], key: str, *, allow_empty: bool) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"mask job field '{key}' must be a list")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"mask job field '{key}[{index}]' must be a string")
        if item.strip():
            items.append(item)
    if not allow_empty and not items:
        raise ValueError(f"mask job field '{key}' is required")
    return items


def _require_int_list(payload: Mapping[str, Any], key: str, *, allow_empty: bool) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"mask job field '{key}' must be a list")
    items: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"mask job field '{key}[{index}]' must be an integer")
        items.append(item)
    if not allow_empty and not items:
        raise ValueError(f"mask job field '{key}' is required")
    return items


def _base_module_cmd(python_executable: str, module_name: str) -> list[str]:
    return [python_executable, "-u", "-m", module_name]


def _append_image_list(cmd: list[str], payload: Mapping[str, Any]) -> None:
    image_list = str(payload.get("image_list") or "")
    if image_list:
        cmd.extend(["--image-list", image_list])


def _yolo_sam_command(python_executable: str, payload: Mapping[str, Any]) -> list[str]:
    cmd = [
        *_base_module_cmd(python_executable, "core.yolo_mask"),
        str(payload["images"]),
        str(payload["masks"]),
        "--quality",
        str(payload["quality"]),
        "--expand",
        str(payload["expand"]),
        "--projection",
        str(payload["projection"]),
        "--merge-mode",
        str(payload["merge_mode"]),
    ]
    classes = _require_int_list(payload, "classes", allow_empty=False)
    if classes:
        cmd.extend(["--classes", ",".join(str(item) for item in classes)])
    _append_image_list(cmd, payload)
    cmd.extend(_require_str_list(payload, "extra_args", allow_empty=True))
    return cmd


def _sky_command(python_executable: str, payload: Mapping[str, Any]) -> list[str]:
    cmd = [
        *_base_module_cmd(python_executable, "core.sky_mask"),
        str(payload["images"]),
        str(payload["masks"]),
        "--backend",
        str(payload["backend"]),
        "--projection",
        str(payload["projection"]),
        "--quality",
        str(payload["quality"]),
        "--inference-size",
        str(payload["inference_size"]),
        "--expand",
        str(payload["expand"]),
        "--min-score",
        _format_float(float(payload["min_score"])),
    ]
    cmd.extend(["--merge-mode", str(payload["merge_mode"])])
    cmd.extend(["--min-area-ratio", _format_float(float(payload["min_area_ratio"]))])
    if bool(payload["top_connected"]):
        cmd.append("--top-connected")
    if payload["backend"] == BACKEND_YOLO26_SEM:
        cmd.extend(["--labels", ",".join(_require_str_list(payload, "labels", allow_empty=False))])
    for prompt in _require_str_list(payload, "sam_prompts", allow_empty=True):
        cmd.extend(["--sam-prompt", prompt])
    for prompt in _require_str_list(payload, "sam_subtract_prompts", allow_empty=True):
        cmd.extend(["--subtract-sam-prompt", prompt])
    if bool(payload["replace"]):
        cmd.append("--replace")
    _append_image_list(cmd, payload)
    if bool(payload["safe_batch"]):
        cmd.append("--safe-batch")
    return cmd


def _init_command(python_executable: str, payload: Mapping[str, Any]) -> list[str]:
    cmd = [*_base_module_cmd(python_executable, "core.init_masks"), str(payload["images"]), str(payload["masks"])]
    _append_image_list(cmd, payload)
    return cmd


def _stitch_command(python_executable: str, payload: Mapping[str, Any]) -> list[str]:
    cmd = [
        *_base_module_cmd(python_executable, "core.stitch_mask"),
        str(payload["masks"]),
        str(payload["masks"]),
        "--boundary-width",
        _format_float(float(payload["boundary_width"])),
        "--workers",
        str(payload["workers"]),
    ]
    _append_image_list(cmd, payload)
    return cmd


def _overexposure_command(python_executable: str, payload: Mapping[str, Any]) -> list[str]:
    cmd = [
        *_base_module_cmd(python_executable, "core.overexposure_mask"),
        str(payload["images"]),
        str(payload["masks"]),
        "--threshold",
        str(payload["threshold"]),
        "--dilate",
        str(payload["dilate"]),
        "--workers",
        str(payload["workers"]),
        "--merge-mode",
        str(payload["merge_mode"]),
    ]
    if bool(payload["replace"]):
        cmd.append("--replace")
    _append_image_list(cmd, payload)
    return cmd


def _custom_command(python_executable: str, payload: Mapping[str, Any]) -> list[str]:
    cmd = [
        *_base_module_cmd(python_executable, "core.custom_mask"),
        str(payload["images"]),
        str(payload["masks"]),
        str(payload["custom_mask"]),
        "--merge-mode",
        str(payload["merge_mode"]),
    ]
    if bool(payload["replace"]):
        cmd.append("--replace")
    _append_image_list(cmd, payload)
    return cmd


def _format_float(value: float) -> str:
    return f"{value:g}"
