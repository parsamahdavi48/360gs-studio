from __future__ import annotations

import json
from collections.abc import Mapping
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
    require_views,
)

DATASET_JOB_SCHEMA_VERSION = 1
JOB_KIND_METASHAPE_COLMAP = "metashape_colmap_dataset"
JOB_KIND_METASHAPE_NERF = "metashape_nerf_dataset"
JOB_KIND_REALITYSCAN_LFS_COLMAP = "realityscan_lfs_colmap"


def metashape_colmap_job(
    *,
    scene_dir: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path | None,
    xml_path: str | Path,
    ply_path: str | Path | None,
    output_dir: str | Path,
    views: list[dict[str, Any]],
    output_scale: float,
    output_format: str,
    output_bit_depth: str,
    jpg_quality: int,
    undistort_alpha: float,
    axis_transform: str = "none",
    final_orientation: str = "none",
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_METASHAPE_COLMAP,
        "scene_dir": str(scene_dir),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir) if masks_dir else "",
        "xml_path": str(xml_path),
        "ply_path": str(ply_path) if ply_path else "",
        "output_dir": str(output_dir),
        "views": [dict(view) for view in views],
        "output_scale": float(output_scale),
        "output_format": str(output_format),
        "output_bit_depth": str(output_bit_depth),
        "jpg_quality": int(jpg_quality),
        "undistort_alpha": float(undistort_alpha),
        "axis_transform": str(axis_transform),
        "final_orientation": str(final_orientation),
    }


def metashape_nerf_job(
    *,
    scene_dir: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path | None,
    xml_path: str | Path,
    ply_path: str | Path | None,
    output_dir: str | Path,
    views: list[dict[str, Any]],
    output_scale: float,
    output_format: str,
    output_bit_depth: str,
    jpg_quality: int,
    undistort_alpha: float,
    axis_transform: str,
    final_orientation: str,
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_METASHAPE_NERF,
        "scene_dir": str(scene_dir),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir) if masks_dir else "",
        "xml_path": str(xml_path),
        "ply_path": str(ply_path) if ply_path else "",
        "output_dir": str(output_dir),
        "views": [dict(view) for view in views],
        "output_scale": float(output_scale),
        "output_format": str(output_format),
        "output_bit_depth": str(output_bit_depth),
        "jpg_quality": int(jpg_quality),
        "undistort_alpha": float(undistort_alpha),
        "axis_transform": str(axis_transform),
        "final_orientation": str(final_orientation),
    }


def realityscan_lfs_colmap_job(
    *,
    csv_path: str | Path,
    output_dir: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path | None,
    ply_path: str | Path | None,
    skip_missing_images: bool,
    pre_undistort_distorted_images: bool,
    undistort_alpha: float,
    camera_rotation_x_deg: float = 90.0,
    pointcloud_rotation_x_deg: float = 90.0,
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_JOB_SCHEMA_VERSION,
        "kind": JOB_KIND_REALITYSCAN_LFS_COLMAP,
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir) if masks_dir else "",
        "ply_path": str(ply_path) if ply_path else "",
        "skip_missing_images": bool(skip_missing_images),
        "pre_undistort_distorted_images": bool(pre_undistort_distorted_images),
        "undistort_alpha": float(undistort_alpha),
        "camera_rotation_x_deg": float(camera_rotation_x_deg),
        "pointcloud_rotation_x_deg": float(pointcloud_rotation_x_deg),
    }


def write_dataset_job(path: str | Path, payload: dict[str, Any]) -> Path:
    job_path = Path(path)
    validate_dataset_job_payload(payload)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return job_path


def load_dataset_job(path: str | Path, *, expected_kind: str = "") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset job must be a JSON object: {path}")
    validate_dataset_job_payload(payload)
    if expected_kind and payload["kind"] != expected_kind:
        raise ValueError(f"Dataset job kind must be {expected_kind}: {payload['kind']}")
    return payload


def validate_dataset_job_payload(payload: dict[str, Any]) -> None:
    data = require_mapping(payload, label="dataset")
    require_schema_version(data, expected=DATASET_JOB_SCHEMA_VERSION, label="dataset")
    kind = require_kind(
        data,
        allowed={JOB_KIND_METASHAPE_COLMAP, JOB_KIND_METASHAPE_NERF, JOB_KIND_REALITYSCAN_LFS_COLMAP},
        label="dataset",
    )
    if kind in {JOB_KIND_METASHAPE_COLMAP, JOB_KIND_METASHAPE_NERF}:
        _validate_metashape_dataset_job(data)
    elif kind == JOB_KIND_REALITYSCAN_LFS_COLMAP:
        _validate_realityscan_lfs_colmap_job(data)


def _validate_metashape_dataset_job(payload: Mapping[str, Any]) -> None:
    for key in ("scene_dir", "images_dir", "xml_path", "output_dir", "output_format", "output_bit_depth"):
        require_str(payload, key, label="dataset")
    require_str(payload, "masks_dir", label="dataset", allow_empty=True)
    require_str(payload, "ply_path", label="dataset", allow_empty=True)
    require_str(payload, "axis_transform", label="dataset")
    require_str(payload, "final_orientation", label="dataset")
    require_views(payload, label="dataset")
    require_finite_float(payload, "output_scale", label="dataset", min_value=0.0, max_value=1.0, min_inclusive=False)
    require_finite_float(payload, "undistort_alpha", label="dataset", min_value=0.0, max_value=1.0)
    require_int_range(payload, "jpg_quality", label="dataset", min_value=1, max_value=100)


def _validate_realityscan_lfs_colmap_job(payload: Mapping[str, Any]) -> None:
    for key in ("csv_path", "output_dir", "images_dir"):
        require_str(payload, key, label="dataset")
    require_str(payload, "masks_dir", label="dataset", allow_empty=True)
    require_str(payload, "ply_path", label="dataset", allow_empty=True)
    require_bool(payload, "skip_missing_images", label="dataset")
    require_bool(payload, "pre_undistort_distorted_images", label="dataset")
    require_finite_float(payload, "undistort_alpha", label="dataset", min_value=0.0, max_value=1.0)
    require_finite_float(payload, "camera_rotation_x_deg", label="dataset")
    require_finite_float(payload, "pointcloud_rotation_x_deg", label="dataset")
