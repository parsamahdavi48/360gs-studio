"""Coordinate profile handling for read-only scene preview datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.orientation_correction import (
    FINAL_ORIENTATION_REALITYSCAN,
    REALITYSCAN_FINAL_ORIENTATION_MATRIX,
)

COORDINATE_PROFILE_LICHTFELD_CUBE6 = "lichtfeld_cube6"
COORDINATE_PROFILE_POSTSHOT_CUBE6 = "postshot_cube6"
COORDINATE_PROFILE_BRUSH_CUBE6 = "brush_cube6"
COORDINATE_PROFILE_REALITYSCAN = "realityscan"
COORDINATE_PROFILE_CUSTOM = "custom"

LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP = np.linalg.inv(REALITYSCAN_FINAL_ORIENTATION_MATRIX)
REALITYSCAN_LFS_FILE_TO_PREVIEW_Y_UP = (
    REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP @ np.linalg.inv(REALITYSCAN_FINAL_ORIENTATION_MATRIX)
)


@dataclass(frozen=True)
class ScenePreviewDisplayTransform:
    profile: str
    note: str
    camera_matrix: np.ndarray | None = None
    pointcloud_matrix: np.ndarray | None = None


def step4_output_display_transform(settings: Mapping[str, Any]) -> ScenePreviewDisplayTransform | None:
    """Return display transforms that match the Step 4/AprilTag scene viewer contract."""
    if not settings:
        return None

    profile = _profile_from_settings(settings)
    if profile == COORDINATE_PROFILE_LICHTFELD_CUBE6:
        return ScenePreviewDisplayTransform(
            profile=profile,
            note=f"step4_output / {profile}",
            camera_matrix=LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT.copy(),
            pointcloud_matrix=None,
        )
    return ScenePreviewDisplayTransform(
        profile=profile,
        note=f"step4_output / {profile}",
    )


def transforms_dataset_display_transform(
    transforms_json: str | Path,
    *,
    fallback_settings: Mapping[str, Any] | None = None,
) -> ScenePreviewDisplayTransform | None:
    """Return read-only display transforms for a concrete transforms.json dataset."""
    path = Path(transforms_json)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    source = data.get("source")
    if isinstance(source, dict) and str(source.get("type") or "").strip().lower() == "realityscan_csv":
        transform = _realityscan_transforms_display_transform(source)
        if transform is not None:
            return transform

    final_orientation = _nested_text(data, "postprocess", "final_orientation") or _nested_text(source, "final_orientation")
    if str(final_orientation).strip().lower() == FINAL_ORIENTATION_REALITYSCAN:
        orientation_matrix = _matrix4_or_none(_nested_value(data, "postprocess", "final_orientation_matrix"))
        if orientation_matrix is None:
            orientation_matrix = REALITYSCAN_FINAL_ORIENTATION_MATRIX.copy()
        return ScenePreviewDisplayTransform(
            profile=COORDINATE_PROFILE_REALITYSCAN,
            note=f"transforms_json / {COORDINATE_PROFILE_REALITYSCAN}",
            camera_matrix=np.linalg.inv(orientation_matrix),
            pointcloud_matrix=None,
        )

    axis_transform = _nested_text(source, "axis_transform")
    target_profile = _nested_text(source, "target_profile")
    if str(final_orientation).strip().lower() == "lichtfeld" or (
        str(axis_transform).strip().lower() == "none" and str(target_profile).strip().lower() == "lichtfeld"
    ):
        return ScenePreviewDisplayTransform(
            profile=COORDINATE_PROFILE_LICHTFELD_CUBE6,
            note=f"transforms_json / {COORDINATE_PROFILE_LICHTFELD_CUBE6}",
            camera_matrix=LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT.copy(),
            pointcloud_matrix=None,
        )

    if fallback_settings:
        return step4_output_display_transform(fallback_settings)
    return None


def realityscan_csv_display_transform() -> ScenePreviewDisplayTransform:
    return ScenePreviewDisplayTransform(
        profile=COORDINATE_PROFILE_REALITYSCAN,
        note=f"realityscan_csv / {COORDINATE_PROFILE_REALITYSCAN}",
        camera_matrix=REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP.copy(),
        pointcloud_matrix=REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP.copy(),
    )


def realityscan_lfs_colmap_display_transform() -> ScenePreviewDisplayTransform:
    return ScenePreviewDisplayTransform(
        profile=COORDINATE_PROFILE_LICHTFELD_CUBE6,
        note="colmap / realityscan_lfs / preview_y_up",
        camera_matrix=REALITYSCAN_LFS_FILE_TO_PREVIEW_Y_UP.copy(),
        pointcloud_matrix=REALITYSCAN_LFS_FILE_TO_PREVIEW_Y_UP.copy(),
    )


def realityscan_colmap_export_display_transform() -> ScenePreviewDisplayTransform:
    return ScenePreviewDisplayTransform(
        profile=COORDINATE_PROFILE_REALITYSCAN,
        note="colmap / realityscan / preview_y_up",
        camera_matrix=REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP.copy(),
        pointcloud_matrix=REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP.copy(),
    )


def _profile_from_settings(settings: Mapping[str, Any]) -> str:
    effective_profile = str(
        settings.get("effective_profile") or settings.get("target_profile") or ""
    ).strip().lower()
    axis_transform = str(settings.get("axis_transform") or "").strip().lower()
    if effective_profile == COORDINATE_PROFILE_REALITYSCAN:
        return COORDINATE_PROFILE_REALITYSCAN
    if axis_transform == "brush" or effective_profile == "brush":
        return COORDINATE_PROFILE_BRUSH_CUBE6
    if axis_transform == "postshot" or effective_profile == "postshot":
        return COORDINATE_PROFILE_POSTSHOT_CUBE6
    if axis_transform == "none" or effective_profile == "lichtfeld":
        return COORDINATE_PROFILE_LICHTFELD_CUBE6
    return COORDINATE_PROFILE_CUSTOM


def _realityscan_transforms_display_transform(source: Mapping[str, Any]) -> ScenePreviewDisplayTransform | None:
    target_profile = str(source.get("target_profile") or "").strip().lower()
    if target_profile != "lichtfeld":
        return None
    camera_matrix = _matrix4_or_none(source.get("target_profile_matrix"))
    pointcloud_matrix = _matrix4_or_none(source.get("pointcloud_profile_matrix"))
    if camera_matrix is None or pointcloud_matrix is None:
        return ScenePreviewDisplayTransform(
            profile=COORDINATE_PROFILE_LICHTFELD_CUBE6,
            note="transforms_json / realityscan / lichtfeld",
        )
    return ScenePreviewDisplayTransform(
        profile=COORDINATE_PROFILE_LICHTFELD_CUBE6,
        note="transforms_json / realityscan / lichtfeld",
        camera_matrix=None,
        pointcloud_matrix=camera_matrix @ np.linalg.inv(pointcloud_matrix),
    )


def _matrix4_or_none(value: object) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _nested_text(value: object, *keys: str) -> str:
    current = _nested_value(value, *keys)
    return str(current or "")


def _nested_value(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current
