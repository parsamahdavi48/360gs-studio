"""Coordinate profile handling for read-only scene preview datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

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
