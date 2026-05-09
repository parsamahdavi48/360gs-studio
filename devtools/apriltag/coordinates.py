"""Coordinate profile metadata for AprilTag development datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoordinateProfile:
    id: str
    label: str
    pointcloud_note: str
    pointcloud_display_matrix: np.ndarray | None = None
    image_ray_matrix: np.ndarray | None = None


LICHTFELD_FINAL_ORIENTATION = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
LICHTFELD_PRE_FINAL_POINTCLOUD_ALIGNMENT = LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT @ LICHTFELD_FINAL_ORIENTATION

COORDINATE_PROFILE_LICHTFELD_CUBE6 = "lichtfeld_cube6"
COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY = "lichtfeld_cube6_pre_final_ply"
COORDINATE_PROFILE_POSTSHOT_CUBE6 = "postshot_cube6"
COORDINATE_PROFILE_BRUSH_CUBE6 = "brush_cube6"
COORDINATE_PROFILE_CUSTOM = "custom"
DEFAULT_COORDINATE_PROFILE = COORDINATE_PROFILE_LICHTFELD_CUBE6

COORDINATE_PROFILES: tuple[CoordinateProfile, ...] = (
    CoordinateProfile(
        id=COORDINATE_PROFILE_LICHTFELD_CUBE6,
        label="LichtFeld Cube6出力",
        pointcloud_note="output/pointcloud.plyをJSON表示座標へ重ねるため、表示時だけY 180度回転を適用します。",
        pointcloud_display_matrix=LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT,
        image_ray_matrix=LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT,
    ),
    CoordinateProfile(
        id=COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY,
        label="LichtFeld Cube6 JSON + 補正前PLY",
        pointcloud_note="scene直下の補正前pointcloud.plyをJSON表示座標へ重ねるため、表示時だけ複合補正を適用します。",
        pointcloud_display_matrix=LICHTFELD_PRE_FINAL_POINTCLOUD_ALIGNMENT,
        image_ray_matrix=LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT,
    ),
    CoordinateProfile(
        id=COORDINATE_PROFILE_POSTSHOT_CUBE6,
        label="Postshot Cube6",
        pointcloud_note="表示用の追加PLY補正は行いません。",
    ),
    CoordinateProfile(
        id=COORDINATE_PROFILE_BRUSH_CUBE6,
        label="Brush Cube6",
        pointcloud_note="表示用の追加PLY補正は行いません。",
    ),
    CoordinateProfile(
        id=COORDINATE_PROFILE_CUSTOM,
        label="カスタム",
        pointcloud_note="表示用の追加PLY補正は行いません。",
    ),
)
COORDINATE_PROFILE_BY_ID = {profile.id: profile for profile in COORDINATE_PROFILES}


def normalize_coordinate_profile(value: str | None) -> str:
    text = str(value or "").strip()
    if text in COORDINATE_PROFILE_BY_ID:
        return text
    return DEFAULT_COORDINATE_PROFILE


def coordinate_profile_label(value: str | None) -> str:
    return COORDINATE_PROFILE_BY_ID[normalize_coordinate_profile(value)].label


def coordinate_profile_note(value: str | None) -> str:
    return COORDINATE_PROFILE_BY_ID[normalize_coordinate_profile(value)].pointcloud_note


def pointcloud_display_matrix(value: str | None) -> np.ndarray | None:
    matrix = COORDINATE_PROFILE_BY_ID[normalize_coordinate_profile(value)].pointcloud_display_matrix
    return None if matrix is None else matrix.copy()


def image_ray_matrix(value: str | None) -> np.ndarray | None:
    matrix = COORDINATE_PROFILE_BY_ID[normalize_coordinate_profile(value)].image_ray_matrix
    return None if matrix is None else matrix.copy()
