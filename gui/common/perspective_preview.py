"""Shared equirectangular-to-perspective preview helpers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

PREVIEW_PROJECTION_EQUIRECT = "equirect"
PREVIEW_PROJECTION_PERSPECTIVE = "perspective"
PREVIEW_PROJECTIONS = {PREVIEW_PROJECTION_EQUIRECT, PREVIEW_PROJECTION_PERSPECTIVE}
PERSPECTIVE_PREVIEW_FOV_DEG = 90.0
PERSPECTIVE_LOOK_DEG_PER_PIXEL = 0.18


@dataclass(frozen=True)
class PerspectiveParams:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    fov_deg: float = PERSPECTIVE_PREVIEW_FOV_DEG
    roll_deg: float = 0.0


def normalize_yaw_deg(yaw_deg: float) -> float:
    return ((float(yaw_deg) + 180.0) % 360.0) - 180.0


def clamp_pitch_deg(pitch_deg: float) -> float:
    return max(-89.0, min(89.0, float(pitch_deg)))


def params_from_drag(
    params: PerspectiveParams,
    delta_x: float,
    delta_y: float,
    *,
    degrees_per_pixel: float = PERSPECTIVE_LOOK_DEG_PER_PIXEL,
) -> PerspectiveParams:
    return PerspectiveParams(
        yaw_deg=normalize_yaw_deg(params.yaw_deg - float(delta_x) * degrees_per_pixel),
        pitch_deg=clamp_pitch_deg(params.pitch_deg - float(delta_y) * degrees_per_pixel),
        fov_deg=params.fov_deg,
        roll_deg=params.roll_deg,
    )


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ry @ rx @ rz


@lru_cache(maxsize=8)
def perspective_remap_maps(
    source_width: int,
    source_height: int,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    output_size = max(1, int(output_size))
    fov_deg = max(1.0, min(179.0, float(fov_deg)))

    coords = (np.arange(output_size, dtype=np.float64) + 0.5) / float(output_size)
    u = coords * 2.0 - 1.0
    v = 1.0 - coords * 2.0
    uu, vv = np.meshgrid(u, v)

    focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    rays = np.stack([uu, vv, np.full_like(uu, focal)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays @ _rotation_matrix(float(yaw_deg), float(pitch_deg), float(roll_deg)).T

    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))

    map_x = ((lon / np.pi + 1.0) * 0.5 * source_width).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * source_height).astype(np.float32)
    map_x = np.mod(map_x, float(source_width))
    map_y = np.clip(map_y, 0.0, float(source_height - 1))
    return map_x, map_y


def perspective_output_size(image: np.ndarray) -> int:
    height, width = image.shape[:2]
    return max(1, min(int(width), int(height)))


def equirect_to_perspective(
    image: np.ndarray,
    params: PerspectiveParams | None = None,
    *,
    output_size: int | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    params = params or PerspectiveParams()
    if image is None or image.size == 0:
        return image
    height, width = image.shape[:2]
    size = perspective_output_size(image) if output_size is None else max(1, int(output_size))
    map_x, map_y = perspective_remap_maps(
        int(width),
        int(height),
        int(size),
        round(float(params.yaw_deg), 6),
        round(float(params.pitch_deg), 6),
        round(float(params.roll_deg), 6),
        round(float(params.fov_deg), 6),
    )
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_WRAP,
    )
