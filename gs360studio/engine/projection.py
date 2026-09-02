"""Equirectangular-to-perspective projection with bounded map caching."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from gs360studio.domain.models import ViewSpec

_CV_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rotate_yaw = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rotate_pitch = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)))
    rotate_roll = np.array(((cr, -sr, 0.0), (sr, cr, 0.0), (0.0, 0.0, 1.0)))
    return rotate_yaw @ rotate_pitch @ rotate_roll


def build_projection_map(input_size: tuple[int, int], view: ViewSpec) -> tuple[np.ndarray, np.ndarray]:
    input_width, input_height = (int(input_size[0]), int(input_size[1]))
    if input_width <= 0 or input_height <= 0:
        raise ValueError("input dimensions must be positive")

    xs, ys = np.meshgrid(
        np.arange(view.width, dtype=np.float64),
        np.arange(view.height, dtype=np.float64),
    )
    normalized_x = (2.0 * (xs + 0.5) / view.width) - 1.0
    normalized_y = 1.0 - (2.0 * (ys + 0.5) / view.height)
    horizontal_scale = np.tan(np.deg2rad(view.hfov_deg) / 2.0)
    vertical_scale = np.tan(np.deg2rad(view.effective_vfov_deg) / 2.0)
    rays = np.stack(
        (
            normalized_x * horizontal_scale,
            normalized_y * vertical_scale,
            np.ones_like(normalized_x),
        ),
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays @ _rotation_matrix(view.yaw_deg, view.pitch_deg, view.roll_deg).T
    dx, dy, dz = rays[..., 0], rays[..., 1], rays[..., 2]
    longitude = np.arctan2(dx, dz)
    latitude = np.arctan2(dy, np.sqrt(dx * dx + dz * dz))
    map_x = ((longitude / np.pi + 1.0) * 0.5 * input_width).astype(np.float32)
    map_y = ((0.5 - latitude / np.pi) * input_height).astype(np.float32)
    return map_x, map_y


@dataclass(frozen=True, slots=True)
class _MapKey:
    input_size: tuple[int, int]
    view: tuple[object, ...]


class ProjectionMapCache:
    def __init__(self, max_entries: int = 16) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._maps: OrderedDict[_MapKey, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    @staticmethod
    def key(input_size: tuple[int, int], view: ViewSpec) -> _MapKey:
        return _MapKey(
            (int(input_size[0]), int(input_size[1])),
            (
                view.width,
                view.height,
                round(view.hfov_deg, 6),
                round(view.effective_vfov_deg, 6),
                round(view.yaw_deg, 6),
                round(view.pitch_deg, 6),
                round(view.roll_deg, 6),
            ),
        )

    def get(self, input_size: tuple[int, int], view: ViewSpec) -> tuple[np.ndarray, np.ndarray]:
        key = self.key(input_size, view)
        cached = self._maps.pop(key, None)
        if cached is None:
            cached = build_projection_map(input_size, view)
        self._maps[key] = cached
        while len(self._maps) > self.max_entries:
            self._maps.popitem(last=False)
        return cached

    def clear(self) -> None:
        self._maps.clear()

    def __len__(self) -> int:
        return len(self._maps)


def project_equirectangular(
    image: np.ndarray,
    view: ViewSpec,
    *,
    cache: ProjectionMapCache | None = None,
    border_value: int | tuple[int, ...] = 0,
) -> np.ndarray:
    if image is None or image.ndim not in (2, 3):
        raise ValueError("image must be a grayscale or color array")
    input_size = (int(image.shape[1]), int(image.shape[0]))
    active_cache = cache if cache is not None else ProjectionMapCache(max_entries=1)
    map_x, map_y = active_cache.get(input_size, view)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=_CV_INTERPOLATION[view.interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
