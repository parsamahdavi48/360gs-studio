"""Shared view recipes for primary mask generation.

The quality level controls which source views are fed to the selected model.
Projection-specific helpers keep 360-degree pole handling out of individual
backends so YOLO/SAM2.1, Mask2Former, and SAM3.1 can share the same inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

QUALITY_STANDARD = "standard"
QUALITY_HIGH = "high"
QUALITY_BEST = "best"
QUALITY_CHOICES = (QUALITY_STANDARD, QUALITY_HIGH, QUALITY_BEST)
DEFAULT_QUALITY = QUALITY_HIGH
PROJECTION_EQUIRECT = "equirect"
PROJECTION_NORMAL = "normal"


@dataclass(frozen=True)
class TileSpec:
    columns: int
    rows: int
    top_ratio: float
    bottom_ratio: float
    pad_px: int = 20


@dataclass(frozen=True)
class TileRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    index: int
    total: int
    purpose: str = "center"


@dataclass(frozen=True)
class MaskViewRecipe:
    quality: str
    projection: str
    yolo_level: int
    direct: bool
    tile_spec: TileSpec | None
    top_view: bool
    bottom_view: bool
    bottom_rotations: tuple[int, ...]
    bottom_conf: float
    bottom_model: str
    bottom_filter: bool


_top_extract_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
_bottom_extract_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
_top_back_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_bottom_back_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def normalize_projection(projection: str) -> str:
    value = str(projection).strip().lower()
    return PROJECTION_NORMAL if value == PROJECTION_NORMAL else PROJECTION_EQUIRECT


def quality_from_legacy_level(level: int | str | None) -> str:
    if level is None:
        return DEFAULT_QUALITY
    try:
        value = int(level)
    except (TypeError, ValueError):
        return normalize_quality(str(level))
    if value <= 1:
        return QUALITY_STANDARD
    if value == 2:
        return QUALITY_HIGH
    return QUALITY_BEST


def normalize_quality(quality: str | None, *, legacy_level: int | str | None = None) -> str:
    if quality is None or str(quality).strip() == "":
        return quality_from_legacy_level(legacy_level)
    value = str(quality).strip().lower().replace("_", "-")
    aliases = {
        "std": QUALITY_STANDARD,
        "normal": QUALITY_STANDARD,
        "fast": QUALITY_STANDARD,
        "1": QUALITY_STANDARD,
        "quality": QUALITY_HIGH,
        "high-quality": QUALITY_HIGH,
        "high": QUALITY_HIGH,
        "2": QUALITY_HIGH,
        "max": QUALITY_BEST,
        "best": QUALITY_BEST,
        "3": QUALITY_BEST,
    }
    return aliases.get(value, value if value in QUALITY_CHOICES else DEFAULT_QUALITY)


def recipe_for(quality: str | None, projection: str, *, legacy_level: int | str | None = None) -> MaskViewRecipe:
    q = normalize_quality(quality, legacy_level=legacy_level)
    p = normalize_projection(projection)

    if p == PROJECTION_NORMAL:
        if q == QUALITY_STANDARD:
            return MaskViewRecipe(q, p, 1, True, None, False, False, (), 0.30, "same", False)
        if q == QUALITY_HIGH:
            return MaskViewRecipe(q, p, 2, True, TileSpec(4, 2, 0.0, 1.0), False, False, (), 0.30, "same", False)
        return MaskViewRecipe(q, p, 3, True, TileSpec(8, 4, 0.0, 1.0), False, False, (), 0.30, "same", False)

    if q == QUALITY_STANDARD:
        return MaskViewRecipe(q, p, 1, True, None, False, True, (0,), 0.30, "same", False)
    if q == QUALITY_HIGH:
        return MaskViewRecipe(q, p, 2, True, TileSpec(4, 1, 0.25, 0.75), True, True, (0, 90, 180, 270), 0.15, "same", True)
    return MaskViewRecipe(q, p, 3, True, TileSpec(8, 2, 0.20, 0.80), True, True, (0, 90, 180, 270), 0.10, "x", True)


def iter_tile_regions(width: int, height: int, spec: TileSpec | None) -> list[TileRegion]:
    if spec is None or width <= 0 or height <= 0:
        return []
    columns = max(1, int(spec.columns))
    rows = max(1, int(spec.rows))
    top_y = int(height * max(0.0, min(1.0, spec.top_ratio)))
    bottom_y = int(height * max(0.0, min(1.0, spec.bottom_ratio)))
    if bottom_y <= top_y:
        return []
    tile_w = max(1, width // columns)
    tile_h = max(1, (bottom_y - top_y) // rows)
    pad = max(0, int(spec.pad_px))
    total = columns * rows
    regions: list[TileRegion] = []
    for i in range(columns):
        raw_x1 = i * tile_w
        raw_x2 = width if i == columns - 1 else (i + 1) * tile_w
        x1 = max(0, raw_x1 - pad)
        x2 = min(width, raw_x2 + pad)
        for j in range(rows):
            raw_y1 = top_y + j * tile_h
            raw_y2 = bottom_y if j == rows - 1 else top_y + (j + 1) * tile_h
            y1 = max(0, raw_y1 - pad)
            y2 = min(height, raw_y2 + pad)
            if x2 > x1 and y2 > y1:
                regions.append(TileRegion(x1, y1, x2, y2, len(regions), total))
    return regions


def auto_view_size(width: int, height: int) -> int:
    return max(512, min(2048, int(width) // 4 if width > 0 else int(height)))


def extract_top_view(pano_img: np.ndarray, size: int) -> np.ndarray:
    return extract_cube_pole_view(pano_img, size, pole="top")


def extract_bottom_view(pano_img: np.ndarray, size: int) -> np.ndarray:
    return extract_cube_pole_view(pano_img, size, pole="bottom")


def extract_cube_pole_view(pano_img: np.ndarray, size: int, *, pole: str) -> np.ndarray:
    height, width = pano_img.shape[:2]
    size = int(size)
    cache = _top_extract_cache if pole == "top" else _bottom_extract_cache
    key = (width, height, size)
    cached = cache.get(key)
    if cached is None:
        u = np.linspace(-1, 1, size, dtype=np.float32)
        v = np.linspace(-1, 1, size, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)
        x = u_grid
        y = v_grid
        z = np.ones_like(u_grid) if pole == "top" else -np.ones_like(u_grid)
        lon = np.arctan2(y, x)
        lat = np.arctan2(z, np.sqrt(x**2 + y**2))
        map_x = ((lon + np.pi) / (2 * np.pi) * (width - 1)).astype(np.float32)
        map_y = ((np.pi / 2 - lat) / np.pi * (height - 1)).astype(np.float32)
        cache[key] = (map_x, map_y)
    else:
        map_x, map_y = cached
    return cv2.remap(pano_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def back_project_top_mask(top_mask: np.ndarray, pano_width: int, pano_height: int) -> np.ndarray:
    return back_project_cube_pole_mask(top_mask, pano_width, pano_height, pole="top")


def back_project_bottom_mask(bottom_mask: np.ndarray, pano_width: int, pano_height: int) -> np.ndarray:
    return back_project_cube_pole_mask(bottom_mask, pano_width, pano_height, pole="bottom")


def back_project_cube_pole_mask(pole_mask: np.ndarray, pano_width: int, pano_height: int, *, pole: str) -> np.ndarray:
    sign = 1.0 if pole == "top" else -1.0
    size = int(pole_mask.shape[0])
    cache = _top_back_cache if pole == "top" else _bottom_back_cache
    key = (int(pano_width), int(pano_height), size)
    cached = cache.get(key)
    if cached is None:
        lon = np.linspace(-np.pi, np.pi, int(pano_width), dtype=np.float32)
        lat = np.linspace(np.pi / 2, -np.pi / 2, int(pano_height), dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.cos(lat_grid) * np.sin(lon_grid)
        z = np.sin(lat_grid)
        abs_z = np.abs(z)
        is_pole = (sign * z > 0) & (abs_z >= np.abs(x)) & (abs_z >= np.abs(y))
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[is_pole] = x[is_pole] / abs_z[is_pole]
        v[is_pole] = y[is_pole] / abs_z[is_pole]
        map_x = ((u + 1) / 2 * (size - 1)).astype(np.float32)
        map_y = ((v + 1) / 2 * (size - 1)).astype(np.float32)
        cache[key] = (map_x, map_y, is_pole)
    else:
        map_x, map_y, is_pole = cached

    mapped = cv2.remap(pole_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if pole_mask.ndim == 3:
        result = np.zeros((int(pano_height), int(pano_width), pole_mask.shape[2]), dtype=np.uint8)
    else:
        result = np.zeros((int(pano_height), int(pano_width)), dtype=np.uint8)
    result[is_pole] = mapped[is_pole]
    return result


def rotate_quarter_turn(image: np.ndarray, angle: int) -> np.ndarray:
    angle = int(angle) % 360
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("angle must be a multiple of 90")


def quarter_turn_angles(rotation_count: int) -> tuple[int, ...]:
    if rotation_count == 1:
        return (0,)
    if rotation_count == 2:
        return (0, 180)
    if rotation_count == 4:
        return (0, 90, 180, 270)
    raise ValueError("rotation count must be one of: 1, 2, 4")


def transform_bbox_from_rotated_view(
    bbox: Iterable[float],
    angle: int,
    *,
    width: int,
    height: int,
) -> list[float] | None:
    angle = int(angle) % 360
    x1, y1, x2, y2 = [float(v) for v in bbox]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def to_original(x: float, y: float) -> tuple[float, float]:
        if angle == 0:
            return x, y
        if angle == 90:
            return y, height - x
        if angle == 180:
            return width - x, height - y
        if angle == 270:
            return width - y, x
        raise ValueError("angle must be a multiple of 90")

    transformed = [to_original(x, y) for x, y in corners]
    xs = [x for x, _y in transformed]
    ys = [y for _x, y in transformed]
    out_x1 = max(0.0, min(float(width), min(xs)))
    out_y1 = max(0.0, min(float(height), min(ys)))
    out_x2 = max(0.0, min(float(width), max(xs)))
    out_y2 = max(0.0, min(float(height), max(ys)))
    if out_x2 <= out_x1 or out_y2 <= out_y1:
        return None
    return [out_x1, out_y1, out_x2, out_y2]
