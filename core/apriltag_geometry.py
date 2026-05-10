"""Shared pinhole-frame geometry for AprilTag scale development."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.apriltag_cubemap import CubemapViewMetadata


@dataclass(frozen=True)
class PinholeFrame:
    """One pinhole frame from transforms.json."""

    frame_id: str
    file_path: str
    image_path: Path
    width: int
    height: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    transform_matrix: np.ndarray

    @property
    def camera_position_sfm(self) -> np.ndarray:
        return self.transform_matrix[:3, 3]

    @property
    def camera_to_world_rotation(self) -> np.ndarray:
        return self.transform_matrix[:3, :3]

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fl_x, 0.0, self.cx],
                [0.0, self.fl_y, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


def _frame_intrinsics(frame: dict, data: dict) -> tuple[float, float, float, float, int, int]:
    w = int(frame.get("w", data.get("w", 0)))
    h = int(frame.get("h", data.get("h", 0)))
    fl_x = float(frame.get("fl_x", data.get("fl_x", 0.0)))
    fl_y = float(frame.get("fl_y", data.get("fl_y", fl_x)))
    cx = float(frame.get("cx", data.get("cx", (w - 1) / 2.0)))
    cy = float(frame.get("cy", data.get("cy", (h - 1) / 2.0)))
    if w <= 0 or h <= 0 or fl_x <= 0.0 or fl_y <= 0.0:
        raise ValueError("Invalid or missing pinhole intrinsics in transforms.json")
    return fl_x, fl_y, cx, cy, w, h


def _resolve_image_path(image_root: Path, file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path
    return image_root / file_path


def load_pinhole_frames(
    transforms_json: Path,
    image_root: Path | None = None,
    *,
    normalize_cubemap: bool = True,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
    discover_cubemap_views: bool = True,
) -> tuple[PinholeFrame, ...]:
    """Load PINHOLE or SIMPLE_PINHOLE frames from a transforms.json file."""
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    if data.get("camera_model") not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError("AprilTag scale estimation expects PINHOLE or SIMPLE_PINHOLE transforms.json")

    root = image_root if image_root is not None else transforms_json.parent
    frames: list[PinholeFrame] = []
    for index, frame in enumerate(data.get("frames", [])):
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        fl_x, fl_y, cx, cy, w, h = _frame_intrinsics(frame, data)
        transform = np.asarray(frame.get("transform_matrix"), dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"Invalid transform_matrix for {file_path}")
        frames.append(
            PinholeFrame(
                frame_id=str(frame.get("name") or file_path or index),
                file_path=file_path,
                image_path=_resolve_image_path(root, file_path),
                width=w,
                height=h,
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
                transform_matrix=transform,
            )
        )
    loaded = tuple(frames)
    if normalize_cubemap:
        from core.apriltag_cubemap import discover_cubemap_view_metadata, normalize_standard_cubemap_frames

        view_params = cubemap_view_params
        if view_params is None and discover_cubemap_views:
            view_params = discover_cubemap_view_metadata(transforms_json)
        loaded = normalize_standard_cubemap_frames(loaded, view_params=view_params)
    return loaded


def normalized(vec: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError(f"{name} must not be a zero vector")
    return vec / norm


def tag_corners_sfm(
    center: np.ndarray,
    normal: np.ndarray,
    up: np.ndarray,
    tag_size_m: float,
    true_scale: float,
) -> np.ndarray:
    """Return synthetic tag corners in SfM units."""
    normal = normalized(normal, "--tag-normal-sfm")
    up = up - normal * float(up @ normal)
    up = normalized(up, "--tag-up-sfm")
    right = normalized(np.cross(normal, up), "computed tag right")
    half_size = float(tag_size_m) / float(true_scale) / 2.0
    return np.array(
        [
            center - right * half_size + up * half_size,
            center + right * half_size + up * half_size,
            center + right * half_size - up * half_size,
            center - right * half_size - up * half_size,
        ],
        dtype=float,
    )


def project_sfm_points(frame: PinholeFrame, points_sfm: np.ndarray) -> np.ndarray | None:
    """Project SfM-world points into a pinhole frame."""
    camera_points = (points_sfm - frame.camera_position_sfm) @ frame.camera_to_world_rotation
    z = camera_points[:, 2]
    if np.any(z <= 1e-8):
        return None
    u = frame.fl_x * (camera_points[:, 0] / z) + frame.cx
    v = frame.cy - frame.fl_y * (camera_points[:, 1] / z)
    return np.stack([u, v], axis=1).astype(np.float32)


def points_intersect_image(points: np.ndarray, width: int, height: int) -> bool:
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    return bool(max_xy[0] >= 0 and max_xy[1] >= 0 and min_xy[0] < width and min_xy[1] < height)
