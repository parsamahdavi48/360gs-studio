"""Cubemap-frame preview helpers for the AprilTag placement dev GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.apriltag_geometry import PinholeFrame, load_pinhole_frames
from core.image_io import imread_unicode

_FACE_NAMES = ("px", "nx", "pz", "nz", "top", "bottom", "py", "ny")
_REFERENCE_FACE_ORDER = ("pz", "px", "nz", "nx", "top", "bottom", "py", "ny")


@dataclass(frozen=True)
class CubemapFrameGroup:
    name: str
    frames_by_face: dict[str, PinholeFrame]

    @property
    def frames(self) -> tuple[PinholeFrame, ...]:
        return tuple(self.frames_by_face.values())

    @property
    def reference_frame(self) -> PinholeFrame:
        for face in _REFERENCE_FACE_ORDER:
            frame = self.frames_by_face.get(face)
            if frame is not None:
                return frame
        return next(iter(self.frames_by_face.values()))

    @property
    def camera_position_sfm(self) -> np.ndarray:
        return self.reference_frame.camera_position_sfm


def split_cubemap_face(file_path: str) -> tuple[str, str] | None:
    path = Path(file_path)
    stem = path.stem
    lower = stem.lower()
    for face in _FACE_NAMES:
        suffix = f"_{face}"
        if lower.endswith(suffix):
            prefix = stem[: -len(suffix)]
            return prefix, face
    return None


def load_cubemap_frame_groups(transforms_json: Path) -> tuple[CubemapFrameGroup, ...]:
    groups: dict[str, dict[str, PinholeFrame]] = {}
    for frame in load_pinhole_frames(transforms_json):
        parsed = split_cubemap_face(frame.file_path)
        if parsed is None:
            continue
        prefix, face = parsed
        groups.setdefault(prefix, {})[face] = frame
    return tuple(
        CubemapFrameGroup(name=name, frames_by_face=frames)
        for name, frames in sorted(groups.items())
        if len(frames) >= 4
    )


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
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
    return ry @ rx


def virtual_camera_rotation(group: CubemapFrameGroup, *, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Return a camera-to-world rotation for the interactive preview view."""
    return _rotation_matrix(yaw_deg, pitch_deg) @ group.reference_frame.camera_to_world_rotation


def _view_rays(output_size: int, fov_deg: float) -> np.ndarray:
    size = max(1, int(output_size))
    xs, ys = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    cx = xs - (size - 1) / 2.0
    cy = ys - (size - 1) / 2.0
    focal = 0.5 * size / np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    rays = np.stack([cx, -cy, np.full_like(cx, focal)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def view_pixel_to_world_ray(
    group: CubemapFrameGroup,
    *,
    x_px: float,
    y_px: float,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
) -> np.ndarray:
    size = max(1, int(output_size))
    cx = float(x_px) - (size - 1) / 2.0
    cy = float(y_px) - (size - 1) / 2.0
    focal = 0.5 * size / np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    ray = np.array([cx, -cy, focal], dtype=np.float64)
    ray /= max(float(np.linalg.norm(ray)), 1e-12)
    return ray @ virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg).T


def view_up_world(group: CubemapFrameGroup, *, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up = up @ virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg).T
    return up / max(float(np.linalg.norm(up)), 1e-12)


def render_cubemap_perspective(
    group: CubemapFrameGroup,
    *,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float = 90.0,
    output_size: int = 768,
    image_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray:
    size = max(1, int(output_size))
    rays = _view_rays(size, fov_deg)
    world_rays = rays @ virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg).T

    output = np.full((size, size, 3), 16, dtype=np.uint8)
    best_score = np.full((size, size), -np.inf, dtype=np.float64)
    for frame in group.frames:
        image = None
        if image_cache is not None:
            image = image_cache.get(frame.image_path)
        if image is None:
            image = imread_unicode(frame.image_path)
            if image_cache is not None and image is not None:
                image_cache[frame.image_path] = image
        if image is None:
            continue
        image_bgr = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        local = world_rays @ frame.camera_to_world_rotation
        z = local[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            map_x = frame.fl_x * (local[:, :, 0] / z) + frame.cx
            map_y = frame.cy - frame.fl_y * (local[:, :, 1] / z)
        valid = (
            (z > 1e-8)
            & (map_x >= 0.0)
            & (map_y >= 0.0)
            & (map_x < frame.width - 1)
            & (map_y < frame.height - 1)
        )
        better = valid & (z > best_score)
        if not np.any(better):
            continue
        sampled = cv2.remap(
            image_bgr,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(16, 16, 16),
        )
        output[better] = sampled[better]
        best_score[better] = z[better]
    return output


def _equirect_world_rays(width: int, height: int, reference_rotation: np.ndarray) -> np.ndarray:
    xs = (np.arange(width, dtype=np.float64) + 0.5) / max(width, 1)
    ys = (np.arange(height, dtype=np.float64) + 0.5) / max(height, 1)
    lon = (xs * 2.0 - 1.0) * np.pi
    lat = (0.5 - ys) * np.pi
    cos_lat = np.cos(lat)
    rays_ref = np.stack(
        [
            cos_lat[:, None] * np.sin(lon)[None, :],
            np.sin(lat)[:, None] * np.ones((1, width), dtype=np.float64),
            cos_lat[:, None] * np.cos(lon)[None, :],
        ],
        axis=-1,
    )
    return rays_ref @ reference_rotation.T


def render_cubemap_equirect(
    group: CubemapFrameGroup,
    *,
    output_width: int = 2048,
    output_height: int = 1024,
    image_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray:
    """Rebuild a lightweight equirectangular preview texture from cubemap faces.

    This is intended as a one-time texture build per selected frame group. The
    interactive perspective view can then use the existing OpenGL equirect shader.
    """
    width = max(1, int(output_width))
    height = max(1, int(output_height))
    world_rays = _equirect_world_rays(width, height, group.reference_frame.camera_to_world_rotation)

    output = np.full((height, width, 3), 16, dtype=np.uint8)
    best_score = np.full((height, width), -np.inf, dtype=np.float64)
    for frame in group.frames:
        image = None
        if image_cache is not None:
            image = image_cache.get(frame.image_path)
        if image is None:
            image = imread_unicode(frame.image_path)
            if image_cache is not None and image is not None:
                image_cache[frame.image_path] = image
        if image is None:
            continue
        image_bgr = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        local = world_rays @ frame.camera_to_world_rotation
        z = local[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            map_x = frame.fl_x * (local[:, :, 0] / z) + frame.cx
            map_y = frame.cy - frame.fl_y * (local[:, :, 1] / z)
        valid = (
            (z > 1e-8)
            & (map_x >= 0.0)
            & (map_y >= 0.0)
            & (map_x < frame.width - 1)
            & (map_y < frame.height - 1)
        )
        better = valid & (z > best_score)
        if not np.any(better):
            continue
        sampled = cv2.remap(
            image_bgr,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(16, 16, 16),
        )
        output[better] = sampled[better]
        best_score[better] = z[better]
    return output
