from __future__ import annotations

import numpy as np

from core.cubemap_view_spec import build_remap_spec


def quantize_yaw_offset(yaw_offset: float) -> float:
    """Return the normalized yaw offset used by remap-cache keys."""
    return round(float(yaw_offset) % 360.0, 3)


def remap_cache_key(input_size: tuple[int, int], yaw_offset: float) -> tuple[int, int, float]:
    return int(input_size[0]), int(input_size[1]), quantize_yaw_offset(yaw_offset)


def rot4(r3: np.ndarray) -> np.ndarray:
    r4 = np.eye(4)
    r4[:3, :3] = r3
    return r4


def rotation_matrix(yaw_deg: float, pitch_deg: float, forward: bool) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ]
    )

    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )

    r = rx @ ry if forward else ry @ rx
    r[np.abs(r) < 1e-10] = 0.0
    return r


def build_remap(
    input_size: tuple[int, int],
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    spec = build_remap_spec(
        input_size=input_size,
        output_size=output_size,
        fov_deg=fov_deg,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
    )
    input_size = spec.input_size
    output_size = spec.output_size
    fov_deg = spec.fov_deg
    yaw_deg = spec.yaw_deg
    pitch_deg = spec.pitch_deg

    xs, ys = np.meshgrid(
        np.arange(output_size, dtype=np.float64),
        np.arange(output_size, dtype=np.float64),
    )
    cx = xs - (output_size - 1) / 2.0
    cy = ys - (output_size - 1) / 2.0

    focal = 0.5 * output_size / np.tan(np.deg2rad(fov_deg) / 2.0)

    rays = np.stack([cx, -cy, np.full_like(cx, focal)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    r = rotation_matrix(yaw_deg, pitch_deg, False)
    rays = rays @ r.T

    dx, dy, dz = rays[..., 0], rays[..., 1], rays[..., 2]
    lon = np.arctan2(dx, dz)
    lat = np.arctan2(dy, np.sqrt(dx * dx + dz * dz))

    map_x = (lon / np.pi + 1.0) * 0.5 * input_size[0]
    map_y = (0.5 - lat / np.pi) * input_size[1]

    return map_x.astype(np.float32), map_y.astype(np.float32)
