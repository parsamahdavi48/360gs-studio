"""Cubemap preview helpers used by the production scene preview UI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.apriltag_cubemap import (
    CubemapViewMetadata,
    cubemap_face_from_path,
    cubemap_face_rotation,
    infer_generated_cubemap_face_rotations,
)
from core.apriltag_geometry import PinholeFrame, load_pinhole_frames
from core.image_io import imread_unicode

_REFERENCE_FACE_ORDER = ("pz", "px", "nz", "nx", "top", "bottom", "py", "ny")
_STANDARD_SIDE_FACES = frozenset({"px", "nx", "pz", "nz"})
_STANDARD_VERTICAL_FACE_SETS = (frozenset({"top", "bottom"}), frozenset({"py", "ny"}))
_STANDARD_FACE_VIEW_PARAMS: dict[str, tuple[float, float]] = {
    "pz": (0.0, 0.0),
    "px": (90.0, 0.0),
    "nz": (180.0, 0.0),
    "nx": (-90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    "py": (0.0, -90.0),
    "ny": (0.0, 90.0),
}
_STANDARD_FACE_LOOK_PARAMS: dict[str, tuple[float, float]] = {
    "pz": (0.0, 0.0),
    "px": (90.0, 0.0),
    "nz": (180.0, 0.0),
    "nx": (-90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    "py": (0.0, -90.0),
    "ny": (0.0, 90.0),
}


@dataclass(frozen=True)
class CubemapFrameGroup:
    name: str
    frames_by_face: dict[str, PinholeFrame]
    group_index: int = 0

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


def split_cubemap_face(file_path: str) -> tuple[str, str] | None:
    return cubemap_face_from_path(file_path)


def load_cubemap_frame_groups(
    transforms_json: Path,
    *,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
    normalize_cubemap: bool = True,
) -> tuple[CubemapFrameGroup, ...]:
    groups: dict[str, dict[str, PinholeFrame]] = {}
    for frame in load_pinhole_frames(
        transforms_json,
        cubemap_view_params=cubemap_view_params,
        normalize_cubemap=normalize_cubemap,
    ):
        parsed = split_cubemap_face(frame.file_path)
        if parsed is None:
            continue
        prefix, face = parsed
        groups.setdefault(prefix, {})[face] = frame
    group_indices = {name: index for index, name in enumerate(groups)}
    return tuple(
        CubemapFrameGroup(name=name, frames_by_face=frames, group_index=group_indices[name])
        for name, frames in sorted(groups.items())
        if len(frames) >= 4
    )


def cubemap_frame_groups_from_preview_cameras(cameras: Iterable[object]) -> tuple[CubemapFrameGroup, ...]:
    """Build preview cubemap groups from already-loaded scene preview cameras."""
    groups: dict[str, dict[str, PinholeFrame]] = {}
    for camera in cameras:
        if str(getattr(camera, "projection", "") or "").lower() != "pinhole":
            continue
        parsed = _preview_camera_cubemap_face(camera)
        if parsed is None:
            continue
        prefix, face = parsed
        frame = _preview_camera_to_pinhole_frame(camera)
        if frame is None:
            continue
        groups.setdefault(prefix, {})[face] = frame
    group_indices = {name: index for index, name in enumerate(groups)}
    return tuple(
        CubemapFrameGroup(name=name, frames_by_face=frames, group_index=group_indices[name])
        for name, frames in sorted(groups.items())
        if len(frames) >= 4
    )


def face_view_params(group: CubemapFrameGroup, face: str, *, fov_deg: float = 90.0) -> tuple[float, float, float] | None:
    frame = group.frames_by_face.get(face)
    if frame is None:
        return None
    standard_rotations = _standard_cube6_face_rotations(group)
    if standard_rotations is not None and face in standard_rotations:
        if face in _STANDARD_SIDE_FACES:
            center_ray = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ standard_rotations[face].T
            center_ray /= max(float(np.linalg.norm(center_ray)), 1e-12)
            yaw = float(np.rad2deg(np.arctan2(center_ray[0], center_ray[2])))
            pitch = float(-np.rad2deg(np.arcsin(np.clip(center_ray[1], -1.0, 1.0))))
            return yaw, pitch, float(fov_deg)
        if face in _STANDARD_FACE_LOOK_PARAMS:
            yaw, pitch = _STANDARD_FACE_LOOK_PARAMS[face]
            return float(yaw), float(pitch), float(fov_deg)
    forward_world = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ frame.camera_to_world_rotation.T
    local = forward_world @ group.reference_frame.camera_to_world_rotation
    local /= max(float(np.linalg.norm(local)), 1e-12)
    yaw = float(np.rad2deg(np.arctan2(local[0], local[2])))
    pitch = float(-np.rad2deg(np.arcsin(np.clip(local[1], -1.0, 1.0))))
    return yaw, pitch, float(fov_deg)


def render_cubemap_equirect(
    group: CubemapFrameGroup,
    *,
    output_width: int = 2048,
    output_height: int = 1024,
    image_cache: dict[Path, np.ndarray] | None = None,
    ray_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild a lightweight equirectangular preview texture from cubemap faces."""
    width = max(1, int(output_width))
    height = max(1, int(output_height))
    standard_rotations = _standard_cube6_face_rotations(group)
    rays = (
        _equirect_local_rays(width, height)
        if standard_rotations is not None
        else _equirect_world_rays(width, height, group.reference_frame.camera_to_world_rotation)
    )
    if ray_transform is not None:
        transform = np.asarray(ray_transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("ray_transform must be a 4x4 matrix")
        rays = rays @ transform[:3, :3]

    output = np.full((height, width, 3), 16, dtype=np.uint8)
    best_score = np.full((height, width), -np.inf, dtype=np.float64)
    for face, frame in group.frames_by_face.items():
        face_rotation = standard_rotations.get(face) if standard_rotations is not None else None
        if standard_rotations is not None and face_rotation is None:
            continue
        image = image_cache.get(frame.image_path) if image_cache is not None else None
        if image is None:
            image = imread_unicode(frame.image_path)
            if image_cache is not None and image is not None:
                image_cache[frame.image_path] = image
        if image is None:
            continue
        local = rays @ face_rotation if face_rotation is not None else rays @ frame.camera_to_world_rotation
        _sample_frame_to_output(
            frame=frame,
            image=image,
            local_rays=local,
            output=output,
            best_score=best_score,
        )
    return output


def _preview_camera_cubemap_face(camera: object) -> tuple[str, str] | None:
    values = [
        getattr(camera, "label", ""),
        getattr(camera, "camera_id", ""),
    ]
    image_path = getattr(camera, "image_path", None)
    if image_path is not None:
        path = Path(image_path)
        values.extend([path.name, str(path)])
    for value in values:
        parsed = split_cubemap_face(str(value or ""))
        if parsed is not None:
            return parsed
    return None


def _preview_camera_to_pinhole_frame(camera: object) -> PinholeFrame | None:
    image_path = getattr(camera, "image_path", None)
    if image_path is None:
        return None
    width = int(getattr(camera, "width", 0) or 0)
    height = int(getattr(camera, "height", 0) or 0)
    fl_x = float(getattr(camera, "fl_x", 0.0) or 0.0)
    fl_y = float(getattr(camera, "fl_y", 0.0) or 0.0)
    if width <= 0 or height <= 0 or fl_x <= 0.0 or fl_y <= 0.0:
        return None
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = np.asarray(camera.right, dtype=np.float64).reshape(3)
    matrix[:3, 1] = np.asarray(camera.up, dtype=np.float64).reshape(3)
    matrix[:3, 2] = np.asarray(camera.forward, dtype=np.float64).reshape(3)
    matrix[:3, 3] = np.asarray(camera.position, dtype=np.float64).reshape(3)
    return PinholeFrame(
        frame_id=str(getattr(camera, "camera_id", "") or getattr(camera, "label", "")),
        file_path=_preview_camera_file_path(camera),
        image_path=Path(image_path),
        width=width,
        height=height,
        fl_x=fl_x,
        fl_y=fl_y,
        cx=float(getattr(camera, "cx", 0.0) or 0.0),
        cy=float(getattr(camera, "cy", 0.0) or 0.0),
        transform_matrix=matrix,
        distortion_coeffs=None,
    )


def _preview_camera_file_path(camera: object) -> str:
    image_path = getattr(camera, "image_path", None)
    if image_path is not None:
        return Path(image_path).as_posix()
    return str(getattr(camera, "label", "") or getattr(camera, "camera_id", ""))


def _fixed_standard_cube6_face_rotations(group: CubemapFrameGroup) -> dict[str, np.ndarray] | None:
    faces = set(group.frames_by_face)
    if not _STANDARD_SIDE_FACES.issubset(faces):
        return None
    vertical_faces = next((pair for pair in _STANDARD_VERTICAL_FACE_SETS if pair.issubset(faces)), None)
    if vertical_faces is None:
        return None
    rotations: dict[str, np.ndarray] = {}
    for face in _REFERENCE_FACE_ORDER:
        if face not in _STANDARD_SIDE_FACES and face not in vertical_faces:
            continue
        if face not in group.frames_by_face:
            continue
        rotation = cubemap_face_rotation(face)
        if rotation is not None:
            rotations[face] = rotation
    return rotations


def _standard_cube6_face_rotations(group: CubemapFrameGroup) -> dict[str, np.ndarray] | None:
    generated = infer_generated_cubemap_face_rotations(group.frames_by_face)
    if generated is not None:
        return generated
    fixed = _fixed_standard_cube6_face_rotations(group)
    if fixed is None:
        return None
    derived = _transform_relative_face_rotations(group, fixed)
    return derived if derived is not None else fixed


def _transform_relative_face_rotations(
    group: CubemapFrameGroup,
    fixed_rotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    reference_rotation = np.asarray(group.reference_frame.camera_to_world_rotation, dtype=np.float64)
    if not _is_rotation_like(reference_rotation):
        return None
    rotations: dict[str, np.ndarray] = {}
    for face in fixed_rotations:
        frame = group.frames_by_face.get(face)
        if frame is None:
            return None
        face_rotation = np.asarray(frame.camera_to_world_rotation, dtype=np.float64)
        if not _is_rotation_like(face_rotation):
            return None
        relative = _orthonormalized_rotation(reference_rotation.T @ face_rotation)
        if relative is None:
            return None
        rotations[face] = relative
    if not _has_distinct_cube_face_centers(rotations):
        return None
    return rotations


def _is_rotation_like(value: np.ndarray) -> bool:
    return value.shape == (3, 3) and bool(np.all(np.isfinite(value)))


def _orthonormalized_rotation(value: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(value, dtype=np.float64)
    if np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
        return matrix if float(np.linalg.det(matrix)) > 0.0 else None
    try:
        u, _s, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        return None
    return rotation


def _has_distinct_cube_face_centers(rotations: dict[str, np.ndarray]) -> bool:
    if len(rotations) < 4:
        return False
    centers = []
    for rotation in rotations.values():
        center = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ rotation.T
        center /= max(float(np.linalg.norm(center)), 1e-12)
        centers.append(center)
    for index, center in enumerate(centers):
        for other in centers[index + 1 :]:
            if float(center @ other) > 0.985:
                return False
    return True


def _equirect_world_rays(width: int, height: int, reference_rotation: np.ndarray) -> np.ndarray:
    return _equirect_local_rays(width, height) @ reference_rotation.T


def _equirect_local_rays(width: int, height: int) -> np.ndarray:
    xs = (np.arange(width, dtype=np.float64) + 0.5) / max(width, 1)
    ys = (np.arange(height, dtype=np.float64) + 0.5) / max(height, 1)
    lon = (xs * 2.0 - 1.0) * np.pi
    lat = (0.5 - ys) * np.pi
    cos_lat = np.cos(lat)
    return np.stack(
        [
            cos_lat[:, None] * np.sin(lon)[None, :],
            np.sin(lat)[:, None] * np.ones((1, width), dtype=np.float64),
            cos_lat[:, None] * np.cos(lon)[None, :],
        ],
        axis=-1,
    )


def _sample_frame_to_output(
    *,
    frame: PinholeFrame,
    image: np.ndarray,
    local_rays: np.ndarray,
    output: np.ndarray,
    best_score: np.ndarray,
) -> None:
    image_bgr = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    z = local_rays[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = frame.fl_x * (local_rays[:, :, 0] / z) + frame.cx
        map_y = frame.cy - frame.fl_y * (local_rays[:, :, 1] / z)
    valid = (
        (z > 1e-8)
        & (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x < frame.width - 1)
        & (map_y < frame.height - 1)
    )
    better = valid & (z > best_score)
    if not np.any(better):
        return
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
