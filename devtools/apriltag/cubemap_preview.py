"""Cubemap-frame preview helpers for the AprilTag placement dev GUI."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from core.apriltag_cubemap import (
    CubemapViewMetadata,
    cubemap_view_params_for_group,
    infer_generated_cubemap_face_rotations,
)
from core.apriltag_geometry import PinholeFrame, load_pinhole_frames, points_intersect_image, project_sfm_points
from core.image_io import imread_unicode

_FACE_NAMES = ("px", "nx", "pz", "nz", "top", "bottom", "py", "ny")
_REFERENCE_FACE_ORDER = ("pz", "px", "nz", "nx", "top", "bottom", "py", "ny")
_STANDARD_FACE_ORDER = ("pz", "px", "nz", "nx", "top", "bottom", "py", "ny")
_STANDARD_SIDE_FACES = frozenset({"px", "nx", "pz", "nz"})
_STANDARD_VERTICAL_FACE_SETS = (frozenset({"top", "bottom"}), frozenset({"py", "ny"}))
_STANDARD_FACE_VIEW_PARAMS: dict[str, tuple[float, float]] = {
    "pz": (0.0, 0.0),
    "px": (90.0, 0.0),
    "nz": (180.0, 0.0),
    "nx": (-90.0, 0.0),
    # Legacy GUI Cube6 output used top/bottom names with generated pitch rows.
    # Keep them for old scenes; new GUI output uses py/ny.
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

    @property
    def camera_position_sfm(self) -> np.ndarray:
        return self.reference_frame.camera_position_sfm


@dataclass(frozen=True)
class CubemapPreviewSamplerFace:
    """One loaded cubemap face for direct preview sampling.

    ``preview_to_face_rotation`` maps the interactive preview ray into the
    pinhole camera space of this face using row-vector convention.
    """

    face: str
    frame: PinholeFrame
    image_bgr: np.ndarray
    preview_to_face_rotation: np.ndarray


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


def load_metashape_camera_labels(xml_path: Path) -> tuple[str, ...]:
    if not xml_path.is_file():
        return ()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return ()
    labels: list[str] = []
    for camera in root.findall(".//camera"):
        label = str(camera.attrib.get("label") or "").strip()
        if label:
            labels.append(Path(label).stem)
    return tuple(labels)


def order_groups_by_labels(
    groups: tuple[CubemapFrameGroup, ...],
    labels: tuple[str, ...],
) -> tuple[CubemapFrameGroup, ...]:
    if not labels:
        return groups
    order = {label: index for index, label in enumerate(labels)}
    fallback = len(order)
    return tuple(sorted(groups, key=lambda group: (order.get(group.name, fallback), group.name)))


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))
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


def _export_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
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
    return rx @ ry


def image_space_cubemap_frame_group(
    group: CubemapFrameGroup,
    *,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
) -> CubemapFrameGroup:
    """Return face poses whose pinhole rays match the saved Cube6 images.

    ``cubemap_transforms_json.py`` writes image pixels with ``rotation_matrix(...,
    forward=False)`` but writes ``transforms.json`` face poses with
    ``rotation_matrix(..., forward=True).T``. The debug viewer is an image/point
    inspection tool, so its face buttons and image preview must follow the saved
    jpg ray directions rather than the downstream trainer's raw pose convention.
    """
    params = cubemap_view_params_for_group(cubemap_view_params, group.group_index)
    if params is None:
        fixed = _fixed_standard_cube6_face_rotations(group)
        if fixed is None:
            return group
        params = {
            face: _STANDARD_FACE_VIEW_PARAMS[face]
            for face in fixed
            if face in _STANDARD_FACE_VIEW_PARAMS
        }
    image_rotations: dict[str, np.ndarray] = {}
    export_rotations: dict[str, np.ndarray] = {}
    for face, (yaw, pitch) in params.items():
        if face not in group.frames_by_face:
            continue
        image_rotations[face] = _rotation_matrix(yaw, pitch)
        export_rotations[face] = _export_rotation_matrix(yaw, pitch)
    base = _image_space_base_rotation(group, export_rotations)
    if base is None:
        return group
    position = group.reference_frame.camera_position_sfm
    return CubemapFrameGroup(
        name=group.name,
        frames_by_face={
            face: _replace_frame_pose(frame, base @ image_rotations[face], position)
            for face, frame in group.frames_by_face.items()
            if face in image_rotations
        },
        group_index=group.group_index,
    )


def _image_space_base_rotation(
    group: CubemapFrameGroup,
    export_rotations: dict[str, np.ndarray],
) -> np.ndarray | None:
    bases: list[np.ndarray] = []
    for face, export_rotation in export_rotations.items():
        frame = group.frames_by_face.get(face)
        if frame is None:
            continue
        frame_rotation = np.asarray(frame.camera_to_world_rotation, dtype=np.float64)
        if not _is_rotation_like(frame_rotation):
            return None
        bases.append(frame_rotation @ export_rotation)
    if len(bases) < 4:
        return None
    base = _average_rotations(bases)
    errors = [_rotation_angle(base, candidate) for candidate in bases]
    if max(errors) > np.deg2rad(0.1):
        return None
    return base


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    matrix = np.mean(np.stack(rotations, axis=0), axis=0)
    try:
        u, _s, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return rotations[0]
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    value = (float(np.trace(delta)) - 1.0) * 0.5
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def _replace_frame_pose(frame: PinholeFrame, rotation: np.ndarray, position: np.ndarray) -> PinholeFrame:
    transform = np.array(frame.transform_matrix, dtype=np.float64, copy=True)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return replace(frame, transform_matrix=transform)


def _fixed_standard_cube6_face_rotations(group: CubemapFrameGroup) -> dict[str, np.ndarray] | None:
    faces = set(group.frames_by_face)
    if not _STANDARD_SIDE_FACES.issubset(faces):
        return None
    vertical_faces = next((pair for pair in _STANDARD_VERTICAL_FACE_SETS if pair.issubset(faces)), None)
    if vertical_faces is None:
        return None
    return {
        face: _rotation_matrix(*_STANDARD_FACE_VIEW_PARAMS[face])
        for face in _STANDARD_FACE_ORDER
        if face in _STANDARD_SIDE_FACES or face in vertical_faces
        if face in group.frames_by_face
    }


def _standard_cube6_face_rotations(group: CubemapFrameGroup) -> dict[str, np.ndarray] | None:
    generated = infer_generated_cubemap_face_rotations(group.frames_by_face)
    if generated is not None:
        return generated
    fixed = _fixed_standard_cube6_face_rotations(group)
    if fixed is None:
        return None
    derived = _transform_relative_face_rotations(group, fixed)
    return derived if derived is not None else fixed


def cubemap_image_face_rotations(
    group: CubemapFrameGroup,
    *,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray] | None:
    """Return generated face-local image rotations for a Cube6 group."""
    params = cubemap_view_params_for_group(cubemap_view_params, group.group_index)
    if params is not None:
        rotations = {
            face: _rotation_matrix(yaw, pitch)
            for face, (yaw, pitch) in params.items()
            if face in group.frames_by_face
        }
        if rotations:
            return rotations
    rotations = _standard_cube6_face_rotations(group)
    if rotations is None:
        return None
    return {face: rotation.copy() for face, rotation in rotations.items()}


def source_equirect_base_rotation(
    group: CubemapFrameGroup,
    *,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray | None:
    """Return the source equirectangular camera rotation from Cube6 poses.

    ``cubemap_transforms_json.py`` writes each pinhole face as
    ``source_camera @ export_rotation.T``. Averaging the inverse relation over
    the generated faces gives the source equirectangular camera-to-world
    rotation in the same world basis as ``group``.
    """
    params = cubemap_view_params_for_group(cubemap_view_params, group.group_index)
    if params is None:
        fixed = _fixed_standard_cube6_face_rotations(group)
        if fixed is None:
            return None
        params = {
            face: _STANDARD_FACE_VIEW_PARAMS[face]
            for face in fixed
            if face in _STANDARD_FACE_VIEW_PARAMS
        }
    export_rotations = {
        face: _export_rotation_matrix(yaw, pitch)
        for face, (yaw, pitch) in params.items()
        if face in group.frames_by_face
    }
    if len(export_rotations) < 4:
        return None
    base = _image_space_base_rotation(group, export_rotations)
    return None if base is None else base.copy()


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


def _direction_matrix_sfm_to_preview(sfm_to_preview_matrix: np.ndarray | None) -> np.ndarray:
    if sfm_to_preview_matrix is None:
        return np.eye(3, dtype=np.float64)
    transform = np.asarray(sfm_to_preview_matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("sfm_to_preview_matrix must be a 4x4 matrix")
    return transform[:3, :3].T


def _direction_matrix_preview_to_sfm(sfm_to_preview_matrix: np.ndarray | None) -> np.ndarray:
    if sfm_to_preview_matrix is None:
        return np.eye(3, dtype=np.float64)
    transform = np.asarray(sfm_to_preview_matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("sfm_to_preview_matrix must be a 4x4 matrix")
    return np.linalg.inv(transform[:3, :3]).T


def _best_face_for_sfm_ray(group: CubemapFrameGroup, ray_sfm: np.ndarray) -> str | None:
    best_face: str | None = None
    best_z = -np.inf
    for face, frame in group.frames_by_face.items():
        local = np.asarray(ray_sfm, dtype=np.float64) @ frame.camera_to_world_rotation
        z = float(local[2])
        if z > best_z:
            best_face = face
            best_z = z
    return best_face if best_z > 1e-8 else None


def _view_ray_local(
    *,
    x_px: float,
    y_px: float,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    size = max(1, int(output_size))
    cx = float(x_px) - (size - 1) / 2.0
    cy = float(y_px) - (size - 1) / 2.0
    focal = 0.5 * size / np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    ray = np.array([cx, -cy, focal], dtype=np.float64)
    ray /= max(float(np.linalg.norm(ray)), 1e-12)
    return ray @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T


def _best_standard_face(
    group: CubemapFrameGroup,
    preview_ray: np.ndarray,
) -> tuple[str, PinholeFrame, np.ndarray, np.ndarray] | None:
    rotations = _standard_cube6_face_rotations(group)
    if rotations is None:
        return None
    best_face: str | None = None
    best_frame: PinholeFrame | None = None
    best_local: np.ndarray | None = None
    best_rotation: np.ndarray | None = None
    best_z = -np.inf
    for face, rotation in rotations.items():
        frame = group.frames_by_face.get(face)
        if frame is None:
            continue
        local = preview_ray @ rotation
        z = float(local[2])
        if z > best_z:
            best_face = face
            best_frame = frame
            best_local = local
            best_rotation = rotation
            best_z = z
    if best_face is None or best_frame is None or best_local is None or best_rotation is None or best_z <= 1e-8:
        return None
    return best_face, best_frame, best_local, best_rotation


def virtual_camera_rotation(group: CubemapFrameGroup, *, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Return a camera-to-world rotation for the interactive preview view."""
    return virtual_camera_rotation_with_roll(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=0.0)


def virtual_camera_rotation_with_roll(
    group: CubemapFrameGroup,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Return a camera-to-world rotation for the interactive preview view."""
    local_rotation = _rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    base_rotation = _preview_base_rotation(group)
    if base_rotation is not None:
        return base_rotation @ local_rotation
    return group.reference_frame.camera_to_world_rotation @ local_rotation


def _preview_base_rotation(group: CubemapFrameGroup) -> np.ndarray | None:
    rotations = _standard_cube6_face_rotations(group)
    if rotations is None:
        return None
    bases: list[np.ndarray] = []
    for face, rotation in rotations.items():
        frame = group.frames_by_face.get(face)
        if frame is None:
            continue
        bases.append(frame.camera_to_world_rotation @ rotation.T)
    if not bases:
        return None
    matrix = np.mean(np.stack(bases, axis=0), axis=0)
    try:
        u, _s, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return bases[0]
    base = u @ vt
    if float(np.linalg.det(base)) < 0.0:
        u[:, -1] *= -1.0
        base = u @ vt
    return base


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
    ray, _up, _face = view_pixel_to_world_ray_and_up(
        group,
        x_px=x_px,
        y_px=y_px,
        output_size=output_size,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
    )
    return ray


def preview_frustum_rays(
    group: CubemapFrameGroup,
    *,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    size = max(1, int(output_size))
    center = (size - 1) / 2.0
    center_ray = view_pixel_to_world_ray(
        group,
        x_px=center,
        y_px=center,
        output_size=size,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
    )
    corners = np.asarray(
        [
            view_pixel_to_world_ray(
                group,
                x_px=x,
                y_px=y,
                output_size=size,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                fov_deg=fov_deg,
            )
            for x, y in ((0.0, 0.0), (size - 1.0, 0.0), (size - 1.0, size - 1.0), (0.0, size - 1.0))
        ],
        dtype=np.float64,
    )
    return center_ray, corners


def axis_preview_frustum_rays(
    *,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    size = max(1, int(output_size))
    center = (size - 1) / 2.0
    center_ray = _view_ray_local(
        x_px=center,
        y_px=center,
        output_size=size,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
        roll_deg=roll_deg,
    )
    corners = np.asarray(
        [
            _view_ray_local(
                x_px=x,
                y_px=y,
                output_size=size,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                fov_deg=fov_deg,
                roll_deg=roll_deg,
            )
            for x, y in ((0.0, 0.0), (size - 1.0, 0.0), (size - 1.0, size - 1.0), (0.0, size - 1.0))
        ],
        dtype=np.float64,
    )
    return center_ray, corners


def view_pixel_to_world_ray_and_up(
    group: CubemapFrameGroup,
    *,
    x_px: float,
    y_px: float,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    ray = _view_ray_local(
        x_px=x_px,
        y_px=y_px,
        output_size=output_size,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
    )
    up_ray = _view_ray_local(
        x_px=x_px,
        y_px=y_px - 1.0,
        output_size=output_size,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
    )
    standard_face = _best_standard_face(group, ray)
    if standard_face is not None:
        face, frame, local, face_rotation = standard_face
        world_ray = local @ frame.camera_to_world_rotation.T
        world_ray = world_ray / max(float(np.linalg.norm(world_ray)), 1e-12)
        local_up_ray = up_ray @ face_rotation
        world_up_ray = local_up_ray @ frame.camera_to_world_rotation.T
        up = world_up_ray - world_ray * float(world_up_ray @ world_ray)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        return world_ray, up, face
    world_ray = ray @ virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg).T
    world_ray /= max(float(np.linalg.norm(world_ray)), 1e-12)
    world_up_ray = up_ray @ virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg).T
    up = world_up_ray - world_ray * float(world_up_ray @ world_ray)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    return world_ray, up, None


def view_pixel_to_axis_world_ray_and_up(
    group: CubemapFrameGroup,
    *,
    x_px: float,
    y_px: float,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    roll_deg: float = 0.0,
    sfm_to_preview_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Map a fixed-axis preview pixel back into the original SFM world."""
    preview_to_sfm = _direction_matrix_preview_to_sfm(sfm_to_preview_matrix)
    ray_preview = _view_ray_local(
        x_px=x_px,
        y_px=y_px,
        output_size=output_size,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
        roll_deg=roll_deg,
    )
    up_preview = _view_ray_local(
        x_px=x_px,
        y_px=y_px - 1.0,
        output_size=output_size,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
        roll_deg=roll_deg,
    )
    ray_sfm = ray_preview @ preview_to_sfm
    ray_sfm /= max(float(np.linalg.norm(ray_sfm)), 1e-12)
    up_sfm_ray = up_preview @ preview_to_sfm
    up = up_sfm_ray - ray_sfm * float(up_sfm_ray @ ray_sfm)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    return ray_sfm, up, _best_face_for_sfm_ray(group, ray_sfm)


def view_up_world(group: CubemapFrameGroup, *, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    rotation = _rotation_matrix(yaw_deg, pitch_deg)
    center_ray = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ rotation.T
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64) @ rotation.T
    standard_face = _best_standard_face(group, center_ray)
    if standard_face is not None:
        _face, frame, _local_center, face_rotation = standard_face
        up = (up @ face_rotation) @ frame.camera_to_world_rotation.T
        return up / max(float(np.linalg.norm(up)), 1e-12)
    up = up @ virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg).T
    return up / max(float(np.linalg.norm(up)), 1e-12)


def _preview_ray_from_face_pixel(frame: PinholeFrame, face_rotation: np.ndarray, point_px: np.ndarray) -> np.ndarray:
    local = np.array(
        [
            (float(point_px[0]) - frame.cx) / frame.fl_x,
            (frame.cy - float(point_px[1])) / frame.fl_y,
            1.0,
        ],
        dtype=np.float64,
    )
    local /= max(float(np.linalg.norm(local)), 1e-12)
    preview_ray = local @ face_rotation.T
    return preview_ray / max(float(np.linalg.norm(preview_ray)), 1e-12)


def _standard_preview_ray_for_world_point(group: CubemapFrameGroup, point_sfm: np.ndarray) -> np.ndarray | None:
    rotations = _standard_cube6_face_rotations(group)
    if rotations is None:
        return None
    vector_world = np.asarray(point_sfm, dtype=np.float64) - group.camera_position_sfm
    norm = float(np.linalg.norm(vector_world))
    if norm <= 1e-12:
        return None
    best_ray: np.ndarray | None = None
    best_z = -np.inf
    for face, rotation in rotations.items():
        frame = group.frames_by_face.get(face)
        if frame is None:
            continue
        local = vector_world @ frame.camera_to_world_rotation
        z = float(local[2])
        if z <= 1e-8 or z <= best_z:
            continue
        best_ray = (local / norm) @ rotation.T
        best_z = z
    if best_ray is None:
        return None
    return best_ray / max(float(np.linalg.norm(best_ray)), 1e-12)


def _project_sfm_points_to_standard_preview_rays(
    group: CubemapFrameGroup,
    points_sfm: np.ndarray,
) -> list[np.ndarray] | None:
    rotations = _standard_cube6_face_rotations(group)
    if rotations is None:
        return None
    center = np.asarray(points_sfm, dtype=np.float64).mean(axis=0)
    best: tuple[float, PinholeFrame, np.ndarray, np.ndarray] | None = None
    for face, rotation in rotations.items():
        frame = group.frames_by_face.get(face)
        if frame is None:
            continue
        projected = project_sfm_points(frame, points_sfm)
        if projected is None or not points_intersect_image(projected, frame.width, frame.height):
            continue
        center_local = (center - frame.camera_position_sfm) @ frame.camera_to_world_rotation
        z = float(center_local[2])
        if z <= 1e-8:
            continue
        if best is None or z > best[0]:
            best = (z, frame, rotation, projected)
    if best is None:
        return None
    _z, frame, rotation, projected = best
    return [_preview_ray_from_face_pixel(frame, rotation, point) for point in projected]


def project_sfm_points_to_preview_points(
    group: CubemapFrameGroup,
    points_sfm: np.ndarray,
    *,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
) -> np.ndarray | None:
    """Project independent SfM points into the current square perspective preview."""
    points = np.asarray(points_sfm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_sfm must be an Nx3 array")
    size = max(1, int(output_size))
    focal = 0.5 * size / np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    center = (size - 1) / 2.0
    view_rotation = _rotation_matrix(yaw_deg, pitch_deg)
    standard = _standard_cube6_face_rotations(group) is not None
    virtual_rotation = virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg)

    projected: list[tuple[float, float]] = []
    for point in points:
        if standard:
            preview_ray = _standard_preview_ray_for_world_point(group, point)
            if preview_ray is None:
                return None
            view_local = preview_ray @ view_rotation
        else:
            view_local = (point - group.camera_position_sfm) @ virtual_rotation
            norm = float(np.linalg.norm(view_local))
            if norm <= 1e-12:
                return None
            view_local = view_local / norm
        if float(view_local[2]) <= 1e-8:
            return None
        x = center + focal * float(view_local[0] / view_local[2])
        y = center - focal * float(view_local[1] / view_local[2])
        projected.append((x, y))
    return np.asarray(projected, dtype=np.float32)


def project_sfm_points_to_axis_preview_points(
    group: CubemapFrameGroup,
    points_sfm: np.ndarray,
    *,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    roll_deg: float = 0.0,
    sfm_to_preview_matrix: np.ndarray | None = None,
) -> np.ndarray | None:
    """Project SFM points into a fixed-axis square perspective preview."""
    points = np.asarray(points_sfm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_sfm must be an Nx3 array")
    vectors = points - group.camera_position_sfm.reshape(1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        return None
    sfm_to_preview = _direction_matrix_sfm_to_preview(sfm_to_preview_matrix)
    preview_rays = (vectors / norms[:, None]) @ sfm_to_preview

    size = max(1, int(output_size))
    focal = 0.5 * size / np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    center = (size - 1) / 2.0
    view_rotation = _rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    projected: list[tuple[float, float]] = []
    for preview_ray in preview_rays:
        view_local = preview_ray @ view_rotation
        if float(view_local[2]) <= 1e-8:
            return None
        x = center + focal * float(view_local[0] / view_local[2])
        y = center - focal * float(view_local[1] / view_local[2])
        projected.append((x, y))
    return np.asarray(projected, dtype=np.float32)


def project_sfm_points_to_preview(
    group: CubemapFrameGroup,
    points_sfm: np.ndarray,
    *,
    output_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
) -> np.ndarray | None:
    """Project SfM points into the current square perspective preview."""
    points = np.asarray(points_sfm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_sfm must be an Nx3 array")
    size = max(1, int(output_size))
    focal = 0.5 * size / np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    center = (size - 1) / 2.0
    view_rotation = _rotation_matrix(yaw_deg, pitch_deg)

    projected: list[tuple[float, float]] = []
    standard = _standard_cube6_face_rotations(group) is not None
    virtual_rotation = virtual_camera_rotation(group, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    if standard:
        preview_rays = _project_sfm_points_to_standard_preview_rays(group, points)
        if preview_rays is None:
            return None
    else:
        preview_rays = []
        for point in points:
            view_local = (point - group.camera_position_sfm) @ virtual_rotation
            norm = float(np.linalg.norm(view_local))
            if norm <= 1e-12:
                return None
            preview_rays.append(view_local / norm)

    for preview_ray in preview_rays:
        if standard:
            view_local = preview_ray @ view_rotation
        else:
            view_local = preview_ray
        if float(view_local[2]) <= 1e-8:
            return None
        x = center + focal * float(view_local[0] / view_local[2])
        y = center - focal * float(view_local[1] / view_local[2])
        projected.append((x, y))
    return np.asarray(projected, dtype=np.float32)


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
    # PerspectiveImageView's shader applies positive pitch toward local -Y.
    # Convert transform-space local +Y/-Y to the same UI convention.
    pitch = float(-np.rad2deg(np.arcsin(np.clip(local[1], -1.0, 1.0))))
    return yaw, pitch, float(fov_deg)


def axis_face_view_params(
    group: CubemapFrameGroup,
    face: str,
    *,
    fov_deg: float = 90.0,
    sfm_to_preview_matrix: np.ndarray | None = None,
) -> tuple[float, float, float, float] | None:
    """Return yaw/pitch/roll/fov for a cubemap face in fixed preview axes."""
    frame = group.frames_by_face.get(face)
    if frame is None:
        return None
    sfm_to_preview = _direction_matrix_sfm_to_preview(sfm_to_preview_matrix)
    forward_sfm = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ frame.camera_to_world_rotation.T
    up_sfm = np.array([0.0, 1.0, 0.0], dtype=np.float64) @ frame.camera_to_world_rotation.T
    forward = forward_sfm @ sfm_to_preview
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    up = up_sfm @ sfm_to_preview
    up = up - forward * float(up @ forward)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    right = np.cross(up, forward)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(forward, right)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    view_rotation = np.column_stack([right, up, forward])
    yaw = float(np.rad2deg(np.arctan2(view_rotation[0, 2], view_rotation[2, 2])))
    pitch = float(-np.rad2deg(np.arcsin(np.clip(view_rotation[1, 2], -1.0, 1.0))))
    cp = max(float(np.hypot(view_rotation[0, 2], view_rotation[2, 2])), 1e-12)
    if cp <= 1e-8:
        roll = 0.0
    else:
        roll = float(np.rad2deg(np.arctan2(view_rotation[1, 0], view_rotation[1, 1])))
    return yaw, pitch, roll, float(fov_deg)


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


def cubemap_preview_sampler_faces(
    group: CubemapFrameGroup,
    *,
    image_cache: dict[Path, np.ndarray] | None = None,
) -> tuple[CubemapPreviewSamplerFace, ...]:
    """Load face textures for direct preview sampling.

    The normal camera-image preview needs to show the actual Cube6 pinhole
    images without rebuilding an intermediate equirectangular texture. This
    returns the face images plus the same transform-relative face rotations used
    by click/grid/tag projection.
    """
    standard_rotations = _standard_cube6_face_rotations(group)
    if standard_rotations is None:
        return ()
    faces: list[CubemapPreviewSamplerFace] = []
    for face in _STANDARD_FACE_ORDER:
        frame = group.frames_by_face.get(face)
        rotation = standard_rotations.get(face)
        if frame is None or rotation is None:
            continue
        image = _load_frame_image(frame, image_cache=image_cache)
        if image is None:
            continue
        faces.append(
            CubemapPreviewSamplerFace(
                face=face,
                frame=frame,
                image_bgr=image,
                preview_to_face_rotation=np.asarray(rotation, dtype=np.float64),
            )
        )
    return tuple(faces)


def cubemap_world_sampler_faces(
    group: CubemapFrameGroup,
    *,
    image_cache: dict[Path, np.ndarray] | None = None,
) -> tuple[CubemapPreviewSamplerFace, ...]:
    """Load face textures using each frame's world-space camera pose directly."""
    faces: list[CubemapPreviewSamplerFace] = []
    ordered = list(_STANDARD_FACE_ORDER) + sorted(set(group.frames_by_face) - set(_STANDARD_FACE_ORDER))
    for face in ordered:
        frame = group.frames_by_face.get(face)
        if frame is None:
            continue
        image = _load_frame_image(frame, image_cache=image_cache)
        if image is None:
            continue
        faces.append(
            CubemapPreviewSamplerFace(
                face=face,
                frame=frame,
                image_bgr=image,
                preview_to_face_rotation=np.asarray(frame.camera_to_world_rotation, dtype=np.float64),
            )
        )
    return tuple(faces)


def render_cubemap_direct_preview(
    group: CubemapFrameGroup,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    fov_deg: float = 90.0,
    output_size: int = 768,
    image_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray:
    """CPU reference renderer for the direct Cube6 preview shader."""
    faces = cubemap_preview_sampler_faces(group, image_cache=image_cache)
    if not faces:
        raise ValueError("Direct cubemap preview requires a standard Cube6 image group")
    size = max(1, int(output_size))
    preview_rays = _view_rays(size, fov_deg) @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T

    output = np.full((size, size, 3), 16, dtype=np.uint8)
    best_score = np.full((size, size), -np.inf, dtype=np.float64)
    for sampler_face in faces:
        local = preview_rays @ sampler_face.preview_to_face_rotation
        _sample_frame_to_output(
            frame=sampler_face.frame,
            image=sampler_face.image_bgr,
            local_rays=local,
            output=output,
            best_score=best_score,
        )
    return output


def render_cubemap_world_perspective(
    group: CubemapFrameGroup,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    fov_deg: float = 90.0,
    output_size: int = 768,
    image_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray:
    """Render saved Cube6 images by projecting preview rays into final world poses."""
    size = max(1, int(output_size))
    preview_rays = _view_rays(size, fov_deg) @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T

    output = np.full((size, size, 3), 16, dtype=np.uint8)
    best_score = np.full((size, size), -np.inf, dtype=np.float64)
    for frame in group.frames:
        image = _load_frame_image(frame, image_cache=image_cache)
        if image is None:
            continue
        local = preview_rays @ frame.camera_to_world_rotation
        _sample_frame_to_output(
            frame=frame,
            image=image,
            local_rays=local,
            output=output,
            best_score=best_score,
        )
    return output


def visible_cubemap_preview_face_indices(
    faces: tuple[CubemapPreviewSamplerFace, ...],
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    fov_deg: float,
    sample_grid: int = 7,
) -> tuple[int, ...]:
    """Return direct-preview face indices that can contribute to the current view."""
    if not faces:
        return ()
    grid = max(3, int(sample_grid))
    if grid % 2 == 0:
        grid += 1
    preview_rays = _view_rays(grid, fov_deg) @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T
    visible: list[int] = []
    for index, face in enumerate(faces):
        frame = face.frame
        local = preview_rays @ face.preview_to_face_rotation
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
        if bool(np.any(valid)):
            visible.append(index)
    return tuple(visible) if visible else tuple(range(len(faces)))


def _load_frame_image(
    frame: PinholeFrame,
    *,
    image_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray | None:
    image = image_cache.get(frame.image_path) if image_cache is not None else None
    if image is None:
        image = imread_unicode(frame.image_path)
        if image_cache is not None and image is not None:
            image_cache[frame.image_path] = image
    if image is None:
        return None
    return image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _equirect_world_rays(width: int, height: int, reference_rotation: np.ndarray) -> np.ndarray:
    return _equirect_local_rays(width, height) @ reference_rotation.T


def _equirect_local_rays(width: int, height: int) -> np.ndarray:
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
    return rays_ref


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


def render_cubemap_equirect(
    group: CubemapFrameGroup,
    *,
    output_width: int = 2048,
    output_height: int = 1024,
    image_cache: dict[Path, np.ndarray] | None = None,
    ray_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild a lightweight equirectangular preview texture from cubemap faces.

    This is intended as a one-time texture build per selected frame group. The
    interactive perspective view can then use the existing OpenGL equirect shader.
    """
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
        # The matrix maps image/source coordinates into the displayed world
        # profile. Sampling needs the inverse; for these orthonormal profile
        # transforms that is the non-transposed 3x3 block in row-vector form.
        rays = rays @ transform[:3, :3]

    output = np.full((height, width, 3), 16, dtype=np.uint8)
    best_score = np.full((height, width), -np.inf, dtype=np.float64)
    for face, frame in group.frames_by_face.items():
        face_rotation = standard_rotations.get(face) if standard_rotations is not None else None
        if standard_rotations is not None and face_rotation is None:
            continue
        image = None
        if image_cache is not None:
            image = image_cache.get(frame.image_path)
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


def render_cubemap_axis_equirect(
    group: CubemapFrameGroup,
    *,
    output_width: int = 2048,
    output_height: int = 1024,
    image_cache: dict[Path, np.ndarray] | None = None,
    sfm_to_preview_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild an equirect texture whose axes match the 3D debug viewport."""
    width = max(1, int(output_width))
    height = max(1, int(output_height))
    preview_to_sfm = _direction_matrix_preview_to_sfm(sfm_to_preview_matrix)
    rays_sfm = _equirect_local_rays(width, height) @ preview_to_sfm

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
        local = rays_sfm @ frame.camera_to_world_rotation
        _sample_frame_to_output(
            frame=frame,
            image=image,
            local_rays=local,
            output=output,
            best_score=best_score,
        )
    return output


def render_generated_cubemap_source_axis(
    group: CubemapFrameGroup,
    source_world_to_image_rotation: np.ndarray,
    *,
    cubemap_view_params: CubemapViewMetadata | Mapping[str, tuple[float, float]] | None = None,
    output_width: int = 2048,
    output_height: int = 1024,
    image_cache: dict[Path, np.ndarray] | None = None,
    sfm_to_preview_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild the source-panorama preview axes from generated Cube6 faces."""
    rotation = np.asarray(source_world_to_image_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("source_world_to_image_rotation must be a 3x3 matrix")
    face_rotations = cubemap_image_face_rotations(group, cubemap_view_params=cubemap_view_params)
    if face_rotations is None:
        raise ValueError("Cube6 image reconstruction requires generated view yaw/pitch metadata")

    width = max(1, int(output_width))
    height = max(1, int(output_height))
    preview_to_sfm = _direction_matrix_preview_to_sfm(sfm_to_preview_matrix)
    rays_sfm = _equirect_local_rays(width, height) @ preview_to_sfm
    source_local = rays_sfm @ rotation

    output = np.full((height, width, 3), 16, dtype=np.uint8)
    best_score = np.full((height, width), -np.inf, dtype=np.float64)
    for face, frame in group.frames_by_face.items():
        face_rotation = face_rotations.get(face)
        if face_rotation is None:
            continue
        image = None
        if image_cache is not None:
            image = image_cache.get(frame.image_path)
        if image is None:
            image = imread_unicode(frame.image_path)
            if image_cache is not None and image is not None:
                image_cache[frame.image_path] = image
        if image is None:
            continue
        local = source_local @ face_rotation
        _sample_frame_to_output(
            frame=frame,
            image=image,
            local_rays=local,
            output=output,
            best_score=best_score,
        )
    return output


def render_source_equirect_axis(
    image: np.ndarray,
    source_world_to_image_rotation: np.ndarray,
    *,
    output_width: int = 2048,
    output_height: int = 1024,
    sfm_to_preview_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Resample a source equirectangular image into the fixed preview axes."""
    if image is None or image.size == 0:
        raise ValueError("Source equirect image is empty")
    rotation = np.asarray(source_world_to_image_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("source_world_to_image_rotation must be a 3x3 matrix")

    width = max(1, int(output_width))
    height = max(1, int(output_height))
    preview_to_sfm = _direction_matrix_preview_to_sfm(sfm_to_preview_matrix)
    rays_sfm = _equirect_local_rays(width, height) @ preview_to_sfm
    source_local = rays_sfm @ rotation

    dx = source_local[..., 0]
    dy = source_local[..., 1]
    dz = source_local[..., 2]
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))

    source = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    source_h, source_w = source.shape[:2]
    map_x = ((lon / np.pi + 1.0) * 0.5 * source_w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * source_h).astype(np.float32)
    map_x = np.mod(map_x, float(source_w))
    map_y = np.clip(map_y, 0.0, float(source_h - 1))
    return cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def render_source_equirect_perspective(
    image: np.ndarray,
    source_world_to_image_rotation: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    fov_deg: float = 90.0,
    output_size: int = 768,
    sfm_to_preview_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Render a perspective crop from a source equirect image using preview rays."""
    if image is None or image.size == 0:
        raise ValueError("Source equirect image is empty")
    rotation = np.asarray(source_world_to_image_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("source_world_to_image_rotation must be a 3x3 matrix")

    size = max(1, int(output_size))
    preview_to_sfm = _direction_matrix_preview_to_sfm(sfm_to_preview_matrix)
    rays_sfm = _view_rays(size, fov_deg) @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T @ preview_to_sfm
    source_local = rays_sfm @ rotation

    dx = source_local[..., 0]
    dy = source_local[..., 1]
    dz = source_local[..., 2]
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))

    source = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    source_h, source_w = source.shape[:2]
    map_x = ((lon / np.pi + 1.0) * 0.5 * source_w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * source_h).astype(np.float32)
    map_x = np.mod(map_x, float(source_w))
    map_y = np.clip(map_y, 0.0, float(source_h - 1))
    return cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
