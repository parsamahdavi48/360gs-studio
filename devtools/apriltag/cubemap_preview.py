"""Cubemap-frame preview helpers for the AprilTag placement dev GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

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
    # GUI Cube6 output names currently follow the generated pitch rows:
    # top is the +90 pitch face, bottom is the -90 pitch face.
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    # Legacy/default converter names.
    "py": (0.0, -90.0),
    "ny": (0.0, 90.0),
}
_STANDARD_FACE_LOOK_PARAMS: dict[str, tuple[float, float]] = {
    "pz": (0.0, 0.0),
    "px": (90.0, 0.0),
    "nz": (180.0, 0.0),
    "nx": (-90.0, 0.0),
    # Human-facing preview pitch: negative looks up, positive looks down.
    # Keep this separate from the exported face layout above.
    "top": (0.0, -90.0),
    "bottom": (0.0, 90.0),
    "py": (0.0, -90.0),
    "ny": (0.0, 90.0),
}


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


def _view_ray_local(
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
    return ray @ _rotation_matrix(yaw_deg, pitch_deg).T


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
    return group.reference_frame.camera_to_world_rotation @ _rotation_matrix(yaw_deg, pitch_deg)


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
