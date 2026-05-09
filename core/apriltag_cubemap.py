"""Cubemap pose normalization helpers for AprilTag workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.apriltag_geometry import PinholeFrame

_FACE_NAMES = ("px", "nx", "pz", "nz", "top", "bottom", "py", "ny")
_REFERENCE_FACE_ORDER = ("pz", "px", "nz", "nx", "top", "bottom", "py", "ny")
_SIDE_FACES = frozenset({"px", "nx", "pz", "nz"})
_VERTICAL_FACE_SETS = (frozenset({"top", "bottom"}), frozenset({"py", "ny"}))
_FACE_VIEW_PARAMS: dict[str, tuple[float, float]] = {
    "pz": (0.0, 0.0),
    "px": (90.0, 0.0),
    "nz": (180.0, 0.0),
    "nx": (-90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    "py": (0.0, -90.0),
    "ny": (0.0, 90.0),
}
_GUI_CUBE6_VIEW_PARAMS: dict[str, tuple[float, float]] = {
    # Mirrors gui.cubemap.view_config._CUBE6_VIEW_CELLS with four yaw slots:
    # px=slot0, nz=slot1, nx=slot2, pz=slot3, top/bottom=slot3.
    "px": (0.0, 0.0),
    "nz": (90.0, 0.0),
    "nx": (180.0, 0.0),
    "pz": (-90.0, 0.0),
    "top": (-90.0, 90.0),
    "bottom": (-90.0, -90.0),
}


@dataclass(frozen=True)
class _GeneratedCubemapLayout:
    name: str
    view_params: dict[str, tuple[float, float]]
    face_order: tuple[str, ...]


@dataclass(frozen=True)
class _GeneratedCubemapMatch:
    layout: _GeneratedCubemapLayout
    mode: str
    base_rotation: np.ndarray
    image_rotations: dict[str, np.ndarray]
    max_error: float


_GENERATED_CUBEMAP_LAYOUTS = (
    _GeneratedCubemapLayout("standard", _FACE_VIEW_PARAMS, _REFERENCE_FACE_ORDER),
    _GeneratedCubemapLayout("gui_cube6", _GUI_CUBE6_VIEW_PARAMS, ("bottom", "px", "nz", "nx", "pz", "top")),
)


def cubemap_face_from_path(file_path: str) -> tuple[str, str] | None:
    path = Path(file_path)
    stem = path.stem
    lower = stem.lower()
    for face in _FACE_NAMES:
        suffix = f"_{face}"
        if lower.endswith(suffix):
            return stem[: -len(suffix)], face
    return None


def cubemap_face_rotation(face: str) -> np.ndarray | None:
    params = _FACE_VIEW_PARAMS.get(face)
    if params is None:
        return None
    return _rotation_matrix(*params)


def infer_generated_cubemap_face_rotations(face_frames: dict[str, PinholeFrame]) -> dict[str, np.ndarray] | None:
    """Infer image-space face rotations from converter-generated cube frames.

    ``cubemap_transforms_json.py`` writes cubemap frame transforms with the
    export rotation transposed, while the images themselves are generated with
    the preview/build-remap rotation. Matching against the actual generated
    layout lets AprilTag tools recover image-space pinhole poses without
    treating the face suffix as a hard-coded world direction.
    """
    match = _match_generated_cubemap_layout(face_frames)
    return None if match is None else {face: rotation.copy() for face, rotation in match.image_rotations.items()}


def normalize_standard_cubemap_frames(frames: tuple[PinholeFrame, ...]) -> tuple[PinholeFrame, ...]:
    """Return frames whose rotations match generated cubemap face images.

    The cubemap converter stores face rotations in the downstream 3DGS dataset
    convention. AprilTag detection and synthetic injection need image-space
    pinhole poses: a ray visible in the ``px`` face image must also project into
    the ``px`` frame. Non-cubemap and already-normalized groups are unchanged.
    """
    grouped: dict[str, dict[str, PinholeFrame]] = {}
    for frame in frames:
        parsed = cubemap_face_from_path(frame.file_path)
        if parsed is None:
            continue
        prefix, face = parsed
        grouped.setdefault(prefix, {})[face] = frame

    replacements: dict[str, PinholeFrame] = {}
    for face_frames in grouped.values():
        generated = _match_generated_cubemap_layout(face_frames)
        if generated is not None:
            if generated.mode == "transform":
                for face, frame in face_frames.items():
                    rotation = generated.image_rotations.get(face)
                    if rotation is None:
                        continue
                    transform = np.array(frame.transform_matrix, dtype=float, copy=True)
                    transform[:3, :3] = generated.base_rotation @ rotation
                    transform[:3, 3] = _generated_base_position(face_frames)
                    replacements[frame.file_path] = replace(frame, transform_matrix=transform)
            continue

        rotations = _standard_rotations_for_faces(set(face_frames))
        if rotations is None:
            continue
        reference = _reference_frame(face_frames)
        if not _same_camera_center(face_frames, reference):
            continue
        mode = _normalization_mode(face_frames, reference, rotations)
        if mode != "transposed":
            continue
        for face, rotation in rotations.items():
            frame = face_frames.get(face)
            if frame is None:
                continue
            transform = np.array(frame.transform_matrix, dtype=float, copy=True)
            transform[:3, :3] = reference.camera_to_world_rotation @ rotation
            transform[:3, 3] = reference.camera_position_sfm
            replacements[frame.file_path] = replace(frame, transform_matrix=transform)

    if not replacements:
        return frames
    return tuple(replacements.get(frame.file_path, frame) for frame in frames)


def _generated_base_position(face_frames: dict[str, PinholeFrame]) -> np.ndarray:
    reference = _reference_frame(face_frames)
    return reference.camera_position_sfm


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


def _layout_image_rotations(layout: _GeneratedCubemapLayout, faces: set[str]) -> dict[str, np.ndarray]:
    return {
        face: _rotation_matrix(*layout.view_params[face])
        for face in _REFERENCE_FACE_ORDER
        if face in faces and face in layout.view_params
    }


def _layout_export_rotations(layout: _GeneratedCubemapLayout, faces: set[str]) -> dict[str, np.ndarray]:
    return {
        face: _export_rotation_matrix(*layout.view_params[face])
        for face in _REFERENCE_FACE_ORDER
        if face in faces and face in layout.view_params
    }


def _match_generated_cubemap_layout(face_frames: dict[str, PinholeFrame]) -> _GeneratedCubemapMatch | None:
    if not _same_camera_center(face_frames, _reference_frame(face_frames)):
        return None
    faces = set(face_frames)
    if not _SIDE_FACES.issubset(faces):
        return None
    if next((pair for pair in _VERTICAL_FACE_SETS if pair.issubset(faces)), None) is None:
        return None

    candidates: list[_GeneratedCubemapMatch] = []
    for layout in _GENERATED_CUBEMAP_LAYOUTS:
        image_rotations = _layout_image_rotations(layout, faces)
        export_rotations = _layout_export_rotations(layout, faces)
        if len(image_rotations) < 4 or set(image_rotations) != set(export_rotations):
            continue
        transform_mode = _match_layout_mode(face_frames, layout, export_rotations, mode="transform")
        if transform_mode is not None:
            candidates.append(
                _GeneratedCubemapMatch(
                    layout=layout,
                    mode="transform",
                    base_rotation=transform_mode[0],
                    image_rotations=image_rotations,
                    max_error=transform_mode[1],
                )
            )
        image_mode = _match_layout_mode(face_frames, layout, image_rotations, mode="image")
        if image_mode is not None:
            candidates.append(
                _GeneratedCubemapMatch(
                    layout=layout,
                    mode="image",
                    base_rotation=image_mode[0],
                    image_rotations=image_rotations,
                    max_error=image_mode[1],
                )
            )
    if not candidates:
        return None
    source_order = tuple(face for face in face_frames if face in faces)
    candidates.sort(
        key=lambda item: (
            item.max_error,
            _layout_order_error(item.layout, source_order),
            0 if item.mode == "image" else 1,
        )
    )
    return candidates[0]


def _layout_order_error(layout: _GeneratedCubemapLayout, source_order: tuple[str, ...]) -> int:
    expected = tuple(face for face in layout.face_order if face in source_order)
    if not expected:
        return len(source_order)
    return sum(1 for actual, wanted in zip(source_order, expected, strict=False) if actual != wanted) + abs(
        len(source_order) - len(expected)
    )


def _match_layout_mode(
    face_frames: dict[str, PinholeFrame],
    _layout: _GeneratedCubemapLayout,
    rotations: dict[str, np.ndarray],
    *,
    mode: str,
) -> tuple[np.ndarray, float] | None:
    bases: list[np.ndarray] = []
    for face, rotation in rotations.items():
        frame = face_frames.get(face)
        if frame is None:
            continue
        frame_rotation = np.asarray(frame.camera_to_world_rotation, dtype=np.float64)
        if not _is_rotation_like(frame_rotation):
            return None
        if mode == "transform":
            # transform_json writes base @ export_rotation.T.
            base = frame_rotation @ rotation
        elif mode == "image":
            # Normalized/image-space poses are base @ image_rotation.
            base = frame_rotation @ rotation.T
        else:
            raise ValueError(f"unsupported generated cubemap match mode: {mode}")
        bases.append(base)
    if len(bases) < 4:
        return None
    base = _average_rotations(bases)
    errors = [_rotation_angle(base, candidate) for candidate in bases]
    max_error = max(errors)
    if max_error > np.deg2rad(0.1):
        return None
    return base, max_error


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


def _is_rotation_like(value: np.ndarray) -> bool:
    return value.shape == (3, 3) and bool(np.all(np.isfinite(value)))


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    value = (float(np.trace(delta)) - 1.0) * 0.5
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def _same_camera_center(frames: dict[str, PinholeFrame], reference: PinholeFrame) -> bool:
    for frame in frames.values():
        if float(np.linalg.norm(frame.camera_position_sfm - reference.camera_position_sfm)) > 1e-6:
            return False
    return True


def _standard_rotations_for_faces(faces: set[str]) -> dict[str, np.ndarray] | None:
    if not _SIDE_FACES.issubset(faces):
        return None
    vertical_faces = next((pair for pair in _VERTICAL_FACE_SETS if pair.issubset(faces)), None)
    if vertical_faces is None:
        return None
    rotations: dict[str, np.ndarray] = {}
    for face in _REFERENCE_FACE_ORDER:
        if face not in faces or (face not in _SIDE_FACES and face not in vertical_faces):
            continue
        rotation = cubemap_face_rotation(face)
        if rotation is not None:
            rotations[face] = rotation
    return rotations


def _reference_frame(frames: dict[str, PinholeFrame]) -> PinholeFrame:
    for face in _REFERENCE_FACE_ORDER:
        frame = frames.get(face)
        if frame is not None:
            return frame
    return next(iter(frames.values()))


def _normalization_mode(
    frames: dict[str, PinholeFrame],
    reference: PinholeFrame,
    rotations: dict[str, np.ndarray],
) -> str | None:
    direct_errors: list[float] = []
    transposed_errors: list[float] = []
    for face, rotation in rotations.items():
        frame = frames.get(face)
        if frame is None:
            continue
        direct_errors.append(_rotation_angle(frame.camera_to_world_rotation, reference.camera_to_world_rotation @ rotation))
        transposed_errors.append(
            _rotation_angle(frame.camera_to_world_rotation, reference.camera_to_world_rotation @ rotation.T)
        )
    if not direct_errors or not transposed_errors:
        return None
    direct = max(direct_errors)
    transposed = max(transposed_errors)
    tolerance = np.deg2rad(0.1)
    if direct <= tolerance:
        return "direct"
    if transposed <= tolerance:
        return "transposed"
    return None
