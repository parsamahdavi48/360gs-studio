"""Cubemap pose normalization helpers for AprilTag workflows."""

from __future__ import annotations

from dataclasses import replace
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
