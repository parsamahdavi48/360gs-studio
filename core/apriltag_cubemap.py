"""Cubemap pose normalization helpers for AprilTag workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.apriltag_geometry import PinholeFrame

CubemapViewParams = Mapping[str, tuple[float, float]]

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
    # px=slot0, nz=slot1, nx=slot2, pz=slot3, py/ny=slot3.
    "px": (0.0, 0.0),
    "nz": (90.0, 0.0),
    "nx": (180.0, 0.0),
    "pz": (-90.0, 0.0),
    "py": (-90.0, -90.0),
    "ny": (-90.0, 90.0),
}
_LEGACY_GUI_CUBE6_VIEW_PARAMS: dict[str, tuple[float, float]] = {
    # Older GUI Cube6 output used top/bottom filenames. Keep this only for
    # reading existing scenes; new GUI output uses py/ny.
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
    image_view_params: dict[str, tuple[float, float]]
    export_view_params: dict[str, tuple[float, float]]
    face_order: tuple[str, ...]


@dataclass(frozen=True)
class _GeneratedCubemapMatch:
    layout: _GeneratedCubemapLayout
    mode: str
    base_rotation: np.ndarray
    image_rotations: dict[str, np.ndarray]
    max_error: float


@dataclass(frozen=True)
class CubemapViewMetadata:
    view_params: dict[str, tuple[float, float]]
    yaw_offset_per_frame: float = 0.0
    image_pose_profile: str = ""


_GENERATED_CUBEMAP_LAYOUTS = (
    _GeneratedCubemapLayout("standard", _FACE_VIEW_PARAMS, _FACE_VIEW_PARAMS, _REFERENCE_FACE_ORDER),
    _GeneratedCubemapLayout("gui_cube6", _GUI_CUBE6_VIEW_PARAMS, _GUI_CUBE6_VIEW_PARAMS, ("py", "px", "nz", "nx", "pz", "ny")),
    _GeneratedCubemapLayout(
        "gui_cube6_legacy_top_bottom",
        _LEGACY_GUI_CUBE6_VIEW_PARAMS,
        _LEGACY_GUI_CUBE6_VIEW_PARAMS,
        ("bottom", "px", "nz", "nx", "pz", "top"),
    ),
)
_IMAGE_POSE_PROFILE_LICHTFELD_CUBE6 = "lichtfeld_cube6"
_SOURCE_EQUIRECT_LOCAL_FROM_LICHTFELD_LOCAL = np.diag([1.0, -1.0, -1.0])
_SAVED_RASTER_Y_FLIP = np.diag([1.0, -1.0, 1.0])
_LICHTFELD_VERTICAL_FACE_MAP = {
    "top": "bottom",
    "bottom": "top",
    "py": "ny",
    "ny": "py",
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


def infer_generated_cubemap_face_rotations(
    face_frames: dict[str, PinholeFrame],
    *,
    view_params: CubemapViewParams | None = None,
) -> dict[str, np.ndarray] | None:
    """Infer image-space face rotations from converter-generated cube frames.

    ``cubemap_transforms_json.py`` writes cubemap frame transforms with the
    export rotation transposed, while the images themselves are generated with
    the preview/build-remap rotation. Matching against the actual generated
    layout lets AprilTag tools recover image-space pinhole poses without
    treating the face suffix as a hard-coded world direction.
    """
    match = _match_generated_cubemap_layout(face_frames, view_params=view_params)
    if match is None:
        return None
    if match.mode == "image":
        source_order = tuple(face for face in face_frames)
        if _layout_order_error(match.layout, source_order) != 0:
            return None
    elif match.mode != "transform":
        return None
    return {face: rotation.copy() for face, rotation in match.image_rotations.items()}


def normalize_standard_cubemap_frames(
    frames: tuple[PinholeFrame, ...],
    *,
    view_params: CubemapViewMetadata | CubemapViewParams | None = None,
) -> tuple[PinholeFrame, ...]:
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

    metadata = _coerce_view_metadata(view_params)
    replacements: dict[str, PinholeFrame] = {}
    for group_index, face_frames in enumerate(grouped.values()):
        generated = _match_generated_cubemap_layout(
            face_frames,
            view_params=_view_params_for_group(metadata, group_index),
        )
        if generated is not None:
            if generated.mode == "transform":
                for face, frame in face_frames.items():
                    rotation = _image_rotation_for_saved_raster(generated.image_rotations, face, metadata)
                    if rotation is None:
                        continue
                    transform = np.array(frame.transform_matrix, dtype=float, copy=True)
                    transform[:3, :3] = _saved_raster_camera_rotation(generated.base_rotation, rotation, metadata)
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
        face: _rotation_matrix(*layout.image_view_params[face])
        for face in _REFERENCE_FACE_ORDER
        if face in faces and face in layout.image_view_params
    }


def _layout_export_rotations(layout: _GeneratedCubemapLayout, faces: set[str]) -> dict[str, np.ndarray]:
    return {
        face: _export_rotation_matrix(*layout.export_view_params[face])
        for face in _REFERENCE_FACE_ORDER
        if face in faces and face in layout.export_view_params
    }


def _coerce_view_metadata(
    view_params: CubemapViewMetadata | CubemapViewParams | None,
) -> CubemapViewMetadata | None:
    if view_params is None:
        return None
    if isinstance(view_params, CubemapViewMetadata):
        return view_params
    params = _valid_view_params(view_params)
    return None if params is None else CubemapViewMetadata(params)


def _valid_view_params(view_params: CubemapViewParams | None) -> dict[str, tuple[float, float]] | None:
    if not view_params:
        return None
    params: dict[str, tuple[float, float]] = {}
    for face, raw in view_params.items():
        if face not in _FACE_NAMES:
            continue
        if not isinstance(raw, tuple | list) or len(raw) != 2:
            continue
        yaw, pitch = float(raw[0]), float(raw[1])
        if not np.isfinite(yaw) or not np.isfinite(pitch):
            continue
        params[face] = (yaw, pitch)
    if len(_SIDE_FACES.intersection(params)) < 4:
        return None
    return params


def _frame_yaw_offset(frame_index: int, step_deg: float) -> float:
    if step_deg == 0.0:
        return 0.0
    return (float(frame_index) * float(step_deg)) % 360.0


def _view_params_for_group(
    metadata: CubemapViewMetadata | None,
    group_index: int,
) -> dict[str, tuple[float, float]] | None:
    if metadata is None:
        return None
    offset = _frame_yaw_offset(group_index, metadata.yaw_offset_per_frame)
    if offset == 0.0:
        return metadata.view_params
    return {face: (yaw + offset, pitch) for face, (yaw, pitch) in metadata.view_params.items()}


def _uses_lichtfeld_saved_raster_adapter(metadata: CubemapViewMetadata | None) -> bool:
    return metadata is not None and metadata.image_pose_profile == _IMAGE_POSE_PROFILE_LICHTFELD_CUBE6


def _image_rotation_for_saved_raster(
    image_rotations: dict[str, np.ndarray],
    face: str,
    metadata: CubemapViewMetadata | None,
) -> np.ndarray | None:
    key = _LICHTFELD_VERTICAL_FACE_MAP.get(face, face) if _uses_lichtfeld_saved_raster_adapter(metadata) else face
    return image_rotations.get(key)


def _saved_raster_camera_rotation(
    base_rotation: np.ndarray,
    image_rotation: np.ndarray,
    metadata: CubemapViewMetadata | None,
) -> np.ndarray:
    if not _uses_lichtfeld_saved_raster_adapter(metadata):
        return base_rotation @ image_rotation
    return base_rotation @ _SOURCE_EQUIRECT_LOCAL_FROM_LICHTFELD_LOCAL @ image_rotation @ _SAVED_RASTER_Y_FLIP


def cubemap_view_params_for_group(
    metadata: CubemapViewMetadata | CubemapViewParams | None,
    group_index: int,
) -> dict[str, tuple[float, float]] | None:
    """Return generated face yaw/pitch values for one cubemap camera group."""
    return _view_params_for_group(_coerce_view_metadata(metadata), group_index)


def _metadata_layout(view_params: CubemapViewParams | None) -> _GeneratedCubemapLayout | None:
    params = _valid_view_params(view_params)
    if params is None:
        return None
    return _GeneratedCubemapLayout("metadata", params, params, tuple(params))


def _candidate_generated_layouts(
    view_params: CubemapViewParams | None,
) -> tuple[_GeneratedCubemapLayout, ...]:
    metadata = _metadata_layout(view_params)
    if metadata is None:
        return _GENERATED_CUBEMAP_LAYOUTS
    return (metadata, *_GENERATED_CUBEMAP_LAYOUTS)


def _match_generated_cubemap_layout(
    face_frames: dict[str, PinholeFrame],
    *,
    view_params: CubemapViewParams | None = None,
) -> _GeneratedCubemapMatch | None:
    if not _same_camera_center(face_frames, _reference_frame(face_frames)):
        return None
    faces = set(face_frames)
    if not _SIDE_FACES.issubset(faces):
        return None
    if next((pair for pair in _VERTICAL_FACE_SETS if pair.issubset(faces)), None) is None:
        return None

    candidates: list[_GeneratedCubemapMatch] = []
    for layout in _candidate_generated_layouts(view_params):
        layout_faces = set(layout.image_view_params)
        layout_vertical_faces = next((pair for pair in _VERTICAL_FACE_SETS if pair.issubset(layout_faces)), None)
        if layout_vertical_faces is None or not layout_vertical_faces.issubset(faces):
            continue
        image_rotations = _layout_image_rotations(layout, faces)
        export_rotations = _layout_export_rotations(layout, faces)
        if len(image_rotations) < 6 or set(image_rotations) != set(export_rotations):
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
            0 if item.layout.name == "metadata" else 1,
            item.max_error,
            _layout_order_error(item.layout, source_order),
            0 if item.mode == "image" else 1,
        )
    )
    return candidates[0]


def _views_from_payload(data: object) -> dict[str, tuple[float, float]] | None:
    if not isinstance(data, dict):
        return None
    raw_views = data.get("views")
    if raw_views is None and isinstance(data.get("views_config_snapshot"), dict):
        raw_views = data["views_config_snapshot"].get("views")
    if raw_views is None and isinstance(data.get("view_config"), dict):
        raw_views = data["view_config"].get("views")
    if not isinstance(raw_views, list):
        return None
    params: dict[str, tuple[float, float]] = {}
    for raw in raw_views:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if name not in _FACE_NAMES:
            continue
        if raw.get("enabled", True) is False:
            continue
        try:
            yaw = float(raw["yaw"])
            pitch = float(raw["pitch"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(yaw) and np.isfinite(pitch):
            params[name] = (yaw, pitch)
    return params or None


def _yaw_offset_per_frame_from_payload(data: object) -> float:
    if not isinstance(data, dict):
        return 0.0
    raw = data.get("yaw_offset_per_frame")
    if raw is None and isinstance(data.get("conversion"), dict):
        raw = data["conversion"].get("yaw_offset_per_frame")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _image_pose_profile_from_payload(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    profile = str(data.get("effective_profile") or data.get("target_profile") or "").strip().lower()
    axis_transform = str(data.get("axis_transform") or "").strip().lower()
    output_shape = str(data.get("output_shape") or "").strip().lower()
    postprocess = data.get("postprocess")
    has_lichtfeld_final_fix = isinstance(postprocess, dict) and bool(
        postprocess.get("lichtfeld_final_orientation_correction")
    )
    if (
        profile == "lichtfeld"
        and axis_transform in {"none", ""}
        and output_shape in {"projected", "projection_views", ""}
        and (has_lichtfeld_final_fix or axis_transform == "none")
    ):
        return _IMAGE_POSE_PROFILE_LICHTFELD_CUBE6
    return ""


def load_cubemap_view_metadata(path: Path) -> CubemapViewMetadata | None:
    """Load face yaw/pitch metadata from Step 4 settings or views_config JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    params = _views_from_payload(data)
    if not params:
        return None
    return CubemapViewMetadata(
        params,
        yaw_offset_per_frame=_yaw_offset_per_frame_from_payload(data),
        image_pose_profile=_image_pose_profile_from_payload(data),
    )


def load_cubemap_view_params(path: Path) -> dict[str, tuple[float, float]] | None:
    """Load base face yaw/pitch metadata without per-frame yaw offsets."""
    metadata = load_cubemap_view_metadata(path)
    return None if metadata is None else metadata.view_params


def discover_cubemap_view_metadata(transforms_json: Path) -> CubemapViewMetadata | None:
    """Find the view metadata that generated a cubemap transforms.json, if available."""
    transforms_json = Path(transforms_json)
    roots = [transforms_json.parent]
    if transforms_json.parent.name.lower() in {"output", "metashape_import"}:
        roots.append(transforms_json.parent.parent)
    roots.append(transforms_json.parent.parent)

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "view_export_settings.json",
                root / "_stechdrive" / "step4" / "export_settings.json",
                root / "_stechdrive" / "step4" / "views_config.json",
            ]
        )
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if not candidate.is_file():
            continue
        metadata = load_cubemap_view_metadata(candidate)
        if metadata:
            return metadata
    return None


def discover_cubemap_view_params(transforms_json: Path) -> dict[str, tuple[float, float]] | None:
    """Find base face yaw/pitch metadata without per-frame yaw offsets."""
    metadata = discover_cubemap_view_metadata(transforms_json)
    return None if metadata is None else metadata.view_params


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
