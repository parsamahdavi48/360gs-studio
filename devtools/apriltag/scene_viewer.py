"""Scene viewer for Cube6 AprilTag development datasets."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.apriltag_cubemap import CubemapViewMetadata, discover_cubemap_view_metadata
from core.apriltag_geometry import (
    PinholeFrame,
    load_pinhole_frames,
    points_intersect_image,
    project_sfm_points,
    tag_corners_sfm,
)
from core.orientation_correction import FINAL_ORIENTATION_LICHTFELD, final_orientation_is_applied
from core.apriltag_pipeline import collect_observations
from core.apriltag_scale import estimate_scene_scale
from core.image_io import imread_unicode
from devtools.apriltag.case import AprilTagDevCase, load_case_or_scene, save_case
from devtools.apriltag.coordinates import (
    COORDINATE_PROFILE_LICHTFELD_CUBE6,
    COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY,
    COORDINATE_PROFILES,
    DEFAULT_COORDINATE_PROFILE,
    coordinate_profile_label,
    normalize_coordinate_profile,
    pointcloud_display_matrix,
    world_display_matrix,
)
from devtools.apriltag.cubemap_preview import (
    CubemapFrameGroup,
    axis_face_view_params,
    cubemap_image_face_rotations,
    image_space_cubemap_frame_group,
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    render_cubemap_axis_equirect,
    render_generated_cubemap_source_axis,
    render_source_equirect_axis,
    source_equirect_base_rotation,
    split_cubemap_face,
)
from devtools.apriltag.printable import create_printable_target
from devtools.apriltag.synthetic import SyntheticAprilTagConfig, inject_synthetic_apriltag
from devtools.apriltag.world_debug_view import (
    AprilTagWorldDebugView,
    PointCloudSample,
    load_point_cloud_sample,
    transform_point_cloud_sample,
)
from gui.common.drag_spinbox import DragDoubleSpinBox
from gui.common.perspective_image_view import PerspectiveImageView, PerspectiveLabelOverlay
from gui.common.perspective_preview import (
    PERSPECTIVE_LOOK_DEG_PER_PIXEL,
    PerspectiveParams,
    clamp_pitch_deg,
    normalize_yaw_deg,
    params_from_drag,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FACE_ORDER = ("pz", "px", "nx", "nz", "top", "bottom", "py", "ny")
SIDE_FACE_ORDER = frozenset({"pz", "px", "nx", "nz"})
RAY_BASIS_WORLD = "world"
RAY_BASIS_IMAGE = "image"
RAY_BASIS_BOTH = "both"
RIGHT_VIEW_IMAGE_POINTCLOUD = "image_pointcloud"
RIGHT_VIEW_POINTCLOUD = "pointcloud"
RIGHT_VIEW_SOURCE_EQUIRECT = "image"
RIGHT_VIEW_RECONSTRUCTED_CUBE6 = "cube6_reconstruct"
RIGHT_VIEW_IMAGE_MODES = {
    RIGHT_VIEW_IMAGE_POINTCLOUD,
    RIGHT_VIEW_SOURCE_EQUIRECT,
    RIGHT_VIEW_RECONSTRUCTED_CUBE6,
}
LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES = {
    COORDINATE_PROFILE_LICHTFELD_CUBE6,
    COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY,
}
SOURCE_EQUIRECT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")
SOURCE_EQUIRECT_LOCAL_FROM_LICHTFELD_LOCAL = np.diag([1.0, -1.0, -1.0])
SOURCE_PREVIEW_LOCAL_Y_FLIP = np.diag([1.0, -1.0, 1.0])
SYNTHETIC_IMAGE_RASTER_Y_FLIP = np.diag([1.0, -1.0, 1.0])
SYNTHETIC_OUTPUT_FACE_ROTATION_FACE = {
    "top": "bottom",
    "bottom": "top",
    "py": "ny",
    "ny": "py",
}
DEFAULT_EQUIRECT_PREVIEW_SIZE = (4096, 2048)


@dataclass(frozen=True)
class SyntheticTagFrameCandidate:
    frame: PinholeFrame
    projected_points: np.ndarray
    distance_sfm: float
    view_angle_deg: float
    area_px: float


@dataclass(frozen=True)
class SyntheticScaleValidationRequest:
    case_dir: Path
    transforms_json: Path
    run_dir: Path
    cubemap_view_metadata: CubemapViewMetadata | None
    frame_transform_overrides: dict[str, np.ndarray]
    tag_family: str
    tag_id: int
    tag_size_m: float
    true_scale: float
    expected_scale: float
    tag_center_sfm: np.ndarray
    tag_normal_sfm: np.ndarray
    tag_up_sfm: np.ndarray
    selected_paths: frozenset[str]
    candidate_count: int
    total_frame_count: int
    candidate_report: list[dict]
    tag_display: dict[str, object]
    candidate_filters: dict[str, float]


class SyntheticScaleValidationWorker(QObject):
    progress = Signal(str, bool)
    finished = Signal(object, str)

    def __init__(self, request: SyntheticScaleValidationRequest) -> None:
        super().__init__()
        self._request = request

    @Slot()
    def run(self) -> None:
        request = self._request
        try:
            self.progress.emit(
                f"実行中 2/6: タグ画像を生成中... 候補 {request.candidate_count}/{request.total_frame_count}",
                True,
            )
            target = create_printable_target(
                request.run_dir / "assets",
                family=request.tag_family,
                tag_id=request.tag_id,
                tag_size_m=request.tag_size_m,
            )
            self.progress.emit(
                f"実行中 3/6: Cube6画像へタグを合成中... 候補 {request.candidate_count}/{request.total_frame_count}",
                True,
            )
            synthetic_report = inject_synthetic_apriltag(
                SyntheticAprilTagConfig(
                    input_transforms=request.transforms_json,
                    output_dir=request.run_dir,
                    tag_image=target.marker_png,
                    tag_size_m=request.tag_size_m,
                    true_scale=request.true_scale,
                    tag_center_sfm=request.tag_center_sfm,
                    tag_normal_sfm=request.tag_normal_sfm,
                    tag_up_sfm=request.tag_up_sfm,
                    frame_file_paths=request.selected_paths,
                    copy_unselected_frames=False,
                    output_tagged_only=True,
                    cubemap_view_params=request.cubemap_view_metadata,
                    frame_transform_overrides=request.frame_transform_overrides,
                    write_normalized_transforms=True,
                )
            )
            self.progress.emit(
                f"実行中 4/6: AprilTagを検出中... 合成 {synthetic_report['frames_written']} / "
                f"書き出し対象 {request.candidate_count} frames",
                True,
            )
            frames, frame_detections, observations = collect_observations(
                request.run_dir / "transforms.json",
                image_root=None,
                tag_size_m=request.tag_size_m,
                family=request.tag_family,
                tag_ids={int(request.tag_id)},
                normalize_cubemap=False,
            )
            self.progress.emit(
                f"実行中 5/6: スケールを推定中... 観測 {len(observations)}",
                True,
            )
            estimate = None
            estimate_error = ""
            try:
                estimate = estimate_scene_scale(observations)
            except ValueError as e:
                estimate_error = str(e)

            detected_frames = sum(1 for item in frame_detections if item.detections)
            detection_count = sum(len(item.detections) for item in frame_detections)
            result: dict[str, object] = {
                "schema_version": 1,
                "case_dir": str(request.case_dir),
                "run_dir": str(request.run_dir),
                "input_transforms": str(request.transforms_json),
                "cubemap_view_metadata": _cubemap_view_metadata_report(request.cubemap_view_metadata),
                "frame_transform_override_count": len(request.frame_transform_overrides),
                "expected_scale": request.expected_scale,
                "tag_display": request.tag_display,
                "tag_sfm": {
                    "center": request.tag_center_sfm.tolist(),
                    "normal": request.tag_normal_sfm.tolist(),
                    "up": request.tag_up_sfm.tolist(),
                    "true_scale": request.true_scale,
                },
                "candidate_filters": request.candidate_filters,
                "candidate_count": request.candidate_count,
                "total_frame_count": request.total_frame_count,
                "candidates": request.candidate_report,
                "synthetic_report": synthetic_report,
                "loaded_frame_count": len(frames),
                "detected_frame_count": detected_frames,
                "detection_count": detection_count,
                "observation_count": len(observations),
                "estimate_error": estimate_error,
            }
            if estimate is not None:
                error_pct = (
                    (estimate.scale - request.expected_scale) / request.expected_scale * 100.0
                    if request.expected_scale
                    else 0.0
                )
                result["estimate"] = {
                    "scale": estimate.scale,
                    "pair_count": estimate.pair_count,
                    "inlier_count": estimate.inlier_count,
                    "rms_residual_m": estimate.rms_residual_m,
                    "median_pair_scale": estimate.median_pair_scale,
                    "mad_pair_scale": estimate.mad_pair_scale,
                    "error_pct": error_pct,
                }
                status = (
                    f"scale={estimate.scale:.6g} m/SfM "
                    f"(期待 {request.expected_scale:.6g}, 誤差 {error_pct:+.2f}%) / "
                    f"候補 {request.candidate_count}/{request.total_frame_count}, "
                    f"合成 {synthetic_report['frames_written']}, "
                    f"検出 {detection_count}, obs {len(observations)}, pairs {estimate.pair_count}"
                )
            else:
                status = (
                    f"推定不可: {estimate_error} / "
                    f"候補 {request.candidate_count}/{request.total_frame_count}, "
                    f"合成 {synthetic_report['frames_written']}, "
                    f"検出 {detection_count}, obs {len(observations)}"
                )
            self.progress.emit("実行中 6/6: レポートを書き出し中...", True)
            (request.run_dir / "viewer_scale_validation_report.json").write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
            self.finished.emit({"status": status, "run_dir": str(request.run_dir), "result": result}, "")
        except Exception as e:
            self.finished.emit(None, str(e))


def _transform_points(points: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if matrix is None:
        return values.copy()
    transform = np.asarray(matrix, dtype=np.float64)
    return values @ transform[:3, :3].T + transform[:3, 3]


def _transform_points_from_world_display(points: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if matrix is None:
        return values.copy()
    transform = np.asarray(matrix, dtype=np.float64)
    inverse_linear = np.linalg.inv(transform[:3, :3])
    return (values - transform[:3, 3]) @ inverse_linear.T


def _transform_vectors_from_world_display(vectors: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if matrix is None:
        return values.copy()
    inverse_linear = np.linalg.inv(np.asarray(matrix, dtype=np.float64)[:3, :3])
    return values @ inverse_linear.T


def _polygon_area_px(points: np.ndarray) -> float:
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 3:
        return 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _transform_frame_for_world_display(
    frame: PinholeFrame,
    matrix: np.ndarray | None,
) -> PinholeFrame:
    if matrix is None:
        transform = np.eye(4, dtype=np.float64)
    else:
        transform = np.asarray(matrix, dtype=np.float64)
    output = np.array(frame.transform_matrix, dtype=np.float64, copy=True)
    output[:3, :3] = transform[:3, :3] @ frame.camera_to_world_rotation
    output[:3, 3] = _transform_points(frame.camera_position_sfm.reshape(1, 3), matrix)[0]
    return replace(frame, transform_matrix=output)


def transform_group_for_world_display(
    group: CubemapFrameGroup,
    matrix: np.ndarray | None,
    *,
    image_ray_correction: np.ndarray | None = None,
    cubemap_view_params: CubemapViewMetadata | None = None,
) -> CubemapFrameGroup:
    # Kept for backward-compatible callers. Image sampling corrections must not
    # be folded into world camera poses; doing so mirrors the displayed Cube6
    # images and makes face-ray gizmos left-handed.
    _ = image_ray_correction, cubemap_view_params
    if matrix is None:
        return group
    return CubemapFrameGroup(
        name=group.name,
        frames_by_face={
            face: _transform_frame_for_world_display(frame, matrix)
            for face, frame in group.frames_by_face.items()
        },
        group_index=group.group_index,
    )


def image_ray_display_matrix_for_profile(
    profile: str | None,
    world_matrix: np.ndarray | None,
) -> np.ndarray | None:
    """Return the display transform for generated Cube6 image rays.

    LichtFeld camera poses need ``world_display_matrix`` to line up with the
    corrected PLY. The saved Cube6 image rays come from the Metashape panorama
    pixels, so after ``image_space_cubemap_frame_group`` they are already in
    the display/Metashape orientation. Applying the LichtFeld world display
    matrix again rotates the image preview to the opposite horizontal faces.
    """
    if normalize_coordinate_profile(profile) in LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES:
        return None
    return None if world_matrix is None else world_matrix.copy()


def image_preview_group_for_profile(
    profile: str | None,
    world_group: CubemapFrameGroup,
    image_group: CubemapFrameGroup,
) -> CubemapFrameGroup:
    _ = profile, world_group
    return image_group


def compose_display_matrices(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None:
        return None if second is None else second.copy()
    if second is None:
        return first.copy()
    return first @ second


def case_cubemap_view_metadata(case: AprilTagDevCase) -> CubemapViewMetadata | None:
    """Return the view yaw/pitch metadata that generated this dev case."""
    candidates = [case.source_transforms, case.transforms_for_processing()]
    for transforms in candidates:
        metadata = discover_cubemap_view_metadata(transforms)
        if metadata:
            return metadata
    return None


def _cubemap_view_metadata_report(metadata: CubemapViewMetadata | None) -> dict[str, object] | None:
    if metadata is None:
        return None
    return {
        "view_params": {face: [float(yaw), float(pitch)] for face, (yaw, pitch) in metadata.view_params.items()},
        "yaw_offset_per_frame": float(metadata.yaw_offset_per_frame),
    }


def _apply_frame_transform_overrides(
    frames: tuple[PinholeFrame, ...],
    overrides: Mapping[str, np.ndarray],
) -> tuple[PinholeFrame, ...]:
    if not overrides:
        return frames
    return tuple(
        replace(frame, transform_matrix=np.asarray(overrides[frame.file_path], dtype=np.float64))
        if frame.file_path in overrides
        else frame
        for frame in frames
    )


def _scene_root_candidates(case: AprilTagDevCase) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for path in (
        case.image_root,
        case.source_transforms.parent,
        case.transforms_for_processing().parent,
        case.source_metashape_xml.parent if case.source_metashape_xml else None,
    ):
        if path is None:
            continue
        candidates.append(path)
        if path.name.lower() == "output":
            candidates.append(path.parent)
        parent = path.parent
        if parent.name.lower() == "output":
            candidates.append(parent.parent)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).lower()
        except OSError:
            key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _normalized_size(width: object, height: object) -> tuple[int, int] | None:
    try:
        w = int(width)  # type: ignore[arg-type]
        h = int(height)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h


def _size_from_object(value: object) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        return _normalized_size(
            value.get("w", value.get("width")),
            value.get("h", value.get("height")),
        )
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "")
        if "x" in text:
            width, height = text.split("x", 1)
            return _normalized_size(width, height)
    if isinstance(value, list | tuple) and len(value) >= 2:
        return _normalized_size(value[0], value[1])
    return None


def _source_equirect_size_from_payload(payload: object) -> tuple[int, int] | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("input_size", "source_size", "source_equirect_size", "equirect_size", "video"):
        size = _size_from_object(payload.get(key))
        if size is not None:
            return size
    sessions = payload.get("sessions")
    if isinstance(sessions, list):
        for session in reversed(sessions):
            if not isinstance(session, Mapping):
                continue
            size = _size_from_object(session.get("video"))
            if size is not None:
                return size
    return None


def _source_equirect_size_from_json(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _source_equirect_size_from_payload(payload)


def _case_source_equirect_metadata_size(case: AprilTagDevCase) -> tuple[int, int] | None:
    candidates: list[Path] = []
    for root in _scene_root_candidates(case):
        candidates.extend(
            [
                root / "view_export_settings.json",
                root / "_stechdrive" / "frames" / "extract_report.json",
                root / "_stechdrive" / "frames" / "extract_sessions.json",
                root / "_stechdrive" / "step4" / "export_settings.json",
            ]
        )
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).lower()
        except OSError:
            key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        size = _source_equirect_size_from_json(candidate)
        if size is not None:
            return size
    return None


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return _normalized_size(image.width, image.height)
    except Exception:
        return None


def _image_array_size(image: np.ndarray | None) -> tuple[int, int] | None:
    if image is None or image.size == 0 or image.ndim < 2:
        return None
    height, width = image.shape[:2]
    return _normalized_size(width, height)


def _cubemap_equirect_preview_size(group: CubemapFrameGroup | None) -> tuple[int, int] | None:
    if group is None:
        return None
    face_size = 0
    for frame in group.frames:
        face_size = max(face_size, int(frame.width), int(frame.height))
    if face_size <= 0:
        return None
    return face_size * 4, face_size * 2


def _clamp_equirect_preview_size(
    source_size: tuple[int, int] | None,
    cubemap_size: tuple[int, int] | None,
    *,
    fallback_size: tuple[int, int] = DEFAULT_EQUIRECT_PREVIEW_SIZE,
) -> tuple[int, int]:
    fallback = _size_from_object(fallback_size) or DEFAULT_EQUIRECT_PREVIEW_SIZE
    source = _size_from_object(source_size) if source_size is not None else None
    cubemap = _size_from_object(cubemap_size) if cubemap_size is not None else None
    if source is not None and cubemap is not None:
        return max(1, min(source[0], cubemap[0])), max(1, min(source[1], cubemap[1]))
    if source is not None:
        return source
    if cubemap is not None:
        return max(1, min(cubemap[0], fallback[0])), max(1, min(cubemap[1], fallback[1]))
    return fallback


def _selected_frame_output_map(scene_root: Path) -> dict[str, Path]:
    csv_path = scene_root / "_stechdrive" / "frames" / "selected_frames.csv"
    if not csv_path.is_file():
        return {}
    mapping: dict[str, Path] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                output_file = str(row.get("output_file") or "").strip()
                if not output_file:
                    continue
                path = scene_root / Path(output_file)
                if path.is_file():
                    mapping[Path(output_file).stem] = path
    except OSError:
        return {}
    return mapping


def _resolve_source_equirect_paths(case: AprilTagDevCase, group_names: tuple[str, ...]) -> dict[str, Path]:
    names = tuple(dict.fromkeys(str(name) for name in group_names if name))
    if not names:
        return {}
    resolved: dict[str, Path] = {}
    for scene_root in _scene_root_candidates(case):
        from_csv = _selected_frame_output_map(scene_root)
        for name in names:
            path = from_csv.get(name)
            if path is not None and path.is_file():
                resolved.setdefault(name, path)
        image_dir = scene_root / "images"
        if image_dir.is_dir():
            for name in names:
                if name in resolved:
                    continue
                for ext in SOURCE_EQUIRECT_IMAGE_EXTS:
                    path = image_dir / f"{name}{ext}"
                    if path.is_file():
                        resolved[name] = path
                        break
        if len(resolved) == len(names):
            break
    return resolved


def _source_equirect_rotations_from_groups(
    groups: tuple[CubemapFrameGroup, ...],
    metadata: CubemapViewMetadata | None,
    *,
    undo_legacy_lichtfeld_local_flip: bool = True,
) -> dict[str, np.ndarray]:
    rotations: dict[str, np.ndarray] = {}
    for group in groups:
        rotation = source_equirect_base_rotation(group, cubemap_view_params=metadata)
        if rotation is not None:
            if undo_legacy_lichtfeld_local_flip:
                # The source raster local-axis adapter is independent from the
                # final dataset orientation. final_orientation is already
                # folded into transforms.json/pointcloud.ply and therefore
                # into source_equirect_base_rotation().
                rotations[group.name] = rotation @ SOURCE_EQUIRECT_LOCAL_FROM_LICHTFELD_LOCAL
            else:
                rotations[group.name] = rotation
    return rotations


def _case_has_lichtfeld_final_orientation(case: AprilTagDevCase | None) -> bool:
    if case is None:
        return False
    for path in (case.transforms_for_processing(), case.source_transforms):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if final_orientation_is_applied(payload, FINAL_ORIENTATION_LICHTFELD):
            return True
    return False


def _source_raster_frame_group(
    group: CubemapFrameGroup,
    source_world_to_image_rotation: np.ndarray,
    metadata: CubemapViewMetadata | None,
) -> CubemapFrameGroup | None:
    face_rotations = cubemap_image_face_rotations(group, cubemap_view_params=metadata)
    if face_rotations is None:
        return None
    rotation = np.asarray(source_world_to_image_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        return None
    position = group.reference_frame.camera_position_sfm
    frames: dict[str, PinholeFrame] = {}
    for face, frame in group.frames_by_face.items():
        face_rotation = _source_raster_face_rotation(face_rotations, face, metadata)
        if face_rotation is None:
            continue
        transform = np.array(frame.transform_matrix, dtype=np.float64, copy=True)
        transform[:3, :3] = rotation @ face_rotation
        transform[:3, 3] = position
        frames[face] = replace(frame, transform_matrix=transform)
    if len(frames) < 4:
        return None
    return CubemapFrameGroup(name=group.name, frames_by_face=frames, group_index=group.group_index)


def _source_raster_face_rotation(
    face_rotations: dict[str, np.ndarray],
    face: str,
    metadata: CubemapViewMetadata | None,
) -> np.ndarray | None:
    if metadata is not None and metadata.image_pose_profile == COORDINATE_PROFILE_LICHTFELD_CUBE6:
        raster_face = _synthetic_output_face_rotation_face(face)
        rotation = face_rotations.get(raster_face)
        if rotation is None:
            return None
        return rotation @ SYNTHETIC_IMAGE_RASTER_Y_FLIP
    return face_rotations.get(face)


def _source_preview_render_rotation(rotation: np.ndarray, profile: str | None) -> np.ndarray:
    if normalize_coordinate_profile(profile) in LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES:
        # This is a raster sampling adapter, not a world/camera-pose transform.
        # The left view and pointcloud mode stay in the proper right-handed
        # display world; only the source panorama latitude lookup is flipped so
        # pitch motion samples the same physical direction as the frustum.
        return np.asarray(rotation, dtype=np.float64) @ SOURCE_PREVIEW_LOCAL_Y_FLIP
    return np.asarray(rotation, dtype=np.float64)


def _synthetic_output_face_rotation_face(face: str) -> str:
    """Return the generated face rotation key for synthetic output projection.

    ``SOURCE_EQUIRECT_LOCAL_FROM_LICHTFELD_LOCAL`` undoes the source panorama's
    local Y/Z flip for image lookup. That makes the pole directions opposite
    for projection into saved Cube6 JPG rasters: the file path stays unchanged,
    but vertical faces need the opposite generated pitch when deciding which
    image receives synthetic pixels. ``top/bottom`` are legacy names; new GUI
    Cube6 output uses ``py/ny``.
    """
    return SYNTHETIC_OUTPUT_FACE_ROTATION_FACE.get(face, face)


def _source_equirect_preview_params(
    params: PerspectiveParams,
    active_face: str,
    ray_basis_mode: str,
    anchor_params: PerspectiveParams | None = None,
) -> PerspectiveParams:
    """Return the camera params used by source-panorama image previews.

    The right image view must show the same world rays as the left frustum and
    right pointcloud view. Raster orientation can be unusual for some Cube6
    faces; do not compensate by changing the camera direction here.
    """
    _ = active_face, ray_basis_mode, anchor_params
    return params


def _uses_source_preview_screen_axis_adapter(active_face: str, ray_basis_mode: str) -> bool:
    _ = active_face, ray_basis_mode
    return True


def _params_from_grab_drag(
    params: PerspectiveParams,
    delta_x: float,
    delta_y: float,
    *,
    source_preview_screen_axis_adapter: bool,
    screen_right: np.ndarray | None = None,
    screen_up: np.ndarray | None = None,
) -> PerspectiveParams:
    """Apply viewport drags as grab/pan-style view movement.

    When the active camera has roll, screen up/right no longer match raw
    yaw/pitch. If the current viewport basis is available, solve the yaw/pitch
    delta from the visible screen axes so dragging stays touch-like in monitor
    space without changing any world or image-ray transforms.
    """
    if screen_right is not None and screen_up is not None:
        return _params_from_screen_drag(
            params,
            delta_x,
            delta_y,
            screen_right=screen_right,
            screen_up=screen_up,
            horizontal_drag_sign=-1.0 if source_preview_screen_axis_adapter else 1.0,
        )
    if source_preview_screen_axis_adapter:
        return params_from_drag(params, -delta_x, delta_y)
    return params_from_drag(params, delta_x, delta_y)


def _params_from_screen_drag(
    params: PerspectiveParams,
    delta_x: float,
    delta_y: float,
    *,
    screen_right: np.ndarray,
    screen_up: np.ndarray,
    horizontal_drag_sign: float = 1.0,
    degrees_per_pixel: float = PERSPECTIVE_LOOK_DEG_PER_PIXEL,
) -> PerspectiveParams:
    yaw = np.deg2rad(float(params.yaw_deg))
    pitch = np.deg2rad(float(params.pitch_deg))
    screen_right = _normalized_vector(np.asarray(screen_right, dtype=np.float64), (1.0, 0.0, 0.0))
    screen_up = _normalized_vector(np.asarray(screen_up, dtype=np.float64), (0.0, 1.0, 0.0))
    desired = np.deg2rad(float(degrees_per_pixel)) * (
        float(horizontal_drag_sign) * float(delta_x) * screen_right
        + float(delta_y) * screen_up
    )

    d_forward_d_yaw = np.array(
        [np.cos(yaw) * np.cos(pitch), 0.0, -np.sin(yaw) * np.cos(pitch)],
        dtype=np.float64,
    )
    d_forward_d_pitch = np.array(
        [-np.sin(yaw) * np.sin(pitch), -np.cos(pitch), -np.cos(yaw) * np.sin(pitch)],
        dtype=np.float64,
    )
    jacobian = np.column_stack([d_forward_d_yaw, d_forward_d_pitch])
    try:
        delta, _residuals, _rank, _singular = np.linalg.lstsq(jacobian, desired, rcond=None)
    except np.linalg.LinAlgError:
        delta = np.array([0.0, 0.0], dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        delta = np.array([0.0, 0.0], dtype=np.float64)
    return PerspectiveParams(
        yaw_deg=normalize_yaw_deg(float(params.yaw_deg) + float(np.rad2deg(delta[0]))),
        pitch_deg=clamp_pitch_deg(float(params.pitch_deg) + float(np.rad2deg(delta[1]))),
        fov_deg=params.fov_deg,
        roll_deg=params.roll_deg,
    )


def _load_metashape_camera_transforms(xml_path: Path) -> dict[str, np.ndarray]:
    if not xml_path.is_file():
        return {}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {}
    transforms: dict[str, np.ndarray] = {}
    for camera in root.findall(".//camera"):
        label = str(camera.attrib.get("label") or "").strip()
        transform_text = camera.findtext("transform")
        if not label or not transform_text:
            continue
        values = [float(value) for value in transform_text.split()]
        if len(values) != 16:
            continue
        transform = np.asarray(values, dtype=np.float64).reshape(4, 4)
        if np.all(np.isfinite(transform)):
            transforms[Path(label).stem] = transform.copy()
    return transforms


def _load_metashape_camera_positions(xml_path: Path) -> dict[str, np.ndarray]:
    return {
        label: transform[:3, 3].copy()
        for label, transform in _load_metashape_camera_transforms(xml_path).items()
    }


def estimate_world_display_matrix_from_metashape(
    groups: tuple[CubemapFrameGroup, ...],
    xml_path: Path | None,
) -> tuple[np.ndarray, float, int] | None:
    if xml_path is None:
        return None
    metashape_positions = _load_metashape_camera_positions(xml_path)
    if not metashape_positions:
        return None
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    for group in groups:
        metashape_position = metashape_positions.get(group.name)
        if metashape_position is None:
            continue
        source.append(group.camera_position_sfm)
        target.append(metashape_position)
    if len(source) < 3:
        return None
    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    linear = np.linalg.lstsq(source_points, target_points, rcond=None)[0].T
    try:
        u, _s, vt = np.linalg.svd(linear)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    transformed = source_points @ rotation.T
    rmse = float(np.sqrt(np.mean((transformed - target_points) ** 2)))
    span = float(np.max(np.ptp(target_points, axis=0)))
    tolerance = max(1e-5, span * 1e-4)
    if rmse > tolerance:
        return None
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return matrix, rmse, len(source)


def rotation_from_perspective_params(params: PerspectiveParams) -> np.ndarray:
    yaw = np.deg2rad(float(params.yaw_deg))
    pitch = np.deg2rad(float(params.pitch_deg))
    roll = np.deg2rad(float(params.roll_deg))
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


def camera_pose_from_perspective_params(
    group: CubemapFrameGroup,
    params: PerspectiveParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = rotation_from_perspective_params(params)
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64) @ rotation.T
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64) @ rotation.T
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ rotation.T
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = up - right * float(up @ right)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    forward = forward - right * float(forward @ right) - up * float(forward @ up)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    return np.asarray(group.camera_position_sfm, dtype=np.float64), right, up, forward


def _normalized_vector(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.asarray(fallback, dtype=np.float64)
    return value / norm


def _right_view_screen_basis(
    *,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the perspective screen basis used by AprilTagWorldDebugView."""
    forward = _normalized_vector(np.asarray(forward, dtype=np.float64), (0.0, 0.0, 1.0))
    back = -forward
    right = np.asarray(right, dtype=np.float64) - back * float(np.asarray(right, dtype=np.float64) @ back)
    if float(np.linalg.norm(right)) <= 1e-12:
        right = np.cross(np.asarray(up, dtype=np.float64), back)
    right = _normalized_vector(right, (1.0, 0.0, 0.0))
    up_from_basis = _normalized_vector(np.cross(back, right), (0.0, 1.0, 0.0))
    if float(up_from_basis @ np.asarray(up, dtype=np.float64)) < 0.0:
        right = -right
        up_from_basis = _normalized_vector(np.cross(back, right), (0.0, 1.0, 0.0))
    return right, up_from_basis, forward


def right_image_screen_x_sign(*, use_source_equirect: bool, use_reconstructed_cube6: bool) -> float:
    """Screen X adapter for source-axis image previews.

    The right pointcloud view follows AprilTagWorldDebugView's screen contract
    where camera-local +X appears on monitor-left for the fixed frustum view.
    Source equirect and reconstructed Cube6 previews are already rendered into
    the correct camera rays, so only their final viewport X mapping needs this
    display adapter.
    """
    return -1.0 if use_source_equirect or use_reconstructed_cube6 else 1.0


def project_world_points_to_right_view(
    group: CubemapFrameGroup,
    params: PerspectiveParams,
    points: np.ndarray,
    *,
    output_size: int,
) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera, right, up, forward = camera_pose_from_perspective_params(group, params)
    right, up, forward = _right_view_screen_basis(right=right, up=up, forward=forward)
    rel = points - camera.reshape(1, 3)
    depth = rel @ forward
    if np.any(~np.isfinite(depth)) or np.any(depth <= 1e-8):
        return None
    size = max(1, int(output_size))
    focal = 0.5 * float(size) / np.tan(np.deg2rad(float(params.fov_deg)) * 0.5)
    center = float(size) * 0.5
    x = center + focal * ((rel @ right) / depth)
    y = center - focal * ((rel @ up) / depth)
    return np.column_stack([x, y]).astype(np.float32)


def project_world_points_to_image_view(
    group: CubemapFrameGroup,
    params: PerspectiveParams,
    points: np.ndarray,
    *,
    output_size: int,
) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera, right, up, forward = camera_pose_from_perspective_params(group, params)
    rel = points - camera.reshape(1, 3)
    depth = rel @ forward
    if np.any(~np.isfinite(depth)) or np.any(depth <= 1e-8):
        return None
    size = max(1, int(output_size))
    focal = 0.5 * float(size) / np.tan(np.deg2rad(float(params.fov_deg)) * 0.5)
    center = float(size) * 0.5
    x = center + focal * ((rel @ right) / depth)
    y = center - focal * ((rel @ up) / depth)
    return np.column_stack([x, y]).astype(np.float32)


def project_world_points_to_right_view_visible(
    group: CubemapFrameGroup,
    params: PerspectiveParams,
    points: np.ndarray,
    *,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    camera, right, up, forward = camera_pose_from_perspective_params(group, params)
    right, up, forward = _right_view_screen_basis(right=right, up=up, forward=forward)
    rel = points - camera.reshape(1, 3)
    depth = rel @ forward
    valid = np.isfinite(depth) & (depth > 1e-8)
    if not np.any(valid):
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    valid_indices = np.nonzero(valid)[0]
    size = max(1, int(output_size))
    focal = 0.5 * float(size) / np.tan(np.deg2rad(float(params.fov_deg)) * 0.5)
    center = float(size) * 0.5
    rel = rel[valid]
    depth = depth[valid]
    x = center + focal * ((rel @ right) / depth)
    y = center - focal * ((rel @ up) / depth)
    xy = np.column_stack([x, y]).astype(np.float32)
    finite = np.isfinite(xy).all(axis=1)
    return xy[finite], depth[finite], valid_indices[finite]


def project_world_points_to_image_view_visible(
    group: CubemapFrameGroup,
    params: PerspectiveParams,
    points: np.ndarray,
    *,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    camera, right, up, forward = camera_pose_from_perspective_params(group, params)
    rel = points - camera.reshape(1, 3)
    depth = rel @ forward
    valid = np.isfinite(depth) & (depth > 1e-8)
    if not np.any(valid):
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    valid_indices = np.nonzero(valid)[0]
    size = max(1, int(output_size))
    focal = 0.5 * float(size) / np.tan(np.deg2rad(float(params.fov_deg)) * 0.5)
    center = float(size) * 0.5
    rel = rel[valid]
    depth = depth[valid]
    x = center + focal * ((rel @ right) / depth)
    y = center - focal * ((rel @ up) / depth)
    xy = np.column_stack([x, y]).astype(np.float32)
    finite = np.isfinite(xy).all(axis=1)
    return xy[finite], depth[finite], valid_indices[finite]


def face_forward_ray(group: CubemapFrameGroup, face: str) -> np.ndarray | None:
    frame = group.frames_by_face.get(face)
    if frame is None:
        return None
    ray = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ frame.camera_to_world_rotation.T
    norm = float(np.linalg.norm(ray))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    return ray / norm


def closest_image_face_for_world_face(
    world_group: CubemapFrameGroup,
    image_group: CubemapFrameGroup,
    world_face: str,
) -> tuple[str, float, float] | None:
    world_ray = face_forward_ray(world_group, world_face)
    if world_ray is None:
        return None
    best: tuple[str, float] | None = None
    same_label_angle = np.nan
    for image_face in FACE_ORDER:
        image_ray = face_forward_ray(image_group, image_face)
        if image_ray is None:
            continue
        dot = float(np.clip(world_ray @ image_ray, -1.0, 1.0))
        angle = float(np.rad2deg(np.arccos(dot)))
        if image_face == world_face:
            same_label_angle = angle
        if best is None or angle < best[1]:
            best = (image_face, angle)
    if best is None:
        return None
    return best[0], best[1], same_label_angle


def opposite_image_face_for_world_face(
    world_group: CubemapFrameGroup,
    image_group: CubemapFrameGroup,
    world_face: str,
) -> tuple[str, float] | None:
    world_ray = face_forward_ray(world_group, world_face)
    if world_ray is None:
        return None
    best: tuple[str, float] | None = None
    for image_face in FACE_ORDER:
        image_ray = face_forward_ray(image_group, image_face)
        if image_ray is None:
            continue
        dot = float(np.clip((-world_ray) @ image_ray, -1.0, 1.0))
        angle = float(np.rad2deg(np.arccos(dot)))
        if best is None or angle < best[1]:
            best = (image_face, angle)
    return best


def face_mapping_summary(
    world_group: CubemapFrameGroup,
    image_group: CubemapFrameGroup,
) -> str:
    parts: list[str] = []
    for face in FACE_ORDER:
        if face not in world_group.frames_by_face:
            continue
        closest = closest_image_face_for_world_face(world_group, image_group, face)
        if closest is None:
            continue
        image_face, angle, same_label_angle = closest
        opposite = opposite_image_face_for_world_face(world_group, image_group, face)
        opposite_text = "" if opposite is None else f", reverse={opposite[0]} {opposite[1]:.1f}deg"
        same = "" if not np.isfinite(same_label_angle) else f", same={same_label_angle:.1f}deg"
        parts.append(f"{face}->img {image_face} ({angle:.1f}deg{same}{opposite_text})")
    return "; ".join(parts)


class AprilTagSceneViewerWindow(QWidget):
    """Two-pane debug viewer for Cube6 transforms and point clouds."""

    scene_loaded = Signal()
    validation_finished = Signal()

    def __init__(self, *, initial_case: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("AprilTag Cube6 Scene Viewer")
        icon_path = REPO_ROOT / "gui" / "assets" / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1500, 900)

        self.case: AprilTagDevCase | None = None
        self._raw_groups: tuple[CubemapFrameGroup, ...] = ()
        self._world_groups: tuple[CubemapFrameGroup, ...] = ()
        self._image_ray_groups: tuple[CubemapFrameGroup, ...] = ()
        self._image_preview_groups: tuple[CubemapFrameGroup, ...] = ()
        self._world_pointcloud: PointCloudSample | None = None
        self._world_matrix: np.ndarray | None = None
        self._metashape_alignment: tuple[float, int] | None = None
        self._world_ray_source = ""
        self._image_ray_source = ""
        self._image_cache: dict[Path, np.ndarray] = {}
        self._image_size_cache: dict[Path, tuple[int, int]] = {}
        self._equirect_cache: dict[str, np.ndarray] = {}
        self._source_equirect_paths: dict[str, Path] = {}
        self._source_equirect_rotations: dict[str, np.ndarray] = {}
        self._source_equirect_metadata_size: tuple[int, int] | None = None
        self._displayed_image_key = ""
        self._params = PerspectiveParams(fov_deg=90.0)
        self._active_face = "pz"
        self._tag_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._tag_yaw_deg = 0.0
        self._tag_pitch_deg = 0.0
        self._tag_roll_deg = 0.0
        self._tag_size_sfm = 0.64
        self._tag_physical_size_m = 0.160
        self._last_validation_scale_text = ""
        self._last_status_detail = ""
        self._validation_running = False
        self._validation_thread: QThread | None = None
        self._validation_worker: SyntheticScaleValidationWorker | None = None
        self._right_view_mode_user_selected = False
        self._setting_default_right_view_mode = False

        self._build_ui()
        self._connect_signals()
        case_dir = initial_case
        if case_dir is not None and ((case_dir / "case.json").is_file() or (case_dir / "output" / "transforms.json").is_file()):
            self.load_case_dir(case_dir)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.case_label = QLabel("シーン未選択")
        self.case_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.case_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.open_case_button = QPushButton("シーンを開く")
        self.reload_button = QPushButton("再読み込み")
        header.addWidget(self.case_label, 1)
        header.addWidget(self.open_case_button)
        header.addWidget(self.reload_button)
        root.addLayout(header)

        controls = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.prev_button = QPushButton("前")
        self.next_button = QPushButton("次")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("点群", RIGHT_VIEW_POINTCLOUD)
        self.mode_combo.addItem("元360画像", RIGHT_VIEW_SOURCE_EQUIRECT)
        self.mode_combo.addItem("Cube6再構築", RIGHT_VIEW_RECONSTRUCTED_CUBE6)
        self.ray_basis_combo = QComboBox()
        self.ray_basis_combo.addItem("画像レイ", RAY_BASIS_IMAGE)
        self.ray_basis_combo.addItem("JSON Face(デバッグ)", RAY_BASIS_WORLD)
        self.ray_basis_combo.addItem("両方(デバッグ)", RAY_BASIS_BOTH)
        self.profile_combo = QComboBox()
        for profile in COORDINATE_PROFILES:
            self.profile_combo.addItem(profile.label, profile.id)
        controls.addWidget(QLabel("カメラ"))
        controls.addWidget(self.camera_combo, 1)
        controls.addWidget(self.prev_button)
        controls.addWidget(self.next_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel("右ビュー"))
        controls.addWidget(self.mode_combo)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Face線基準"))
        controls.addWidget(self.ray_basis_combo)
        controls.addSpacing(12)
        controls.addWidget(QLabel("座標解釈"))
        controls.addWidget(self.profile_combo)
        root.addLayout(controls)

        content_splitter = QSplitter(Qt.Horizontal)
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setMinimumWidth(300)
        self.sidebar_scroll.setMaximumWidth(420)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        sidebar_layout.setSpacing(8)

        self.face_box = QGroupBox("FOV 90° Face")
        face_layout = QGridLayout(self.face_box)
        face_layout.setContentsMargins(8, 8, 8, 8)
        face_layout.setHorizontalSpacing(6)
        face_layout.setVerticalSpacing(6)
        self.face_buttons: dict[str, QPushButton] = {}
        for index, face in enumerate(FACE_ORDER):
            button = QPushButton(face)
            button.setCheckable(True)
            self.face_buttons[face] = button
            face_layout.addWidget(button, index // 4, index % 4)
        sidebar_layout.addWidget(self.face_box)

        self.tag_controls_box = self._build_tag_controls()
        sidebar_layout.addWidget(self.tag_controls_box)

        result_row = QHBoxLayout()
        result_row.setContentsMargins(0, 0, 0, 0)
        result_row.setSpacing(6)
        result_label = QLabel("結果")
        result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        result_row.addWidget(result_label)
        self.validation_status_label = QLabel("検証未実行")
        self.validation_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.validation_status_label.setWordWrap(True)
        self.validation_status_label.setMinimumWidth(0)
        self.validation_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        result_row.addWidget(self.validation_status_label, 1)
        sidebar_layout.addLayout(result_row)

        sidebar_layout.addStretch(1)
        self.sidebar_scroll.setWidget(sidebar)
        content_splitter.addWidget(self.sidebar_scroll)

        splitter = QSplitter(Qt.Horizontal)
        self.world_view = AprilTagWorldDebugView()
        self.world_view.setMinimumSize(520, 420)
        self.right_stack = QStackedWidget()
        self.point_view = AprilTagWorldDebugView()
        self.point_view.set_fixed_navigation_enabled(False)
        self.point_view.set_fixed_screen_zoom_enabled(True)
        self.point_view.setMinimumSize(520, 420)
        self.image_view = PerspectiveImageView("Cubemap画像を読み込みます")
        self.image_view.setMinimumSize(520, 420)
        self.image_view.set_drag_mode("look")
        self.right_stack.addWidget(self.point_view)
        self.right_stack.addWidget(self.image_view)
        splitter.addWidget(self.world_view)
        splitter.addWidget(self.right_stack)
        splitter.setSizes([760, 740])
        self.view_splitter = splitter
        content_splitter.addWidget(splitter)
        content_splitter.setSizes([340, 1180])
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        self.content_splitter = content_splitter
        root.addWidget(content_splitter, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(80)
        self.log.setMaximumHeight(130)
        root.addWidget(self.log)

    @staticmethod
    def _compact_fields_row(title: str, fields: tuple[tuple[str, QWidget], ...]) -> QWidget:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        title_label = QLabel(title)
        title_label.setFixedWidth(44)
        row.addWidget(title_label)
        for label_text, widget in fields:
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(label)
            widget.setMinimumWidth(52)
            row.addWidget(widget, 1)
        return row_widget

    def _build_tag_controls(self) -> QGroupBox:
        group = QGroupBox("AprilTag検証")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.tag_x_spin = self._double_spin(-1_000_000.0, 1_000_000.0, 0.0, decimals=3, step=0.05)
        self.tag_y_spin = self._double_spin(-1_000_000.0, 1_000_000.0, 0.0, decimals=3, step=0.05)
        self.tag_z_spin = self._double_spin(-1_000_000.0, 1_000_000.0, 0.0, decimals=3, step=0.05)
        layout.addWidget(
            self._compact_fields_row(
                "位置",
                (("X", self.tag_x_spin), ("Y", self.tag_y_spin), ("Z", self.tag_z_spin)),
            )
        )

        self.tag_yaw_spin = self._double_spin(-180.0, 180.0, 0.0, decimals=1, step=5.0)
        self.tag_pitch_spin = self._double_spin(-89.0, 89.0, 0.0, decimals=1, step=5.0)
        self.tag_roll_spin = self._double_spin(-180.0, 180.0, 0.0, decimals=1, step=5.0)
        layout.addWidget(
            self._compact_fields_row(
                "角度",
                (("Yw", self.tag_yaw_spin), ("Pt", self.tag_pitch_spin), ("Rl", self.tag_roll_spin)),
            )
        )

        self.tag_size_sfm_spin = self._double_spin(0.0001, 1_000_000.0, self._tag_size_sfm, decimals=4, step=0.05)
        self.tag_physical_size_spin = self._double_spin(0.001, 100.0, self._tag_physical_size_m, decimals=3, step=0.01, suffix=" m")
        self.tag_size_sfm_spin.setToolTip("左ビューのSfM空間に配置するAprilTag一辺の長さです。")
        self.tag_physical_size_spin.setToolTip("検出器へ渡す現実のAprilTag一辺の長さです。")
        self.reset_tag_button = QPushButton("原点へ戻す")
        layout.addWidget(
            self._compact_fields_row(
                "サイズ",
                (("タグ", self.tag_size_sfm_spin), ("物理", self.tag_physical_size_spin)),
            )
        )
        layout.addWidget(self.reset_tag_button)

        validation_form = QFormLayout()
        validation_form.setLabelAlignment(Qt.AlignRight)
        self.validation_distance_spin = self._double_spin(
            0.0,
            1_000_000.0,
            self._default_validation_distance_sfm(),
            decimals=3,
            step=0.5,
        )
        self.validation_angle_spin = self._double_spin(0.0, 180.0, 75.0, decimals=1, step=5.0)
        self.validation_min_area_spin = self._double_spin(0.0, 1_000_000.0, 64.0, decimals=0, step=64.0)
        self.validation_distance_spin.setToolTip("タグ中心からカメラまでの距離がこのSfM値以内の画像だけを合成対象にします。0では距離で除外しません。")
        self.validation_angle_spin.setToolTip("タグ正面とカメラ方向の角度がこの値以内の画像だけを合成対象にします。")
        self.validation_min_area_spin.setToolTip("画像上に投影されたタグ四角形の面積がこのpx^2未満の画像を合成対象から外します。")
        self.run_validation_button = QPushButton("検出")
        self.copy_validation_scale_button = QPushButton("scaleコピー")
        self.copy_validation_scale_button.setEnabled(False)
        validation_actions = QHBoxLayout()
        validation_actions.setContentsMargins(0, 0, 0, 0)
        validation_actions.setSpacing(6)
        validation_actions.addWidget(self.run_validation_button)
        validation_actions.addWidget(self.copy_validation_scale_button)
        validation_form.addRow("認識範囲", self.validation_distance_spin)
        validation_form.addRow("最大角度", self.validation_angle_spin)
        validation_form.addRow("最小投影面積", self.validation_min_area_spin)
        validation_form.addRow("", validation_actions)
        layout.addLayout(validation_form)
        return group

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int,
        step: float,
        suffix: str = "",
    ) -> DragDoubleSpinBox:
        spin = DragDoubleSpinBox(
            minimum=float(minimum),
            maximum=float(maximum),
            step=float(step),
            decimals=int(decimals),
            value=float(value),
            suffix=suffix,
            drag_pixels_per_step=6.0,
        )
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    def _connect_signals(self) -> None:
        self.open_case_button.clicked.connect(self._choose_case)
        self.reload_button.clicked.connect(self.reload)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.prev_button.clicked.connect(lambda: self._step_camera(-1))
        self.next_button.clicked.connect(lambda: self._step_camera(1))
        self.mode_combo.currentIndexChanged.connect(self._on_right_view_mode_changed)
        self.ray_basis_combo.currentIndexChanged.connect(self._on_ray_basis_changed)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.world_view.camera_clicked.connect(self.select_camera_by_name)
        self.point_view.fixed_view_dragged.connect(self._on_right_view_dragged)
        self.world_view.gpu_pointcloud_failed.connect(lambda: self._on_gpu_pointcloud_failed("左ビュー"))
        self.point_view.gpu_pointcloud_failed.connect(lambda: self._on_gpu_pointcloud_failed("右点群ビュー"))
        self.image_view.look_dragged.connect(self._on_right_view_dragged)
        for face, button in self.face_buttons.items():
            button.clicked.connect(lambda _checked=False, value=face: self.set_active_face(value))
        for spin in (
            self.tag_x_spin,
            self.tag_y_spin,
            self.tag_z_spin,
            self.tag_yaw_spin,
            self.tag_pitch_spin,
            self.tag_roll_spin,
            self.tag_size_sfm_spin,
            self.tag_physical_size_spin,
        ):
            spin.valueChanged.connect(lambda _value: self._on_tag_transform_changed())
        self.validation_distance_spin.valueChanged.connect(lambda _value: self._sync_tag_views())
        self.reset_tag_button.clicked.connect(self.reset_tag_transform)
        self.run_validation_button.clicked.connect(self.run_synthetic_scale_validation)
        self.copy_validation_scale_button.clicked.connect(self._copy_validation_scale)

    def _on_gpu_pointcloud_failed(self, label: str) -> None:
        self._append_log_once(f"gpu-pointcloud-failed:{label}", f"{label}: GPU点群描画を初期化できないためCPU描画へ戻しました。")
        self._update_status()

    def load_case_dir(self, case_dir: Path) -> None:
        try:
            self.case = load_case_or_scene(case_dir)
        except Exception as e:
            QMessageBox.critical(self, "シーン読み込みエラー", str(e))
            return
        self._right_view_mode_user_selected = False
        self._set_profile_combo(self.case.coordinate_profile)
        self._set_tag_defaults_from_case(self.case)
        self.reload()

    def _on_right_view_mode_changed(self, _index: int) -> None:
        if not self._setting_default_right_view_mode:
            self._right_view_mode_user_selected = True
        self._sync_views()

    def reload(self) -> None:
        if self.case is None:
            return
        self._image_cache.clear()
        self._image_size_cache.clear()
        self._equirect_cache.clear()
        self._source_equirect_paths = {}
        self._source_equirect_rotations = {}
        self._source_equirect_metadata_size = None
        self._displayed_image_key = ""
        try:
            metadata = case_cubemap_view_metadata(self.case)
            self._source_equirect_metadata_size = _case_source_equirect_metadata_size(self.case)
            groups = load_cubemap_frame_groups(
                self.case.transforms_for_processing(),
                cubemap_view_params=metadata,
                normalize_cubemap=False,
            )
            labels = load_metashape_camera_labels(self.case.source_metashape_xml) if self.case.source_metashape_xml else ()
            self._raw_groups = order_groups_by_labels(groups, labels)
            self._source_equirect_paths = _resolve_source_equirect_paths(
                self.case,
                tuple(group.name for group in self._raw_groups),
            )
            self._source_equirect_rotations = _source_equirect_rotations_from_groups(
                self._raw_groups,
                metadata,
            )
            self._update_world_matrix()
            self._world_groups = tuple(
                transform_group_for_world_display(
                    group,
                    self._world_matrix,
                    cubemap_view_params=metadata,
                )
                for group in self._raw_groups
            )
            fallback_image_ray_display_matrix = image_ray_display_matrix_for_profile(
                self.case.coordinate_profile,
                self._world_matrix,
            )
            use_source_raster_image_rays = (
                normalize_coordinate_profile(self.case.coordinate_profile) in LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES
            )
            image_ray_groups: list[CubemapFrameGroup] = []
            for group in self._raw_groups:
                source_rotation = (
                    self._source_equirect_rotations.get(group.name)
                    if use_source_raster_image_rays
                    else None
                )
                source_raster_group = (
                    _source_raster_frame_group(group, source_rotation, metadata)
                    if source_rotation is not None
                    else None
                )
                if source_raster_group is not None:
                    image_ray_groups.append(
                        transform_group_for_world_display(
                            source_raster_group,
                            self._world_matrix,
                            cubemap_view_params=metadata,
                        )
                    )
                else:
                    image_ray_groups.append(
                        transform_group_for_world_display(
                            image_space_cubemap_frame_group(group, cubemap_view_params=metadata),
                            fallback_image_ray_display_matrix,
                            cubemap_view_params=metadata,
                        )
                    )
            self._image_ray_groups = tuple(image_ray_groups)
            self._image_preview_groups = tuple(
                image_preview_group_for_profile(
                    self.case.coordinate_profile,
                    world_group,
                    image_group,
                )
                for world_group, image_group in zip(self._world_groups, self._image_ray_groups, strict=True)
            )
            self._world_ray_source = "transforms.json face +Z"
            self._image_ray_source = "Cube6 export yaw/pitch"
            self._load_pointcloud()
        except Exception as e:
            QMessageBox.critical(self, "シーン読み込みエラー", str(e))
            return
        self._populate_camera_combo()
        self._select_default_right_view_mode()
        self._sync_views()
        self.scene_loaded.emit()

    def _select_default_right_view_mode(self) -> None:
        if self._right_view_mode_user_selected:
            return
        preferred = (
            RIGHT_VIEW_RECONSTRUCTED_CUBE6
            if self._source_equirect_rotations
            else RIGHT_VIEW_POINTCLOUD
        )
        index = self.mode_combo.findData(preferred)
        if index < 0:
            return
        self._setting_default_right_view_mode = True
        try:
            self.mode_combo.setCurrentIndex(index)
        finally:
            self._setting_default_right_view_mode = False

    def select_camera_by_name(self, name: str) -> None:
        index = self.camera_combo.findData(str(name))
        if index < 0:
            self._append_log(f"Camera not found: {name}")
            return
        self.camera_combo.setCurrentIndex(index)

    def set_active_face(self, face: str) -> None:
        if face not in FACE_ORDER:
            return
        self._active_face = face
        self._set_params_from_active_face()
        self._sync_views()

    def selected_world_group(self) -> CubemapFrameGroup | None:
        if not self._world_groups:
            return None
        index = self.camera_combo.currentIndex()
        if index < 0 or index >= len(self._world_groups):
            return self._world_groups[0]
        return self._world_groups[index]

    def selected_raw_group(self) -> CubemapFrameGroup | None:
        if not self._raw_groups:
            return None
        index = self.camera_combo.currentIndex()
        if index < 0 or index >= len(self._raw_groups):
            return self._raw_groups[0]
        return self._raw_groups[index]

    def selected_image_ray_group(self) -> CubemapFrameGroup | None:
        if not self._image_ray_groups:
            return None
        index = self.camera_combo.currentIndex()
        if index < 0 or index >= len(self._image_ray_groups):
            return self._image_ray_groups[0]
        return self._image_ray_groups[index]

    def selected_image_preview_group(self) -> CubemapFrameGroup | None:
        if not self._image_preview_groups:
            return self.selected_image_ray_group()
        index = self.camera_combo.currentIndex()
        if index < 0 or index >= len(self._image_preview_groups):
            return self._image_preview_groups[0]
        return self._image_preview_groups[index]

    def selected_image_render_group(self) -> CubemapFrameGroup | None:
        if self._ray_basis_mode() == RAY_BASIS_IMAGE:
            return self.selected_image_ray_group()
        return self.selected_image_preview_group()

    def set_tag_transform(
        self,
        *,
        center: tuple[float, float, float] | np.ndarray | None = None,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
        roll_deg: float | None = None,
        size_sfm: float | None = None,
    ) -> None:
        if center is not None:
            values = np.asarray(center, dtype=np.float64).reshape(3)
            self._set_spin_value_blocked(self.tag_x_spin, float(values[0]))
            self._set_spin_value_blocked(self.tag_y_spin, float(values[1]))
            self._set_spin_value_blocked(self.tag_z_spin, float(values[2]))
        if yaw_deg is not None:
            self._set_spin_value_blocked(self.tag_yaw_spin, float(yaw_deg))
        if pitch_deg is not None:
            self._set_spin_value_blocked(self.tag_pitch_spin, float(pitch_deg))
        if roll_deg is not None:
            self._set_spin_value_blocked(self.tag_roll_spin, float(roll_deg))
        if size_sfm is not None:
            self._set_spin_value_blocked(self.tag_size_sfm_spin, max(0.0001, float(size_sfm)))
        self._read_tag_transform_from_controls()
        self._sync_views()

    def reset_tag_transform(self) -> None:
        self.set_tag_transform(
            center=(0.0, 0.0, 0.0),
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            size_sfm=self._default_tag_size_sfm(self.case),
        )

    @staticmethod
    def _set_spin_value_blocked(spin: DragDoubleSpinBox, value: float) -> None:
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    @staticmethod
    def _default_tag_size_sfm(case: AprilTagDevCase | None) -> float:
        if case is None:
            return 0.64
        return max(0.0001, float(case.default_tag_size_m) / max(float(case.true_scale), 1e-12))

    def _default_validation_distance_sfm(self) -> float:
        return max(2.0, float(self._tag_size_sfm) * 30.0)

    def _set_tag_defaults_from_case(self, case: AprilTagDevCase) -> None:
        self._set_spin_value_blocked(self.tag_x_spin, 0.0)
        self._set_spin_value_blocked(self.tag_y_spin, 0.0)
        self._set_spin_value_blocked(self.tag_z_spin, 0.0)
        self._set_spin_value_blocked(self.tag_yaw_spin, 0.0)
        self._set_spin_value_blocked(self.tag_pitch_spin, 0.0)
        self._set_spin_value_blocked(self.tag_roll_spin, 0.0)
        self._set_spin_value_blocked(self.tag_size_sfm_spin, self._default_tag_size_sfm(case))
        self._set_spin_value_blocked(self.tag_physical_size_spin, float(case.default_tag_size_m))
        self._read_tag_transform_from_controls()
        if hasattr(self, "validation_distance_spin"):
            self._set_spin_value_blocked(self.validation_distance_spin, self._default_validation_distance_sfm())
        if hasattr(self, "validation_status_label"):
            self.validation_status_label.setText("検証未実行")
        if hasattr(self, "copy_validation_scale_button"):
            self._last_validation_scale_text = ""
            self.copy_validation_scale_button.setEnabled(False)

    def _on_tag_transform_changed(self) -> None:
        self._read_tag_transform_from_controls()
        self._sync_views()

    def _read_tag_transform_from_controls(self) -> None:
        self._tag_center = np.array(
            [
                self.tag_x_spin.value(),
                self.tag_y_spin.value(),
                self.tag_z_spin.value(),
            ],
            dtype=np.float64,
        )
        self._tag_yaw_deg = float(self.tag_yaw_spin.value())
        self._tag_pitch_deg = float(self.tag_pitch_spin.value())
        self._tag_roll_deg = float(self.tag_roll_spin.value())
        self._tag_size_sfm = max(0.0001, float(self.tag_size_sfm_spin.value()))
        self._tag_physical_size_m = max(0.001, float(self.tag_physical_size_spin.value()))

    def _tag_normal_up(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = rotation_from_perspective_params(
            PerspectiveParams(
                yaw_deg=self._tag_yaw_deg,
                pitch_deg=self._tag_pitch_deg,
                roll_deg=self._tag_roll_deg,
                fov_deg=90.0,
            )
        )
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float64) @ rotation.T
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64) @ rotation.T
        return normal, up

    def _tag_corners_world_display(self) -> np.ndarray:
        normal, up = self._tag_normal_up()
        return tag_corners_sfm(
            self._tag_center,
            normal,
            up,
            self._tag_size_sfm,
            1.0,
        )

    def _sync_tag_views(self) -> None:
        normal, up = self._tag_normal_up()
        for view in (self.world_view, self.point_view):
            view.set_tag(
                center=self._tag_center,
                normal=normal,
                up=up,
                tag_size_m=self._tag_size_sfm,
                true_scale=1.0,
            )
        distance = float(self.validation_distance_spin.value()) if hasattr(self, "validation_distance_spin") else 0.0
        self.world_view.set_tag_validation_distance(distance)
        self.point_view.set_tag_validation_distance(None)

    def _tag_image_overlays(
        self,
        group: CubemapFrameGroup | None,
        params: PerspectiveParams,
        *,
        output_size: int = 768,
        color_bgr: tuple[int, int, int] | None = None,
        image_view_basis: bool = False,
    ) -> list[PerspectiveLabelOverlay]:
        if group is None:
            return []
        try:
            projector = (
                project_world_points_to_image_view
                if image_view_basis
                else project_world_points_to_right_view
            )
            projected = projector(
                group,
                params,
                self._tag_corners_world_display(),
                output_size=output_size,
            )
        except Exception:
            return []
        if projected is None or not np.all(np.isfinite(projected)):
            return []
        min_xy = np.floor(projected.min(axis=0)).astype(int)
        max_xy = np.ceil(projected.max(axis=0)).astype(int)
        size = int(output_size)
        if max_xy[0] < 0 or max_xy[1] < 0 or min_xy[0] > size or min_xy[1] > size:
            return []
        if int(max(max_xy - min_xy)) > size * 3:
            return []
        origin_y = max(18, int(min_xy[1]) - 8)
        color = color_bgr if color_bgr is not None else self._tag_overlay_color(projected)
        return [
            PerspectiveLabelOverlay(
                label="tag",
                box=(int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])),
                origin=(int(min_xy[0]), origin_y),
                color_bgr=color,
                highlighted=True,
                polygon=tuple((float(x), float(y)) for x, y in projected),
                fill_alpha=0.16,
            )
        ]

    def _pointcloud_image_overlays(
        self,
        group: CubemapFrameGroup | None,
        params: PerspectiveParams,
        *,
        output_size: int = 768,
        max_points: int = 30_000,
        image_view_basis: bool = False,
    ) -> list[PerspectiveLabelOverlay]:
        if group is None or self._world_pointcloud is None or len(self._world_pointcloud.points) == 0:
            return []
        points = self._world_pointcloud.points
        colors = self._world_pointcloud.colors
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points = points[indices]
            colors = colors[indices] if colors is not None else None
        visible_projector = (
            project_world_points_to_image_view_visible
            if image_view_basis
            else project_world_points_to_right_view_visible
        )
        projected, _depth, indices = visible_projector(
            group,
            params,
            points,
            output_size=output_size,
        )
        if len(projected) == 0:
            return []
        size = float(max(1, int(output_size)))
        mask = (
            (projected[:, 0] >= 0.0)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 0] < size)
            & (projected[:, 1] < size)
        )
        if not np.any(mask):
            return []
        projected = projected[mask]
        colors = colors[indices[mask]] if colors is not None else None
        if colors is None:
            return [
                PerspectiveLabelOverlay(
                    label="",
                    box=(0, 0, 0, 0),
                    origin=(0, 0),
                    color_bgr=(164, 155, 145),
                    points=tuple((float(x), float(y)) for x, y in projected),
                    point_alpha=0.58,
                    point_radius=1.0,
                )
            ]

        overlays: list[PerspectiveLabelOverlay] = []
        quantized = (np.clip(colors, 0, 255).astype(np.uint8) // 64) * 64 + 32
        keys = (
            quantized[:, 0].astype(np.uint32) << 16
            | quantized[:, 1].astype(np.uint32) << 8
            | quantized[:, 2].astype(np.uint32)
        )
        for key in np.unique(keys):
            group_mask = keys == key
            rgb = quantized[group_mask][0]
            overlays.append(
                PerspectiveLabelOverlay(
                    label="",
                    box=(0, 0, 0, 0),
                    origin=(0, 0),
                    color_bgr=(int(rgb[2]), int(rgb[1]), int(rgb[0])),
                    points=tuple((float(x), float(y)) for x, y in projected[group_mask]),
                    point_alpha=0.62,
                    point_radius=1.0,
                )
            )
        return overlays

    def _tag_overlay_color(self, points: np.ndarray) -> tuple[int, int, int]:
        min_area = float(self.validation_min_area_spin.value()) if hasattr(self, "validation_min_area_spin") else 0.0
        if min_area > 0.0 and _polygon_area_px(np.asarray(points, dtype=np.float64)) < min_area:
            return (0, 64, 255)
        return (0, 255, 180)

    def _tag_projected_overlay(
        self,
        projected: np.ndarray,
        *,
        width: int,
        height: int,
        output_size: int = 768,
    ) -> list[PerspectiveLabelOverlay]:
        points = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            return []
        scale = float(max(1, int(output_size))) / float(max(1, min(int(width), int(height))))
        scaled = points * scale
        min_xy = np.floor(scaled.min(axis=0)).astype(int)
        max_xy = np.ceil(scaled.max(axis=0)).astype(int)
        size = int(output_size)
        if max_xy[0] < 0 or max_xy[1] < 0 or min_xy[0] > size or min_xy[1] > size:
            return []
        if int(max(max_xy - min_xy)) > size * 3:
            return []
        origin_y = max(18, int(min_xy[1]) - 8)
        color = self._tag_overlay_color(scaled)
        return [
            PerspectiveLabelOverlay(
                label="tag",
                box=(int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])),
                origin=(int(min_xy[0]), origin_y),
                color_bgr=color,
                highlighted=True,
                polygon=tuple((float(x), float(y)) for x, y in scaled),
                fill_alpha=0.16,
            )
        ]

    def _tag_output_image_overlays(self, *, output_size: int = 768) -> list[PerspectiveLabelOverlay]:
        if self.case is None:
            return []
        overrides = self._synthetic_frame_transform_overrides()
        if not overrides:
            return []
        raw_group = self.selected_raw_group()
        if raw_group is None:
            return []
        candidates, _total = self._synthetic_tag_candidates()
        selected = [
            candidate
            for candidate in candidates
            if (parsed := split_cubemap_face(candidate.frame.file_path)) is not None and parsed[0] == raw_group.name
        ]
        if not selected:
            return []
        selected.sort(key=lambda candidate: (-candidate.area_px, candidate.frame.file_path))
        candidate = selected[0]
        return self._tag_projected_overlay(
            candidate.projected_points,
            width=candidate.frame.width,
            height=candidate.frame.height,
            output_size=output_size,
        )

    def _tag_front_faces_group(self, group: CubemapFrameGroup) -> bool:
        normal, _up = self._tag_normal_up()
        camera_delta = np.asarray(group.camera_position_sfm, dtype=np.float64) - self._tag_center
        distance = float(np.linalg.norm(camera_delta))
        if distance <= 1e-12 or not np.isfinite(distance):
            return False
        return float(normal @ (camera_delta / distance)) > 0.0

    def _tag_viewport_image_overlays(
        self,
        world_group: CubemapFrameGroup,
        *,
        projection_params: PerspectiveParams | None = None,
        output_size: int = 768,
        image_view_basis: bool = False,
    ) -> list[PerspectiveLabelOverlay]:
        color = None if self._tag_front_faces_group(world_group) else (0, 64, 255)
        params = self._params if projection_params is None else projection_params
        return self._tag_image_overlays(
            world_group,
            params,
            output_size=output_size,
            color_bgr=color,
            image_view_basis=image_view_basis,
        )

    def _right_image_tag_overlays(
        self,
        world_group: CubemapFrameGroup,
        *,
        use_output_projection: bool,
        projection_params: PerspectiveParams | None = None,
        output_size: int = 768,
        image_view_basis: bool = False,
    ) -> list[PerspectiveLabelOverlay]:
        params = self._params if projection_params is None else projection_params
        if use_output_projection:
            overlays = self._tag_viewport_image_overlays(
                world_group,
                projection_params=params,
                output_size=output_size,
                image_view_basis=image_view_basis,
            )
            if overlays or self._synthetic_frame_transform_overrides():
                return overlays
        return self._tag_image_overlays(
            world_group,
            params,
            output_size=output_size,
            image_view_basis=image_view_basis,
        )

    def _synthetic_tag_placement_sfm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.case is None:
            raise ValueError("ケースが読み込まれていません")
        normal_display, up_display = self._tag_normal_up()
        center_sfm = _transform_points_from_world_display(
            self._tag_center.reshape(1, 3),
            self._world_matrix,
        )[0]
        vectors_sfm = _transform_vectors_from_world_display(
            np.vstack([normal_display, up_display]),
            self._world_matrix,
        )
        normal_sfm = _normalized_vector(vectors_sfm[0], (0.0, 0.0, -1.0))
        up_sfm = vectors_sfm[1] - normal_sfm * float(vectors_sfm[1] @ normal_sfm)
        up_sfm = _normalized_vector(up_sfm, (0.0, 1.0, 0.0))
        true_scale = float(self._tag_physical_size_m) / max(float(self._tag_size_sfm), 1e-12)
        return center_sfm, normal_sfm, up_sfm, true_scale

    def _synthetic_frame_transform_overrides(self) -> dict[str, np.ndarray]:
        if self.case is None:
            return {}
        if normalize_coordinate_profile(self.case.coordinate_profile) not in LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES:
            return {}
        metadata = case_cubemap_view_metadata(self.case)
        if metadata is None or not self._raw_groups:
            return {}
        source_rotations = _source_equirect_rotations_from_groups(
            self._raw_groups,
            metadata,
        )
        overrides: dict[str, np.ndarray] = {}
        for group in self._raw_groups:
            source_rotation = source_rotations.get(group.name)
            face_rotations = cubemap_image_face_rotations(group, cubemap_view_params=metadata)
            if source_rotation is None or face_rotations is None:
                continue
            position = group.reference_frame.camera_position_sfm
            for face, frame in group.frames_by_face.items():
                face_rotation = _source_raster_face_rotation(face_rotations, face, metadata)
                if face_rotation is None:
                    continue
                transform = np.array(frame.transform_matrix, dtype=np.float64, copy=True)
                # The reconstructed preview samples Cube6 JPG pixels in the
                # source equirect basis, but the saved raster Y axis is the
                # inverse of the pinhole +Y convention used by
                # project_sfm_points(). This override is a raster projection
                # adapter so written pixels land where the viewer displays
                # them; the synthetic compositor preserves marker chirality
                # when this adapter produces negative polygon winding.
                transform[:3, :3] = source_rotation @ face_rotation
                transform[:3, 3] = position
                overrides[frame.file_path] = transform
        return overrides

    def _synthetic_tag_candidates(self) -> tuple[tuple[SyntheticTagFrameCandidate, ...], int]:
        if self.case is None:
            return (), 0
        center_sfm, normal_sfm, up_sfm, true_scale = self._synthetic_tag_placement_sfm()
        metadata = case_cubemap_view_metadata(self.case)
        frames = load_pinhole_frames(
            self.case.transforms_for_processing(),
            cubemap_view_params=metadata,
        )
        frames = _apply_frame_transform_overrides(frames, self._synthetic_frame_transform_overrides())
        corners = tag_corners_sfm(
            center_sfm,
            normal_sfm,
            up_sfm,
            float(self._tag_physical_size_m),
            true_scale,
        )
        max_distance = float(self.validation_distance_spin.value()) if hasattr(self, "validation_distance_spin") else 0.0
        max_angle_deg = float(self.validation_angle_spin.value()) if hasattr(self, "validation_angle_spin") else 180.0
        min_area = float(self.validation_min_area_spin.value()) if hasattr(self, "validation_min_area_spin") else 0.0
        min_cos = float(np.cos(np.deg2rad(max(0.0, min(180.0, max_angle_deg)))))
        candidates: list[SyntheticTagFrameCandidate] = []
        for frame in frames:
            camera_delta = np.asarray(frame.camera_position_sfm, dtype=np.float64) - center_sfm
            distance = float(np.linalg.norm(camera_delta))
            if distance <= 1e-12 or not np.isfinite(distance):
                continue
            if max_distance > 0.0 and distance > max_distance:
                continue
            view_dot = float(np.clip(normal_sfm @ (camera_delta / distance), -1.0, 1.0))
            if view_dot < min_cos:
                continue
            projected = project_sfm_points(frame, corners)
            if projected is None or not points_intersect_image(projected, frame.width, frame.height):
                continue
            area = _polygon_area_px(projected)
            if area < min_area:
                continue
            candidates.append(
                SyntheticTagFrameCandidate(
                    frame=frame,
                    projected_points=projected,
                    distance_sfm=distance,
                    view_angle_deg=float(np.rad2deg(np.arccos(view_dot))),
                    area_px=area,
                )
            )
        return tuple(candidates), len(frames)

    def _next_validation_run_dir(self) -> Path:
        if self.case is None:
            raise ValueError("ケースが読み込まれていません")
        self.case.runs_dir.mkdir(parents=True, exist_ok=True)
        stem = "viewer_synthetic_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.case.runs_dir / stem
        if not candidate.exists():
            return candidate
        for index in range(2, 1000):
            numbered = self.case.runs_dir / f"{stem}_{index:03d}"
            if not numbered.exists():
                return numbered
        raise RuntimeError(f"Could not create a unique validation run directory under {self.case.runs_dir}")

    def _candidate_report(self, candidates: tuple[SyntheticTagFrameCandidate, ...]) -> list[dict]:
        return [
            {
                "frame_id": candidate.frame.frame_id,
                "file_path": candidate.frame.file_path,
                "distance_sfm": candidate.distance_sfm,
                "view_angle_deg": candidate.view_angle_deg,
                "area_px": candidate.area_px,
                "projected_points": candidate.projected_points.tolist(),
            }
            for candidate in candidates
        ]

    def _set_validation_status(self, text: str, *, log: bool = False) -> None:
        self.validation_status_label.setText(text)
        if log:
            self._append_log(f"[validation] {text}")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _set_validation_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.tag_x_spin,
            self.tag_y_spin,
            self.tag_z_spin,
            self.tag_yaw_spin,
            self.tag_pitch_spin,
            self.tag_roll_spin,
            self.tag_size_sfm_spin,
            self.validation_distance_spin,
            self.validation_angle_spin,
            self.validation_min_area_spin,
            self.tag_physical_size_spin,
            self.reset_tag_button,
            self.run_validation_button,
            self.copy_validation_scale_button,
        ):
            widget.setEnabled(bool(enabled))
        if enabled and not self._last_validation_scale_text:
            self.copy_validation_scale_button.setEnabled(False)

    def _finish_validation_run(self) -> None:
        self._validation_running = False
        self._set_validation_controls_enabled(True)
        self.run_validation_button.setText("検出")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def run_synthetic_scale_validation(self) -> None:
        if self.case is None:
            QMessageBox.warning(self, "AprilTag検証", "ケースを読み込んでください。")
            return
        if self._validation_running:
            self._set_validation_status("実行中: 前回の合成検出がまだ完了していません。")
            return
        case = self.case
        self._validation_running = True
        self._last_validation_scale_text = ""
        self.copy_validation_scale_button.setEnabled(False)
        self._set_validation_controls_enabled(False)
        self.run_validation_button.setText("実行中...")
        try:
            self._set_validation_status("実行中 1/6: 候補フレームを選別中...", log=True)
            center_sfm, normal_sfm, up_sfm, true_scale = self._synthetic_tag_placement_sfm()
            candidates, total_frames = self._synthetic_tag_candidates()
            if not candidates:
                self._set_validation_status("合成候補なし")
                self._append_log("[validation] 合成候補がありません。タグ表面、距離、投影面積を確認してください。")
                self._finish_validation_run()
                self.validation_finished.emit()
                return

            run_dir = self._next_validation_run_dir()
            selected_paths = frozenset(candidate.frame.file_path for candidate in candidates)
            expected_scale = float(self._tag_physical_size_m) / max(float(self._tag_size_sfm), 1e-12)
            metadata = case_cubemap_view_metadata(case)
            frame_transform_overrides = self._synthetic_frame_transform_overrides()
            request = SyntheticScaleValidationRequest(
                case_dir=case.case_dir,
                transforms_json=case.transforms_for_processing(),
                run_dir=run_dir,
                cubemap_view_metadata=metadata,
                frame_transform_overrides=frame_transform_overrides,
                tag_family=case.tag_family,
                tag_id=int(case.tag_id),
                tag_size_m=float(self._tag_physical_size_m),
                true_scale=true_scale,
                expected_scale=expected_scale,
                tag_center_sfm=center_sfm,
                tag_normal_sfm=normal_sfm,
                tag_up_sfm=up_sfm,
                selected_paths=selected_paths,
                candidate_count=len(candidates),
                total_frame_count=total_frames,
                candidate_report=self._candidate_report(candidates),
                tag_display={
                    "center": self._tag_center.tolist(),
                    "yaw_deg": self._tag_yaw_deg,
                    "pitch_deg": self._tag_pitch_deg,
                    "roll_deg": self._tag_roll_deg,
                    "size_sfm": self._tag_size_sfm,
                    "physical_size_m": self._tag_physical_size_m,
                },
                candidate_filters={
                    "max_distance_sfm": float(self.validation_distance_spin.value()),
                    "max_view_angle_deg": float(self.validation_angle_spin.value()),
                    "min_area_px": float(self.validation_min_area_spin.value()),
                },
            )
            self._start_validation_worker(request)
        except Exception as e:
            self._set_validation_status(f"検証エラー: {e}")
            QMessageBox.critical(self, "AprilTag検証エラー", str(e))
            self._finish_validation_run()
            self.validation_finished.emit()

    def _start_validation_worker(self, request: SyntheticScaleValidationRequest) -> None:
        thread = QThread(self)
        worker = SyntheticScaleValidationWorker(request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_validation_worker_progress)
        worker.finished.connect(self._on_validation_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_validation_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._validation_thread = thread
        self._validation_worker = worker
        thread.start()

    def _on_validation_worker_progress(self, text: str, log: bool) -> None:
        self._set_validation_status(str(text), log=bool(log))

    def _on_validation_worker_finished(self, payload: object, error: str) -> None:
        if error:
            self._set_validation_status(f"検証エラー: {error}")
            QMessageBox.critical(self, "AprilTag検証エラー", error)
        elif isinstance(payload, dict):
            status = str(payload.get("status", "検証完了"))
            run_dir = str(payload.get("run_dir", ""))
            self._set_validation_status(status)
            self._append_log(f"[validation] {status}")
            result = payload.get("result")
            estimate = result.get("estimate") if isinstance(result, dict) else None
            if isinstance(estimate, dict) and "scale" in estimate:
                self._last_validation_scale_text = f"{float(estimate['scale']):.12g}"
                self.copy_validation_scale_button.setEnabled(True)
            else:
                self._last_validation_scale_text = ""
                self.copy_validation_scale_button.setEnabled(False)
            if run_dir:
                self._append_log(f"[validation] output={run_dir}")
        else:
            self._set_validation_status("検証エラー: worker result is invalid")
            QMessageBox.critical(self, "AprilTag検証エラー", "worker result is invalid")

    def _on_validation_thread_finished(self) -> None:
        self._validation_thread = None
        self._validation_worker = None
        self._finish_validation_run()
        self.validation_finished.emit()

    def _copy_validation_scale(self) -> None:
        if not self._last_validation_scale_text:
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(self._last_validation_scale_text)
        self._set_validation_status(f"scale={self._last_validation_scale_text} をコピーしました")

    def selected_face_basis_group(self) -> CubemapFrameGroup | None:
        if self._ray_basis_mode() == RAY_BASIS_IMAGE:
            return self.selected_image_ray_group() or self.selected_world_group()
        return self.selected_world_group()

    def _source_equirect_for_group(self, group: CubemapFrameGroup | None) -> tuple[Path, np.ndarray] | None:
        if group is None or self.case is None:
            return None
        if normalize_coordinate_profile(self.case.coordinate_profile) not in LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES:
            return None
        path = self._source_equirect_paths.get(group.name)
        rotation = self._source_equirect_rotations.get(group.name)
        if path is None or rotation is None:
            return None
        return path, rotation

    def _source_equirect_size_for_group(self, group: CubemapFrameGroup | None) -> tuple[int, int] | None:
        if group is not None:
            path = self._source_equirect_paths.get(group.name)
            if path is not None:
                size = self._image_size_cache.get(path)
                if size is None:
                    size = _image_size(path)
                    if size is not None:
                        self._image_size_cache[path] = size
                if size is not None:
                    return size
        return self._source_equirect_metadata_size

    def _source_equirect_rotation_for_group(self, group: CubemapFrameGroup | None) -> np.ndarray | None:
        if group is None or self.case is None:
            return None
        if normalize_coordinate_profile(self.case.coordinate_profile) not in LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES:
            return None
        return self._source_equirect_rotations.get(group.name)

    def _right_image_equirect_size(
        self,
        *,
        world_group: CubemapFrameGroup | None,
        cubemap_group: CubemapFrameGroup | None,
        use_source_equirect: bool,
        source_image: np.ndarray | None = None,
    ) -> tuple[int, int]:
        source_size = _image_array_size(source_image) or self._source_equirect_size_for_group(world_group)
        if use_source_equirect:
            return source_size or DEFAULT_EQUIRECT_PREVIEW_SIZE
        return _clamp_equirect_preview_size(source_size, _cubemap_equirect_preview_size(cubemap_group))

    def _choose_case(self) -> None:
        start = str(self._case_dialog_start_dir())
        chosen = QFileDialog.getExistingDirectory(self, "シーンまたはAprilTagケースを選択", start)
        if chosen:
            self.load_case_dir(Path(chosen))

    def _case_dialog_start_dir(self) -> Path:
        if self.case is None:
            return Path.home()
        if self.case.validation_runs_dir is not None:
            scene_dir = self.case.validation_runs_dir.parent.parent
            return scene_dir.parent if scene_dir.parent != scene_dir else scene_dir
        case_parent = self.case.case_dir.parent
        return case_parent if case_parent != self.case.case_dir else self.case.case_dir

    def _set_profile_combo(self, profile: str | None) -> None:
        normalized = normalize_coordinate_profile(profile)
        index = self.profile_combo.findData(normalized)
        if index < 0:
            index = self.profile_combo.findData(DEFAULT_COORDINATE_PROFILE)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(max(0, index))
        self.profile_combo.blockSignals(False)

    def _on_profile_changed(self, _index: int) -> None:
        if self.case is None:
            return
        profile = normalize_coordinate_profile(self.profile_combo.currentData())
        self.case = replace(self.case, coordinate_profile=profile)
        try:
            save_case(self.case)
        except Exception as e:
            self._append_log(f"Case profile save failed: {e}")
        self.reload()

    def _on_camera_changed(self, _index: int) -> None:
        self._set_params_from_active_face()
        self._sync_views()

    def _on_ray_basis_changed(self, _index: int) -> None:
        self._set_params_from_active_face()
        self._sync_views()

    def _ray_basis_mode(self) -> str:
        data = str(self.ray_basis_combo.currentData() or RAY_BASIS_BOTH)
        if data in {RAY_BASIS_WORLD, RAY_BASIS_IMAGE, RAY_BASIS_BOTH}:
            return data
        return RAY_BASIS_BOTH

    def _ray_basis_label(self) -> str:
        mode = self._ray_basis_mode()
        if mode == RAY_BASIS_IMAGE:
            return "Cube6 image ray"
        if mode == RAY_BASIS_WORLD:
            return "transforms.json face +Z"
        return "both (active=transforms.json face +Z)"

    def _step_camera(self, delta: int) -> None:
        count = self.camera_combo.count()
        if count <= 0:
            return
        self.camera_combo.setCurrentIndex((self.camera_combo.currentIndex() + int(delta)) % count)

    def _populate_camera_combo(self) -> None:
        current = self.camera_combo.currentData()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for group in self._world_groups:
            self.camera_combo.addItem(f"{group.name} ({len(group.frames)} faces)", group.name)
        if current:
            index = self.camera_combo.findData(current)
            if index >= 0:
                self.camera_combo.setCurrentIndex(index)
        self.camera_combo.blockSignals(False)
        if self.camera_combo.currentIndex() < 0 and self.camera_combo.count():
            self.camera_combo.setCurrentIndex(0)
        self._set_params_from_active_face()

    def _update_world_matrix(self) -> None:
        self._world_matrix = None
        self._metashape_alignment = None
        if self.case is None:
            return
        profile_matrix = world_display_matrix(self.case.coordinate_profile)
        if profile_matrix is not None:
            self._world_matrix = profile_matrix
            return
        estimated = estimate_world_display_matrix_from_metashape(
            self._raw_groups,
            self.case.source_metashape_xml,
        )
        if estimated is not None:
            matrix, rmse, count = estimated
            self._world_matrix = matrix
            self._metashape_alignment = (rmse, count)
            return
        self._world_matrix = None

    def _load_pointcloud(self) -> None:
        if self.case is None:
            return
        pointcloud_path = self.case.input_dir / "pointcloud.ply"
        if not pointcloud_path.is_file():
            pointcloud_path = self.case.source_pointcloud or pointcloud_path
        if pointcloud_path is None or not pointcloud_path.is_file():
            self._world_pointcloud = None
            return
        matrix = compose_display_matrices(
            self._world_matrix,
            pointcloud_display_matrix(self.case.coordinate_profile),
        )
        sample = load_point_cloud_sample(pointcloud_path, max_points=None)
        self._world_pointcloud = transform_point_cloud_sample(sample, matrix)

    def _set_params_from_active_face(self) -> None:
        group = self.selected_face_basis_group()
        if group is None:
            return
        params = axis_face_view_params(group, self._active_face, fov_deg=90.0)
        if params is None:
            fallback = next((face for face in FACE_ORDER if face in group.frames_by_face), None)
            if fallback is None:
                return
            self._active_face = fallback
            params = axis_face_view_params(group, fallback, fov_deg=90.0)
        if params is None:
            return
        yaw, pitch, roll, fov = params
        self._params = PerspectiveParams(
            yaw_deg=normalize_yaw_deg(yaw),
            pitch_deg=clamp_pitch_deg(pitch),
            roll_deg=roll,
            fov_deg=float(fov),
        )

    def _sync_views(self) -> None:
        group = self.selected_world_group()
        basis_group = self.selected_face_basis_group()
        selected_name = group.name if group is not None else ""
        ray_mode = self._ray_basis_mode()
        world_preview_params = self._world_view_preview_params(group)
        for view in (self.world_view, self.point_view):
            view.set_groups(self._world_groups)
            view.set_image_ray_groups(self._image_ray_groups)
            view.set_face_ray_mode(ray_mode)
            view.set_selected_group(selected_name)
            view.set_pointcloud(self._world_pointcloud)
        self.world_view.set_preview_params(
            yaw_deg=world_preview_params.yaw_deg,
            pitch_deg=world_preview_params.pitch_deg,
            roll_deg=world_preview_params.roll_deg,
            fov_deg=world_preview_params.fov_deg,
        )
        self.point_view.set_preview_params(
            yaw_deg=self._params.yaw_deg,
            pitch_deg=self._params.pitch_deg,
            roll_deg=self._params.roll_deg,
            fov_deg=self._params.fov_deg,
        )
        self._sync_tag_views()
        if basis_group is not None:
            self._sync_point_view(basis_group)
        self._sync_mode_visibility()
        self._sync_face_buttons()
        self._update_status()

    def _world_view_preview_params(self, world_group: CubemapFrameGroup | None) -> PerspectiveParams:
        _ = world_group
        return self._params

    def _right_image_projection_modes(self, mode: str, world_group: CubemapFrameGroup | None) -> tuple[bool, bool]:
        source_equirect = self._source_equirect_for_group(world_group)
        source_rotation = self._source_equirect_rotation_for_group(world_group)
        use_source_equirect = (
            mode in {RIGHT_VIEW_SOURCE_EQUIRECT, RIGHT_VIEW_IMAGE_POINTCLOUD}
            and source_equirect is not None
        )
        use_reconstructed_cube6 = mode == RIGHT_VIEW_RECONSTRUCTED_CUBE6 or (
            mode == RIGHT_VIEW_IMAGE_POINTCLOUD and source_equirect is None and source_rotation is not None
        )
        return use_source_equirect, use_reconstructed_cube6

    def _source_projection_anchor_params(self, world_group: CubemapFrameGroup) -> PerspectiveParams | None:
        basis_group = self.selected_face_basis_group() or world_group
        params = axis_face_view_params(basis_group, self._active_face, fov_deg=self._params.fov_deg)
        if params is None:
            return None
        yaw, pitch, roll, fov = params
        return PerspectiveParams(
            yaw_deg=normalize_yaw_deg(yaw),
            pitch_deg=clamp_pitch_deg(pitch),
            roll_deg=roll,
            fov_deg=float(fov),
        )

    def _source_projection_preview_params(self, world_group: CubemapFrameGroup) -> PerspectiveParams:
        return _source_equirect_preview_params(
            self._params,
            self._active_face,
            self._ray_basis_mode(),
            self._source_projection_anchor_params(world_group),
        )

    def _sync_point_view(self, group: CubemapFrameGroup) -> None:
        camera, right, up, forward = camera_pose_from_perspective_params(group, self._params)
        self.point_view.set_fixed_perspective_view(
            camera_position=camera,
            right=right,
            up=up,
            forward=forward,
            fov_deg=self._params.fov_deg,
        )
        self.point_view.set_preview_to_world_matrix(rotation_from_perspective_params(self._params).T)

    def _sync_mode_visibility(self) -> None:
        mode = str(self.mode_combo.currentData() or RIGHT_VIEW_POINTCLOUD)
        if mode in RIGHT_VIEW_IMAGE_MODES:
            self.right_stack.setCurrentWidget(self.image_view)
            self._render_image_view()
        else:
            self.right_stack.setCurrentWidget(self.point_view)

    def _render_image_view(self) -> None:
        mode = str(self.mode_combo.currentData() or RIGHT_VIEW_POINTCLOUD)
        world_group = self.selected_world_group()
        raw_group = self.selected_raw_group()
        image_group = self.selected_image_render_group()
        if world_group is None or raw_group is None or image_group is None:
            self.image_view.setText("Cubemap画像グループがありません")
            return
        source_equirect = self._source_equirect_for_group(world_group)
        source_rotation_for_group = self._source_equirect_rotation_for_group(world_group)
        use_pointcloud_overlay = mode == RIGHT_VIEW_IMAGE_POINTCLOUD
        use_source_equirect, use_reconstructed_cube6 = self._right_image_projection_modes(mode, world_group)
        screen_x_sign = right_image_screen_x_sign(
            use_source_equirect=use_source_equirect,
            use_reconstructed_cube6=use_reconstructed_cube6,
        )
        equirect_group = raw_group if use_reconstructed_cube6 else image_group
        equirect_width, equirect_height = self._right_image_equirect_size(
            world_group=world_group,
            cubemap_group=equirect_group,
            use_source_equirect=use_source_equirect,
        )
        source_path = source_equirect[0] if source_equirect is not None else None
        source_rotation = (
            source_equirect[1]
            if use_source_equirect
            else source_rotation_for_group
        )
        render_source_rotation = (
            _source_preview_render_rotation(source_rotation, self.case.coordinate_profile if self.case else None)
            if source_rotation is not None
            else None
        )
        view_params = (
            self._source_projection_preview_params(world_group)
            if use_source_equirect or use_reconstructed_cube6
            else self._params
        )
        image_source = (
            "source-equirect"
            if use_source_equirect
            else "cube6-reconstruct"
            if use_reconstructed_cube6
            else "cube6-axis"
        )
        key = (
            f"{self.case.coordinate_profile if self.case else ''}:"
            f"{image_source}:"
            f"{self._ray_basis_mode()}:{image_group.name}:"
            f"{source_path if use_source_equirect else ''}:"
            f"{equirect_width}x{equirect_height}"
        )
        overlays = []
        if use_pointcloud_overlay:
            overlays.extend(
                self._pointcloud_image_overlays(
                    world_group,
                    view_params,
                    output_size=768,
                    image_view_basis=True,
                )
            )
        overlays.extend(
            self._right_image_tag_overlays(
                world_group,
                use_output_projection=use_source_equirect or use_reconstructed_cube6,
                projection_params=view_params,
                output_size=768,
                image_view_basis=True,
            )
        )
        if self._displayed_image_key == key and self.image_view.set_perspective_params(view_params):
            self.image_view.set_perspective_screen_x_sign(screen_x_sign)
            self.image_view.set_drag_mode("look")
            self.image_view.set_perspective_label_overlays(overlays)
            return
        image = self._equirect_cache.get(key)
        if image is None:
            try:
                if use_source_equirect:
                    assert source_path is not None
                    assert render_source_rotation is not None
                    source = self._image_cache.get(source_path)
                    if source is None:
                        source = imread_unicode(source_path)
                        if source is not None:
                            self._image_cache[source_path] = source
                    if source is None:
                        raise OSError(f"Cannot read source equirect image: {source_path}")
                    equirect_width, equirect_height = self._right_image_equirect_size(
                        world_group=world_group,
                        cubemap_group=equirect_group,
                        use_source_equirect=True,
                        source_image=source,
                    )
                    image = render_source_equirect_axis(
                        source,
                        render_source_rotation,
                        output_width=equirect_width,
                        output_height=equirect_height,
                        sfm_to_preview_matrix=self._world_matrix,
                    )
                elif use_reconstructed_cube6:
                    if render_source_rotation is None:
                        raise ValueError("Cube6再構築には source camera rotation が必要です")
                    image = render_generated_cubemap_source_axis(
                        raw_group,
                        render_source_rotation,
                        cubemap_view_params=case_cubemap_view_metadata(self.case) if self.case else None,
                        output_width=equirect_width,
                        output_height=equirect_height,
                        image_cache=self._image_cache,
                        sfm_to_preview_matrix=self._world_matrix,
                    )
                else:
                    image = render_cubemap_axis_equirect(
                        image_group,
                        output_width=equirect_width,
                        output_height=equirect_height,
                        image_cache=self._image_cache,
                    )
            except Exception as e:
                self.image_view.setText(f"Cubemap画像生成エラー: {e}")
                return
            self._equirect_cache[key] = image
        shown = self.image_view.set_perspective_image_bgr(
            image,
            view_params,
            overlays=overlays,
            logical_size=QSize(768, 768),
            screen_x_sign=screen_x_sign,
        )
        self.image_view.set_drag_mode("look")
        self._displayed_image_key = key if shown else ""
        if not shown:
            self.image_view.setText("GPU透視投影プレビューを初期化できませんでした")

    def _sync_face_buttons(self) -> None:
        group = self.selected_face_basis_group() or self.selected_world_group()
        faces = set(group.frames_by_face) if group is not None else set()
        for face, button in self.face_buttons.items():
            button.blockSignals(True)
            button.setEnabled(face in faces)
            button.setChecked(face == self._active_face)
            button.blockSignals(False)

    def _update_status(self) -> None:
        if self.case is None:
            self.case_label.setText("シーン未選択")
            self._last_status_detail = ""
            return
        mode = str(self.mode_combo.currentData() or RIGHT_VIEW_POINTCLOUD)
        group = self.selected_world_group()
        basis_group = self.selected_face_basis_group()
        image_group = self.selected_image_render_group()
        point_count = 0 if self._world_pointcloud is None else len(self._world_pointcloud.points)
        source_point_count = 0 if self._world_pointcloud is None else int(self._world_pointcloud.source_count)
        point_text = (
            str(point_count)
            if source_point_count <= 0 or source_point_count == point_count
            else f"{point_count}/{source_point_count}"
        )
        alignment = ""
        if self._metashape_alignment is not None:
            rmse, count = self._metashape_alignment
            alignment = f" / Metashape XML alignment count={count}, rmse={rmse:.3g}"
        ray_source = (
            f" / world rays={self._world_ray_source or 'transforms.json face +Z'}"
            f" / image rays={self._image_ray_source or 'transforms.json'}"
            f" / active basis={self._ray_basis_label()}"
        )
        group_text = "-" if group is None else group.name
        source_image_text = ""
        source_equirect = self._source_equirect_for_group(group)
        if source_equirect is not None:
            source_image_text = f" / source equirect={source_equirect[0].name}"
        preview_size_text = ""
        if mode in RIGHT_VIEW_IMAGE_MODES and group is not None:
            raw_group = self.selected_raw_group()
            use_source = mode in {RIGHT_VIEW_SOURCE_EQUIRECT, RIGHT_VIEW_IMAGE_POINTCLOUD} and source_equirect is not None
            use_reconstructed = mode == RIGHT_VIEW_RECONSTRUCTED_CUBE6 or (
                mode == RIGHT_VIEW_IMAGE_POINTCLOUD
                and source_equirect is None
                and self._source_equirect_rotation_for_group(group) is not None
            )
            equirect_group = raw_group if use_reconstructed else image_group
            width, height = self._right_image_equirect_size(
                world_group=group,
                cubemap_group=equirect_group,
                use_source_equirect=use_source,
            )
            preview_size_text = f" / preview equirect={width}x{height}"
        mapping = ""
        if mode == RIGHT_VIEW_IMAGE_POINTCLOUD and group is not None:
            if source_equirect is not None:
                mapping = " / image preview=source equirect + pointcloud"
            elif self._source_equirect_rotation_for_group(group) is not None:
                mapping = " / image preview=Cube6 reconstructed + pointcloud"
            else:
                mapping = " / image preview=Cube6 axis + pointcloud"
        elif mode == RIGHT_VIEW_RECONSTRUCTED_CUBE6 and group is not None:
            if self._source_equirect_rotation_for_group(group) is not None:
                mapping = " / image preview=Cube6 reconstructed"
        elif mode == RIGHT_VIEW_SOURCE_EQUIRECT and source_equirect is not None:
            mapping = " / image preview=source equirect direct"
        if basis_group is not None and image_group is not None:
            closest = closest_image_face_for_world_face(
                basis_group,
                image_group,
                self._active_face,
            )
            if closest is not None:
                image_face, angle, same_label_angle = closest
                opposite = opposite_image_face_for_world_face(
                    basis_group,
                    image_group,
                    self._active_face,
                )
                opposite_text = "" if opposite is None else f", reverse={opposite[0]} {opposite[1]:.1f}deg"
                same = "" if not np.isfinite(same_label_angle) else f", same-label={same_label_angle:.1f}deg"
                mapping += f" / active {self._active_face}->image {image_face} ({angle:.1f}deg{same}{opposite_text})"
        display_path = self.case.image_root if self.case.input_mode == "scene" and self.case.image_root else self.case.case_dir
        self.case_label.setText(str(display_path))
        self._last_status_detail = (
            f"カメラ: {group_text} / "
            f"点群: {point_text} / 点群描画: {self.world_view.pointcloud_renderer_label()} / "
            f"座標: {coordinate_profile_label(self.case.coordinate_profile)}"
            f"{alignment}{ray_source}{source_image_text}{mapping}{preview_size_text}"
        )
        self._append_log_once(
            "scene",
            f"Loaded {len(self._world_groups)} camera groups, points={point_text}. "
            f" World rays: {self._world_ray_source or 'transforms.json face +Z'}."
            f" Image rays: {self._image_ray_source or 'transforms.json'}."
            f" Source equirect images: {len(self._source_equirect_paths)}.",
        )
        if group is not None and image_group is not None and source_equirect is None:
            self._append_log_once(
                f"mapping:{group.name}",
                f"Face/image ray mapping for {group.name}: {face_mapping_summary(group, image_group)}",
            )

    def _on_right_view_dragged(self, delta_x: float, delta_y: float) -> None:
        if not self._world_groups:
            return
        screen_basis = self._right_view_drag_screen_basis()
        screen_right: np.ndarray | None = None
        screen_up: np.ndarray | None = None
        if screen_basis is not None:
            screen_right, screen_up = screen_basis
        self._params = _params_from_grab_drag(
            self._params,
            delta_x,
            delta_y,
            source_preview_screen_axis_adapter=self._right_view_uses_source_preview_screen_axis_adapter(),
            screen_right=screen_right,
            screen_up=screen_up,
        )
        self._sync_views()

    def _right_view_uses_source_preview_screen_axis_adapter(self) -> bool:
        if self.right_stack.currentWidget() is not self.image_view:
            return False
        mode = str(self.mode_combo.currentData() or RIGHT_VIEW_POINTCLOUD)
        if mode == RIGHT_VIEW_IMAGE_POINTCLOUD:
            source_preview = (
                self._source_equirect_for_group(self.selected_world_group()) is not None
                or self._source_equirect_rotation_for_group(self.selected_world_group()) is not None
            )
        elif mode == RIGHT_VIEW_RECONSTRUCTED_CUBE6:
            source_preview = self._source_equirect_rotation_for_group(self.selected_world_group()) is not None
        elif mode == RIGHT_VIEW_SOURCE_EQUIRECT:
            source_preview = self._source_equirect_for_group(self.selected_world_group()) is not None
        else:
            source_preview = False
        return bool(source_preview and _uses_source_preview_screen_axis_adapter(self._active_face, self._ray_basis_mode()))

    def _right_view_drag_screen_basis(self) -> tuple[np.ndarray, np.ndarray] | None:
        group = self.selected_world_group()
        if group is None:
            return None
        _camera, right, up, forward = camera_pose_from_perspective_params(group, self._params)
        if self.right_stack.currentWidget() is self.point_view:
            screen_right, screen_up, _forward = _right_view_screen_basis(
                right=right,
                up=up,
                forward=forward,
            )
            return screen_right, screen_up
        if self.right_stack.currentWidget() is self.image_view:
            mode = str(self.mode_combo.currentData() or RIGHT_VIEW_POINTCLOUD)
            use_source_equirect, use_reconstructed_cube6 = self._right_image_projection_modes(mode, group)
            screen_x_sign = right_image_screen_x_sign(
                use_source_equirect=use_source_equirect,
                use_reconstructed_cube6=use_reconstructed_cube6,
            )
            return (
                _normalized_vector(right * screen_x_sign, (1.0, 0.0, 0.0)),
                _normalized_vector(up, (0.0, 1.0, 0.0)),
            )
        return None

    def _append_log(self, text: str) -> None:
        self.log.append(text)

    def _append_log_once(self, key: str, text: str) -> None:
        marker = f"[{key}] "
        current = self.log.toPlainText()
        if marker in current:
            return
        self._append_log(marker + text)
