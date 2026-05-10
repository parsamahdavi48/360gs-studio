"""Scene viewer for Cube6 AprilTag development datasets."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.apriltag_cubemap import CubemapViewMetadata, discover_cubemap_view_metadata
from core.apriltag_geometry import PinholeFrame
from core.image_io import imread_unicode
from devtools.apriltag.case import DEFAULT_CASE_ROOT, AprilTagDevCase, load_case, save_case
from devtools.apriltag.coordinates import (
    COORDINATE_PROFILE_LICHTFELD_CUBE6,
    COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY,
    COORDINATE_PROFILES,
    DEFAULT_COORDINATE_PROFILE,
    coordinate_profile_label,
    coordinate_profile_note,
    normalize_coordinate_profile,
    pointcloud_display_matrix,
    world_display_matrix,
)
from devtools.apriltag.cubemap_preview import (
    CubemapFrameGroup,
    axis_face_view_params,
    image_space_cubemap_frame_group,
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    render_cubemap_axis_equirect,
    render_source_equirect_axis,
    source_equirect_base_rotation,
)
from devtools.apriltag.world_debug_view import (
    AprilTagWorldDebugView,
    PointCloudSample,
    load_point_cloud_sample,
    transform_point_cloud_sample,
)
from gui.common.perspective_image_view import PerspectiveImageView
from gui.common.perspective_preview import (
    PerspectiveParams,
    clamp_pitch_deg,
    normalize_yaw_deg,
    params_from_drag,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEWER_CASE = DEFAULT_CASE_ROOT / "current"
FACE_ORDER = ("pz", "px", "nx", "nz", "top", "bottom", "py", "ny")
RAY_BASIS_WORLD = "world"
RAY_BASIS_IMAGE = "image"
RAY_BASIS_BOTH = "both"
LICHTFELD_IMAGE_RAY_DISPLAY_PROFILES = {
    COORDINATE_PROFILE_LICHTFELD_CUBE6,
    COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY,
}
SOURCE_EQUIRECT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")
SOURCE_EQUIRECT_LOCAL_FROM_LICHTFELD_LOCAL = np.diag([1.0, -1.0, -1.0])


def _transform_points(points: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if matrix is None:
        return values.copy()
    transform = np.asarray(matrix, dtype=np.float64)
    return values @ transform[:3, :3].T + transform[:3, 3]


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


def _scene_root_candidates(case: AprilTagDevCase) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for path in (
        case.image_root,
        case.source_transforms.parent,
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
) -> dict[str, np.ndarray]:
    rotations: dict[str, np.ndarray] = {}
    for group in groups:
        rotation = source_equirect_base_rotation(group, cubemap_view_params=metadata)
        if rotation is not None:
            # ``source_equirect_base_rotation`` recovers the source camera basis
            # from final Cube6 poses. Those poses include
            # metashape_360_lfs.py::transform_camera_matrix Step 2, which flips
            # local Y/Z for LichtFeld/OpenGL. The source panorama pixels are
            # sampled in the original equirectangular local basis, so undo that
            # local flip only for image lookup.
            rotations[group.name] = rotation @ SOURCE_EQUIRECT_LOCAL_FROM_LICHTFELD_LOCAL
    return rotations


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
        self._equirect_cache: dict[str, np.ndarray] = {}
        self._source_equirect_paths: dict[str, Path] = {}
        self._source_equirect_rotations: dict[str, np.ndarray] = {}
        self._displayed_image_key = ""
        self._params = PerspectiveParams(fov_deg=90.0)
        self._active_face = "pz"

        self._build_ui()
        self._connect_signals()
        case_dir = initial_case or DEFAULT_VIEWER_CASE
        if case_dir is not None and (case_dir / "case.json").is_file():
            self.load_case_dir(case_dir)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.case_label = QLabel("ケース未選択")
        self.case_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.case_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.open_case_button = QPushButton("ケースを開く")
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
        self.mode_combo.addItem("点群モード", "pointcloud")
        self.mode_combo.addItem("Cubemap画像モード", "image")
        self.ray_basis_combo = QComboBox()
        self.ray_basis_combo.addItem("両方", RAY_BASIS_BOTH)
        self.ray_basis_combo.addItem("JSON Face", RAY_BASIS_WORLD)
        self.ray_basis_combo.addItem("画像レイ", RAY_BASIS_IMAGE)
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

        face_box = QGroupBox("FOV 90° 面へ移動")
        face_layout = QHBoxLayout(face_box)
        face_layout.setContentsMargins(8, 8, 8, 8)
        self.face_buttons: dict[str, QPushButton] = {}
        for face in FACE_ORDER:
            button = QPushButton(face)
            button.setCheckable(True)
            self.face_buttons[face] = button
            face_layout.addWidget(button)
        face_layout.addStretch(1)
        root.addWidget(face_box)

        splitter = QSplitter(Qt.Horizontal)
        self.world_view = AprilTagWorldDebugView()
        self.world_view.setMinimumSize(520, 420)
        self.right_stack = QStackedWidget()
        self.point_view = AprilTagWorldDebugView()
        self.point_view.setMinimumSize(520, 420)
        self.image_view = PerspectiveImageView("Cubemap画像を読み込みます")
        self.image_view.setMinimumSize(520, 420)
        self.image_view.set_drag_mode("look")
        self.right_stack.addWidget(self.point_view)
        self.right_stack.addWidget(self.image_view)
        splitter.addWidget(self.world_view)
        splitter.addWidget(self.right_stack)
        splitter.setSizes([760, 740])
        root.addWidget(splitter, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        root.addWidget(self.log)

    def _connect_signals(self) -> None:
        self.open_case_button.clicked.connect(self._choose_case)
        self.reload_button.clicked.connect(self.reload)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.prev_button.clicked.connect(lambda: self._step_camera(-1))
        self.next_button.clicked.connect(lambda: self._step_camera(1))
        self.mode_combo.currentIndexChanged.connect(lambda _index: self._sync_views())
        self.ray_basis_combo.currentIndexChanged.connect(self._on_ray_basis_changed)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.world_view.camera_clicked.connect(self.select_camera_by_name)
        self.point_view.fixed_view_dragged.connect(self._on_right_view_dragged)
        self.image_view.look_dragged.connect(self._on_right_view_dragged)
        for face, button in self.face_buttons.items():
            button.clicked.connect(lambda _checked=False, value=face: self.set_active_face(value))

    def load_case_dir(self, case_dir: Path) -> None:
        try:
            self.case = load_case(case_dir)
        except Exception as e:
            QMessageBox.critical(self, "ケース読み込みエラー", str(e))
            return
        self._set_profile_combo(self.case.coordinate_profile)
        self.reload()

    def reload(self) -> None:
        if self.case is None:
            return
        self._image_cache.clear()
        self._equirect_cache.clear()
        self._source_equirect_paths = {}
        self._source_equirect_rotations = {}
        self._displayed_image_key = ""
        try:
            metadata = case_cubemap_view_metadata(self.case)
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
            image_ray_display_matrix = image_ray_display_matrix_for_profile(
                self.case.coordinate_profile,
                self._world_matrix,
            )
            self._image_ray_groups = tuple(
                transform_group_for_world_display(
                    image_space_cubemap_frame_group(group, cubemap_view_params=metadata),
                    image_ray_display_matrix,
                    cubemap_view_params=metadata,
                )
                for group in self._raw_groups
            )
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
        self._sync_views()
        self.scene_loaded.emit()

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

    def _choose_case(self) -> None:
        start = str((self.case.case_dir if self.case else DEFAULT_VIEWER_CASE).parent)
        chosen = QFileDialog.getExistingDirectory(self, "AprilTagケースを選択", start)
        if chosen:
            self.load_case_dir(Path(chosen))

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
        sample = load_point_cloud_sample(pointcloud_path)
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
        for view in (self.world_view, self.point_view):
            view.set_groups(self._world_groups)
            view.set_image_ray_groups(self._image_ray_groups)
            view.set_face_ray_mode(ray_mode)
            view.set_selected_group(selected_name)
            view.set_pointcloud(self._world_pointcloud)
            view.set_preview_params(
                yaw_deg=self._params.yaw_deg,
                pitch_deg=self._params.pitch_deg,
                roll_deg=self._params.roll_deg,
                fov_deg=self._params.fov_deg,
            )
        if basis_group is not None:
            self._sync_point_view(basis_group)
        self._sync_mode_visibility()
        self._sync_face_buttons()
        self._update_status()

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
        mode = str(self.mode_combo.currentData() or "pointcloud")
        if mode == "image":
            self.right_stack.setCurrentWidget(self.image_view)
            self._render_image_view()
        else:
            self.right_stack.setCurrentWidget(self.point_view)

    def _render_image_view(self) -> None:
        world_group = self.selected_world_group()
        image_group = self.selected_image_render_group()
        if world_group is None or image_group is None:
            self.image_view.setText("Cubemap画像グループがありません")
            return
        source_equirect = self._source_equirect_for_group(world_group)
        use_source_equirect = source_equirect is not None
        source_path = source_equirect[0] if source_equirect is not None else None
        source_rotation = source_equirect[1] if source_equirect is not None else None
        key = (
            f"{self.case.coordinate_profile if self.case else ''}:"
            f"{'source-equirect' if use_source_equirect else 'cube6-axis'}:"
            f"{self._ray_basis_mode()}:{image_group.name}:"
            f"{source_path if use_source_equirect else ''}"
        )
        if self._displayed_image_key == key and self.image_view.set_perspective_params(self._params):
            self.image_view.set_drag_mode("look")
            return
        image = self._equirect_cache.get(key)
        if image is None:
            try:
                if use_source_equirect:
                    assert source_path is not None
                    assert source_rotation is not None
                    source = self._image_cache.get(source_path)
                    if source is None:
                        source = imread_unicode(source_path)
                        if source is not None:
                            self._image_cache[source_path] = source
                    if source is None:
                        raise OSError(f"Cannot read source equirect image: {source_path}")
                    image = render_source_equirect_axis(
                        source,
                        source_rotation,
                        output_width=2048,
                        output_height=1024,
                        sfm_to_preview_matrix=self._world_matrix,
                    )
                else:
                    image = render_cubemap_axis_equirect(
                        image_group,
                        output_width=2048,
                        output_height=1024,
                        image_cache=self._image_cache,
                    )
            except Exception as e:
                self.image_view.setText(f"Cubemap画像生成エラー: {e}")
                return
            self._equirect_cache[key] = image
        shown = self.image_view.set_perspective_image_bgr(
            image,
            self._params,
            overlays=[],
            logical_size=QSize(768, 768),
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
            self.case_label.setText("ケース未選択")
            return
        group = self.selected_world_group()
        basis_group = self.selected_face_basis_group()
        image_group = self.selected_image_render_group()
        point_count = 0 if self._world_pointcloud is None else len(self._world_pointcloud.points)
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
        mapping = ""
        if source_equirect is not None:
            mapping = " / image preview=source equirect direct"
        elif basis_group is not None and image_group is not None:
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
                mapping = f" / active {self._active_face}->image {image_face} ({angle:.1f}deg{same}{opposite_text})"
        self.case_label.setText(
            f"ケース: {self.case.case_dir} / カメラ: {group_text} / "
            f"点群: {point_count} sampled / 座標: {coordinate_profile_label(self.case.coordinate_profile)}"
            f"{alignment}{ray_source}{source_image_text}{mapping}"
        )
        self._append_log_once(
            "scene",
            f"Loaded {len(self._world_groups)} camera groups, point sample={point_count}. "
            f"{coordinate_profile_note(self.case.coordinate_profile)}"
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
        self._params = params_from_drag(self._params, delta_x, delta_y)
        self._sync_views()

    def _append_log(self, text: str) -> None:
        self.log.append(text)

    def _append_log_once(self, key: str, text: str) -> None:
        marker = f"[{key}] "
        current = self.log.toPlainText()
        if marker in current:
            return
        self._append_log(marker + text)
