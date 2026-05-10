"""Scene viewer for Cube6 AprilTag development datasets."""

from __future__ import annotations

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

from core.apriltag_cubemap import CubemapViewMetadata, cubemap_view_params_for_group, discover_cubemap_view_metadata
from core.apriltag_geometry import PinholeFrame
from devtools.apriltag.case import DEFAULT_CASE_ROOT, AprilTagDevCase, load_case, save_case
from devtools.apriltag.coordinates import (
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
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    render_cubemap_axis_equirect,
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


def _transform_points(points: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if matrix is None:
        return values.copy()
    transform = np.asarray(matrix, dtype=np.float64)
    return values @ transform[:3, :3].T + transform[:3, 3]


def _transform_frame_for_world_display(frame: PinholeFrame, matrix: np.ndarray | None) -> PinholeFrame:
    if matrix is None:
        return frame
    transform = np.asarray(matrix, dtype=np.float64)
    output = np.array(frame.transform_matrix, dtype=np.float64, copy=True)
    output[:3, :3] = transform[:3, :3] @ frame.camera_to_world_rotation
    output[:3, 3] = _transform_points(frame.camera_position_sfm.reshape(1, 3), matrix)[0]
    return replace(frame, transform_matrix=output)


def transform_group_for_world_display(group: CubemapFrameGroup, matrix: np.ndarray | None) -> CubemapFrameGroup:
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


def _cubemap_image_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
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


def metashape_image_ray_groups(
    groups: tuple[CubemapFrameGroup, ...],
    xml_path: Path | None,
    metadata: CubemapViewMetadata | None,
) -> tuple[CubemapFrameGroup, ...] | None:
    """Build face poses from Metashape camera orientation and Cube6 image views.

    ``transforms.json`` stores face poses in the downstream training coordinate
    convention. For the debug viewer, the face rays must instead describe the
    actual pinhole images generated by Step 4. The source Metashape camera pose
    plus the recorded view yaw/pitch gives that image-ray pose directly.
    """
    if xml_path is None or metadata is None:
        return None
    metashape_transforms = _load_metashape_camera_transforms(xml_path)
    if not metashape_transforms:
        return None
    output: list[CubemapFrameGroup] = []
    for group in groups:
        camera_transform = metashape_transforms.get(group.name)
        view_params = cubemap_view_params_for_group(metadata, group.group_index)
        if camera_transform is None or view_params is None:
            return None
        frames_by_face: dict[str, PinholeFrame] = {}
        for face, frame in group.frames_by_face.items():
            params = view_params.get(face)
            if params is None:
                continue
            transform = np.array(frame.transform_matrix, dtype=np.float64, copy=True)
            transform[:3, :3] = camera_transform[:3, :3] @ _cubemap_image_rotation(*params)
            transform[:3, 3] = camera_transform[:3, 3]
            frames_by_face[face] = replace(frame, transform_matrix=transform)
        if len(frames_by_face) < 4:
            return None
        output.append(CubemapFrameGroup(group.name, frames_by_face, group_index=group.group_index))
    return tuple(output)


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
        self._world_pointcloud: PointCloudSample | None = None
        self._world_matrix: np.ndarray | None = None
        self._metashape_alignment: tuple[float, int] | None = None
        self._using_metashape_image_rays = False
        self._image_cache: dict[Path, np.ndarray] = {}
        self._equirect_cache: dict[str, np.ndarray] = {}
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
        try:
            metadata = case_cubemap_view_metadata(self.case)
            groups = load_cubemap_frame_groups(
                self.case.transforms_for_processing(),
                cubemap_view_params=metadata,
            )
            labels = load_metashape_camera_labels(self.case.source_metashape_xml) if self.case.source_metashape_xml else ()
            self._raw_groups = order_groups_by_labels(groups, labels)
            self._update_world_matrix()
            metashape_groups = metashape_image_ray_groups(
                self._raw_groups,
                self.case.source_metashape_xml,
                metadata,
            )
            if metashape_groups is not None:
                self._world_groups = metashape_groups
                self._using_metashape_image_rays = True
            else:
                self._world_groups = tuple(
                    transform_group_for_world_display(group, self._world_matrix)
                    for group in self._raw_groups
                )
                self._using_metashape_image_rays = False
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
        estimated = estimate_world_display_matrix_from_metashape(
            self._raw_groups,
            self.case.source_metashape_xml,
        )
        if estimated is not None:
            matrix, rmse, count = estimated
            self._world_matrix = matrix
            self._metashape_alignment = (rmse, count)
            return
        self._world_matrix = world_display_matrix(self.case.coordinate_profile)

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
        group = self.selected_world_group()
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
        selected_name = group.name if group is not None else ""
        for view in (self.world_view, self.point_view):
            view.set_groups(self._world_groups)
            view.set_selected_group(selected_name)
            view.set_pointcloud(self._world_pointcloud)
            view.set_preview_params(
                yaw_deg=self._params.yaw_deg,
                pitch_deg=self._params.pitch_deg,
                roll_deg=self._params.roll_deg,
                fov_deg=self._params.fov_deg,
            )
        if group is not None:
            self._sync_point_view(group)
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
        group = self.selected_world_group()
        if group is None:
            self.image_view.setText("Cubemap画像グループがありません")
            return
        key = f"{self.case.coordinate_profile if self.case else ''}:{group.name}"
        image = self._equirect_cache.get(key)
        if image is None:
            try:
                image = render_cubemap_axis_equirect(
                    group,
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
        if not shown:
            self.image_view.setText("GPU透視投影プレビューを初期化できませんでした")

    def _sync_face_buttons(self) -> None:
        group = self.selected_world_group()
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
        point_count = 0 if self._world_pointcloud is None else len(self._world_pointcloud.points)
        alignment = ""
        if self._metashape_alignment is not None:
            rmse, count = self._metashape_alignment
            alignment = f" / Metashape XML alignment count={count}, rmse={rmse:.3g}"
        ray_source = " / image rays=Metashape XML" if self._using_metashape_image_rays else " / image rays=transforms.json"
        group_text = "-" if group is None else group.name
        self.case_label.setText(
            f"ケース: {self.case.case_dir} / カメラ: {group_text} / "
            f"点群: {point_count} sampled / 座標: {coordinate_profile_label(self.case.coordinate_profile)}"
            f"{alignment}{ray_source}"
        )
        self._append_log_once(
            "scene",
            f"Loaded {len(self._world_groups)} camera groups, point sample={point_count}. "
            f"{coordinate_profile_note(self.case.coordinate_profile)}"
            f" Image rays: {'Metashape XML + Cube6 view metadata' if self._using_metashape_image_rays else 'transforms.json fallback'}.",
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
