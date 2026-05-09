#!/usr/bin/env python3
"""Standalone development GUI for AprilTag synthetic-injection cases."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.apriltag_detection import available_families
from core.apriltag_geometry import PinholeFrame, tag_corners_sfm
from devtools.apriltag.case import (
    DEFAULT_CASE_ROOT,
    AprilTagDevCase,
    AprilTagPlacement,
    create_case,
    load_case,
    run_dir_for_placement,
    save_case,
    save_placement,
)
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
    face_view_params,
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    project_sfm_points_to_axis_preview_points,
    render_cubemap_axis_equirect,
    view_pixel_to_axis_world_ray_and_up,
    virtual_camera_rotation_with_roll,
)
from devtools.apriltag.printable import create_printable_target
from devtools.apriltag.world_debug_view import (
    AprilTagWorldDebugView,
    PointCloudSample,
    load_point_cloud_sample,
    transform_point_cloud_sample,
)
from gui.common.browse_widget import BrowseWidget
from gui.common.perspective_image_view import PerspectiveImageView, PerspectiveLabelOverlay, bgr_to_qimage
from gui.common.perspective_preview import PerspectiveParams, clamp_pitch_deg, normalize_yaw_deg, params_from_drag
from gui.theme import apply_theme


# PerspectiveLabelOverlay stores OpenCV-style BGR values. These match the
# highlighted X/Z grid axes in AprilTagWorldDebugView.
GRID_X_AXIS_BGR = (90, 180, 245)
GRID_Z_AXIS_BGR = (245, 175, 90)
AXIS_GIZMO_X_BGR = (92, 92, 255)
AXIS_GIZMO_Y_BGR = (130, 245, 120)
AXIS_GIZMO_Z_BGR = (255, 170, 96)


def _preview_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
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


def _transform_points_for_world_display(points: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
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
    linear = transform[:3, :3]
    translation = transform[:3, 3]
    return (values - translation) @ np.linalg.inv(linear).T


def _transform_vectors_for_world_display(vectors: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if matrix is None:
        return values.copy()
    transform = np.asarray(matrix, dtype=np.float64)
    return values @ transform[:3, :3].T


def _transform_vectors_from_world_display(vectors: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if matrix is None:
        return values.copy()
    transform = np.asarray(matrix, dtype=np.float64)
    return values @ np.linalg.inv(transform[:3, :3]).T


def _transform_frame_for_world_display(frame: PinholeFrame, matrix: np.ndarray | None) -> PinholeFrame:
    if matrix is None:
        return frame
    transform = np.asarray(matrix, dtype=np.float64)
    output = np.array(frame.transform_matrix, dtype=np.float64, copy=True)
    output[:3, :3] = transform[:3, :3] @ frame.camera_to_world_rotation
    output[:3, 3] = _transform_points_for_world_display(frame.camera_position_sfm.reshape(1, 3), matrix)[0]
    return replace(frame, transform_matrix=output)


def _transform_group_for_world_display(group: CubemapFrameGroup, matrix: np.ndarray | None) -> CubemapFrameGroup:
    if matrix is None:
        return group
    return CubemapFrameGroup(
        name=group.name,
        frames_by_face={
            face: _transform_frame_for_world_display(frame, matrix)
            for face, frame in group.frames_by_face.items()
        },
    )


def _compose_display_matrices(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None:
        return None if second is None else second.copy()
    if second is None:
        return first.copy()
    return first @ second


def _load_metashape_camera_positions(xml_path: Path) -> dict[str, np.ndarray]:
    if not xml_path.is_file():
        return {}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {}
    positions: dict[str, np.ndarray] = {}
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
            positions[Path(label).stem] = transform[:3, 3].copy()
    return positions


def _estimate_world_display_matrix_from_metashape(
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
GRID_LINE_BGR = (130, 130, 130)


class Vec3Editor(QWidget):
    value_changed = Signal()

    def __init__(self, values: tuple[float, float, float], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._spins: list[QDoubleSpinBox] = []
        for value in values:
            spin = QDoubleSpinBox()
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setValue(float(value))
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            spin.valueChanged.connect(lambda _value: self.value_changed.emit())
            self._spins.append(spin)
            layout.addWidget(spin)

    def value(self) -> tuple[float, float, float]:
        return tuple(float(spin.value()) for spin in self._spins)  # type: ignore[return-value]

    def set_value(self, values: tuple[float, float, float]) -> None:
        for spin, value in zip(self._spins, values, strict=True):
            spin.setValue(float(value))


class DevAprilTagPlacerWindow(QWidget):
    def __init__(
        self,
        *,
        initial_case: Path | None = None,
        source_transforms: Path | None = None,
        source_pointcloud: Path | None = None,
        source_metashape_xml: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("AprilTag Synthetic Injection Dev GUI")
        icon_path = REPO_ROOT / "gui" / "assets" / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1280, 900)
        self.case: AprilTagDevCase | None = None
        self._process: QProcess | None = None
        self._queue: list[tuple[str, list[str], Callable[[], None] | None]] = []
        self._last_preview_path: Path | None = None
        self._cubemap_groups: tuple[CubemapFrameGroup, ...] = ()
        self._cubemap_image_cache = {}
        self._equirect_preview_cache = {}
        self._scene_preview_params = PerspectiveParams()
        self._scene_preview_size = 768
        self._last_click_state: tuple[str, np.ndarray, np.ndarray] | None = None
        self._world_pointcloud: PointCloudSample | None = None
        self._metashape_world_display_matrix: np.ndarray | None = None
        self._metashape_world_alignment_rmse: float | None = None

        self._build_ui()
        self._connect_signals()
        if initial_case is not None and (initial_case / "case.json").is_file():
            self._load_case(initial_case)
        elif initial_case is not None:
            self.case_root_browse.set_text(str(initial_case.parent))
            self.case_name_edit.setText(initial_case.name)
        if source_transforms is not None:
            self.transforms_browse.set_text(str(source_transforms))
        if source_pointcloud is not None:
            self.pointcloud_browse.set_text(str(source_pointcloud))
        if source_metashape_xml is not None:
            self.xml_browse.set_text(str(source_metashape_xml))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QLabel("AprilTag 仮想タグ注入デバッグ")
        header.setObjectName("stepHeader")
        root.addWidget(header)

        main = QHBoxLayout()
        main.setSpacing(10)
        root.addLayout(main, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        main.addWidget(left, stretch=0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        main.addWidget(right, stretch=1)

        left_layout.addWidget(self._build_case_group())
        left_layout.addWidget(self._build_tag_group())
        left_layout.addWidget(self._build_placement_group())
        left_layout.addWidget(self._build_action_group())
        left_layout.addStretch(1)

        self.status_label = QLabel("ケース未選択")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        right_layout.addWidget(self._build_scene_preview_group())

        viewport_splitter = QSplitter(Qt.Horizontal)
        right_layout.addWidget(viewport_splitter, stretch=1)

        world_group = QGroupBox("3Dワールド")
        world_layout = QVBoxLayout(world_group)
        world_layout.setContentsMargins(6, 6, 6, 6)
        self.world_debug_view = AprilTagWorldDebugView()
        self.world_debug_view.setToolTip("カメラ点をクリックすると対応する画像セットへ移動します。左ドラッグで回転、右ドラッグで平行移動します。")
        world_layout.addWidget(self.world_debug_view)
        viewport_splitter.addWidget(world_group)

        image_group = QGroupBox("カメラ画像")
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(6, 6, 6, 6)
        self.preview_label = PerspectiveImageView("プレビュー未作成")
        self.preview_label.setMinimumSize(520, 360)
        self.preview_label.setStyleSheet("background-color: #101316; border: 1px solid #3a424d;")
        self.preview_label.look_dragged.connect(self._on_scene_preview_dragged)
        self.preview_label.image_clicked.connect(self._on_scene_preview_clicked)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.preview_label)
        self.camera_debug_view = AprilTagWorldDebugView()
        self.camera_debug_view.setMinimumSize(520, 360)
        self.camera_debug_view.setToolTip("選択カメラの向きに固定した3Dワールド表示です。左の3Dワールドと同じ描画経路を使います。")
        self.camera_debug_view.camera_clicked.connect(self._select_frame_group_by_name)
        self.camera_debug_view.fixed_view_dragged.connect(self._on_camera_debug_view_dragged)
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(scroll)
        self.preview_stack.addWidget(self.camera_debug_view)
        image_layout.addWidget(self.preview_stack)
        viewport_splitter.addWidget(image_group)
        viewport_splitter.setStretchFactor(0, 1)
        viewport_splitter.setStretchFactor(1, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        right_layout.addWidget(self.log)

    def _build_case_group(self) -> QGroupBox:
        group = QGroupBox("入力データセット")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        self.case_root_browse = BrowseWidget(mode="dir")
        self.case_root_browse.set_text(str(DEFAULT_CASE_ROOT))
        self.case_name_edit = QLineEdit("current")
        self.transforms_browse = BrowseWidget(mode="file", filter_str="JSON (*.json);;All files (*)")
        self.pointcloud_browse = BrowseWidget(mode="file", filter_str="PLY (*.ply);;All files (*)")
        self.xml_browse = BrowseWidget(mode="file", filter_str="XML (*.xml);;All files (*)")
        self.coordinate_profile_combo = QComboBox()
        for profile in COORDINATE_PROFILES:
            self.coordinate_profile_combo.addItem(profile.label, profile.id)
        self.coordinate_profile_combo.setCurrentIndex(self.coordinate_profile_combo.findData(DEFAULT_COORDINATE_PROFILE))
        self.coordinate_profile_combo.setToolTip(
            "仮想タグデバッグでJSONとPLYをどういう出力座標系として重ねるかを指定します。既定はLichtFeld Cube6です。"
        )
        self.copy_images_check = QCheckBox("画像もケース内へコピーする")
        self.copy_images_check.setToolTip("通常はオフ推奨。Cubemap画像が大きい場合は参照だけにします。")
        form.addRow("ケース保存先", self.case_root_browse)
        form.addRow("ケース名", self.case_name_edit)
        form.addRow("transforms.json", self.transforms_browse)
        form.addRow("pointcloud.ply", self.pointcloud_browse)
        form.addRow("Metashape XML", self.xml_browse)
        form.addRow("座標解釈", self.coordinate_profile_combo)
        form.addRow("", self.copy_images_check)
        row = QHBoxLayout()
        self.create_case_btn = QPushButton("このセットでテストケースを作成")
        self.open_case_btn = QPushButton("既存ケースを開く")
        row.addWidget(self.create_case_btn)
        row.addWidget(self.open_case_btn)
        form.addRow("", row)
        return group

    def _build_scene_preview_group(self) -> QGroupBox:
        group = QGroupBox("配置プレビュー")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.frame_group_combo = QComboBox()
        self.frame_group_combo.setMinimumWidth(260)
        self.prev_camera_btn = QPushButton("前")
        self.next_camera_btn = QPushButton("次")
        self.reload_groups_btn = QPushButton("画像リスト更新")
        self.render_scene_preview_btn = QPushButton("プレビュー表示")
        self.grid_only_preview_check = QCheckBox("画像OFF")
        self.grid_only_preview_check.setToolTip("カメラ画像を隠し、同じ投影経路でグリッドとタグだけを確認します。")
        self.pointcloud_preview_check = QCheckBox("点群ビュー")
        self.pointcloud_preview_check.setToolTip(
            "Cubemap画像を貼らず、3Dワールド上の点群とカメラ点を選択カメラから透視投影します。"
        )
        row.addWidget(QLabel("カメラ位置"))
        row.addWidget(self.prev_camera_btn)
        row.addWidget(self.frame_group_combo, stretch=1)
        row.addWidget(self.next_camera_btn)
        row.addWidget(self.reload_groups_btn)
        row.addWidget(self.render_scene_preview_btn)
        row.addWidget(self.grid_only_preview_check)
        row.addWidget(self.pointcloud_preview_check)
        layout.addLayout(row)

        self.camera_status_label = QLabel("-")
        layout.addWidget(self.camera_status_label)

        params = QHBoxLayout()
        self.look_yaw_spin = QDoubleSpinBox()
        self.look_yaw_spin.setRange(-180.0, 180.0)
        self.look_yaw_spin.setDecimals(1)
        self.look_yaw_spin.setSingleStep(5.0)
        self.look_pitch_spin = QDoubleSpinBox()
        self.look_pitch_spin.setRange(-89.0, 89.0)
        self.look_pitch_spin.setDecimals(1)
        self.look_pitch_spin.setSingleStep(5.0)
        self.look_fov_spin = QDoubleSpinBox()
        self.look_fov_spin.setRange(20.0, 120.0)
        self.look_fov_spin.setDecimals(1)
        self.look_fov_spin.setSingleStep(5.0)
        self.look_fov_spin.setValue(90.0)
        self.placement_depth_spin = QDoubleSpinBox()
        self.placement_depth_spin.setRange(0.01, 10000.0)
        self.placement_depth_spin.setDecimals(3)
        self.placement_depth_spin.setSingleStep(1.0)
        self.placement_depth_spin.setValue(10.0)
        self.place_click_check = QCheckBox("クリックで配置を更新")
        self.place_click_check.setChecked(True)
        self.place_click_check.setToolTip("オンのときだけプレビュークリックでタグ位置を更新します。配置後は自動でオフになります。")
        for label, widget in (
            ("yaw", self.look_yaw_spin),
            ("pitch", self.look_pitch_spin),
            ("FOV", self.look_fov_spin),
            ("クリック深度SfM", self.placement_depth_spin),
        ):
            params.addWidget(QLabel(label))
            params.addWidget(widget)
        params.addWidget(self.place_click_check)
        params.addStretch(1)
        layout.addLayout(params)

        face_row = QHBoxLayout()
        face_row.addWidget(QLabel("面へ移動"))
        self.face_buttons: dict[str, QPushButton] = {}
        for face in ("pz", "px", "nx", "nz", "top", "bottom"):
            button = QPushButton(face)
            button.setFixedWidth(58)
            self.face_buttons[face] = button
            face_row.addWidget(button)
        face_row.addStretch(1)
        layout.addLayout(face_row)

        grid_row = QHBoxLayout()
        self.grid_overlay_check = QCheckBox("XZグリッド")
        self.grid_overlay_check.setChecked(True)
        self.grid_step_spin = QDoubleSpinBox()
        self.grid_step_spin.setRange(0.1, 1000.0)
        self.grid_step_spin.setDecimals(2)
        self.grid_step_spin.setSingleStep(0.5)
        self.grid_step_spin.setValue(2.0)
        self.grid_extent_spin = QDoubleSpinBox()
        self.grid_extent_spin.setRange(1.0, 100000.0)
        self.grid_extent_spin.setDecimals(1)
        self.grid_extent_spin.setSingleStep(5.0)
        self.grid_extent_spin.setValue(20.0)
        grid_row.addWidget(self.grid_overlay_check)
        grid_row.addWidget(QLabel("間隔SfM"))
        grid_row.addWidget(self.grid_step_spin)
        grid_row.addWidget(QLabel("最大距離SfM"))
        grid_row.addWidget(self.grid_extent_spin)
        grid_row.addStretch(1)
        layout.addLayout(grid_row)

        hint = QLabel("Cubemap 6面から疑似360画像を一度構築してカメラ画像を表示します。ドラッグで視点回転、クリックで深度値に沿って中心SfM/法線/上方向を入力します。3Dワールドのカメラ点クリックで画像セットを切り替えます。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return group

    def _build_tag_group(self) -> QGroupBox:
        group = QGroupBox("タグ設定")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        self.family_combo = QComboBox()
        self.family_combo.addItems(available_families())
        self.family_combo.setCurrentText("tag36h11")
        self.tag_id_spin = QSpinBox()
        self.tag_id_spin.setRange(0, 1_000_000)
        self.tag_id_spin.setValue(7)
        self.tag_size_spin = QDoubleSpinBox()
        self.tag_size_spin.setRange(0.001, 100.0)
        self.tag_size_spin.setDecimals(4)
        self.tag_size_spin.setSingleStep(0.01)
        self.tag_size_spin.setValue(0.160)
        self.true_scale_spin = QDoubleSpinBox()
        self.true_scale_spin.setRange(0.000001, 1000.0)
        self.true_scale_spin.setDecimals(6)
        self.true_scale_spin.setSingleStep(0.01)
        self.true_scale_spin.setValue(0.25)
        self.tag_image_browse = BrowseWidget(mode="file", filter_str="Images (*.png *.jpg *.jpeg);;All files (*)")
        self.create_printable_btn = QPushButton("A4印刷用タグを作成")
        form.addRow("ファミリ", self.family_combo)
        form.addRow("ID", self.tag_id_spin)
        form.addRow("実寸 m", self.tag_size_spin)
        form.addRow("合成真値 scale", self.true_scale_spin)
        form.addRow("注入元画像", self.tag_image_browse)
        form.addRow("", self.create_printable_btn)
        return group

    def _build_placement_group(self) -> QGroupBox:
        group = QGroupBox("仮想配置")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        self.placement_name_edit = QLineEdit("placement_001")
        self.center_editor = Vec3Editor((48.424, 9.482, 1.376))
        self.normal_editor = Vec3Editor((-0.981, -0.193, 0.004))
        self.up_editor = Vec3Editor((0.0, 1.0, 0.0))
        self.reference_frame_edit = QLineEdit()
        self.note_edit = QLineEdit()
        self.demo_pose_btn = QPushButton("D:/3DGS/test 用の仮配置を入れる")
        form.addRow("配置名", self.placement_name_edit)
        form.addRow("中心 SfM", self.center_editor)
        form.addRow("法線 SfM", self.normal_editor)
        form.addRow("上方向 SfM", self.up_editor)
        form.addRow("参照フレーム", self.reference_frame_edit)
        form.addRow("メモ", self.note_edit)
        form.addRow("", self.demo_pose_btn)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("実行")
        grid = QGridLayout(group)
        self.save_placement_btn = QPushButton("配置を保存")
        self.inject_btn = QPushButton("注入画像を生成")
        self.estimate_btn = QPushButton("推定を実行")
        self.preview_btn = QPushButton("プレビュー作成")
        self.run_all_btn = QPushButton("注入→推定→プレビュー")
        self.open_run_btn = QPushButton("結果フォルダを開く")
        grid.addWidget(self.save_placement_btn, 0, 0)
        grid.addWidget(self.inject_btn, 0, 1)
        grid.addWidget(self.estimate_btn, 1, 0)
        grid.addWidget(self.preview_btn, 1, 1)
        grid.addWidget(self.run_all_btn, 2, 0, 1, 2)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        grid.addWidget(line, 3, 0, 1, 2)
        grid.addWidget(self.open_run_btn, 4, 0, 1, 2)
        return group

    def _connect_signals(self) -> None:
        self.create_case_btn.clicked.connect(self._create_case)
        self.open_case_btn.clicked.connect(self._browse_case)
        self.coordinate_profile_combo.currentIndexChanged.connect(self._on_coordinate_profile_changed)
        self.create_printable_btn.clicked.connect(self._create_printable_target)
        self.reload_groups_btn.clicked.connect(self._load_preview_groups)
        self.render_scene_preview_btn.clicked.connect(self._render_scene_preview)
        self.grid_only_preview_check.toggled.connect(lambda _checked: self._render_scene_preview())
        self.pointcloud_preview_check.toggled.connect(lambda _checked: self._render_scene_preview())
        self.frame_group_combo.currentIndexChanged.connect(lambda _index: self._render_scene_preview())
        self.world_debug_view.camera_clicked.connect(self._select_frame_group_by_name)
        self.prev_camera_btn.clicked.connect(lambda: self._step_camera(-1))
        self.next_camera_btn.clicked.connect(lambda: self._step_camera(1))
        for face, button in self.face_buttons.items():
            button.clicked.connect(lambda _checked=False, face=face: self._jump_to_face(face))
        self.look_yaw_spin.valueChanged.connect(self._on_preview_spin_changed)
        self.look_pitch_spin.valueChanged.connect(self._on_preview_spin_changed)
        self.look_fov_spin.valueChanged.connect(self._on_preview_spin_changed)
        self.true_scale_spin.valueChanged.connect(lambda _value: self._update_tag_preview_overlay())
        self.tag_size_spin.valueChanged.connect(lambda _value: self._update_tag_preview_overlay())
        self.tag_id_spin.valueChanged.connect(lambda _value: self._update_tag_preview_overlay())
        self.placement_depth_spin.valueChanged.connect(lambda _value: self._reapply_last_preview_click_depth())
        self.grid_overlay_check.toggled.connect(lambda _checked: self._update_tag_preview_overlay())
        self.grid_step_spin.valueChanged.connect(lambda _value: self._update_tag_preview_overlay())
        self.grid_extent_spin.valueChanged.connect(lambda _value: self._update_tag_preview_overlay())
        for editor in (self.center_editor, self.normal_editor, self.up_editor):
            editor.value_changed.connect(self._update_tag_preview_overlay)
        self.save_placement_btn.clicked.connect(lambda: self._save_current_placement(show_message=True))
        self.inject_btn.clicked.connect(self._run_injection)
        self.estimate_btn.clicked.connect(self._run_estimation)
        self.preview_btn.clicked.connect(self._run_preview)
        self.run_all_btn.clicked.connect(self._run_all)
        self.open_run_btn.clicked.connect(self._open_run_dir)
        self.demo_pose_btn.clicked.connect(self._set_demo_pose)

    def _append_log(self, text: str) -> None:
        self.log.append(text.rstrip())

    def _set_demo_pose(self) -> None:
        self.center_editor.set_value((48.424, 9.482, 1.376))
        self.normal_editor.set_value((-0.981, -0.193, 0.004))
        self.up_editor.set_value((0.0, 1.0, 0.0))

    def _create_case(self) -> None:
        try:
            case = create_case(
                case_root=Path(self.case_root_browse.text()),
                case_name=self.case_name_edit.text(),
                source_transforms=Path(self.transforms_browse.text()),
                source_pointcloud=Path(self.pointcloud_browse.text()) if self.pointcloud_browse.text() else None,
                source_metashape_xml=Path(self.xml_browse.text()) if self.xml_browse.text() else None,
                copy_images=self.copy_images_check.isChecked(),
                tag_family=self.family_combo.currentText(),
                tag_id=self.tag_id_spin.value(),
                default_tag_size_m=self.tag_size_spin.value(),
                true_scale=self.true_scale_spin.value(),
                coordinate_profile=self.coordinate_profile_combo.currentData() or DEFAULT_COORDINATE_PROFILE,
            )
        except Exception as e:
            QMessageBox.critical(self, "ケース作成エラー", str(e))
            return
        self._apply_case(case)
        self._append_log(f"Case created: {case.case_json_path}")

    def _browse_case(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "ケースフォルダを選択", self.case_root_browse.text())
        if path:
            self._load_case(Path(path))

    def _load_case(self, case_dir: Path) -> None:
        try:
            case = load_case(case_dir)
        except Exception as e:
            QMessageBox.critical(self, "ケース読み込みエラー", str(e))
            return
        self._apply_case(case)
        self._append_log(f"Case loaded: {case.case_json_path}")

    def _apply_case(self, case: AprilTagDevCase) -> None:
        self.case = case
        self.case_root_browse.set_text(str(case.case_dir.parent))
        self.case_name_edit.setText(case.name)
        self.transforms_browse.set_text(str(case.source_transforms))
        self.pointcloud_browse.set_text(str(case.source_pointcloud or ""))
        self.xml_browse.set_text(str(case.source_metashape_xml or ""))
        self._set_coordinate_profile(case.coordinate_profile)
        self.family_combo.setCurrentText(case.tag_family)
        self.tag_id_spin.setValue(case.tag_id)
        self.tag_size_spin.setValue(case.default_tag_size_m)
        self.true_scale_spin.setValue(case.true_scale)
        self.status_label.setText(self._case_status_text(case))
        self._cubemap_image_cache.clear()
        self._equirect_preview_cache.clear()
        self._load_preview_groups()

    @staticmethod
    def _case_status_text(case: AprilTagDevCase) -> str:
        return (
            f"ケース: {case.case_dir}\n"
            f"入力: {case.transforms_for_processing()}\n"
            f"画像モード: {'コピー' if case.input_mode == 'copy' else '参照'}\n"
            f"座標解釈: {coordinate_profile_label(case.coordinate_profile)}"
        )

    def _load_world_pointcloud(self, case: AprilTagDevCase) -> None:
        pointcloud_path = case.source_pointcloud
        copied_pointcloud = case.input_dir / "pointcloud.ply"
        if copied_pointcloud.is_file():
            pointcloud_path = copied_pointcloud
        if pointcloud_path is None or not pointcloud_path.is_file():
            self._world_pointcloud = None
            self.world_debug_view.set_pointcloud(None)
            self.camera_debug_view.set_pointcloud(None)
            self._append_log("Point cloud: none")
            return
        try:
            sample = load_point_cloud_sample(pointcloud_path)
            matrix = _compose_display_matrices(
                self._world_display_matrix(),
                pointcloud_display_matrix(case.coordinate_profile),
            )
            self._world_pointcloud = transform_point_cloud_sample(sample, matrix)
        except Exception as e:
            self._world_pointcloud = None
            self.world_debug_view.set_pointcloud(None)
            self.camera_debug_view.set_pointcloud(None)
            self._append_log(f"Point cloud load failed: {e}")
            return
        self.world_debug_view.set_pointcloud(self._world_pointcloud)
        self.camera_debug_view.set_pointcloud(self._world_pointcloud)
        self._append_log(
            f"Point cloud loaded: {pointcloud_path} "
            f"({len(self._world_pointcloud.points)} / {self._world_pointcloud.source_count} points)"
        )
        self._append_log(
            f"Coordinate profile: {coordinate_profile_label(case.coordinate_profile)} - "
            f"{coordinate_profile_note(case.coordinate_profile)}"
        )

    def _world_display_matrix(self) -> np.ndarray | None:
        if self._metashape_world_display_matrix is not None:
            return self._metashape_world_display_matrix.copy()
        profile = self.case.coordinate_profile if self.case is not None else self.coordinate_profile_combo.currentData()
        return world_display_matrix(profile)

    def _update_world_display_alignment(self) -> None:
        self._metashape_world_display_matrix = None
        self._metashape_world_alignment_rmse = None
        if self.case is None or not self._cubemap_groups:
            return
        estimated = _estimate_world_display_matrix_from_metashape(
            self._cubemap_groups,
            self.case.source_metashape_xml,
        )
        if estimated is None:
            return
        matrix, rmse, count = estimated
        self._metashape_world_display_matrix = matrix
        self._metashape_world_alignment_rmse = rmse
        self._append_log(f"Metashape axis alignment: XML camera match, count={count}, rmse={rmse:.6g}")

    def _world_display_groups(self) -> tuple[CubemapFrameGroup, ...]:
        matrix = self._world_display_matrix()
        if matrix is None:
            return self._cubemap_groups
        return tuple(_transform_group_for_world_display(group, matrix) for group in self._cubemap_groups)

    def _world_display_group_for(self, group: CubemapFrameGroup) -> CubemapFrameGroup:
        return _transform_group_for_world_display(group, self._world_display_matrix())

    def _selected_world_display_group(self) -> CubemapFrameGroup | None:
        group = self._selected_group()
        return None if group is None else self._world_display_group_for(group)

    def _set_coordinate_profile(self, value: str | None) -> None:
        profile = normalize_coordinate_profile(value)
        index = self.coordinate_profile_combo.findData(profile)
        if index < 0:
            index = self.coordinate_profile_combo.findData(DEFAULT_COORDINATE_PROFILE)
        self.coordinate_profile_combo.blockSignals(True)
        self.coordinate_profile_combo.setCurrentIndex(max(0, index))
        self.coordinate_profile_combo.blockSignals(False)

    def _on_coordinate_profile_changed(self, _index: int) -> None:
        if self.case is None:
            return
        profile = normalize_coordinate_profile(self.coordinate_profile_combo.currentData())
        self.case = replace(self.case, coordinate_profile=profile)
        try:
            save_case(self.case)
        except Exception as e:
            self._append_log(f"Case coordinate profile save failed: {e}")
        self.status_label.setText(self._case_status_text(self.case))
        self._equirect_preview_cache.clear()
        self._update_world_display_alignment()
        self._load_world_pointcloud(self.case)
        self._sync_world_debug_view()
        self._render_scene_preview()

    def _load_preview_groups(self) -> None:
        case = self._require_case()
        if case is None:
            return
        try:
            groups = load_cubemap_frame_groups(case.transforms_for_processing())
            labels = load_metashape_camera_labels(case.source_metashape_xml) if case.source_metashape_xml else ()
            self._cubemap_groups = order_groups_by_labels(groups, labels)
        except Exception as e:
            QMessageBox.critical(self, "プレビュー読み込みエラー", str(e))
            return
        current = self.frame_group_combo.currentText()
        self.frame_group_combo.blockSignals(True)
        self.frame_group_combo.clear()
        for group in self._cubemap_groups:
            self.frame_group_combo.addItem(f"{group.name} ({len(group.frames)} faces)", group.name)
        if current:
            index = self.frame_group_combo.findText(current)
            if index >= 0:
                self.frame_group_combo.setCurrentIndex(index)
        self.frame_group_combo.blockSignals(False)
        self._append_log(f"Cubemap preview groups: {len(self._cubemap_groups)}")
        self._update_world_display_alignment()
        self._load_world_pointcloud(case)
        self._sync_world_debug_view()
        if self._cubemap_groups:
            self._render_scene_preview()
        else:
            self.camera_status_label.setText("Cubemap画像グループがありません")

    def _create_printable_target(self) -> None:
        case = self._require_case()
        if case is None:
            return
        try:
            target = create_printable_target(
                case.assets_dir,
                family=self.family_combo.currentText(),
                tag_id=self.tag_id_spin.value(),
                tag_size_m=self.tag_size_spin.value(),
            )
        except Exception as e:
            QMessageBox.critical(self, "タグ作成エラー", str(e))
            return
        self.tag_image_browse.set_text(str(target.marker_png))
        self._append_log(f"Printable PNG: {target.a4_png}")
        self._append_log(f"Printable PDF: {target.a4_pdf}")
        self._append_log(f"Marker image for injection: {target.marker_png}")

    def _require_case(self) -> AprilTagDevCase | None:
        if self.case is None:
            QMessageBox.warning(self, "ケース未選択", "先にテストケースを作成または読み込んでください。")
            return None
        return self.case

    def _current_placement(self) -> AprilTagPlacement | None:
        case = self._require_case()
        if case is None:
            return None
        tag_image = Path(self.tag_image_browse.text()) if self.tag_image_browse.text() else None
        if tag_image is None or not tag_image.is_file():
            try:
                target = create_printable_target(
                    case.assets_dir,
                    family=self.family_combo.currentText(),
                    tag_id=self.tag_id_spin.value(),
                    tag_size_m=self.tag_size_spin.value(),
                )
            except Exception as e:
                QMessageBox.critical(self, "タグ作成エラー", str(e))
                return None
            tag_image = target.marker_png
            self.tag_image_browse.set_text(str(tag_image))
            self._append_log(f"Marker image auto-created: {tag_image}")
        return AprilTagPlacement(
            name=self.placement_name_edit.text().strip() or "placement_001",
            tag_family=self.family_combo.currentText(),
            tag_id=self.tag_id_spin.value(),
            tag_image=tag_image,
            tag_size_m=self.tag_size_spin.value(),
            true_scale=self.true_scale_spin.value(),
            tag_center_sfm=self.center_editor.value(),
            tag_normal_sfm=self.normal_editor.value(),
            tag_up_sfm=self.up_editor.value(),
            reference_frame=self.reference_frame_edit.text().strip(),
            note=self.note_edit.text().strip(),
        )

    def _selected_group(self) -> CubemapFrameGroup | None:
        if not self._cubemap_groups:
            return None
        index = self.frame_group_combo.currentIndex()
        if index < 0 or index >= len(self._cubemap_groups):
            return self._cubemap_groups[0]
        return self._cubemap_groups[index]

    def _sync_world_debug_view(self) -> None:
        group = self._selected_group()
        display_matrix = self._world_display_matrix()
        display_group = None if group is None else self._world_display_group_for(group)
        display_groups = self._world_display_groups()
        for view in (self.world_debug_view, self.camera_debug_view):
            view.set_groups(display_groups)
            view.set_selected_group(group.name if group is not None else "")
        if self.pointcloud_preview_check.isChecked() and display_group is not None:
            self.world_debug_view.set_preview_to_world_matrix(
                self._preview_camera_rotation_for_display_group(display_group).T
            )
        else:
            self.world_debug_view.set_preview_to_world_matrix(None)
        self.camera_debug_view.set_preview_to_world_matrix(
            None if display_group is None else self._preview_camera_rotation_for_display_group(display_group).T
        )
        for view in (self.world_debug_view, self.camera_debug_view):
            view.set_preview_params(
                yaw_deg=self._scene_preview_params.yaw_deg,
                pitch_deg=self._scene_preview_params.pitch_deg,
                roll_deg=self._scene_preview_params.roll_deg,
                fov_deg=self._scene_preview_params.fov_deg,
            )
            view.set_grid(
                step=self.grid_step_spin.value(),
                extent=self.grid_extent_spin.value(),
            )
        try:
            center = _transform_points_for_world_display(
                np.asarray(self.center_editor.value(), dtype=float).reshape(1, 3),
                display_matrix,
            )[0]
            normal = _transform_vectors_for_world_display(
                np.asarray(self.normal_editor.value(), dtype=float).reshape(1, 3),
                display_matrix,
            )[0]
            up = _transform_vectors_for_world_display(
                np.asarray(self.up_editor.value(), dtype=float).reshape(1, 3),
                display_matrix,
            )[0]
            for view in (self.world_debug_view, self.camera_debug_view):
                view.set_tag(
                    center=center,
                    normal=normal,
                    up=up,
                    tag_size_m=float(self.tag_size_spin.value()),
                    true_scale=float(self.true_scale_spin.value()),
                )
        except Exception:
            pass
        if self.pointcloud_preview_check.isChecked() and display_group is not None:
            camera, right, up, forward = self._camera_debug_view_pose(display_group)
            self.camera_debug_view.set_fixed_perspective_view(
                camera_position=camera,
                right=right,
                up=up,
                forward=forward,
                fov_deg=self._scene_preview_params.fov_deg,
            )
        else:
            self.camera_debug_view.clear_fixed_view()
        self.preview_stack.setCurrentIndex(1 if self.pointcloud_preview_check.isChecked() else 0)

    def _sync_camera_debug_view_pose_only(self) -> None:
        display_group = self._selected_world_display_group()
        if display_group is None:
            return
        preview_to_world = self._preview_camera_rotation_for_display_group(display_group).T
        self.world_debug_view.set_preview_to_world_matrix(preview_to_world)
        self.camera_debug_view.set_preview_to_world_matrix(preview_to_world)
        for view in (self.world_debug_view, self.camera_debug_view):
            view.set_preview_params(
                yaw_deg=self._scene_preview_params.yaw_deg,
                pitch_deg=self._scene_preview_params.pitch_deg,
                roll_deg=self._scene_preview_params.roll_deg,
                fov_deg=self._scene_preview_params.fov_deg,
            )
        camera, right, up, forward = self._camera_debug_view_pose(display_group)
        self.camera_debug_view.set_fixed_perspective_view(
            camera_position=camera,
            right=right,
            up=up,
            forward=forward,
            fov_deg=self._scene_preview_params.fov_deg,
        )
        self.preview_stack.setCurrentIndex(1)

    def _camera_debug_view_pose(
        self,
        display_group: CubemapFrameGroup,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rotation = self._preview_camera_rotation_for_display_group(display_group)
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64) @ rotation.T
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64) @ rotation.T
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ rotation.T
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up = up - right * float(up @ right)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        forward = forward - right * float(forward @ right) - up * float(forward @ up)
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        camera = np.asarray(display_group.camera_position_sfm, dtype=np.float64)
        return camera, right, up, forward

    def _step_camera(self, delta: int) -> None:
        count = self.frame_group_combo.count()
        if count <= 0:
            return
        index = (self.frame_group_combo.currentIndex() + int(delta)) % count
        self.frame_group_combo.setCurrentIndex(index)

    def _select_frame_group_by_name(self, group_name: str) -> None:
        index = self.frame_group_combo.findData(str(group_name))
        if index < 0:
            self._append_log(f"Camera group not found: {group_name}")
            return
        if index == self.frame_group_combo.currentIndex():
            self._sync_world_debug_view()
            return
        self.frame_group_combo.setCurrentIndex(index)
        self._append_log(f"Selected camera from 3D view: {group_name}")

    def _jump_to_face(self, face: str) -> None:
        group = self._selected_world_display_group()
        if group is None:
            return
        if self.pointcloud_preview_check.isChecked():
            actual_params = face_view_params(group, face, fov_deg=self.look_fov_spin.value())
            params = None if actual_params is None else (actual_params[0], actual_params[1], 0.0, actual_params[2])
        else:
            params = axis_face_view_params(
                group,
                face,
                fov_deg=self.look_fov_spin.value(),
            )
        if params is None:
            self._append_log(f"Face not available in this group: {face}")
            return
        yaw, pitch, roll, fov = params
        self._scene_preview_params = PerspectiveParams(
            yaw_deg=normalize_yaw_deg(yaw),
            pitch_deg=clamp_pitch_deg(pitch),
            fov_deg=fov,
            roll_deg=roll,
        )
        self._sync_preview_spins()
        if self.pointcloud_preview_check.isChecked():
            self._render_scene_preview()
        else:
            self.preview_label.set_perspective_params(self._scene_preview_params)
            self._update_tag_preview_overlay()
        self._append_log(f"Jumped to face {face}: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")

    def _on_preview_spin_changed(self) -> None:
        self._set_scene_preview_params_from_spins()
        if self.pointcloud_preview_check.isChecked():
            self._render_scene_preview()
        else:
            self.preview_label.set_perspective_params(self._scene_preview_params)
            self._update_tag_preview_overlay()

    def _set_scene_preview_params_from_spins(self) -> None:
        self._scene_preview_params = PerspectiveParams(
            yaw_deg=normalize_yaw_deg(self.look_yaw_spin.value()),
            pitch_deg=clamp_pitch_deg(self.look_pitch_spin.value()),
            fov_deg=float(self.look_fov_spin.value()),
            roll_deg=self._scene_preview_params.roll_deg,
        )

    def _sync_preview_spins(self) -> None:
        self.look_yaw_spin.blockSignals(True)
        self.look_pitch_spin.blockSignals(True)
        self.look_fov_spin.blockSignals(True)
        self.look_yaw_spin.setValue(self._scene_preview_params.yaw_deg)
        self.look_pitch_spin.setValue(self._scene_preview_params.pitch_deg)
        self.look_fov_spin.setValue(self._scene_preview_params.fov_deg)
        self.look_yaw_spin.blockSignals(False)
        self.look_pitch_spin.blockSignals(False)
        self.look_fov_spin.blockSignals(False)

    def _render_scene_preview(self) -> None:
        group = self._selected_group()
        display_group = self._selected_world_display_group()
        if group is None:
            self.preview_label.setText("Cubemap画像グループがありません")
            return
        if display_group is None:
            self.preview_label.setText("表示用カメラを作成できません")
            return
        case = self._require_case()
        if case is None:
            return
        self._set_scene_preview_params_from_spins()
        self._sync_world_debug_view()
        try:
            if self.pointcloud_preview_check.isChecked():
                self.preview_stack.setCurrentIndex(1)
                return
            overlays = self._preview_overlays()
            if self.grid_only_preview_check.isChecked():
                image = self._grid_only_equirect_preview()
                shown = self.preview_label.set_perspective_image_bgr(
                    image,
                    self._scene_preview_params,
                    overlays=overlays,
                    logical_size=QSize(self._scene_preview_size, self._scene_preview_size),
                )
            else:
                cache_key = f"world-display-camera:{normalize_coordinate_profile(case.coordinate_profile)}:{group.name}"
                image = self._equirect_preview_cache.get(cache_key)
                if image is None:
                    image = render_cubemap_axis_equirect(
                        display_group,
                        output_width=2048,
                        output_height=1024,
                        image_cache=self._cubemap_image_cache,
                    )
                    self._equirect_preview_cache[cache_key] = image
                shown = self.preview_label.set_perspective_image_bgr(
                    image,
                    self._scene_preview_params,
                    overlays=overlays,
                    logical_size=QSize(self._scene_preview_size, self._scene_preview_size),
                )
        except Exception as e:
            self.preview_label.setText(f"プレビュー生成エラー: {e}")
            return
        self.preview_label.set_drag_mode("look")
        if not shown:
            self.preview_label.setText("GPU透視投影プレビューを初期化できませんでした")
        index = self.frame_group_combo.currentIndex()
        position = group.camera_position_sfm
        self.camera_status_label.setText(
            f"{index + 1} / {len(self._cubemap_groups)}  {group.name}  "
            f"pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
        )
        for face, button in self.face_buttons.items():
            button.setEnabled(face in group.frames_by_face)
        self._sync_world_debug_view()

    def _grid_only_equirect_preview(self) -> np.ndarray:
        cache_key = "__grid_only__"
        image = self._equirect_preview_cache.get(cache_key)
        if image is not None:
            return image
        height, width = 1024, 2048
        image = np.full((height, width, 3), (20, 24, 30), dtype=np.uint8)
        longitude_color = np.array([45, 52, 64], dtype=np.uint8)
        latitude_color = np.array([40, 46, 56], dtype=np.uint8)
        major_color = np.array([76, 88, 105], dtype=np.uint8)
        for x in range(0, width, width // 24):
            image[:, max(0, x - 1) : min(width, x + 1)] = longitude_color
        for y in range(0, height, height // 12):
            image[max(0, y - 1) : min(height, y + 1), :] = latitude_color
        image[:, width // 2 - 2 : width // 2 + 2] = np.array([90, 175, 245], dtype=np.uint8)
        image[height // 2 - 2 : height // 2 + 2, :] = major_color
        self._equirect_preview_cache[cache_key] = image
        return image

    def _render_pointcloud_perspective_preview(
        self,
        display_group: CubemapFrameGroup,
        overlays: list[PerspectiveLabelOverlay],
    ) -> bool:
        image = self._pointcloud_perspective_preview_bgr(display_group)
        self._draw_preview_overlays_bgr(image, overlays)
        pixmap = QPixmap.fromImage(bgr_to_qimage(image))
        if pixmap.isNull():
            return False
        self.preview_label.set_source_pixmap(pixmap)
        return True

    def _pointcloud_perspective_preview_bgr(self, display_group: CubemapFrameGroup) -> np.ndarray:
        size = int(self._scene_preview_size)
        image = np.full((size, size, 3), (14, 18, 24), dtype=np.uint8)
        pointcloud = self._world_pointcloud
        if pointcloud is not None and pointcloud.points.size:
            xy, depth, valid = self._project_world_display_points_for_preview(display_group, pointcloud.points)
            in_frame = (
                valid
                & (xy[:, 0] >= 0.0)
                & (xy[:, 0] < size)
                & (xy[:, 1] >= 0.0)
                & (xy[:, 1] < size)
            )
            visible = np.flatnonzero(in_frame)
            if visible.size:
                order = visible[np.argsort(depth[visible])[::-1]]
                x = np.rint(xy[order, 0]).astype(np.int32)
                y = np.rint(xy[order, 1]).astype(np.int32)
                x = np.clip(x, 0, size - 1)
                y = np.clip(y, 0, size - 1)
                if pointcloud.colors is None:
                    colors = np.full((order.size, 3), (188, 188, 188), dtype=np.uint8)
                else:
                    source_colors = np.asarray(pointcloud.colors, dtype=np.uint8)
                    colors = source_colors[order][:, ::-1]
                image[y, x] = colors

        self._draw_preview_camera_points_bgr(image, display_group)
        center = (size - 1) // 2
        cv2.line(image, (center - 10, center), (center + 10, center), (180, 190, 205), 1, cv2.LINE_AA)
        cv2.line(image, (center, center - 10), (center, center + 10), (180, 190, 205), 1, cv2.LINE_AA)
        return image

    def _draw_preview_camera_points_bgr(self, image: np.ndarray, display_group: CubemapFrameGroup) -> None:
        groups = self._world_display_groups()
        if not groups:
            return
        camera_points = np.asarray([group.camera_position_sfm for group in groups], dtype=np.float64)
        xy, _depth, valid = self._project_world_display_points_for_preview(display_group, camera_points)
        size = int(self._scene_preview_size)
        for group, point, is_valid in zip(groups, xy, valid, strict=True):
            if not is_valid:
                continue
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            if x < 0 or y < 0 or x >= size or y >= size:
                continue
            selected = group.name == display_group.name
            color = (55, 220, 255) if selected else (255, 150, 70)
            radius = 5 if selected else 3
            cv2.circle(image, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)

    def _project_world_display_points_for_preview(
        self,
        display_group: CubemapFrameGroup,
        points_world_display: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = self._pointcloud_preview_points_for_projection(points_world_display)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_world_display must be an Nx3 array")
        vectors = points - display_group.camera_position_sfm.reshape(1, 3)
        view_rotation = self._preview_camera_rotation_for_display_group(display_group)
        view = vectors @ view_rotation
        depth = view[:, 2]
        size = float(self._scene_preview_size)
        focal = 0.5 * size / np.tan(np.deg2rad(float(self._scene_preview_params.fov_deg)) / 2.0)
        center = (size - 1.0) / 2.0
        xy = np.full((len(points), 2), np.nan, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            xy[valid, 0] = center + focal * (view[valid, 0] / depth[valid])
            xy[valid, 1] = center - focal * (view[valid, 1] / depth[valid])
        valid &= np.all(np.isfinite(xy), axis=1)
        return xy, depth, valid

    def _pointcloud_preview_points_for_projection(self, points_world_display: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_display, dtype=np.float64)
        return points

    def _preview_camera_rotation_for_display_group(self, display_group: CubemapFrameGroup) -> np.ndarray:
        return virtual_camera_rotation_with_roll(
            display_group,
            yaw_deg=self._scene_preview_params.yaw_deg,
            pitch_deg=self._scene_preview_params.pitch_deg,
            roll_deg=self._scene_preview_params.roll_deg,
        )

    def _draw_preview_overlays_bgr(self, image: np.ndarray, overlays: list[PerspectiveLabelOverlay]) -> None:
        for item in overlays:
            color = tuple(int(channel) for channel in item.color_bgr)
            width = 3 if item.highlighted else 1
            if item.polygon:
                points = np.rint(np.asarray(item.polygon, dtype=np.float32)).astype(np.int32)
                if len(points) >= 3 and item.fill_alpha > 0.0:
                    layer = image.copy()
                    cv2.fillPoly(layer, [points], color, lineType=cv2.LINE_AA)
                    cv2.addWeighted(layer, float(item.fill_alpha), image, 1.0 - float(item.fill_alpha), 0.0, image)
                if len(points) >= 2:
                    cv2.polylines(image, [points], True, color, width, cv2.LINE_AA)
            if item.polyline:
                points = np.rint(np.asarray(item.polyline, dtype=np.float32)).astype(np.int32)
                for start, end in zip(points, points[1:], strict=False):
                    self._draw_preview_overlay_line(image, tuple(start), tuple(end), color, width, item.dashed)
                if item.point_radius > 0.0 and len(points):
                    center = tuple(int(v) for v in points[-1])
                    radius = max(1, int(round(float(item.point_radius))))
                    cv2.circle(image, center, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
                    cv2.circle(image, center, radius, (245, 245, 245), 1, cv2.LINE_AA)
            if item.label:
                origin = (int(item.origin[0]), int(item.origin[1]))
                cv2.putText(image, item.label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, item.label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_preview_overlay_line(
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
        dashed: bool,
    ) -> None:
        if not dashed:
            cv2.line(image, start, end, color, width, cv2.LINE_AA)
            return
        start_vec = np.asarray(start, dtype=np.float64)
        end_vec = np.asarray(end, dtype=np.float64)
        delta = end_vec - start_vec
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            return
        direction = delta / length
        dash = 10.0
        gap = 7.0
        position = 0.0
        while position < length:
            segment_start = start_vec + direction * position
            segment_end = start_vec + direction * min(position + dash, length)
            cv2.line(
                image,
                tuple(np.rint(segment_start).astype(np.int32)),
                tuple(np.rint(segment_end).astype(np.int32)),
                color,
                width,
                cv2.LINE_AA,
            )
            position += dash + gap

    def _on_scene_preview_dragged(self, delta_x: float, delta_y: float) -> None:
        if not self._cubemap_groups:
            return
        self._scene_preview_params = params_from_drag(self._scene_preview_params, delta_x, delta_y)
        self._sync_preview_spins()
        if self.pointcloud_preview_check.isChecked():
            self._sync_camera_debug_view_pose_only()
        else:
            self.preview_label.set_perspective_params(self._scene_preview_params)
            self._update_tag_preview_overlay()

    def _on_camera_debug_view_dragged(self, delta_x: float, delta_y: float) -> None:
        if not self._cubemap_groups:
            return
        self._scene_preview_params = params_from_drag(self._scene_preview_params, delta_x, delta_y)
        self._sync_preview_spins()
        self._sync_camera_debug_view_pose_only()

    def _on_scene_preview_clicked(self, x: float, y: float) -> None:
        if not self.place_click_check.isChecked():
            self._append_log("Preview click ignored: enable 'クリックで配置を更新' to move the tag.")
            return
        group = self._selected_group()
        display_group = self._selected_world_display_group()
        if group is None or display_group is None:
            return
        size = float(self._scene_preview_size)
        x_px = max(0.0, min(size - 1.0, x))
        y_px = max(0.0, min(size - 1.0, y))
        ray_display, up_display, face = self._preview_pixel_to_world_display_ray_and_up(
            display_group,
            x_px=x_px,
            y_px=y_px,
        )
        display_matrix = self._world_display_matrix()
        ray = _transform_vectors_from_world_display(ray_display.reshape(1, 3), display_matrix)[0]
        ray /= max(float(np.linalg.norm(ray)), 1e-12)
        up = _transform_vectors_from_world_display(up_display.reshape(1, 3), display_matrix)[0]
        up = up - ray * float(up @ ray)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        self._last_click_state = (group.name, ray.copy(), up.copy())
        self._apply_click_placement(group, ray, up)
        self._append_log(
            "Placement filled from preview click: "
            f"group={group.name}, face={face or '-'}, depth_sfm={self.placement_depth_spin.value():.3f}"
        )
        self.place_click_check.setChecked(False)

    def _apply_click_placement(self, group: CubemapFrameGroup, ray: np.ndarray, up: np.ndarray) -> None:
        center = group.camera_position_sfm + ray * float(self.placement_depth_spin.value())
        normal = -ray
        self.center_editor.set_value(tuple(float(v) for v in center))
        self.normal_editor.set_value(tuple(float(v) for v in normal))
        self.up_editor.set_value(tuple(float(v) for v in up))
        self.reference_frame_edit.setText(group.name)
        self._update_tag_preview_overlay()

    def _reapply_last_preview_click_depth(self) -> None:
        if self._last_click_state is None:
            return
        group_name, ray, up = self._last_click_state
        group = next((candidate for candidate in self._cubemap_groups if candidate.name == group_name), None)
        if group is None:
            return
        self._apply_click_placement(group, ray, up)

    def _preview_overlays(self) -> list[PerspectiveLabelOverlay]:
        return [
            *self._grid_preview_overlays(),
            *self._tag_preview_overlays(),
            *self._camera_axis_gizmo_overlays(),
        ]

    def _camera_axis_gizmo_overlays(self) -> list[PerspectiveLabelOverlay]:
        size = float(self._scene_preview_size)
        origin = np.array([54.0, max(46.0, size - 48.0)], dtype=np.float64)
        length = 34.0
        rotation = self._camera_axis_gizmo_rotation()
        axes = (
            ("+X", np.array([1.0, 0.0, 0.0], dtype=np.float64), AXIS_GIZMO_X_BGR),
            ("+Y", np.array([0.0, 1.0, 0.0], dtype=np.float64), AXIS_GIZMO_Y_BGR),
            ("+Z", np.array([0.0, 0.0, 1.0], dtype=np.float64), AXIS_GIZMO_Z_BGR),
        )
        overlays: list[PerspectiveLabelOverlay] = [
            PerspectiveLabelOverlay(
                label="",
                box=(int(origin[0] - 4), int(origin[1] - 4), int(origin[0] + 4), int(origin[1] + 4)),
                origin=(int(origin[0]), int(origin[1])),
                color_bgr=(245, 245, 245),
                highlighted=True,
                polyline=((float(origin[0]), float(origin[1])),),
                point_radius=4.0,
            )
        ]
        for label, axis, color in axes:
            view = axis @ rotation
            screen_delta = np.array([view[0], -view[1]], dtype=np.float64)
            end = origin + screen_delta * length
            label_offset = np.array([5.0, -5.0], dtype=np.float64)
            if float(np.linalg.norm(screen_delta)) < 0.15:
                label_offset = np.array([5.0, 12.0 if float(view[2]) >= 0.0 else -12.0], dtype=np.float64)
            overlays.append(
                PerspectiveLabelOverlay(
                    label=label,
                    box=(int(end[0] - 4), int(end[1] - 4), int(end[0] + 4), int(end[1] + 4)),
                    origin=(int(end[0] + label_offset[0]), int(end[1] + label_offset[1])),
                    color_bgr=color,
                    highlighted=True,
                    polyline=((float(origin[0]), float(origin[1])), (float(end[0]), float(end[1]))),
                    point_radius=3.0,
                )
            )
        return overlays

    def _camera_axis_gizmo_rotation(self) -> np.ndarray:
        group = self._selected_world_display_group()
        if self.pointcloud_preview_check.isChecked() and group is not None:
            return self._preview_camera_rotation_for_display_group(group)
        return _preview_rotation_matrix(
            self._scene_preview_params.yaw_deg,
            self._scene_preview_params.pitch_deg,
            self._scene_preview_params.roll_deg,
        )

    def _project_preview_points(self, points_sfm: np.ndarray) -> np.ndarray | None:
        group = self._selected_world_display_group()
        if group is None:
            return None
        points_display = _transform_points_for_world_display(
            np.asarray(points_sfm, dtype=np.float64),
            self._world_display_matrix(),
        )
        return project_sfm_points_to_axis_preview_points(
            group,
            points_display,
            output_size=self._scene_preview_size,
            yaw_deg=self._scene_preview_params.yaw_deg,
            pitch_deg=self._scene_preview_params.pitch_deg,
            fov_deg=self._scene_preview_params.fov_deg,
            roll_deg=self._scene_preview_params.roll_deg,
        )

    def _grid_preview_overlays(self) -> list[PerspectiveLabelOverlay]:
        if not self.grid_overlay_check.isChecked():
            return []
        group = self._selected_group()
        display_group = self._selected_world_display_group()
        if group is None or display_group is None:
            return []
        try:
            display_matrix = self._world_display_matrix()
            center = _transform_points_for_world_display(
                np.asarray(self.center_editor.value(), dtype=float).reshape(1, 3),
                display_matrix,
            )[0]
            camera = _transform_points_for_world_display(
                group.camera_position_sfm.reshape(1, 3),
                display_matrix,
            )[0]
            step = max(0.1, float(self.grid_step_spin.value()))
            max_distance = max(step, float(self.grid_extent_spin.value()))
        except Exception:
            return []

        overlays: list[PerspectiveLabelOverlay] = []
        x_min, x_max, z_min, z_max = self._visible_ground_bounds(display_group, center, camera, max_distance, step)
        draw_step = step
        while max((x_max - x_min) / draw_step, (z_max - z_min) / draw_step) > 80:
            draw_step *= 2.0
        x_min = np.floor(x_min / draw_step) * draw_step
        x_max = np.ceil(x_max / draw_step) * draw_step
        z_min = np.floor(z_min / draw_step) * draw_step
        z_max = np.ceil(z_max / draw_step) * draw_step
        x_values = np.arange(x_min, x_max + draw_step * 0.5, draw_step)
        z_values = np.arange(z_min, z_max + draw_step * 0.5, draw_step)

        def add_grid_line(points: np.ndarray, color_bgr: tuple[int, int, int], highlighted: bool = False) -> None:
            for segment in self._project_world_display_polyline_segments(points):
                overlays.append(
                    PerspectiveLabelOverlay(
                        label="",
                        box=(0, 0, 0, 0),
                        origin=(0, 0),
                        color_bgr=color_bgr,
                        highlighted=highlighted,
                        polyline=segment,
                    )
                )

        samples = max(8, min(80, int(round(max(x_max - x_min, z_max - z_min) / max(draw_step, 1e-6))) * 2 + 1))
        for x in x_values:
            is_axis = abs(x) <= draw_step * 0.25
            if is_axis:
                continue
            zs = self._line_samples(z_min, z_max, samples, include_zero=is_axis)
            add_grid_line(
                np.column_stack([np.full_like(zs, float(x)), np.zeros_like(zs), zs]),
                GRID_LINE_BGR,
                False,
            )
        for z in z_values:
            is_axis = abs(z) <= draw_step * 0.25
            if is_axis:
                continue
            xs = self._line_samples(x_min, x_max, samples, include_zero=is_axis)
            add_grid_line(
                np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, float(z))]),
                GRID_LINE_BGR,
                False,
            )

        axis_x_min = min(x_min, 0.0, float(center[0]), float(camera[0]))
        axis_x_max = max(x_max, 0.0, float(center[0]), float(camera[0]))
        axis_z_min = min(z_min, 0.0, float(center[2]), float(camera[2]))
        axis_z_max = max(z_max, 0.0, float(center[2]), float(camera[2]))
        axis_samples = max(
            samples,
            max(
                16,
                min(
                    160,
                    int(round(max(axis_x_max - axis_x_min, axis_z_max - axis_z_min) / max(draw_step, 1e-6))) * 2 + 1,
                ),
            ),
        )
        axis_zs = self._line_samples(axis_z_min, axis_z_max, axis_samples, include_zero=True)
        add_grid_line(
            np.column_stack([np.zeros_like(axis_zs), np.zeros_like(axis_zs), axis_zs]),
            GRID_Z_AXIS_BGR,
            True,
        )
        axis_xs = self._line_samples(axis_x_min, axis_x_max, axis_samples, include_zero=True)
        add_grid_line(
            np.column_stack([axis_xs, np.zeros_like(axis_xs), np.zeros_like(axis_xs)]),
            GRID_X_AXIS_BGR,
            True,
        )
        overlays.extend(
            self._positive_axis_marker_overlays(
                axis_x_max=axis_x_max,
                axis_z_max=axis_z_max,
                step=draw_step,
            )
        )

        foot = np.array([center[0], 0.0, center[2]], dtype=float)
        vertical = self._project_world_display_points_to_preview(np.vstack([center, foot]))
        if vertical is not None and np.all(np.isfinite(vertical)):
            overlays.append(
                PerspectiveLabelOverlay(
                    label="",
                    box=(0, 0, 0, 0),
                    origin=(0, 0),
                    color_bgr=(0, 255, 255),
                    highlighted=True,
                    polyline=tuple((float(x), float(y)) for x, y in vertical),
                    dashed=True,
                )
            )
        foot_marker = self._point_marker_overlay(foot, "XZ", (0, 255, 255), radius=7.0, world_display=True)
        if foot_marker is not None:
            overlays.append(foot_marker)
        origin_marker = self._point_marker_overlay(
            np.array([0.0, 0.0, 0.0], dtype=float),
            "O",
            (255, 80, 255),
            radius=8.0,
            world_display=True,
        )
        if origin_marker is not None:
            overlays.append(origin_marker)
        return overlays

    def _positive_axis_marker_overlays(
        self,
        *,
        axis_x_max: float,
        axis_z_max: float,
        step: float,
    ) -> list[PerspectiveLabelOverlay]:
        overlays: list[PerspectiveLabelOverlay] = []
        x_marker = self._axis_marker_distance(axis_x_max, step)
        if x_marker is not None:
            marker = self._point_marker_overlay(
                np.array([x_marker, 0.0, 0.0], dtype=float),
                "+X",
                AXIS_GIZMO_X_BGR,
                radius=5.0,
                world_display=True,
            )
            if marker is not None:
                overlays.append(marker)
        z_marker = self._axis_marker_distance(axis_z_max, step)
        if z_marker is not None:
            marker = self._point_marker_overlay(
                np.array([0.0, 0.0, z_marker], dtype=float),
                "+Z",
                AXIS_GIZMO_Z_BGR,
                radius=5.0,
                world_display=True,
            )
            if marker is not None:
                overlays.append(marker)
        return overlays

    @staticmethod
    def _axis_marker_distance(axis_max: float, step: float) -> float | None:
        if axis_max <= max(0.25, step * 0.5):
            return None
        return min(float(axis_max), max(float(step) * 2.0, float(axis_max) * 0.55))

    @staticmethod
    def _line_samples(start: float, stop: float, count: int, *, include_zero: bool) -> np.ndarray:
        samples = np.linspace(float(start), float(stop), max(2, int(count)))
        if include_zero and float(start) <= 0.0 <= float(stop) and not np.any(np.isclose(samples, 0.0)):
            samples = np.sort(np.append(samples, 0.0))
        return samples

    def _visible_ground_bounds(
        self,
        group: CubemapFrameGroup,
        center: np.ndarray,
        camera: np.ndarray,
        max_distance: float,
        step: float,
    ) -> tuple[float, float, float, float]:
        samples = np.linspace(0.0, float(self._scene_preview_size - 1), 17)
        edge_pixels = (
            [(x, 0.0) for x in samples]
            + [(x, float(self._scene_preview_size - 1)) for x in samples]
            + [(0.0, y) for y in samples[1:-1]]
            + [(float(self._scene_preview_size - 1), y) for y in samples[1:-1]]
        )
        points: list[np.ndarray] = [
            np.array([center[0], 0.0, center[2]], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([camera[0], 0.0, camera[2]], dtype=float),
        ]
        for x_px, y_px in edge_pixels:
            ray, _up, _face = self._preview_pixel_to_world_display_ray_and_up(
                group,
                x_px=x_px,
                y_px=y_px,
            )
            if abs(float(ray[1])) > 1e-8:
                distance = float(-camera[1] / ray[1])
                if 0.0 < distance <= max_distance:
                    points.append(camera + ray * distance)
                    continue
            direction_xz = np.array([ray[0], ray[2]], dtype=float)
            norm = float(np.linalg.norm(direction_xz))
            if norm > 1e-8:
                xz = np.array([camera[0], camera[2]], dtype=float) + direction_xz / norm * max_distance
                points.append(np.array([xz[0], 0.0, xz[1]], dtype=float))
        xz_points = np.asarray([[point[0], point[2]] for point in points], dtype=float)
        margin = max(step * 2.0, max_distance * 0.05)
        x_min = max(float(np.min(xz_points[:, 0]) - margin), float(camera[0] - max_distance))
        x_max = min(float(np.max(xz_points[:, 0]) + margin), float(camera[0] + max_distance))
        z_min = max(float(np.min(xz_points[:, 1]) - margin), float(camera[2] - max_distance))
        z_max = min(float(np.max(xz_points[:, 1]) + margin), float(camera[2] + max_distance))
        if x_max <= x_min:
            x_min, x_max = center[0] - max_distance, center[0] + max_distance
        if z_max <= z_min:
            z_min, z_max = center[2] - max_distance, center[2] + max_distance
        return x_min, x_max, z_min, z_max

    def _project_world_display_points_to_preview(self, points_world_display: np.ndarray) -> np.ndarray | None:
        group = self._selected_world_display_group()
        if group is None:
            return None
        if self.pointcloud_preview_check.isChecked():
            projected, _depth, valid = self._project_world_display_points_for_preview(
                group,
                np.asarray(points_world_display, dtype=np.float64),
            )
            if not np.all(valid):
                return None
            return projected
        return project_sfm_points_to_axis_preview_points(
            group,
            np.asarray(points_world_display, dtype=np.float64),
            output_size=self._scene_preview_size,
            yaw_deg=self._scene_preview_params.yaw_deg,
            pitch_deg=self._scene_preview_params.pitch_deg,
            fov_deg=self._scene_preview_params.fov_deg,
            roll_deg=self._scene_preview_params.roll_deg,
        )

    def _preview_pixel_to_world_display_ray_and_up(
        self,
        group: CubemapFrameGroup,
        *,
        x_px: float,
        y_px: float,
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        if self.pointcloud_preview_check.isChecked():
            ray = self._preview_pixel_to_local_ray(
                x_px=x_px,
                y_px=y_px,
            )
            up_ray = self._preview_pixel_to_local_ray(x_px=x_px, y_px=y_px - 1.0)
            rotation = self._preview_camera_rotation_for_display_group(group)
            ray_display = ray @ rotation.T
            ray_display /= max(float(np.linalg.norm(ray_display)), 1e-12)
            up_display_ray = up_ray @ rotation.T
            up = up_display_ray - ray_display * float(up_display_ray @ ray_display)
            up /= max(float(np.linalg.norm(up)), 1e-12)
            return ray_display, up, None
        return view_pixel_to_axis_world_ray_and_up(
            group,
            x_px=x_px,
            y_px=y_px,
            output_size=self._scene_preview_size,
            yaw_deg=self._scene_preview_params.yaw_deg,
            pitch_deg=self._scene_preview_params.pitch_deg,
            fov_deg=self._scene_preview_params.fov_deg,
            roll_deg=self._scene_preview_params.roll_deg,
        )

    def _preview_pixel_to_local_ray(self, *, x_px: float, y_px: float) -> np.ndarray:
        size = max(1, int(self._scene_preview_size))
        cx = float(x_px) - (size - 1) / 2.0
        cy = float(y_px) - (size - 1) / 2.0
        focal = 0.5 * size / np.tan(np.deg2rad(float(self._scene_preview_params.fov_deg)) / 2.0)
        ray = np.array([cx, -cy, focal], dtype=np.float64)
        ray /= max(float(np.linalg.norm(ray)), 1e-12)
        return ray @ _preview_rotation_matrix(
            self._scene_preview_params.yaw_deg,
            self._scene_preview_params.pitch_deg,
            self._scene_preview_params.roll_deg,
        ).T

    def _project_world_display_polyline_segments(self, points_world_display: np.ndarray) -> list[tuple[tuple[float, float], ...]]:
        size = int(self._scene_preview_size)
        segments: list[tuple[tuple[float, float], ...]] = []
        current: list[tuple[float, float]] = []
        for point in np.asarray(points_world_display, dtype=float):
            projected = self._project_world_display_points_to_preview(point.reshape(1, 3))
            if projected is None or not np.all(np.isfinite(projected)):
                if len(current) >= 2:
                    segments.append(tuple(current))
                current = []
                continue
            x, y = float(projected[0, 0]), float(projected[0, 1])
            if x < -size or y < -size or x > size * 2 or y > size * 2:
                if len(current) >= 2:
                    segments.append(tuple(current))
                current = []
                continue
            current.append((x, y))
        if len(current) >= 2:
            segments.append(tuple(current))
        return segments

    def _point_marker_overlay(
        self,
        point_sfm: np.ndarray,
        label: str,
        color_bgr: tuple[int, int, int],
        *,
        radius: float,
        world_display: bool = False,
    ) -> PerspectiveLabelOverlay | None:
        points = np.asarray(point_sfm, dtype=float).reshape(1, 3)
        projected = self._project_world_display_points_to_preview(points) if world_display else self._project_preview_points(points)
        if projected is None or not np.all(np.isfinite(projected)):
            return None
        x, y = (float(projected[0, 0]), float(projected[0, 1]))
        return PerspectiveLabelOverlay(
            label=label,
            box=(int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
            origin=(int(x + radius + 3), int(y - radius - 3)),
            color_bgr=color_bgr,
            highlighted=True,
            polyline=((x, y),),
            point_radius=radius,
        )

    def _tag_preview_overlays(self) -> list[PerspectiveLabelOverlay]:
        group = self._selected_group()
        if group is None:
            return []
        try:
            corners = tag_corners_sfm(
                np.asarray(self.center_editor.value(), dtype=float),
                np.asarray(self.normal_editor.value(), dtype=float),
                np.asarray(self.up_editor.value(), dtype=float),
                float(self.tag_size_spin.value()),
                float(self.true_scale_spin.value()),
            )
            points = self._project_preview_points(corners)
        except Exception:
            return []
        if points is None or not np.all(np.isfinite(points)):
            return []
        min_xy = np.floor(points.min(axis=0)).astype(int)
        max_xy = np.ceil(points.max(axis=0)).astype(int)
        size = int(self._scene_preview_size)
        if max_xy[0] < 0 or max_xy[1] < 0 or min_xy[0] > size or min_xy[1] > size:
            return []
        if int(max(max_xy - min_xy)) > size * 3:
            return []
        origin_y = max(18, int(min_xy[1]) - 8)
        return [
            PerspectiveLabelOverlay(
                label=f"tag {self.tag_id_spin.value()}",
                box=(int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])),
                origin=(int(min_xy[0]), origin_y),
                color_bgr=(0, 255, 180),
                highlighted=True,
                polygon=tuple((float(x), float(y)) for x, y in points),
                fill_alpha=0.16,
            )
        ]

    def _update_tag_preview_overlay(self) -> None:
        if self.pointcloud_preview_check.isChecked():
            self._render_scene_preview()
            return
        self.preview_label.set_perspective_label_overlays(self._preview_overlays())
        self._sync_world_debug_view()

    def _save_current_placement(self, *, show_message: bool) -> tuple[AprilTagDevCase, AprilTagPlacement, Path] | None:
        case = self._require_case()
        if case is None:
            return None
        placement = self._current_placement()
        if placement is None:
            return None
        try:
            path = save_placement(case, placement)
        except Exception as e:
            QMessageBox.critical(self, "配置保存エラー", str(e))
            return None
        self._append_log(f"Placement saved: {path}")
        if show_message:
            QMessageBox.information(self, "配置を保存", str(path))
        return case, placement, path

    def _injection_command(self, case: AprilTagDevCase, placement: AprilTagPlacement) -> list[str]:
        run_dir = run_dir_for_placement(case, placement)
        return [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "inject_synthetic_apriltag.py"),
            str(case.transforms_for_processing()),
            str(run_dir),
            "--tag-image",
            str(placement.tag_image),
            "--tag-size-m",
            str(placement.tag_size_m),
            "--true-scale",
            str(placement.true_scale),
            "--tag-center-sfm",
            self._vec_text(placement.tag_center_sfm),
            "--tag-normal-sfm",
            self._vec_text(placement.tag_normal_sfm),
            "--tag-up-sfm",
            self._vec_text(placement.tag_up_sfm),
        ]

    def _estimate_command(self, case: AprilTagDevCase, placement: AprilTagPlacement) -> list[str]:
        run_dir = run_dir_for_placement(case, placement)
        return [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "estimate_apriltag_scale.py"),
            str(run_dir / "transforms.json"),
            "--tag-size-m",
            str(placement.tag_size_m),
            "--family",
            placement.tag_family,
            "--tag-id",
            str(placement.tag_id),
            "--report-json",
            str(run_dir / "apriltag_scale_report.json"),
        ]

    def _preview_command(self, case: AprilTagDevCase, placement: AprilTagPlacement) -> list[str]:
        run_dir = run_dir_for_placement(case, placement)
        preview_path = run_dir / "preview_contact_sheet.jpg"
        csv_path = run_dir / "synthetic_injection_frames.csv"
        self._last_preview_path = preview_path
        return [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "dev_apriltag_preview.py"),
            str(run_dir),
            "--family",
            placement.tag_family,
            "--tag-size-m",
            str(placement.tag_size_m),
            "--tag-id",
            str(placement.tag_id),
            "--out",
            str(preview_path),
            "--csv",
            str(csv_path),
        ]

    @staticmethod
    def _vec_text(values: tuple[float, float, float]) -> str:
        return " ".join(f"{value:.9g}" for value in values)

    def _prepared_case_and_placement(self) -> tuple[AprilTagDevCase, AprilTagPlacement] | None:
        saved = self._save_current_placement(show_message=False)
        if saved is None:
            return None
        case, placement, _path = saved
        return case, placement

    def _run_injection(self) -> None:
        prepared = self._prepared_case_and_placement()
        if prepared is None:
            return
        case, placement = prepared
        self._enqueue([("注入画像を生成", self._injection_command(case, placement), None)])

    def _run_estimation(self) -> None:
        prepared = self._prepared_case_and_placement()
        if prepared is None:
            return
        case, placement = prepared
        self._enqueue([("推定を実行", self._estimate_command(case, placement), None)])

    def _run_preview(self) -> None:
        prepared = self._prepared_case_and_placement()
        if prepared is None:
            return
        case, placement = prepared
        self._enqueue([("プレビュー作成", self._preview_command(case, placement), self._load_preview)])

    def _run_all(self) -> None:
        prepared = self._prepared_case_and_placement()
        if prepared is None:
            return
        case, placement = prepared
        self._enqueue(
            [
                ("注入画像を生成", self._injection_command(case, placement), None),
                ("推定を実行", self._estimate_command(case, placement), None),
                ("プレビュー作成", self._preview_command(case, placement), self._load_preview),
            ]
        )

    def _enqueue(self, commands: list[tuple[str, list[str], Callable[[], None] | None]]) -> None:
        if self._process is not None:
            QMessageBox.warning(self, "実行中", "現在の処理が終わってから再実行してください。")
            return
        self._queue = list(commands)
        self._start_next_process()

    def _start_next_process(self) -> None:
        if not self._queue:
            self._set_actions_enabled(True)
            self._append_log("Done.")
            return
        label, command, callback = self._queue.pop(0)
        self._set_actions_enabled(False)
        self._append_log(f"> {label}")
        self._append_log(" ".join(f'"{part}"' if " " in part else part for part in command))
        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._append_process_output(process))
        process.finished.connect(lambda exit_code, _status: self._process_finished(exit_code, callback))
        self._process = process
        process.start()

    def _append_process_output(self, process: QProcess) -> None:
        data = bytes(process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self._append_log(data)

    def _process_finished(self, exit_code: int, callback: Callable[[], None] | None) -> None:
        process = self._process
        if process is not None:
            self._append_process_output(process)
        self._process = None
        if exit_code != 0:
            self._queue.clear()
            self._set_actions_enabled(True)
            self._append_log(f"Process failed with exit code {exit_code}.")
            return
        if callback is not None:
            callback()
        self._start_next_process()

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.create_case_btn,
            self.open_case_btn,
            self.create_printable_btn,
            self.save_placement_btn,
            self.inject_btn,
            self.estimate_btn,
            self.preview_btn,
            self.run_all_btn,
            self.open_run_btn,
        ):
            button.setEnabled(enabled)

    def _load_preview(self) -> None:
        if self._last_preview_path is None or not self._last_preview_path.is_file():
            return
        pixmap = QPixmap(str(self._last_preview_path))
        if pixmap.isNull():
            return
        self.preview_label.set_drag_mode("pan")
        self.preview_label.set_source_pixmap(pixmap)
        self._append_log(f"Preview loaded: {self._last_preview_path}")

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt API
        super().resizeEvent(event)
        if (
            self._last_preview_path
            and self._last_preview_path.is_file()
            and not self.preview_label.is_showing_gpu_perspective()
        ):
            self._load_preview()

    def _open_run_dir(self) -> None:
        prepared = self._prepared_case_and_placement()
        if prepared is None:
            return
        case, placement = prepared
        run_dir = run_dir_for_placement(case, placement)
        run_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(run_dir)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the AprilTag synthetic-injection dev GUI.")
    parser.add_argument("--case", type=Path, default=None, help="Case directory to load or prefill")
    parser.add_argument("--source-transforms", type=Path, default=None, help="Prefill source transforms.json")
    parser.add_argument("--source-pointcloud", type=Path, default=None, help="Prefill source pointcloud.ply")
    parser.add_argument("--source-metashape-xml", type=Path, default=None, help="Prefill Metashape XML")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("STECHDRIVE_APRILTAG_DEVTOOLS", "1")
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = DevAprilTagPlacerWindow(
        initial_case=args.case,
        source_transforms=args.source_transforms,
        source_pointcloud=args.source_pointcloud,
        source_metashape_xml=args.source_metashape_xml,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
