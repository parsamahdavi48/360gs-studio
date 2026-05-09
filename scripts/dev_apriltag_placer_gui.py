#!/usr/bin/env python3
"""Standalone development GUI for AprilTag synthetic-injection cases."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

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
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.apriltag_detection import available_families
from core.apriltag_geometry import tag_corners_sfm
from devtools.apriltag.case import (
    DEFAULT_CASE_ROOT,
    AprilTagDevCase,
    AprilTagPlacement,
    create_case,
    load_case,
    run_dir_for_placement,
    save_placement,
)
from devtools.apriltag.cubemap_preview import (
    CubemapFrameGroup,
    face_view_params,
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    project_sfm_points_to_preview,
    render_cubemap_equirect,
    view_pixel_to_world_ray_and_up,
)
from devtools.apriltag.printable import create_printable_target
from gui.common.browse_widget import BrowseWidget
from gui.common.perspective_image_view import PerspectiveImageView, PerspectiveLabelOverlay
from gui.common.perspective_preview import PerspectiveParams, clamp_pitch_deg, normalize_yaw_deg, params_from_drag
from gui.theme import apply_theme


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

        self.preview_label = PerspectiveImageView("プレビュー未作成")
        self.preview_label.setMinimumSize(520, 360)
        self.preview_label.setStyleSheet("background-color: #101316; border: 1px solid #3a424d;")
        self.preview_label.look_dragged.connect(self._on_scene_preview_dragged)
        self.preview_label.image_clicked.connect(self._on_scene_preview_clicked)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.preview_label)
        right_layout.addWidget(scroll, stretch=1)

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
        self.copy_images_check = QCheckBox("画像もケース内へコピーする")
        self.copy_images_check.setToolTip("通常はオフ推奨。Cubemap画像が大きい場合は参照だけにします。")
        form.addRow("ケース保存先", self.case_root_browse)
        form.addRow("ケース名", self.case_name_edit)
        form.addRow("transforms.json", self.transforms_browse)
        form.addRow("pointcloud.ply", self.pointcloud_browse)
        form.addRow("Metashape XML", self.xml_browse)
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
        row.addWidget(QLabel("カメラ位置"))
        row.addWidget(self.prev_camera_btn)
        row.addWidget(self.frame_group_combo, stretch=1)
        row.addWidget(self.next_camera_btn)
        row.addWidget(self.reload_groups_btn)
        row.addWidget(self.render_scene_preview_btn)
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

        hint = QLabel("Cubemap 6面から擬似360ビューを再構築します。ドラッグで視点回転、クリックで深度値に沿って中心SfM/法線/上方向を入力します。")
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
        self.create_printable_btn.clicked.connect(self._create_printable_target)
        self.reload_groups_btn.clicked.connect(self._load_preview_groups)
        self.render_scene_preview_btn.clicked.connect(self._render_scene_preview)
        self.frame_group_combo.currentIndexChanged.connect(lambda _index: self._render_scene_preview())
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
        self.family_combo.setCurrentText(case.tag_family)
        self.tag_id_spin.setValue(case.tag_id)
        self.tag_size_spin.setValue(case.default_tag_size_m)
        self.true_scale_spin.setValue(case.true_scale)
        self.status_label.setText(
            f"ケース: {case.case_dir}\n"
            f"入力: {case.transforms_for_processing()}\n"
            f"画像モード: {'コピー' if case.input_mode == 'copy' else '参照'}"
        )
        self._cubemap_image_cache.clear()
        self._equirect_preview_cache.clear()
        self._load_preview_groups()

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

    def _step_camera(self, delta: int) -> None:
        count = self.frame_group_combo.count()
        if count <= 0:
            return
        index = (self.frame_group_combo.currentIndex() + int(delta)) % count
        self.frame_group_combo.setCurrentIndex(index)

    def _jump_to_face(self, face: str) -> None:
        group = self._selected_group()
        if group is None:
            return
        params = face_view_params(group, face, fov_deg=self.look_fov_spin.value())
        if params is None:
            self._append_log(f"Face not available in this group: {face}")
            return
        yaw, pitch, fov = params
        self._scene_preview_params = PerspectiveParams(
            yaw_deg=normalize_yaw_deg(yaw),
            pitch_deg=clamp_pitch_deg(pitch),
            fov_deg=fov,
        )
        self._sync_preview_spins()
        self.preview_label.set_perspective_params(self._scene_preview_params)
        self._append_log(f"Jumped to face {face}: yaw={yaw:.1f}, pitch={pitch:.1f}")

    def _on_preview_spin_changed(self) -> None:
        self._scene_preview_params = PerspectiveParams(
            yaw_deg=normalize_yaw_deg(self.look_yaw_spin.value()),
            pitch_deg=clamp_pitch_deg(self.look_pitch_spin.value()),
            fov_deg=float(self.look_fov_spin.value()),
        )
        self.preview_label.set_perspective_params(self._scene_preview_params)
        self._update_tag_preview_overlay()

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
        if group is None:
            self.preview_label.setText("Cubemap画像グループがありません")
            return
        self._on_preview_spin_changed()
        try:
            image = self._equirect_preview_cache.get(group.name)
            if image is None:
                image = render_cubemap_equirect(
                    group,
                    output_width=2048,
                    output_height=1024,
                    image_cache=self._cubemap_image_cache,
                )
                self._equirect_preview_cache[group.name] = image
        except Exception as e:
            self.preview_label.setText(f"プレビュー生成エラー: {e}")
            return
        self.preview_label.set_drag_mode("look")
        shown = self.preview_label.set_perspective_image_bgr(
            image,
            self._scene_preview_params,
            overlays=self._tag_preview_overlays(),
            logical_size=QSize(self._scene_preview_size, self._scene_preview_size),
        )
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

    def _on_scene_preview_dragged(self, delta_x: float, delta_y: float) -> None:
        if not self._cubemap_groups:
            return
        self._scene_preview_params = params_from_drag(self._scene_preview_params, delta_x, delta_y)
        self._sync_preview_spins()
        self.preview_label.set_perspective_params(self._scene_preview_params)
        self._update_tag_preview_overlay()

    def _on_scene_preview_clicked(self, x: float, y: float) -> None:
        if not self.place_click_check.isChecked():
            self._append_log("Preview click ignored: enable 'クリックで配置を更新' to move the tag.")
            return
        group = self._selected_group()
        if group is None:
            return
        size = float(self._scene_preview_size)
        x_px = max(0.0, min(size - 1.0, x))
        y_px = max(0.0, min(size - 1.0, y))
        ray, up, face = view_pixel_to_world_ray_and_up(
            group,
            x_px=x_px,
            y_px=y_px,
            output_size=self._scene_preview_size,
            yaw_deg=self._scene_preview_params.yaw_deg,
            pitch_deg=self._scene_preview_params.pitch_deg,
            fov_deg=self._scene_preview_params.fov_deg,
        )
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
            points = project_sfm_points_to_preview(
                group,
                corners,
                output_size=self._scene_preview_size,
                yaw_deg=self._scene_preview_params.yaw_deg,
                pitch_deg=self._scene_preview_params.pitch_deg,
                fov_deg=self._scene_preview_params.fov_deg,
            )
        except Exception:
            return []
        if points is None or not np.all(np.isfinite(points)):
            return []
        min_xy = np.floor(points.min(axis=0)).astype(int)
        max_xy = np.ceil(points.max(axis=0)).astype(int)
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
        self.preview_label.set_perspective_label_overlays(self._tag_preview_overlays())

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
