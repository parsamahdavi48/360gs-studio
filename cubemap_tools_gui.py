#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from PySide6.QtCore import QProcess, Qt, QTimer
    from PySide6.QtGui import QCloseEvent, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSlider,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    QProcess = None
    Qt = None
    QTimer = None
    QCloseEvent = None
    QImage = None
    QPixmap = None
    QApplication = None
    QCheckBox = None
    QComboBox = None
    QFileDialog = None
    QFormLayout = None
    QGridLayout = None
    QHBoxLayout = None
    QLabel = None
    QLineEdit = None
    QMainWindow = None
    QMessageBox = None
    QPlainTextEdit = None
    QPushButton = None
    QSlider = None
    QTabWidget = None
    QVBoxLayout = None
    QWidget = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None


_CONVERT_RE = re.compile(r"^Converting\s+(\d+)\s+images\.\.\.$")
_MIN_YAW_SLOTS = 4
_MAX_YAW_SLOTS = 8
_DEFAULT_YAW_SLOTS = 6
_MAX_PITCH_ROWS = 9
_WARN_ENABLED_VIEWS = 24
_BLOCK_ENABLED_VIEWS = 40
_PROFILE_POSTSHOT = "postshot"
_PROFILE_LICHTFELD = "lichtfeld"
_PROFILE_CUSTOM = "custom"
_VIEW_MODE_CUSTOM = "custom_views"
_VIEW_MODE_CUBE6 = "cube6"
_TASK_MODE_CUBEMAP = "cubemap"
_TASK_MODE_COLMAP = "colmap"


def _normalize_angle(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    r = ry @ rx
    r[np.abs(r) < 1e-10] = 0.0
    return r


if QMainWindow is not None:
    class CubemapToolsWindow(QMainWindow):
        def __init__(self, initial_scene_dir: str | None = None) -> None:
            super().__init__()
            self.base_dir = Path(__file__).resolve().parent
            self.proc: QProcess | None = None
            self.current_phase = ""
            self._process_buffer = ""
            self._converted_total = 0
            self._processed_sources = 0
            self._preview_pixmap: QPixmap | None = None
            self.preview_images: list[Path] = []
            self._preview_slider_sync = False
            self.cancel_requested = False
            self.pending_steps: list[tuple[str, list[str]]] = []

            self.pitch_rows: list[dict] = []
            self.yaw_slot_labels: list[QLabel] = []
            self._preprocess_widgets: list[QWidget] = []
            self._preprocess_ply_path_widgets: list[QWidget] = []
            self._custom_view_widgets: list[QWidget] = []
            self._cube6_widgets: list[QWidget] = []
            self._colmap_sequential_widgets: list[QWidget] = []
            self._colmap_preset_sync = False

            self.setWindowTitle("Cubemap Tools")
            self.resize(1380, 940)

            self._build_ui(initial_scene_dir or ".")
            self._apply_scene_paths()
            self._apply_pitch_rows()
            self._refresh_action_buttons()
            self._render_preview()

        def _build_ui(self, initial_scene_dir: str) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            form = QFormLayout()

            self.scene_dir_edit = QLineEdit(initial_scene_dir)
            self.scene_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            self.scene_dir_edit.textChanged.connect(lambda _: self._update_output_estimate())
            browse_scene_btn = QPushButton("Browse")
            browse_scene_btn.clicked.connect(self._browse_scene_dir)
            apply_scene_btn = QPushButton("Apply Scene Paths")
            apply_scene_btn.clicked.connect(self._apply_scene_paths)
            row = QHBoxLayout()
            row.addWidget(self.scene_dir_edit)
            row.addWidget(browse_scene_btn)
            row.addWidget(apply_scene_btn)
            form.addRow("Scene Directory", row)

            self.output_dir_edit = QLineEdit()
            self.output_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            browse_output_btn = QPushButton("Browse")
            browse_output_btn.clicked.connect(self._browse_output_dir)
            row = QHBoxLayout()
            row.addWidget(self.output_dir_edit)
            row.addWidget(browse_output_btn)
            form.addRow("Output Directory", row)

            self.json_name_edit = QLineEdit("transforms.json")
            form.addRow("Transforms JSON", self.json_name_edit)

            self.target_profile_combo = QComboBox()
            self.target_profile_combo.addItem("Postshot / Brush", _PROFILE_POSTSHOT)
            self.target_profile_combo.addItem("LichtFeld Studio", _PROFILE_LICHTFELD)
            self.target_profile_combo.addItem("Custom (manual)", _PROFILE_CUSTOM)
            form.addRow("Target Profile", self.target_profile_combo)

            self.target_profile_hint = QLabel("")
            self.target_profile_hint.setWordWrap(True)
            form.addRow("Profile Hint", self.target_profile_hint)

            self.preprocess_enable_check = QCheckBox("Run metashape_360_lfs preprocess")
            self.preprocess_enable_check.setChecked(True)
            self.preprocess_enable_check.toggled.connect(self._on_preprocess_toggle)
            form.addRow("Preprocess", self.preprocess_enable_check)

            self.ms_images_edit = QLineEdit()
            browse_ms_images_btn = QPushButton("Browse")
            browse_ms_images_btn.clicked.connect(self._browse_ms_images_dir)
            row = QHBoxLayout()
            row.addWidget(self.ms_images_edit)
            row.addWidget(browse_ms_images_btn)
            form.addRow("MS Images Dir", row)
            self._preprocess_widgets.extend([self.ms_images_edit, browse_ms_images_btn])

            self.ms_xml_edit = QLineEdit()
            browse_ms_xml_btn = QPushButton("Browse")
            browse_ms_xml_btn.clicked.connect(self._browse_ms_xml_file)
            row = QHBoxLayout()
            row.addWidget(self.ms_xml_edit)
            row.addWidget(browse_ms_xml_btn)
            form.addRow("MS XML", row)
            self._preprocess_widgets.extend([self.ms_xml_edit, browse_ms_xml_btn])

            self.ms_ply_edit = QLineEdit()
            browse_ms_ply_btn = QPushButton("Browse")
            browse_ms_ply_btn.clicked.connect(self._browse_ms_ply_file)
            clear_ms_ply_btn = QPushButton("Clear")
            clear_ms_ply_btn.clicked.connect(lambda: self.ms_ply_edit.setText(""))
            row = QHBoxLayout()
            row.addWidget(self.ms_ply_edit)
            row.addWidget(browse_ms_ply_btn)
            row.addWidget(clear_ms_ply_btn)
            form.addRow("MS PLY (optional)", row)
            self._preprocess_ply_path_widgets.extend([self.ms_ply_edit, browse_ms_ply_btn, clear_ms_ply_btn])
            self._preprocess_widgets.extend([self.ms_ply_edit, browse_ms_ply_btn, clear_ms_ply_btn])

            self.ms_use_ply_check = QCheckBox("Include PLY in preprocess (--ply)")
            self.ms_use_ply_check.setChecked(False)
            self.ms_use_ply_check.toggled.connect(self._on_ms_use_ply_toggle)
            form.addRow("MS PLY Usage", self.ms_use_ply_check)
            self._preprocess_widgets.append(self.ms_use_ply_check)

            self.ms_scale_edit = QLineEdit("1.0")
            form.addRow("MS Scale", self.ms_scale_edit)
            self._preprocess_widgets.append(self.ms_scale_edit)

            self.ms_no_fix_rotation_check = QCheckBox("Disable rotation fix (--no-fix-rotation)")
            self.ms_no_fix_rotation_check.setChecked(False)
            form.addRow("MS Options", self.ms_no_fix_rotation_check)
            self._preprocess_widgets.append(self.ms_no_fix_rotation_check)

            self.mask_dir_edit = QLineEdit()
            self.mask_dir_edit.textChanged.connect(lambda _: self._render_preview())
            browse_mask_btn = QPushButton("Browse")
            browse_mask_btn.clicked.connect(self._browse_mask_dir)
            row = QHBoxLayout()
            row.addWidget(self.mask_dir_edit)
            row.addWidget(browse_mask_btn)
            form.addRow("Mask Directory", row)

            self.view_mode_combo = QComboBox()
            self.view_mode_combo.addItem("Custom Pitch/Yaw", _VIEW_MODE_CUSTOM)
            self.view_mode_combo.addItem("Cube6 (4 sides + top/bottom)", _VIEW_MODE_CUBE6)
            form.addRow("View Mode", self.view_mode_combo)

            self.yaw_offset_edit = QLineEdit("45.0")
            self.yaw_offset_edit.textChanged.connect(self._on_yaw_or_pitch_changed)
            form.addRow("Yaw Offset (deg)", self.yaw_offset_edit)

            self.yaw_slots_combo = QComboBox()
            self.yaw_slots_combo.addItems([str(v) for v in range(_MIN_YAW_SLOTS, _MAX_YAW_SLOTS + 1)])
            self.yaw_slots_combo.setCurrentText(str(_DEFAULT_YAW_SLOTS))
            self.yaw_slots_combo.currentTextChanged.connect(lambda _: self._on_yaw_slots_changed())
            form.addRow("Yaw Slots", self.yaw_slots_combo)
            self._custom_view_widgets.append(self.yaw_slots_combo)

            self.pitch_list_edit = QLineEdit("-30,0,30")
            form.addRow("Pitch Rows (deg CSV)", self.pitch_list_edit)
            self._custom_view_widgets.append(self.pitch_list_edit)

            cube6_row = QHBoxLayout()
            self.cube6_drop_top_check = QCheckBox("Drop Top (+90deg)")
            self.cube6_drop_top_check.setChecked(False)
            self.cube6_drop_top_check.toggled.connect(self._on_view_selection_changed)
            cube6_row.addWidget(self.cube6_drop_top_check)
            self._cube6_widgets.append(self.cube6_drop_top_check)
            self.cube6_drop_bottom_check = QCheckBox("Drop Bottom (-90deg)")
            self.cube6_drop_bottom_check.setChecked(False)
            self.cube6_drop_bottom_check.toggled.connect(self._on_view_selection_changed)
            cube6_row.addWidget(self.cube6_drop_bottom_check)
            self._cube6_widgets.append(self.cube6_drop_bottom_check)
            cube6_row.addStretch(1)
            form.addRow("Cube6 Options", cube6_row)

            self.fov_label = QLabel("90.0 (fixed)")
            form.addRow("FOV", self.fov_label)

            self.sample_image_edit = QLineEdit()
            self.sample_image_edit.textChanged.connect(self._on_sample_image_changed)
            browse_sample_btn = QPushButton("Browse")
            browse_sample_btn.clicked.connect(self._browse_sample_image)
            auto_sample_btn = QPushButton("Auto")
            auto_sample_btn.clicked.connect(self._auto_select_sample_image)
            reload_sample_btn = QPushButton("Reload")
            reload_sample_btn.clicked.connect(self._refresh_preview_image_list)
            row = QHBoxLayout()
            row.addWidget(self.sample_image_edit)
            row.addWidget(browse_sample_btn)
            row.addWidget(auto_sample_btn)
            row.addWidget(reload_sample_btn)
            form.addRow("Preview Image", row)

            self.preview_timeline_slider = QSlider(Qt.Horizontal)
            self.preview_timeline_slider.setRange(0, 0)
            self.preview_timeline_slider.setValue(0)
            self.preview_timeline_slider.setEnabled(False)
            self.preview_timeline_slider.valueChanged.connect(self._on_preview_timeline_changed)
            self.preview_timeline_label = QLabel("0 / 0")
            timeline_row = QHBoxLayout()
            timeline_row.addWidget(self.preview_timeline_slider, stretch=1)
            timeline_row.addWidget(self.preview_timeline_label)
            form.addRow("Preview Timeline", timeline_row)

            self.mask_overlay_slider = QSlider(Qt.Horizontal)
            self.mask_overlay_slider.setRange(0, 100)
            self.mask_overlay_slider.setValue(35)
            self.mask_overlay_slider.valueChanged.connect(lambda _: self._render_preview())
            form.addRow("Mask Overlay (%)", self.mask_overlay_slider)

            self.preview_mask_edit = QLineEdit()
            self.preview_mask_edit.textChanged.connect(lambda _: self._render_preview())
            browse_preview_mask_btn = QPushButton("Browse")
            browse_preview_mask_btn.clicked.connect(self._browse_preview_mask_image)
            clear_preview_mask_btn = QPushButton("Clear")
            clear_preview_mask_btn.clicked.connect(self._clear_preview_mask_image)
            row = QHBoxLayout()
            row.addWidget(self.preview_mask_edit)
            row.addWidget(browse_preview_mask_btn)
            row.addWidget(clear_preview_mask_btn)
            form.addRow("Preview Mask Image", row)

            options_row = QHBoxLayout()
            self.mask_from_alpha_check = QCheckBox("Extract mask from alpha (--mask_from_alpha)")
            self.mask_from_alpha_check.setChecked(False)
            options_row.addWidget(self.mask_from_alpha_check)

            self.no_image_check = QCheckBox("Transforms only (--no_image)")
            self.no_image_check.setChecked(False)
            options_row.addWidget(self.no_image_check)

            self.no_transform_check = QCheckBox("No axis transform (--no_transform)")
            self.no_transform_check.setChecked(False)
            options_row.addWidget(self.no_transform_check)

            self.duplicate_check = QCheckBox("Allow duplicate (--duplicate)")
            self.duplicate_check.setChecked(False)
            options_row.addWidget(self.duplicate_check)

            self.invert_masks_check = QCheckBox("Invert masks (--invert_masks)")
            self.invert_masks_check.setChecked(False)
            options_row.addWidget(self.invert_masks_check)

            options_row.addStretch(1)
            form.addRow("Options", options_row)

            layout.addLayout(form)

            self.workflow_tabs = QTabWidget()
            cubemap_tab = QWidget()
            cubemap_tab_layout = QVBoxLayout(cubemap_tab)
            cubemap_hint = QLabel(
                "Cubemap tab: Run existing preprocess + cubemap conversion pipeline.\n"
                "COLMAP Rig tab: Export COLMAP rig dataset and optionally run SfM."
            )
            cubemap_hint.setWordWrap(True)
            cubemap_tab_layout.addWidget(cubemap_hint)
            cubemap_tab_layout.addStretch(1)
            self.workflow_tabs.addTab(cubemap_tab, "Cubemap")

            colmap_tab = QWidget()
            colmap_form = QFormLayout(colmap_tab)

            self.colmap_output_dir_edit = QLineEdit()
            self.colmap_output_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            self.colmap_output_dir_edit.textChanged.connect(lambda _: self._update_colmap_workspace_hint())
            browse_colmap_output_btn = QPushButton("Browse")
            browse_colmap_output_btn.clicked.connect(self._browse_colmap_output_dir)
            row = QHBoxLayout()
            row.addWidget(self.colmap_output_dir_edit)
            row.addWidget(browse_colmap_output_btn)
            colmap_form.addRow("COLMAP Output Root", row)

            self.colmap_repo_dir_edit = QLineEdit()
            browse_colmap_repo_btn = QPushButton("Browse")
            browse_colmap_repo_btn.clicked.connect(self._browse_colmap_repo_dir)
            row = QHBoxLayout()
            row.addWidget(self.colmap_repo_dir_edit)
            row.addWidget(browse_colmap_repo_btn)
            colmap_form.addRow("COLMAP Repo (optional)", row)

            self.colmap_binary_edit = QLineEdit()
            self.colmap_binary_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            browse_colmap_bin_btn = QPushButton("Browse")
            browse_colmap_bin_btn.clicked.connect(self._browse_colmap_binary)
            auto_colmap_bin_btn = QPushButton("Auto Detect")
            auto_colmap_bin_btn.clicked.connect(self._auto_detect_colmap_binary)
            row = QHBoxLayout()
            row.addWidget(self.colmap_binary_edit)
            row.addWidget(browse_colmap_bin_btn)
            row.addWidget(auto_colmap_bin_btn)
            colmap_form.addRow("COLMAP Binary", row)

            self.colmap_matcher_combo = QComboBox()
            self.colmap_matcher_combo.addItem("Exhaustive Matcher", "exhaustive")
            self.colmap_matcher_combo.addItem("Sequential Matcher", "sequential")
            self.colmap_matcher_combo.currentIndexChanged.connect(self._on_colmap_matcher_changed)
            colmap_form.addRow("Matcher", self.colmap_matcher_combo)

            self.colmap_preset_combo = QComboBox()
            self.colmap_preset_combo.addItem("Fast (Recommended)", "fast")
            self.colmap_preset_combo.addItem("Balanced", "balanced")
            self.colmap_preset_combo.addItem("Loop-Heavy", "loop_heavy")
            self.colmap_preset_combo.addItem("Custom", "custom")
            self.colmap_preset_combo.currentIndexChanged.connect(self._on_colmap_preset_changed)
            colmap_form.addRow("Pipeline Preset", self.colmap_preset_combo)

            self.colmap_seq_overlap_edit = QLineEdit("6")
            self.colmap_seq_overlap_edit.textChanged.connect(self._on_colmap_advanced_changed)
            colmap_form.addRow("Sequential Overlap", self.colmap_seq_overlap_edit)
            self._colmap_sequential_widgets.append(self.colmap_seq_overlap_edit)

            self.colmap_sift_max_features_edit = QLineEdit("4096")
            self.colmap_sift_max_features_edit.textChanged.connect(self._on_colmap_advanced_changed)
            colmap_form.addRow("SIFT Max Features", self.colmap_sift_max_features_edit)

            self.colmap_mapper_ba_global_iter_edit = QLineEdit("20")
            self.colmap_mapper_ba_global_iter_edit.textChanged.connect(self._on_colmap_advanced_changed)
            colmap_form.addRow("Mapper BA Global Iter", self.colmap_mapper_ba_global_iter_edit)

            self.colmap_loop_detection_check = QCheckBox("Sequential loop detection (optional)")
            self.colmap_loop_detection_check.setChecked(False)
            self.colmap_loop_detection_check.toggled.connect(self._on_colmap_advanced_changed)
            colmap_form.addRow("Loop Detection", self.colmap_loop_detection_check)
            self._colmap_sequential_widgets.append(self.colmap_loop_detection_check)

            self.colmap_vocab_tree_edit = QLineEdit()
            self.colmap_vocab_tree_edit.textChanged.connect(self._on_colmap_advanced_changed)
            browse_vocab_btn = QPushButton("Browse")
            browse_vocab_btn.clicked.connect(self._browse_colmap_vocab_tree)
            clear_vocab_btn = QPushButton("Clear")
            clear_vocab_btn.clicked.connect(lambda: self.colmap_vocab_tree_edit.setText(""))
            row = QHBoxLayout()
            row.addWidget(self.colmap_vocab_tree_edit)
            row.addWidget(browse_vocab_btn)
            row.addWidget(clear_vocab_btn)
            colmap_form.addRow("Vocab Tree (optional)", row)
            self._colmap_sequential_widgets.extend([self.colmap_vocab_tree_edit, browse_vocab_btn, clear_vocab_btn])

            self.colmap_run_until_combo = QComboBox()
            self.colmap_run_until_combo.addItem("Export Only", "export")
            self.colmap_run_until_combo.addItem("Feature Extractor", "feature")
            self.colmap_run_until_combo.addItem("Rig Configurator", "rig")
            self.colmap_run_until_combo.addItem("Matcher", "match")
            self.colmap_run_until_combo.addItem("Mapper (SfM)", "mapper")
            self.colmap_run_until_combo.setCurrentIndex(4)
            self.colmap_run_until_combo.currentIndexChanged.connect(lambda _: self._refresh_action_buttons())
            colmap_form.addRow("Run Until", self.colmap_run_until_combo)

            self.colmap_use_masks_check = QCheckBox("Use masks in feature extraction")
            self.colmap_use_masks_check.setChecked(True)
            self.colmap_use_masks_check.toggled.connect(self._on_colmap_advanced_changed)
            colmap_form.addRow("Mask Usage", self.colmap_use_masks_check)

            self.colmap_invert_masks_check = QCheckBox("Invert exported masks")
            self.colmap_invert_masks_check.setChecked(False)
            colmap_form.addRow("Mask Export", self.colmap_invert_masks_check)

            self.colmap_refine_sensor_check = QCheckBox("Mapper: refine sensor from rig")
            self.colmap_refine_sensor_check.setChecked(False)
            self.colmap_refine_sensor_check.toggled.connect(self._on_colmap_advanced_changed)
            colmap_form.addRow("Mapper Option", self.colmap_refine_sensor_check)

            self.colmap_workspace_hint = QLabel("")
            self.colmap_workspace_hint.setWordWrap(True)
            colmap_form.addRow("Output Layout", self.colmap_workspace_hint)

            self.workflow_tabs.addTab(colmap_tab, "COLMAP Rig SfM")
            self.workflow_tabs.currentChanged.connect(self._on_workflow_tab_changed)
            layout.addWidget(self.workflow_tabs)

            view_ctrl_row = QHBoxLayout()
            self.apply_pitch_btn = QPushButton("Apply Pitch Rows")
            self.apply_pitch_btn.clicked.connect(self._apply_pitch_rows)
            view_ctrl_row.addWidget(self.apply_pitch_btn)
            self._custom_view_widgets.append(self.apply_pitch_btn)

            self.all_on_btn = QPushButton("All On")
            self.all_on_btn.clicked.connect(self._all_on)
            view_ctrl_row.addWidget(self.all_on_btn)
            self._custom_view_widgets.append(self.all_on_btn)

            self.all_off_btn = QPushButton("All Off")
            self.all_off_btn.clicked.connect(self._all_off)
            view_ctrl_row.addWidget(self.all_off_btn)
            self._custom_view_widgets.append(self.all_off_btn)

            refresh_preview_btn = QPushButton("Refresh Preview")
            refresh_preview_btn.clicked.connect(self._render_preview)
            view_ctrl_row.addWidget(refresh_preview_btn)

            view_ctrl_row.addStretch(1)

            self.selected_views_label = QLabel("Selected views: 0")
            self.selected_views_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            view_ctrl_row.addWidget(self.selected_views_label)
            layout.addLayout(view_ctrl_row)

            self.estimate_label = QLabel("Estimated output images: -")
            self.estimate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(self.estimate_label)

            self.view_grid_widget = QWidget()
            self.view_grid_layout = QGridLayout(self.view_grid_widget)
            layout.addWidget(self.view_grid_widget)
            self._custom_view_widgets.append(self.view_grid_widget)

            self.preview_label = QLabel("Preview unavailable")
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setMinimumSize(960, 440)
            self.preview_label.setStyleSheet("border: 1px solid palette(mid);")
            layout.addWidget(self.preview_label, stretch=1)

            btn_row = QHBoxLayout()
            self.run_button = QPushButton("Run Cubemap Convert")
            self.run_button.clicked.connect(self._run_convert)
            btn_row.addWidget(self.run_button)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._cancel_running_process)
            self.cancel_button.setEnabled(False)
            btn_row.addWidget(self.cancel_button)
            btn_row.addStretch(1)

            self.status_label = QLabel("Idle")
            self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            btn_row.addWidget(self.status_label)
            layout.addLayout(btn_row)

            self.log_text = QPlainTextEdit()
            self.log_text.setReadOnly(True)
            layout.addWidget(self.log_text, stretch=1)

            self.target_profile_combo.currentIndexChanged.connect(self._on_target_profile_changed)
            self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
            self._on_preprocess_toggle(self.preprocess_enable_check.isChecked())
            self._on_target_profile_changed(self.target_profile_combo.currentIndex())
            self._on_view_mode_changed(self.view_mode_combo.currentIndex())
            self._on_colmap_preset_changed(self.colmap_preset_combo.currentIndex())
            self._on_colmap_matcher_changed(self.colmap_matcher_combo.currentIndex())
            self._on_workflow_tab_changed(self.workflow_tabs.currentIndex())

        def _is_running(self) -> bool:
            return self.proc is not None and self.proc.state() != QProcess.NotRunning

        def _append_log(self, text: str) -> None:
            self.log_text.appendPlainText(text)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_running_state(self, running: bool, status_text: str) -> None:
            self.status_label.setText(status_text)
            if running:
                self.run_button.setEnabled(False)
                self.cancel_button.setEnabled(True)
            else:
                self.cancel_button.setEnabled(False)
                self._refresh_action_buttons()

        def _refresh_action_buttons(self) -> None:
            if not hasattr(self, "run_button") or not hasattr(self, "cancel_button"):
                return
            if self._is_running():
                self.run_button.setEnabled(False)
                self.cancel_button.setEnabled(True)
                return
            scene_ok = Path(self.scene_dir_edit.text().strip()).is_dir()
            if self._task_mode_id() == _TASK_MODE_COLMAP:
                if not hasattr(self, "colmap_output_dir_edit") or not hasattr(self, "colmap_binary_edit"):
                    self.run_button.setEnabled(scene_ok)
                    self.cancel_button.setEnabled(False)
                    return
                output_ok = bool(self.colmap_output_dir_edit.text().strip())
                run_until = self._colmap_run_until_id()
                needs_colmap = run_until != "export"
                colmap_ok = True
                if needs_colmap:
                    colmap_text = self.colmap_binary_edit.text().strip()
                    if colmap_text:
                        colmap_ok = Path(colmap_text).is_file()
                    else:
                        colmap_ok = self._guess_colmap_binary() is not None
                self.run_button.setEnabled(scene_ok and output_ok and colmap_ok)
            else:
                self.run_button.setEnabled(scene_ok)
            self.cancel_button.setEnabled(False)

        def _target_profile_id(self) -> str:
            data = self.target_profile_combo.currentData()
            if isinstance(data, str) and data:
                return data
            return _PROFILE_POSTSHOT

        def _view_mode_id(self) -> str:
            data = self.view_mode_combo.currentData()
            if isinstance(data, str) and data:
                return data
            return _VIEW_MODE_CUSTOM

        def _task_mode_id(self) -> str:
            if not hasattr(self, "workflow_tabs") or self.workflow_tabs is None:
                return _TASK_MODE_CUBEMAP
            idx = self.workflow_tabs.currentIndex()
            if idx == 1:
                return _TASK_MODE_COLMAP
            return _TASK_MODE_CUBEMAP

        def _colmap_run_until_id(self) -> str:
            if not hasattr(self, "colmap_run_until_combo"):
                return "mapper"
            data = self.colmap_run_until_combo.currentData()
            if isinstance(data, str) and data:
                return data
            return "mapper"

        def _colmap_preset_id(self) -> str:
            if not hasattr(self, "colmap_preset_combo"):
                return "fast"
            data = self.colmap_preset_combo.currentData()
            if isinstance(data, str) and data:
                return data
            return "fast"

        def _set_colmap_preset_custom(self) -> None:
            if not hasattr(self, "colmap_preset_combo"):
                return
            if self._colmap_preset_id() == "custom":
                return
            self._colmap_preset_sync = True
            self.colmap_preset_combo.setCurrentIndex(self.colmap_preset_combo.count() - 1)
            self._colmap_preset_sync = False

        def _on_colmap_advanced_changed(self, *_args) -> None:
            if self._colmap_preset_sync:
                return
            self._set_colmap_preset_custom()
            self._refresh_action_buttons()

        def _on_colmap_preset_changed(self, _index: int) -> None:
            if self._colmap_preset_sync:
                return
            preset = self._colmap_preset_id()
            if preset == "custom":
                self._refresh_action_buttons()
                return

            values = {
                "fast": {
                    "matcher": "sequential",
                    "overlap": "6",
                    "sift": "4096",
                    "ba": "20",
                    "loop": False,
                    "use_masks": True,
                    "refine_sensor": False,
                },
                "balanced": {
                    "matcher": "sequential",
                    "overlap": "10",
                    "sift": "8192",
                    "ba": "35",
                    "loop": False,
                    "use_masks": True,
                    "refine_sensor": False,
                },
                "loop_heavy": {
                    "matcher": "sequential",
                    "overlap": "18",
                    "sift": "12000",
                    "ba": "55",
                    "loop": False,
                    "use_masks": True,
                    "refine_sensor": False,
                },
            }.get(preset)
            if values is None:
                self._refresh_action_buttons()
                return

            self._colmap_preset_sync = True
            idx = self.colmap_matcher_combo.findData(values["matcher"])
            if idx >= 0:
                self.colmap_matcher_combo.setCurrentIndex(idx)
            self.colmap_seq_overlap_edit.setText(values["overlap"])
            self.colmap_sift_max_features_edit.setText(values["sift"])
            self.colmap_mapper_ba_global_iter_edit.setText(values["ba"])
            self.colmap_loop_detection_check.setChecked(bool(values["loop"]))
            self.colmap_use_masks_check.setChecked(bool(values["use_masks"]))
            self.colmap_refine_sensor_check.setChecked(bool(values["refine_sensor"]))
            self._on_colmap_matcher_changed(self.colmap_matcher_combo.currentIndex())
            self._colmap_preset_sync = False
            self._refresh_action_buttons()

        def _on_colmap_matcher_changed(self, _index: int) -> None:
            matcher = self.colmap_matcher_combo.currentData()
            is_sequential = matcher == "sequential"
            for w in self._colmap_sequential_widgets:
                w.setEnabled(is_sequential)
            if not self._colmap_preset_sync:
                self._set_colmap_preset_custom()
            self._refresh_action_buttons()

        def _effective_bundle_profile(self) -> str:
            profile = self._target_profile_id()
            if profile in {_PROFILE_POSTSHOT, _PROFILE_LICHTFELD}:
                return profile
            if self.no_transform_check.isChecked() or self.ms_use_ply_check.isChecked():
                return _PROFILE_LICHTFELD
            return _PROFILE_POSTSHOT

        def _on_workflow_tab_changed(self, _index: int) -> None:
            if not hasattr(self, "run_button"):
                return
            is_colmap = self._task_mode_id() == _TASK_MODE_COLMAP
            self.run_button.setText("Run COLMAP Rig Pipeline" if is_colmap else "Run Cubemap Convert")
            self._update_colmap_workspace_hint()
            self._refresh_action_buttons()

        @staticmethod
        def _guess_colmap_binary(repo_dir: Path | None = None) -> Path | None:
            candidates: list[Path] = []
            if repo_dir is not None and str(repo_dir).strip():
                root = repo_dir.resolve()
                candidates.extend(
                    [
                        root / "build" / "src" / "exe" / "colmap",
                        root / "build" / "src" / "exe" / "Release" / "colmap.exe",
                        root / "build" / "src" / "colmap",
                        root / "build" / "src" / "colmap.exe",
                        root / "build" / "src" / "exe" / "Debug" / "colmap.exe",
                    ]
                )

            which_hit = shutil.which("colmap")
            if which_hit:
                candidates.append(Path(which_hit))

            for c in candidates:
                if c.is_file():
                    return c
            return None

        def _update_colmap_workspace_hint(self) -> None:
            if not hasattr(self, "colmap_workspace_hint"):
                return
            root = Path(self.colmap_output_dir_edit.text().strip() or ".")
            dataset = root / "dataset"
            workspace = root / "workspace"
            self.colmap_workspace_hint.setText(
                "All COLMAP files are kept under this root:\n"
                f"- {dataset}\n"
                f"- {workspace / 'database.db'}\n"
                f"- {workspace / 'sparse'}"
            )

        def _on_view_mode_changed(self, _index: int) -> None:
            is_custom = self._view_mode_id() == _VIEW_MODE_CUSTOM
            for w in self._custom_view_widgets:
                w.setEnabled(is_custom)
                if w is self.view_grid_widget:
                    w.setVisible(is_custom)
            for w in self._cube6_widgets:
                w.setEnabled(not is_custom)
            self._update_selected_views_label()
            self._render_preview()

        def _refresh_preprocess_ply_path_widgets(self) -> None:
            enabled = self.preprocess_enable_check.isChecked() and self.ms_use_ply_check.isChecked()
            for w in self._preprocess_ply_path_widgets:
                w.setEnabled(enabled)

        def _on_ms_use_ply_toggle(self, _enabled: bool) -> None:
            self._refresh_preprocess_ply_path_widgets()

        def _on_target_profile_changed(self, _index: int) -> None:
            profile = self._target_profile_id()
            if profile == _PROFILE_LICHTFELD:
                self.no_transform_check.setChecked(True)
                self.ms_use_ply_check.setChecked(True)
                self.no_transform_check.setEnabled(False)
                self.ms_use_ply_check.setEnabled(False)
                self.target_profile_hint.setText(
                    "LichtFeld preset (locked): --no_transform ON, preprocess --ply ON."
                )
            elif profile == _PROFILE_POSTSHOT:
                self.no_transform_check.setChecked(False)
                self.ms_use_ply_check.setChecked(False)
                self.no_transform_check.setEnabled(False)
                self.ms_use_ply_check.setEnabled(False)
                self.target_profile_hint.setText(
                    "Postshot/Brush preset (locked): --no_transform OFF, preprocess --ply OFF."
                )
            else:
                self.no_transform_check.setEnabled(True)
                self.ms_use_ply_check.setEnabled(True)
                self.target_profile_hint.setText(
                    "Custom profile: edit --no_transform and preprocess --ply manually."
                )
            if not self.preprocess_enable_check.isChecked():
                self.ms_use_ply_check.setEnabled(False)
            self._refresh_preprocess_ply_path_widgets()

        def _on_preprocess_toggle(self, enabled: bool) -> None:
            for w in self._preprocess_widgets:
                w.setEnabled(enabled)
            if enabled:
                profile = self._target_profile_id()
                if profile in {_PROFILE_POSTSHOT, _PROFILE_LICHTFELD}:
                    self.ms_use_ply_check.setEnabled(False)
            self._refresh_preprocess_ply_path_widgets()

        def _browse_scene_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select scene directory")
            if path:
                self.scene_dir_edit.setText(path)
                self._refresh_preview_image_list(prefer_current=False)

        def _browse_output_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select output directory")
            if path:
                self.output_dir_edit.setText(path)

        def _browse_colmap_output_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select COLMAP output root")
            if path:
                self.colmap_output_dir_edit.setText(path)
                self._update_colmap_workspace_hint()

        def _browse_colmap_repo_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select COLMAP repository directory")
            if path:
                self.colmap_repo_dir_edit.setText(path)

        def _browse_colmap_binary(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select COLMAP binary",
                "",
                "Executable files (*.exe *.bin *);;All files (*.*)",
            )
            if path:
                self.colmap_binary_edit.setText(path)

        def _auto_detect_colmap_binary(self) -> None:
            repo_text = self.colmap_repo_dir_edit.text().strip()
            repo_dir = Path(repo_text) if repo_text else None
            hit = self._guess_colmap_binary(repo_dir)
            if hit is None:
                QMessageBox.warning(
                    self,
                    "COLMAP Not Found",
                    "Could not auto-detect COLMAP binary from the given repo or PATH.",
                )
                return
            self.colmap_binary_edit.setText(str(hit))

        def _browse_colmap_vocab_tree(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select vocabulary tree file",
                "",
                "Vocab files (*.bin *.txt);;All files (*.*)",
            )
            if path:
                self.colmap_vocab_tree_edit.setText(path)

        def _browse_mask_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select mask directory")
            if path:
                self.mask_dir_edit.setText(path)

        def _browse_ms_images_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select Metashape images directory")
            if path:
                self.ms_images_edit.setText(path)

        def _browse_ms_xml_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Metashape XML",
                "",
                "XML files (*.xml);;All files (*.*)",
            )
            if path:
                self.ms_xml_edit.setText(path)

        def _browse_ms_ply_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Metashape PLY (optional)",
                "",
                "PLY files (*.ply);;All files (*.*)",
            )
            if path:
                self.ms_ply_edit.setText(path)

        @staticmethod
        def _guess_metashape_xml(scene_dir: Path) -> Path:
            candidates = [scene_dir / "metashape.xml", scene_dir / "cameras.xml"]
            for c in candidates:
                if c.is_file():
                    return c
            xmls = sorted([p for p in scene_dir.glob("*.xml") if p.is_file()], key=lambda x: x.name.lower())
            if xmls:
                return xmls[0]
            return scene_dir / "metashape.xml"

        @staticmethod
        def _guess_metashape_ply(scene_dir: Path) -> str:
            candidates = [scene_dir / "metashape.ply", scene_dir / "sparse.ply", scene_dir / "pointcloud.ply"]
            for c in candidates:
                if c.is_file():
                    return str(c)
            plys = sorted([p for p in scene_dir.glob("*.ply") if p.is_file()], key=lambda x: x.name.lower())
            if plys:
                return str(plys[0])
            return ""

        @staticmethod
        def _first_existing(paths: list[Path]) -> Path | None:
            for p in paths:
                if p.is_file():
                    return p
            return None

        @staticmethod
        def _resolve_postshot_ply_source(scene_dir: Path) -> Path | None:
            preferred = [scene_dir / "metashape.ply", scene_dir / "sparse.ply"]
            hit = CubemapToolsWindow._first_existing(preferred)
            if hit is not None:
                return hit

            plys = sorted([p for p in scene_dir.glob("*.ply") if p.is_file()], key=lambda x: x.name.lower())
            non_pointcloud = [p for p in plys if p.name.lower() != "pointcloud.ply"]
            if len(non_pointcloud) == 1:
                return non_pointcloud[0]
            if non_pointcloud:
                return non_pointcloud[0]
            if len(plys) == 1 and plys[0].name.lower() != "pointcloud.ply":
                return plys[0]
            return None

        @staticmethod
        def _resolve_lichtfeld_ply_source(scene_dir: Path) -> Path | None:
            preferred = [scene_dir / "pointcloud.ply"]
            hit = CubemapToolsWindow._first_existing(preferred)
            if hit is not None:
                return hit

            plys = sorted([p for p in scene_dir.glob("*.ply") if p.is_file()], key=lambda x: x.name.lower())
            named = [p for p in plys if "pointcloud" in p.name.lower()]
            if named:
                return named[0]
            if len(plys) == 1:
                return plys[0]
            return None

        def _resolve_profile_ply_source(self) -> Path | None:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            profile = self._effective_bundle_profile()
            if profile == _PROFILE_LICHTFELD:
                return self._resolve_lichtfeld_ply_source(scene_dir)
            return self._resolve_postshot_ply_source(scene_dir)

        def _validate_bundle_requirements(self) -> None:
            profile = self._effective_bundle_profile()
            source = self._resolve_profile_ply_source()
            if source is not None:
                return

            if profile == _PROFILE_LICHTFELD and self.preprocess_enable_check.isChecked() and self.ms_use_ply_check.isChecked():
                # pointcloud.ply will be generated by preprocess step.
                return

            if profile == _PROFILE_LICHTFELD:
                raise ValueError(
                    "LichtFeld profile requires pointcloud.ply in Scene Directory, "
                    "or enable preprocess with MS PLY Usage."
                )
            raise ValueError(
                "Postshot profile requires Metashape PLY in Scene Directory "
                "(e.g. metashape.ply or sparse.ply)."
            )

        def _rewrite_output_ply_file_path(self, output_dir: Path, ply_name: str) -> None:
            transforms_path = output_dir / "transforms.json"
            if not transforms_path.is_file():
                raise FileNotFoundError(f"Output transforms.json not found: {transforms_path}")
            data = json.loads(transforms_path.read_text(encoding="utf-8"))
            data["ply_file_path"] = ply_name
            transforms_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        def _finalize_output_bundle(self) -> None:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            output_dir = Path(self.output_dir_edit.text().strip() or (scene_dir / "cubic"))
            output_dir.mkdir(parents=True, exist_ok=True)

            source = self._resolve_profile_ply_source()
            if source is None:
                raise FileNotFoundError("No source PLY found for selected target profile")

            dest = output_dir / source.name
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
                self._append_log(f"[bundle] copied PLY: {source} -> {dest}")
            else:
                self._append_log(f"[bundle] PLY already in output: {dest}")

            self._rewrite_output_ply_file_path(output_dir, dest.name)
            self._append_log(f"[bundle] set transforms.json ply_file_path: {dest.name}")

        def _browse_sample_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select preview image",
                "",
                "Image files (*.jpg *.jpeg *.png);;All files (*.*)",
            )
            if path:
                self.sample_image_edit.setText(path)

        def _browse_preview_mask_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select preview mask image",
                "",
                "Image files (*.png *.jpg *.jpeg);;All files (*.*)",
            )
            if path:
                self.preview_mask_edit.setText(path)

        def _clear_preview_mask_image(self) -> None:
            self.preview_mask_edit.setText("")

        def _apply_scene_paths(self) -> None:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            self.output_dir_edit.setText(str(scene_dir / "cubic"))
            self.mask_dir_edit.setText(str(scene_dir / "masks"))
            self.ms_images_edit.setText(str(scene_dir / "images"))
            self.ms_xml_edit.setText(str(self._guess_metashape_xml(scene_dir)))
            self.ms_ply_edit.setText(self._guess_metashape_ply(scene_dir))
            self.colmap_output_dir_edit.setText(str(scene_dir / "colmap_rig"))
            if not self.colmap_binary_edit.text().strip():
                hit = self._guess_colmap_binary()
                if hit is not None:
                    self.colmap_binary_edit.setText(str(hit))
            self._update_colmap_workspace_hint()
            self._refresh_preview_image_list(prefer_current=False)
            self._refresh_action_buttons()
            self._update_output_estimate()

        def _auto_select_sample_image(self) -> None:
            if not self.preview_images:
                self._refresh_preview_image_list(prefer_current=False)
                return
            self._set_preview_index(0)

        def _iter_scene_preview_images(self) -> list[Path]:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            images_dir = scene_dir / "images"
            roots = [images_dir] if images_dir.is_dir() else [scene_dir]
            exts = {".jpg", ".jpeg", ".png"}

            candidates: list[Path] = []
            seen: set[str] = set()
            for root in roots:
                if not root.is_dir():
                    continue
                for p in root.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in exts:
                        continue
                    key = str(p.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(p)
            candidates.sort(key=lambda x: str(x).lower())
            return candidates

        def _update_preview_timeline_label(self, idx: int = -1) -> None:
            total = len(self.preview_images)
            if total <= 0:
                self.preview_timeline_label.setText("0 / 0")
                return
            if idx < 0:
                try:
                    idx = self.preview_timeline_slider.value()
                except Exception:
                    idx = 0
            idx = max(0, min(idx, total - 1))
            name = self.preview_images[idx].name
            self.preview_timeline_label.setText(f"{idx + 1} / {total} : {name}")

        def _set_preview_index(self, idx: int) -> None:
            if not self.preview_images:
                self.sample_image_edit.setText("")
                self._update_preview_timeline_label(-1)
                return
            idx = max(0, min(idx, len(self.preview_images) - 1))
            self._preview_slider_sync = True
            self.preview_timeline_slider.setValue(idx)
            self._preview_slider_sync = False
            self.sample_image_edit.setText(str(self.preview_images[idx]))
            self._update_preview_timeline_label(idx)

        def _sync_preview_slider_from_sample(self) -> None:
            sample_text = self.sample_image_edit.text().strip()
            if not sample_text:
                self._update_preview_timeline_label(-1)
                return
            try:
                sample_key = str(Path(sample_text).resolve()).lower()
            except Exception:
                self._update_preview_timeline_label(-1)
                return

            for idx, path in enumerate(self.preview_images):
                if str(path.resolve()).lower() == sample_key:
                    self._preview_slider_sync = True
                    self.preview_timeline_slider.setValue(idx)
                    self._preview_slider_sync = False
                    self._update_preview_timeline_label(idx)
                    return

            total = len(self.preview_images)
            if total > 0:
                self.preview_timeline_label.setText(f"custom / {total} : {Path(sample_text).name}")
            else:
                self.preview_timeline_label.setText("custom / 0")

        def _refresh_preview_image_list(self, prefer_current: bool = True) -> None:
            current_text = self.sample_image_edit.text().strip()
            self.preview_images = self._iter_scene_preview_images()
            total = len(self.preview_images)

            self.preview_timeline_slider.setEnabled(total > 0)
            self.preview_timeline_slider.setRange(0, max(0, total - 1))

            if total <= 0:
                self.preview_timeline_slider.setValue(0)
                self._update_preview_timeline_label(-1)
                if not current_text:
                    self.sample_image_edit.setText("")
                self._render_preview()
                return

            target_idx = 0
            if prefer_current and current_text:
                try:
                    current_key = str(Path(current_text).resolve()).lower()
                    for idx, p in enumerate(self.preview_images):
                        if str(p.resolve()).lower() == current_key:
                            target_idx = idx
                            break
                except Exception:
                    pass

            self._set_preview_index(target_idx)
            self._render_preview()

        def _on_preview_timeline_changed(self, idx: int) -> None:
            if self._preview_slider_sync:
                return
            if idx < 0 or idx >= len(self.preview_images):
                self._update_preview_timeline_label(-1)
                return
            self._set_preview_index(idx)

        def _on_sample_image_changed(self, _text: str) -> None:
            self._sync_preview_slider_from_sample()
            self._render_preview()

        @staticmethod
        def _parse_float(text: str, label: str) -> float:
            try:
                return float(text.strip())
            except Exception as e:
                raise ValueError(f"{label} is invalid: {e}") from e

        @staticmethod
        def _parse_int(text: str, label: str) -> int:
            try:
                return int(text.strip())
            except Exception as e:
                raise ValueError(f"{label} is invalid: {e}") from e

        def _yaw_slot_count(self) -> int:
            text = self.yaw_slots_combo.currentText().strip()
            try:
                value = int(text)
            except Exception as e:
                raise ValueError(f"Yaw Slots is invalid: {e}") from e
            if value < _MIN_YAW_SLOTS or value > _MAX_YAW_SLOTS:
                raise ValueError(f"Yaw Slots must be in [{_MIN_YAW_SLOTS}, {_MAX_YAW_SLOTS}]")
            return value

        def _yaw_step_deg(self) -> float:
            return 360.0 / float(self._yaw_slot_count())

        def _count_input_images(self) -> int:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            images_dir = scene_dir / "images"

            roots: list[Path] = [images_dir] if images_dir.is_dir() else [scene_dir]
            exts = {".jpg", ".jpeg", ".png"}
            seen: set[str] = set()
            count = 0
            for root in roots:
                if not root.is_dir():
                    continue
                for p in root.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in exts:
                        continue
                    key = str(p.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    count += 1
            return count

        def _update_output_estimate(self) -> None:
            try:
                views = self._collect_views(include_disabled=True)
            except Exception:
                self.estimate_label.setText("Estimated output images: -")
                return

            enabled = sum(1 for v in views if v["enabled"])
            source_count = self._count_input_images()
            estimated = source_count * enabled
            warn = ""
            if enabled > _BLOCK_ENABLED_VIEWS:
                warn = " [too many selected]"
            elif enabled > _WARN_ENABLED_VIEWS:
                warn = " [high]"
            self.estimate_label.setText(
                f"Estimated output images: {estimated} ({source_count} x {enabled}){warn}"
            )

        def _parse_pitches(self) -> list[float]:
            raw = self.pitch_list_edit.text().strip()
            if not raw:
                raise ValueError("Pitch rows is empty")

            tokens = [t for t in re.split(r"[,\s]+", raw) if t]
            pitches: list[float] = []
            seen: set[str] = set()
            for token in tokens:
                p = float(token)
                if p < -90.0 or p > 90.0:
                    raise ValueError(f"Pitch out of range [-90, 90]: {p}")
                key = f"{p:.6f}"
                if key in seen:
                    continue
                seen.add(key)
                pitches.append(p)

            if not pitches:
                raise ValueError("No valid pitches")
            if len(pitches) > _MAX_PITCH_ROWS:
                raise ValueError(f"Pitch rows are limited to {_MAX_PITCH_ROWS}")
            return pitches

        def _on_yaw_slots_changed(self) -> None:
            self._apply_pitch_rows()

        def _clear_view_grid(self) -> None:
            while self.view_grid_layout.count():
                item = self.view_grid_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def _on_yaw_or_pitch_changed(self) -> None:
            self._update_yaw_slot_labels()
            self._update_selected_views_label()
            self._render_preview()

        def _apply_pitch_rows(self) -> None:
            old_state: dict[str, list[bool]] = {}
            for row in self.pitch_rows:
                key = f"{row['pitch']:.6f}"
                old_state[key] = [cb.isChecked() for cb in row["checks"]]

            try:
                pitches = self._parse_pitches()
            except Exception as e:
                QMessageBox.critical(self, "Pitch Error", str(e))
                return

            slot_count = self._yaw_slot_count()
            self.pitch_rows = []
            self.yaw_slot_labels = []
            self._clear_view_grid()

            head_pitch = QLabel("Pitch / Slot")
            self.view_grid_layout.addWidget(head_pitch, 0, 0)

            for slot in range(slot_count):
                lab = QLabel(f"S{slot}")
                lab.setAlignment(Qt.AlignCenter)
                self.yaw_slot_labels.append(lab)
                self.view_grid_layout.addWidget(lab, 0, slot + 1)

            for row_idx, pitch in enumerate(pitches, start=1):
                pitch_label = QLabel(f"{pitch:g} deg")
                pitch_label.setAlignment(Qt.AlignCenter)
                self.view_grid_layout.addWidget(pitch_label, row_idx, 0)

                checks: list[QCheckBox] = []
                key = f"{pitch:.6f}"
                restored = old_state.get(key)
                for slot in range(slot_count):
                    cb = QCheckBox()
                    if restored is not None and slot < len(restored):
                        checked = restored[slot]
                    else:
                        checked = abs(pitch) < 1e-6
                    cb.setChecked(checked)
                    cb.toggled.connect(self._on_view_selection_changed)
                    self.view_grid_layout.addWidget(cb, row_idx, slot + 1, alignment=Qt.AlignCenter)
                    checks.append(cb)

                self.pitch_rows.append({"pitch": pitch, "checks": checks})

            self._update_yaw_slot_labels()
            self._update_selected_views_label()
            self._render_preview()

        def _all_on(self) -> None:
            for row in self.pitch_rows:
                for cb in row["checks"]:
                    cb.setChecked(True)
            self._on_view_selection_changed()

        def _all_off(self) -> None:
            for row in self.pitch_rows:
                for cb in row["checks"]:
                    cb.setChecked(False)
            self._on_view_selection_changed()

        def _on_view_selection_changed(self, *_args) -> None:
            self._update_selected_views_label()
            self._render_preview()

        def _update_yaw_slot_labels(self) -> None:
            try:
                yaw_offset = self._parse_float(self.yaw_offset_edit.text(), "Yaw Offset")
                yaw_step = self._yaw_step_deg()
            except Exception:
                for idx, lab in enumerate(self.yaw_slot_labels):
                    lab.setText(f"S{idx}")
                return

            for idx, lab in enumerate(self.yaw_slot_labels):
                yaw = _normalize_angle(yaw_offset + idx * yaw_step)
                lab.setText(f"S{idx}\n{yaw:.1f}deg")

        def _angle_token(self, angle: float) -> str:
            sign = "p" if angle >= 0 else "m"
            value = abs(angle)
            token = f"{value:g}".replace(".", "d")
            return f"{sign}{token}"

        def _cube6_views(self, yaw_offset: float) -> list[dict]:
            drop_top = self.cube6_drop_top_check.isChecked()
            drop_bottom = self.cube6_drop_bottom_check.isChecked()
            return [
                {"name": "px", "yaw": 90.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 0, "label": "px"},
                {"name": "nx", "yaw": -90.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 1, "label": "nx"},
                {"name": "pz", "yaw": 0.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 2, "label": "pz"},
                {"name": "nz", "yaw": 180.0 - yaw_offset, "pitch": 0.0, "enabled": True, "slot": 3, "label": "nz"},
                {
                    "name": "top",
                    "yaw": 0.0 - yaw_offset,
                    "pitch": 90.0,
                    "enabled": (not drop_top),
                    "slot": 4,
                    "label": "top",
                },
                {
                    "name": "bottom",
                    "yaw": 0.0 - yaw_offset,
                    "pitch": -90.0,
                    "enabled": (not drop_bottom),
                    "slot": 5,
                    "label": "bottom",
                },
            ]

        def _collect_views(self, include_disabled: bool) -> list[dict]:
            yaw_offset = self._parse_float(self.yaw_offset_edit.text(), "Yaw Offset")
            if self._view_mode_id() == _VIEW_MODE_CUBE6:
                views = self._cube6_views(yaw_offset)
                if include_disabled:
                    return views
                return [v for v in views if v["enabled"]]

            yaw_step = self._yaw_step_deg()
            views: list[dict] = []

            for row in self.pitch_rows:
                pitch = float(row["pitch"])
                for slot, cb in enumerate(row["checks"]):
                    enabled = cb.isChecked()
                    if not include_disabled and not enabled:
                        continue
                    views.append(
                        {
                            "name": f"pit{self._angle_token(pitch)}_s{slot}",
                            "yaw": float(yaw_offset + slot * yaw_step),
                            "pitch": pitch,
                            "enabled": enabled,
                            "slot": slot,
                            "label": f"p{pitch:g}/s{slot}",
                        }
                    )
            return views

        def _update_selected_views_label(self) -> None:
            try:
                views = self._collect_views(include_disabled=True)
                mode = self._view_mode_id()
                slot_count = self._yaw_slot_count()
            except Exception:
                self.selected_views_label.setText("Selected views: -")
                self._update_output_estimate()
                return

            selected = sum(1 for v in views if v["enabled"])
            warn = ""
            if selected > _BLOCK_ENABLED_VIEWS:
                warn = " [limit exceeded]"
            elif selected > _WARN_ENABLED_VIEWS:
                warn = " [high]"
            if mode == _VIEW_MODE_CUBE6:
                self.selected_views_label.setText(f"Selected views: {selected} / {len(views)} (cube6){warn}")
            else:
                self.selected_views_label.setText(
                    f"Selected views: {selected} / {len(views)} (slots={slot_count}){warn}"
                )
            self._update_output_estimate()

        def _find_mask_for_sample(self, sample_path: Path) -> Path | None:
            mask_dir = Path(self.mask_dir_edit.text().strip())
            if not mask_dir.is_dir():
                return None

            candidates: list[Path] = [
                mask_dir / sample_path.name,
                mask_dir / f"{sample_path.name}.png",
                mask_dir / f"{sample_path.stem}.png",
            ]

            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            images_root = scene_dir / "images"
            if sample_path.is_absolute() and images_root.is_dir():
                try:
                    rel = sample_path.relative_to(images_root)
                    candidates.extend(
                        [
                            mask_dir / rel,
                            mask_dir / rel.with_suffix(".png"),
                            mask_dir / f"{rel.name}.png",
                        ]
                    )
                except Exception:
                    pass

            for c in candidates:
                if c.exists() and c.is_file():
                    return c
            return None

        def _resolve_preview_mask_path(self, sample_path: Path) -> Path | None:
            manual = self.preview_mask_edit.text().strip()
            if manual:
                manual_path = Path(manual)
                if manual_path.exists() and manual_path.is_file():
                    return manual_path
            return self._find_mask_for_sample(sample_path)

        def _view_boundary_segments(
            self,
            width: int,
            height: int,
            yaw_deg: float,
            pitch_deg: float,
            fov_deg: float,
            samples_per_edge: int = 120,
        ) -> list[np.ndarray]:
            t = np.linspace(-1.0, 1.0, samples_per_edge)
            u = np.concatenate([
                t,
                np.ones_like(t),
                t[::-1],
                -np.ones_like(t),
                np.array([-1.0]),
            ])
            v = np.concatenate([
                np.ones_like(t),
                t[::-1],
                -np.ones_like(t),
                t,
                np.array([1.0]),
            ])

            focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
            rays = np.stack([u, v, np.full_like(u, focal)], axis=-1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

            r = _rotation_matrix(yaw_deg, pitch_deg)
            rays = rays @ r.T

            lon = np.arctan2(rays[:, 0], rays[:, 2])
            lat = np.arcsin(np.clip(rays[:, 1], -1.0, 1.0))

            x = (lon / np.pi + 1.0) * 0.5 * width
            y = (0.5 - lat / np.pi) * height
            points = np.stack([x, y], axis=-1)

            segments: list[np.ndarray] = []
            current: list[np.ndarray] = [points[0]]
            threshold = width * 0.45
            for p in points[1:]:
                if abs(float(p[0] - current[-1][0])) > threshold:
                    if len(current) >= 2:
                        segments.append(np.array(current, dtype=np.float32))
                    current = [p]
                else:
                    current.append(p)
            if len(current) >= 2:
                segments.append(np.array(current, dtype=np.float32))
            return segments

        def _render_preview(self) -> None:
            sample_text = self.sample_image_edit.text().strip()
            if not sample_text:
                self.preview_label.setText("Preview image not selected")
                self._preview_pixmap = None
                return

            sample_path = Path(sample_text)
            if not sample_path.exists() or not sample_path.is_file():
                self.preview_label.setText("Preview image not found")
                self._preview_pixmap = None
                return

            img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
            if img is None:
                self.preview_label.setText("Failed to load preview image")
                self._preview_pixmap = None
                return

            max_width = 1900
            if img.shape[1] > max_width:
                scale = max_width / float(img.shape[1])
                img = cv2.resize(
                    img,
                    (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            mask_path = self._resolve_preview_mask_path(sample_path)
            if mask_path is not None:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    alpha = float(self.mask_overlay_slider.value()) / 100.0
                    if alpha > 0:
                        masked = mask < 128
                        overlay = np.zeros_like(img)
                        overlay[:, :, 2] = 255
                        img[masked] = (
                            (1.0 - alpha) * img[masked].astype(np.float32)
                            + alpha * overlay[masked].astype(np.float32)
                        ).astype(np.uint8)

            try:
                views = self._collect_views(include_disabled=True)
            except Exception:
                views = []

            h, w = img.shape[:2]
            for view in views:
                color = (0, 220, 255) if view["enabled"] else (128, 128, 128)
                thickness = 2 if view["enabled"] else 1
                segments = self._view_boundary_segments(w, h, view["yaw"], view["pitch"], 90.0)
                all_points: list[np.ndarray] = []
                for seg in segments:
                    if len(seg) < 2:
                        continue
                    pts = np.round(seg).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)
                    all_points.append(seg)

                if view["enabled"] and all_points:
                    merged = np.concatenate(all_points, axis=0)
                    cx = int(np.clip(np.mean(merged[:, 0]), 0, w - 1))
                    cy = int(np.clip(np.mean(merged[:, 1]), 0, h - 1))
                    label = str(view.get("label", f"p{view['pitch']:g}/s{view['slot']}"))
                    cv2.putText(
                        img,
                        label,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        lineType=cv2.LINE_AA,
                    )

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
            self._preview_pixmap = QPixmap.fromImage(qimg)
            self._update_preview_pixmap()

        def _update_preview_pixmap(self) -> None:
            if self._preview_pixmap is None:
                return
            scaled = self._preview_pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)

        def resizeEvent(self, event) -> None:  # pragma: no cover - UI event
            super().resizeEvent(event)
            self._update_preview_pixmap()

        def _write_views_config(self, output_dir: Path, views: list[dict]) -> Path:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "views_config.json"
            payload = {
                "fov": 90.0,
                "views": [
                    {
                        "name": v["name"],
                        "yaw": float(v["yaw"]),
                        "pitch": float(v["pitch"]),
                        "enabled": bool(v["enabled"]),
                    }
                    for v in views
                ],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path

        def _build_preprocess_cmd(self) -> list[str]:
            script = self.base_dir / "vendor" / "metashape_360_lfs" / "metashape_360_lfs.py"
            if not script.exists():
                raise FileNotFoundError(f"metashape_360_lfs.py not found: {script}")

            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            if not scene_dir.is_dir():
                raise ValueError(f"Scene directory not found: {scene_dir}")

            images_dir = Path(self.ms_images_edit.text().strip())
            if not images_dir.is_dir():
                raise ValueError(f"MS Images Dir not found: {images_dir}")

            xml_path = Path(self.ms_xml_edit.text().strip())
            if not xml_path.is_file():
                raise ValueError(f"MS XML not found: {xml_path}")

            scale = self._parse_float(self.ms_scale_edit.text(), "MS Scale")
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("MS Scale must be a positive finite number")

            cmd = [
                sys.executable,
                str(script),
                "--images",
                str(images_dir),
                "--xml",
                str(xml_path),
                "--output",
                str(scene_dir),
                "--scale",
                f"{scale:g}",
            ]

            if self.ms_use_ply_check.isChecked():
                ply_text = self.ms_ply_edit.text().strip()
                if not ply_text:
                    raise ValueError("MS PLY Usage is enabled, but MS PLY path is empty")
                ply_path = Path(ply_text)
                if not ply_path.is_file():
                    raise ValueError(f"MS PLY not found: {ply_path}")
                cmd.extend(["--ply", str(ply_path)])

            if self.ms_no_fix_rotation_check.isChecked():
                cmd.append("--no-fix-rotation")

            return cmd

        def _build_cmd(self) -> list[str]:
            script = self.base_dir / "cubemap_transforms_json.py"
            if not script.exists():
                raise FileNotFoundError(f"cubemap_transforms_json.py not found: {script}")

            input_dir = Path(self.scene_dir_edit.text().strip() or ".")
            if not input_dir.is_dir():
                raise ValueError(f"Scene directory not found: {input_dir}")

            output_dir = Path(self.output_dir_edit.text().strip() or (input_dir / "cubic"))
            json_name = self.json_name_edit.text().strip() or "transforms.json"

            all_views = self._collect_views(include_disabled=True)
            enabled_count = sum(1 for v in all_views if v["enabled"])
            if enabled_count <= 0:
                raise ValueError("At least one view must be enabled")
            if enabled_count > _BLOCK_ENABLED_VIEWS:
                raise ValueError(
                    f"Too many enabled views ({enabled_count}). "
                    f"Reduce to <= {_BLOCK_ENABLED_VIEWS}."
                )

            views_json = self._write_views_config(output_dir, all_views)

            cmd = [
                sys.executable,
                str(script),
                str(input_dir),
                str(output_dir),
                "--json",
                json_name,
                "--fov",
                "90",
                "--views-json",
                str(views_json),
            ]

            mask_dir = Path(self.mask_dir_edit.text().strip())
            if mask_dir.is_dir():
                cmd.extend(["--mask_dir", str(mask_dir)])

            if self.mask_from_alpha_check.isChecked():
                cmd.append("--mask_from_alpha")
            if self.no_image_check.isChecked():
                cmd.append("--no_image")
            if self.no_transform_check.isChecked():
                cmd.append("--no_transform")
            if self.duplicate_check.isChecked():
                cmd.append("--duplicate")
            if self.invert_masks_check.isChecked():
                cmd.append("--invert_masks")

            return cmd

        def _build_colmap_export_cmd(self) -> list[str]:
            script = self.base_dir / "colmap_rig_export.py"
            if not script.exists():
                raise FileNotFoundError(f"colmap_rig_export.py not found: {script}")

            input_dir = Path(self.scene_dir_edit.text().strip() or ".")
            if not input_dir.is_dir():
                raise ValueError(f"Scene directory not found: {input_dir}")

            output_dir_text = self.colmap_output_dir_edit.text().strip()
            if not output_dir_text:
                raise ValueError("COLMAP Output Root is empty")
            output_dir = Path(output_dir_text)

            all_views = self._collect_views(include_disabled=True)
            enabled_count = sum(1 for v in all_views if v["enabled"])
            if enabled_count <= 0:
                raise ValueError("At least one view must be enabled")
            if enabled_count > _BLOCK_ENABLED_VIEWS:
                raise ValueError(
                    f"Too many enabled views ({enabled_count}). "
                    f"Reduce to <= {_BLOCK_ENABLED_VIEWS}."
                )

            views_json = self._write_views_config(output_dir, all_views)
            json_name = self.json_name_edit.text().strip() or "transforms.json"

            cmd = [
                sys.executable,
                str(script),
                str(input_dir),
                str(output_dir),
                "--json",
                json_name,
                "--fov",
                "90",
                "--views-json",
                str(views_json),
            ]

            mask_dir = Path(self.mask_dir_edit.text().strip())
            if mask_dir.is_dir():
                cmd.extend(["--mask_dir", str(mask_dir)])

            if self.mask_from_alpha_check.isChecked():
                cmd.append("--mask_from_alpha")
            if self.duplicate_check.isChecked():
                cmd.append("--duplicate")
            if self.colmap_invert_masks_check.isChecked():
                cmd.append("--invert_masks")
            return cmd

        def _build_colmap_pipeline_cmd(self) -> list[str]:
            script = self.base_dir / "colmap_rig_pipeline.py"
            if not script.exists():
                raise FileNotFoundError(f"colmap_rig_pipeline.py not found: {script}")

            output_dir_text = self.colmap_output_dir_edit.text().strip()
            if not output_dir_text:
                raise ValueError("COLMAP Output Root is empty")

            colmap_bin = self.colmap_binary_edit.text().strip()
            if not colmap_bin:
                hit = self._guess_colmap_binary()
                if hit is None:
                    raise ValueError("COLMAP binary is not set and could not be auto-detected from PATH")
                colmap_bin = str(hit)
                self.colmap_binary_edit.setText(colmap_bin)
            elif not Path(colmap_bin).is_file():
                raise ValueError(f"COLMAP binary not found: {colmap_bin}")

            matcher = self.colmap_matcher_combo.currentData()
            if not isinstance(matcher, str) or not matcher:
                matcher = "sequential"

            run_until = self._colmap_run_until_id()
            if run_until not in {"feature", "rig", "match", "mapper"}:
                raise ValueError(f"Unsupported COLMAP run stage: {run_until}")

            sift_max_features = self._parse_int(self.colmap_sift_max_features_edit.text(), "SIFT Max Features")
            if sift_max_features <= 0:
                raise ValueError("SIFT Max Features must be > 0")

            mapper_ba_global_iter = self._parse_int(
                self.colmap_mapper_ba_global_iter_edit.text(),
                "Mapper BA Global Iter",
            )
            if mapper_ba_global_iter <= 0:
                raise ValueError("Mapper BA Global Iter must be > 0")

            cmd = [
                sys.executable,
                str(script),
                str(output_dir_text),
                "--colmap_bin",
                colmap_bin,
                "--matcher",
                matcher,
                "--run_until",
                run_until,
                "--sift_max_features",
                str(sift_max_features),
                "--mapper_ba_global_max_iter",
                str(mapper_ba_global_iter),
            ]
            if matcher == "sequential":
                seq_overlap = self._parse_int(self.colmap_seq_overlap_edit.text(), "Sequential Overlap")
                if seq_overlap <= 0:
                    raise ValueError("Sequential Overlap must be > 0")
                cmd.extend(["--seq_overlap", str(seq_overlap)])

                if self.colmap_loop_detection_check.isChecked():
                    cmd.append("--seq_loop_detection")
                vocab_tree = self.colmap_vocab_tree_edit.text().strip()
                if vocab_tree:
                    vocab_path = Path(vocab_tree)
                    if not vocab_path.is_file():
                        raise ValueError(f"Vocab Tree not found: {vocab_path}")
                    cmd.extend(["--vocab_tree_path", str(vocab_path)])
            if self.colmap_use_masks_check.isChecked():
                cmd.append("--use_masks")
            if self.colmap_refine_sensor_check.isChecked():
                cmd.append("--refine_sensor_from_rig")
            return cmd

        def _start_step_queue(self, steps: list[tuple[str, list[str]]]) -> None:
            if self._is_running():
                QMessageBox.warning(self, "Busy", "Another process is running.")
                return
            if not steps:
                return
            self.cancel_requested = False
            self.pending_steps = list(steps)
            self._run_next_step()

        def _run_next_step(self) -> None:
            if not self.pending_steps:
                self._set_running_state(False, "Idle")
                return
            phase, cmd = self.pending_steps.pop(0)
            self._start_process(cmd, phase)

        def _start_process(self, cmd: list[str], phase: str) -> None:
            if self._is_running():
                QMessageBox.warning(self, "Busy", "Another process is running.")
                return

            self.cancel_requested = False
            self.current_phase = phase
            self._process_buffer = ""
            self._converted_total = 0
            self._processed_sources = 0
            self._append_log("$ " + " ".join(cmd))

            proc = QProcess(self)
            proc.setProgram(cmd[0])
            proc.setArguments(cmd[1:])
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(self._on_process_output)
            proc.errorOccurred.connect(self._on_process_error)
            proc.finished.connect(self._on_process_finished)
            self.proc = proc

            self._set_running_state(True, f"Running: {phase}")
            proc.start()

        def _terminate_process_gracefully(self, proc: QProcess, phase: str, timeout_ms: int = 3000) -> None:
            if proc.state() == QProcess.NotRunning:
                return
            self._append_log(f"[{phase}] cancel requested; sending terminate")
            proc.terminate()
            QTimer.singleShot(timeout_ms, lambda p=proc, ph=phase: self._force_kill_if_running(p, ph))

        def _force_kill_if_running(self, proc: QProcess, phase: str) -> None:
            if proc.state() == QProcess.NotRunning:
                return
            self._append_log(f"[{phase}] terminate timeout; killing process")
            proc.kill()

        def _cancel_running_process(self) -> None:
            if not self._is_running() or self.proc is None:
                return
            self.cancel_requested = True
            self.pending_steps = []
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Canceling...")
            self._terminate_process_gracefully(self.proc, self.current_phase)

        def _handle_process_line(self, line: str) -> None:
            if not line:
                return
            self._append_log(line)

            m = _CONVERT_RE.match(line)
            if m:
                self._converted_total = int(m.group(1))
                self._processed_sources = 0
                self.status_label.setText(f"Running: convert (0/{self._converted_total})")
                return

            if line.startswith("Processing:"):
                self._processed_sources += 1
                if self._converted_total > 0:
                    self.status_label.setText(
                        f"Running: convert ({min(self._processed_sources, self._converted_total)}/{self._converted_total})"
                    )
                else:
                    self.status_label.setText(f"Running: convert ({self._processed_sources})")

        def _on_process_output(self) -> None:
            if self.proc is None:
                return
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            data = data.replace("\r", "\n")
            self._process_buffer += data
            while "\n" in self._process_buffer:
                line, self._process_buffer = self._process_buffer.split("\n", 1)
                self._handle_process_line(line.rstrip("\r"))

        def _on_process_error(self, _error) -> None:
            if self.proc is None:
                return
            self._append_log(f"[{self.current_phase}] process error occurred")

        def _on_process_finished(self, exit_code: int, _status) -> None:
            if self._process_buffer:
                tail = self._process_buffer.replace("\r", "\n").strip()
                if tail:
                    for line in tail.splitlines():
                        self._handle_process_line(line)
                self._process_buffer = ""

            phase = self.current_phase
            was_canceled = self.cancel_requested
            self.cancel_requested = False
            if exit_code == 0:
                if was_canceled:
                    self._append_log(f"[{phase}] canceled by user")
                    self.pending_steps = []
                    self._set_running_state(False, f"Canceled: {phase}")
                else:
                    self._append_log(f"[{phase}] completed successfully")
                    self.proc = None
                    if self.pending_steps:
                        self._run_next_step()
                        return
                    if phase == "cubemap":
                        try:
                            self._finalize_output_bundle()
                        except Exception as e:
                            self.pending_steps = []
                            self._append_log(f"[bundle] failed: {e}")
                            self._set_running_state(False, "Failed: bundle")
                            return
                    self._set_running_state(False, f"Done: {phase}")
            else:
                if was_canceled:
                    self._append_log(f"[{phase}] canceled by user")
                    self.pending_steps = []
                    self._set_running_state(False, f"Canceled: {phase}")
                else:
                    self._append_log(f"[{phase}] failed (exit={exit_code})")
                    self.pending_steps = []
                    self._set_running_state(False, f"Failed: {phase}")
            self.proc = None

        def _run_convert(self) -> None:
            try:
                all_views = self._collect_views(include_disabled=True)
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return

            enabled_count = sum(1 for v in all_views if v["enabled"])
            if enabled_count > _WARN_ENABLED_VIEWS and enabled_count <= _BLOCK_ENABLED_VIEWS:
                reply = QMessageBox.question(
                    self,
                    "Large View Count",
                    (
                        f"{enabled_count} views are enabled. "
                        "This may increase processing time and output size.\n\n"
                        "Continue?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            if self._task_mode_id() == _TASK_MODE_COLMAP:
                try:
                    export_cmd = self._build_colmap_export_cmd()
                except Exception as e:
                    QMessageBox.critical(self, "Invalid COLMAP Export Input", str(e))
                    return

                steps: list[tuple[str, list[str]]] = [("colmap_export", export_cmd)]
                if self._colmap_run_until_id() != "export":
                    matcher = self.colmap_matcher_combo.currentData()
                    if matcher == "exhaustive":
                        estimated_pairs = enabled_count * self._count_input_images()
                        reply = QMessageBox.question(
                            self,
                            "Exhaustive Matcher Warning",
                            (
                                "Exhaustive matcher is usually impractical for large rig datasets.\n"
                                f"Estimated exported images: {estimated_pairs}\n\n"
                                "Switch to Sequential matcher for normal use.\n"
                                "Continue with Exhaustive anyway?"
                            ),
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No,
                        )
                        if reply != QMessageBox.Yes:
                            return
                    try:
                        pipeline_cmd = self._build_colmap_pipeline_cmd()
                    except Exception as e:
                        QMessageBox.critical(self, "Invalid COLMAP Pipeline Input", str(e))
                        return
                    steps.append(("colmap_sfm", pipeline_cmd))

                self._start_step_queue(steps)
                return

            try:
                self._validate_bundle_requirements()
            except Exception as e:
                QMessageBox.critical(self, "Missing Bundle Input", str(e))
                return

            try:
                cubemap_cmd = self._build_cmd()
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return

            steps = []
            if self.preprocess_enable_check.isChecked():
                try:
                    preprocess_cmd = self._build_preprocess_cmd()
                except Exception as e:
                    QMessageBox.critical(self, "Invalid Preprocess Input", str(e))
                    return
                steps.append(("metashape", preprocess_cmd))
            steps.append(("cubemap", cubemap_cmd))

            self._start_step_queue(steps)

        def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - UI event
            if self._is_running() and self.proc is not None:
                self.proc.terminate()
                if not self.proc.waitForFinished(2000):
                    self.proc.kill()
                    self.proc.waitForFinished(2000)
            super().closeEvent(event)

else:
    class CubemapToolsWindow:  # pragma: no cover - placeholder when PySide6 missing
        pass


def ensure_gui_deps() -> None:
    if QApplication is None:
        raise RuntimeError(f"PySide6 is required to run this GUI: {_PYSIDE_IMPORT_ERROR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI wrapper for cubemap_transforms_json.py")
    parser.add_argument("--scene-dir", default=".", help="Initial scene directory shown in GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_gui_deps()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = CubemapToolsWindow(initial_scene_dir=args.scene_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
