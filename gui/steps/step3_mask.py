"""Step 3: マスク生成 (YOLO + スティッチ + 白飛び)"""
from __future__ import annotations

import os
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
from apply_frame_decisions import pending_drop_image_paths, untracked_image_paths
from custom_mask import load_custom_mask, merge_custom_mask_for_image
from PySide6.QtCore import QProcess, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.mask.mask_preview import MaskPreviewConfig, MaskPreviewWidget
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)
from image_io import imread_unicode, imwrite_unicode
from overexposure_mask import detect_overexposure, read_image_preserve_depth
from stitch_mask import boundary_width_to_limit_angle, create_angular_stitched_mask

_COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

_MASK_PROGRESS_RE = re.compile(r"\[progress\]\s+(\d+)\s*/\s*(\d+)")
_YOLO_PROCESSED_RE = re.compile(r"^Processed:\s+")
_STITCH_TASK_RE = re.compile(r"^Processing\s+(\d+)\s+images\s+with\s+\d+\s+workers\.\.\.$")
_STITCH_TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
_STITCH_BOUNDARY_MIN = 0.0
_STITCH_BOUNDARY_MAX = 30.0
_STITCH_BOUNDARY_DEFAULT = 5.0
_YOLO_EXPAND_MIN = -16
_YOLO_EXPAND_MAX = 32
_YOLO_EXPAND_DEFAULT = 2
_YOLO_BOTTOM_PRESETS: tuple[tuple[str, list[str]], ...] = (
    ("standard", []),
    (
        "strong",
        [
            "--bottom-conf", "0.15",
            "--bottom-tta-rotations", "4",
            "--bottom-filter",
        ],
    ),
    (
        "max",
        [
            "--bottom-conf", "0.10",
            "--bottom-tta-rotations", "4",
            "--bottom-model", "x",
            "--bottom-filter",
        ],
    ),
)
_OVEREXP_THRESHOLD_MIN = 1
_OVEREXP_THRESHOLD_MAX = 254
_OVEREXP_THRESHOLD_DEFAULT = 254
_OVEREXP_DILATE_MIN = 0
_OVEREXP_DILATE_MAX = 128
_OVEREXP_DILATE_DEFAULT = 1
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_PROJECTION_EQUIRECT = "equirect"
_PROJECTION_NORMAL = "normal"


class MaskStep(BaseStepWidget):
    scene_dir_suggested = Signal(str)

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._phase_total = 0
        self._phase_done = 0
        self._stitch_chunk_total = 0
        self._stitch_chunk_done = 0
        self._stitch_done_before = 0
        self._yolo_preview_proc: QProcess | None = None
        self._yolo_preview_temp: tempfile.TemporaryDirectory[str] | None = None
        self._yolo_preview_image: Path | None = None
        self._yolo_preview_output: Path | None = None
        self._custom_mask_path = ""
        self._current_reprocess_proc: QProcess | None = None
        self._current_reprocess_image: Path | None = None
        self._current_reprocess_mask: Path | None = None
        self._current_reprocess_active = False
        self._current_reprocess_queue: list[Path] = []
        self._current_reprocess_total = 0
        self._current_reprocess_completed = 0
        self._current_reprocess_failed: list[Path] = []
        self._current_reprocess_succeeded: list[Path] = []
        self._current_reprocess_last_success: Path | None = None
        self._mask_preview_render_pending = False
        self._mask_preview_render_timer = QTimer(self)
        self._mask_preview_render_timer.setSingleShot(True)
        self._mask_preview_render_timer.setInterval(50)
        self._mask_preview_render_timer.timeout.connect(self._flush_scheduled_mask_preview)
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        settings_scroll = QScrollArea()
        configure_settings_scroll(settings_scroll)
        settings = QWidget()
        settings.setObjectName("settingsPane")
        layout = QVBoxLayout(settings)
        layout.setContentsMargins(*SETTINGS_PANE_MARGINS)
        layout.setSpacing(8)

        # --- 標準フォルダ ---
        path_form = QFormLayout()
        path_form.setSpacing(6)
        self.images_path_label = QLabel("-")
        self.images_path_label.setToolTip(i18n.tip("IMAGES_DIR"))
        self.images_path_label.setWordWrap(True)
        self.images_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add_tooltip_row(path_form, i18n.IMAGES_DIR, self.images_path_label, i18n.tip("IMAGES_DIR"))
        self.masks_path_label = QLabel("-")
        self.masks_path_label.setToolTip(i18n.tip("MASKS_DIR"))
        self.masks_path_label.setWordWrap(True)
        self.masks_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add_tooltip_row(path_form, i18n.MASKS_DIR, self.masks_path_label, i18n.tip("MASKS_DIR"))
        layout.addLayout(path_form)

        self.ready_status_label = QLabel()
        self.ready_status_label.setWordWrap(True)
        layout.addWidget(self.ready_status_label)

        projection_row = QHBoxLayout()
        projection_row.setSpacing(6)
        self.projection_label = QLabel(i18n.t("MASK_IMAGE_TYPE"))
        self.projection_label.setToolTip(i18n.tip("MASK_IMAGE_TYPE"))
        projection_row.addWidget(self.projection_label)
        self.projection_group = QButtonGroup(self)
        self.projection_group.setExclusive(True)
        self.projection_buttons: dict[str, QPushButton] = {}
        for projection, label, tip_key in [
            (_PROJECTION_EQUIRECT, i18n.t("MASK_IMAGE_TYPE_EQUIRECT"), "MASK_IMAGE_TYPE_EQUIRECT"),
            (_PROJECTION_NORMAL, i18n.t("MASK_IMAGE_TYPE_NORMAL"), "MASK_IMAGE_TYPE_NORMAL"),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("segmentedOption")
            btn.setCheckable(True)
            btn.setToolTip(i18n.tip(tip_key))
            btn.clicked.connect(lambda _checked=False, p=projection: self._set_projection(p))
            projection_row.addWidget(btn, stretch=1)
            self.projection_group.addButton(btn)
            self.projection_buttons[projection] = btn
        layout.addLayout(projection_row)

        self.external_images_panel = QWidget()
        external_layout = QVBoxLayout(self.external_images_panel)
        external_layout.setContentsMargins(0, 0, 0, 0)
        external_layout.setSpacing(6)

        self.external_images_title = QLabel(i18n.t("EXTERNAL_IMAGES_SECTION"))
        self.external_images_title.setToolTip(i18n.tip("EXTERNAL_IMAGES_SECTION"))
        external_layout.addWidget(self.external_images_title)

        self.external_images_hint = QLabel(i18n.t("EXTERNAL_IMAGES_HINT"))
        self.external_images_hint.setObjectName("stickySummaryLabel")
        self.external_images_hint.setWordWrap(True)
        self.external_images_hint.setToolTip(i18n.tip("EXTERNAL_IMAGES_SECTION"))
        external_layout.addWidget(self.external_images_hint)

        external_button_row = QHBoxLayout()
        external_button_row.setSpacing(6)
        self.add_external_images_btn = QPushButton(i18n.t("EXTERNAL_IMAGES_ADD"))
        self.add_external_images_btn.setToolTip(i18n.tip("EXTERNAL_IMAGES_ADD"))
        external_button_row.addWidget(self.add_external_images_btn, stretch=1)
        self.open_images_dir_btn = QPushButton(i18n.t("EXTERNAL_IMAGES_OPEN"))
        self.open_images_dir_btn.setToolTip(i18n.tip("EXTERNAL_IMAGES_OPEN"))
        external_button_row.addWidget(self.open_images_dir_btn, stretch=1)
        external_layout.addLayout(external_button_row)
        layout.addWidget(self.external_images_panel)

        # --- 実行対象 + 実行ボタン ---
        task_row = QHBoxLayout()
        task_row.setSpacing(10)
        task_row.addWidget(QLabel(i18n.t("MASK_TASKS_LABEL")))

        self.run_yolo_cb = QCheckBox(i18n.t("MASK_TASK_YOLO"))
        self.run_yolo_cb.setToolTip(i18n.tip("MASK_TASK_YOLO"))
        self.run_yolo_cb.setChecked(True)
        task_row.addWidget(self.run_yolo_cb)

        self.run_stitch_cb = QCheckBox(i18n.t("MASK_TASK_STITCH"))
        self.run_stitch_cb.setToolTip(i18n.tip("MASK_TASK_STITCH"))
        self.run_stitch_cb.setChecked(False)
        task_row.addWidget(self.run_stitch_cb)

        self.run_overexp_cb = QCheckBox(i18n.t("MASK_TASK_OVEREXPOSURE"))
        self.run_overexp_cb.setToolTip(i18n.tip("MASK_TASK_OVEREXPOSURE"))
        self.run_overexp_cb.setChecked(False)
        task_row.addWidget(self.run_overexp_cb)

        self.run_custom_cb = QCheckBox(i18n.t("MASK_TASK_CUSTOM"))
        self.run_custom_cb.setToolTip(i18n.tip("MASK_TASK_CUSTOM"))
        self.run_custom_cb.setChecked(False)
        task_row.addWidget(self.run_custom_cb)

        task_row.addStretch()

        layout.addLayout(task_row)

        # --- YOLO設定 (折りたたみ) ---
        self.yolo_section = CollapsibleSection(i18n.t("YOLO_SECTION"), expanded=True)
        yolo_settings_row_widget = QWidget()
        yolo_settings_row = QHBoxLayout(yolo_settings_row_widget)
        yolo_settings_row.setContentsMargins(0, 0, 0, 0)
        yolo_settings_row.setSpacing(6)

        self.yolo_level_label = QLabel(i18n.t("YOLO_LEVEL_COMPACT"))
        self.yolo_level_label.setToolTip(i18n.tip("YOLO_LEVEL"))
        yolo_settings_row.addWidget(self.yolo_level_label)

        self.yolo_level_combo = QComboBox()
        self.yolo_level_combo.setToolTip(i18n.tip("YOLO_LEVEL"))
        self.yolo_level_combo.addItems(
            [
                i18n.t("YOLO_LEVEL_FAST"),
                i18n.t("YOLO_LEVEL_STANDARD"),
                i18n.t("YOLO_LEVEL_QUALITY"),
                i18n.t("YOLO_LEVEL_BEST"),
            ]
        )
        self.yolo_level_combo.setCurrentIndex(2)
        self.yolo_level_combo.setFixedWidth(104)
        yolo_settings_row.addWidget(self.yolo_level_combo)

        self.yolo_expand_label = QLabel(i18n.t("YOLO_EXPAND_COMPACT"))
        self.yolo_expand_label.setToolTip(i18n.tip("YOLO_EXPAND"))
        yolo_settings_row.addWidget(self.yolo_expand_label)

        self.yolo_expand_edit = DragSpinBox(
            minimum=_YOLO_EXPAND_MIN,
            maximum=_YOLO_EXPAND_MAX,
            step=1,
            value=_YOLO_EXPAND_DEFAULT,
            suffix=" px",
            drag_pixels_per_step=6.0,
        )
        self.yolo_expand_edit.setToolTip(i18n.tip("YOLO_EXPAND"))
        self.yolo_expand_edit.setFixedWidth(74)
        yolo_settings_row.addWidget(self.yolo_expand_edit)
        yolo_settings_row.addStretch()
        self.yolo_settings_row = yolo_settings_row_widget
        self.yolo_section.content_layout.addWidget(yolo_settings_row_widget)

        bottom_settings_row_widget = QWidget()
        bottom_settings_row = QHBoxLayout(bottom_settings_row_widget)
        bottom_settings_row.setContentsMargins(0, 0, 0, 0)
        bottom_settings_row.setSpacing(6)
        self.yolo_bottom_enhance_label = QLabel(i18n.t("YOLO_BOTTOM_ENHANCE"))
        self.yolo_bottom_enhance_label.setToolTip(i18n.tip("YOLO_BOTTOM_ENHANCE"))
        bottom_settings_row.addWidget(self.yolo_bottom_enhance_label)

        self.yolo_bottom_enhance_combo = QComboBox()
        self.yolo_bottom_enhance_combo.setToolTip(i18n.tip("YOLO_BOTTOM_ENHANCE"))
        self.yolo_bottom_enhance_combo.addItems(
            [
                i18n.t("YOLO_BOTTOM_STANDARD"),
                i18n.t("YOLO_BOTTOM_STRONG"),
                i18n.t("YOLO_BOTTOM_MAX"),
            ]
        )
        self.yolo_bottom_enhance_combo.setCurrentIndex(0)
        self.yolo_bottom_enhance_combo.setFixedWidth(128)
        bottom_settings_row.addWidget(self.yolo_bottom_enhance_combo)
        bottom_settings_row.addStretch()
        self.yolo_bottom_settings_row = bottom_settings_row_widget
        self.yolo_section.content_layout.addWidget(bottom_settings_row_widget)

        class_list_section = CollapsibleSection(i18n.t("YOLO_CLASS_LIST_SECTION"), expanded=False)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(160)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(2)
        self.class_cbs: list[QCheckBox] = []
        cols = 2
        for idx, name in enumerate(_COCO_CLASS_NAMES):
            cb = QCheckBox(f"{idx}: {name}")
            if idx == 0:
                cb.setChecked(True)
            self.class_cbs.append(cb)
            grid.addWidget(cb, idx // cols, idx % cols)
        scroll.setWidget(grid_widget)
        class_list_section.content_layout.addWidget(scroll)
        self.yolo_section.content_layout.addWidget(class_list_section)
        layout.addWidget(self.yolo_section)

        # --- スティッチ+白飛び設定 (折りたたみ) ---
        self.other_section = CollapsibleSection(i18n.t("STITCH_OVEREXP_SECTION"), expanded=False)
        other_form = QFormLayout()
        other_form.setSpacing(6)

        self.stitch_boundary_width_edit = DragDoubleSpinBox(
            minimum=_STITCH_BOUNDARY_MIN,
            maximum=_STITCH_BOUNDARY_MAX,
            step=0.5,
            decimals=1,
            value=_STITCH_BOUNDARY_DEFAULT,
        )
        self.stitch_boundary_width_edit.setToolTip(i18n.tip("STITCH_BOUNDARY_WIDTH"))
        self.stitch_boundary_width_edit.setFixedWidth(80)
        add_tooltip_row(
            other_form,
            i18n.STITCH_BOUNDARY_WIDTH,
            self.stitch_boundary_width_edit,
            i18n.tip("STITCH_BOUNDARY_WIDTH"),
        )

        worker_default = os.cpu_count() or 4
        self.stitch_workers_edit = DragSpinBox(
            minimum=1,
            maximum=max(1, worker_default * 2),
            step=1,
            value=worker_default,
        )
        self.stitch_workers_edit.setToolTip(i18n.tip("STITCH_WORKERS"))
        self.stitch_workers_edit.setFixedWidth(80)
        add_tooltip_row(other_form, i18n.STITCH_WORKERS, self.stitch_workers_edit, i18n.tip("STITCH_WORKERS"))

        sep = QLabel("")
        sep.setFixedHeight(8)
        other_form.addRow(sep)

        self.overexp_threshold_edit = DragSpinBox(
            minimum=_OVEREXP_THRESHOLD_MIN,
            maximum=_OVEREXP_THRESHOLD_MAX,
            step=1,
            value=_OVEREXP_THRESHOLD_DEFAULT,
            drag_pixels_per_step=4.0,
        )
        self.overexp_threshold_edit.setToolTip(i18n.tip("OVEREXPOSURE_THRESHOLD"))
        self.overexp_threshold_edit.setFixedWidth(80)
        add_tooltip_row(
            other_form,
            i18n.OVEREXPOSURE_THRESHOLD,
            self.overexp_threshold_edit,
            i18n.tip("OVEREXPOSURE_THRESHOLD"),
        )

        self.overexp_dilate_edit = DragSpinBox(
            minimum=_OVEREXP_DILATE_MIN,
            maximum=_OVEREXP_DILATE_MAX,
            step=1,
            value=_OVEREXP_DILATE_DEFAULT,
            drag_pixels_per_step=6.0,
        )
        self.overexp_dilate_edit.setToolTip(i18n.tip("OVEREXPOSURE_DILATE"))
        self.overexp_dilate_edit.setFixedWidth(80)
        add_tooltip_row(
            other_form,
            i18n.OVEREXPOSURE_DILATE,
            self.overexp_dilate_edit,
            i18n.tip("OVEREXPOSURE_DILATE"),
        )

        self.other_section.content_layout.addLayout(other_form)
        layout.addWidget(self.other_section)

        self.custom_section = CollapsibleSection(i18n.t("CUSTOM_MASK_SECTION"), expanded=False)
        custom_form = QFormLayout()
        custom_form.setSpacing(6)
        self.custom_mask_path_label = QLabel(i18n.t("CUSTOM_MASK_NOT_SELECTED"))
        self.custom_mask_path_label.setToolTip(i18n.tip("CUSTOM_MASK_FILE"))
        self.custom_mask_path_label.setWordWrap(True)
        self.custom_mask_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add_tooltip_row(
            custom_form,
            i18n.t("CUSTOM_MASK_FILE"),
            self.custom_mask_path_label,
            i18n.tip("CUSTOM_MASK_FILE"),
        )
        self.custom_section.content_layout.addLayout(custom_form)

        custom_button_row = QHBoxLayout()
        custom_button_row.setSpacing(6)
        self.custom_mask_browse_btn = QPushButton(i18n.t("CUSTOM_MASK_BROWSE"))
        self.custom_mask_browse_btn.setToolTip(i18n.tip("CUSTOM_MASK_BROWSE"))
        custom_button_row.addWidget(self.custom_mask_browse_btn, stretch=1)
        self.custom_mask_clear_btn = QPushButton(i18n.t("CUSTOM_MASK_CLEAR"))
        self.custom_mask_clear_btn.setToolTip(i18n.tip("CUSTOM_MASK_CLEAR"))
        custom_button_row.addWidget(self.custom_mask_clear_btn, stretch=1)
        self.custom_section.content_layout.addLayout(custom_button_row)
        layout.addWidget(self.custom_section)

        layout.addStretch()
        self.metashape_notice = QLabel(i18n.METASHAPE_NOTICE)
        self.metashape_notice.setObjectName("workflowNote")
        self.metashape_notice.setWordWrap(True)
        layout.addWidget(self.metashape_notice)

        preview_pane = QWidget()
        preview_pane.setObjectName("workPane")
        self.preview_pane = preview_pane
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)
        self.mask_preview = MaskPreviewWidget()
        preview_header = QHBoxLayout()
        preview_header.setContentsMargins(0, 0, 0, 0)
        preview_header.setSpacing(8)
        preview_title = QLabel(i18n.t("MASK_PREVIEW_SECTION"))
        preview_title.setObjectName("paneTitle")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.mask_preview.mode_toolbar)
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.mask_preview, stretch=1)

        settings_scroll.setWidget(settings)
        splitter.addWidget(settings_scroll)
        splitter.addWidget(preview_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        root_layout.addWidget(splitter)

        for cb in (self.run_yolo_cb, self.run_stitch_cb, self.run_overexp_cb):
            cb.toggled.connect(self._update_task_controls)
        self.run_custom_cb.toggled.connect(self._on_custom_mask_toggled)
        self.custom_mask_browse_btn.clicked.connect(lambda _checked=False: self._browse_custom_mask(activate=True))
        self.custom_mask_clear_btn.clicked.connect(lambda _checked=False: self._clear_custom_mask_path())
        self.stitch_boundary_width_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.overexp_threshold_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.overexp_dilate_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.mask_preview.current_image_changed.connect(lambda: self._schedule_render_mask_preview())
        self.mask_preview.yolo_preview_requested.connect(self._run_yolo_preview)
        self.mask_preview.current_reprocess_requested.connect(self._run_current_image_reprocess)
        self.add_external_images_btn.clicked.connect(self._add_external_images_from_folder)
        self.open_images_dir_btn.clicked.connect(self._open_images_dir)
        self._set_projection(_PROJECTION_EQUIRECT)
        self._update_task_controls()
        self._on_images_dir_changed(self._images_dir_text())
        self._update_ready_status()

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if path:
            p = Path(path)
            self.images_path_label.setText(str(p / "images"))
            self.masks_path_label.setText(str(p / "masks"))
        else:
            self.images_path_label.setText("-")
            self.masks_path_label.setText("-")
        self._on_images_dir_changed(self._images_dir_text())
        self._render_mask_preview()
        self._update_ready_status()

    def primary_action_text(self) -> str:
        return i18n.t("GENERATE")

    def primary_action_tooltip(self) -> str:
        ready, reason = self._readiness()
        return i18n.tip("RUN_MASKS") if ready else reason

    def primary_action_enabled(self) -> bool:
        ready, _reason = self._readiness()
        return ready

    def on_activated(self) -> None:
        self.mask_preview.refresh_image_list(prefer_current=True)
        self._render_mask_preview()
        self._update_ready_status()

    def _images_dir_text(self) -> str:
        if not self.scene_dir:
            return ""
        return str(Path(self.scene_dir) / "images")

    def _masks_dir_text(self) -> str:
        if not self.scene_dir:
            return ""
        return str(Path(self.scene_dir) / "masks")

    def _selected_mask_tasks(self) -> list[str]:
        requested_steps = []
        if self.run_yolo_cb.isChecked():
            requested_steps.append("yolo")
        if self._projection() == _PROJECTION_EQUIRECT and self.run_stitch_cb.isChecked():
            requested_steps.append("stitch")
        if self.run_overexp_cb.isChecked():
            requested_steps.append("overexposure")
        if self.run_custom_cb.isChecked():
            requested_steps.append("custom")
        return requested_steps

    def _projection(self) -> str:
        for projection, btn in self.projection_buttons.items():
            if btn.isChecked():
                return projection
        return _PROJECTION_EQUIRECT

    def _set_projection(self, projection: str) -> None:
        if projection not in {_PROJECTION_EQUIRECT, _PROJECTION_NORMAL}:
            projection = _PROJECTION_EQUIRECT
        btn = self.projection_buttons.get(projection)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self.yolo_level_combo.setCurrentIndex(1 if projection == _PROJECTION_NORMAL else 2)
        self._update_task_controls()

    def _scene_csv_path(self) -> Path:
        return Path(self.scene_dir) / "selected_frames.csv"

    def _has_image_files(self) -> bool:
        images = Path(self._images_dir_text())
        if not images.is_dir():
            return False
        return any(
            path.is_file() and path.suffix.lower() in _IMAGE_EXTS
            for path in images.rglob("*")
        )

    def _readiness(self) -> tuple[bool, str]:
        if not self.scene_dir:
            return False, i18n.t("SCENE_REQUIRED_ACTION_HINT")
        if not Path(self.scene_dir).is_dir():
            return False, i18n.t("MASK_READY_SCENE_NOT_FOUND")
        images = Path(self._images_dir_text())
        if not images.is_dir():
            return False, i18n.t("MASK_READY_NO_IMAGES_DIR")
        if not self._has_image_files():
            return False, i18n.t("MASK_READY_NO_IMAGES")
        if self.run_custom_cb.isChecked():
            custom_mask = self._custom_mask_path_text()
            if not custom_mask:
                return False, i18n.t("CUSTOM_MASK_REQUIRED")
            if not Path(custom_mask).is_file():
                return False, i18n.t("CUSTOM_MASK_NOT_FOUND").format(path=custom_mask)
        if not self._selected_mask_tasks():
            return False, i18n.t("MASK_TASK_REQUIRED")
        if not self._scene_csv_path().is_file():
            return True, i18n.t("MASK_READY_EXTERNAL_IMAGES")
        return True, i18n.t("MASK_READY_OK")

    def _update_ready_status(self) -> None:
        ready, reason = self._readiness()
        self.ready_status_label.setText(reason)
        if ready:
            self.ready_status_label.setStyleSheet(
                "color: #7ecf7e; border: 1px solid #2f6f2f; border-radius: 6px; padding: 8px;"
            )
        else:
            self.ready_status_label.setStyleSheet(
                "color: #d8a24a; border: 1px solid #7a5b25; border-radius: 6px; padding: 8px;"
            )
        self.primary_action_state_changed.emit()

    def _selected_classes(self) -> list[int]:
        return [i for i, cb in enumerate(self.class_cbs) if cb.isChecked()]

    def _yolo_expand_arg(self) -> str:
        return str(self.yolo_expand_edit.value())

    def _bottom_enhance_args(self) -> list[str]:
        if self._projection() != _PROJECTION_EQUIRECT:
            return []
        idx = max(0, min(self.yolo_bottom_enhance_combo.currentIndex(), len(_YOLO_BOTTOM_PRESETS) - 1))
        return list(_YOLO_BOTTOM_PRESETS[idx][1])

    def _update_task_controls(self) -> None:
        yolo_enabled = self.run_yolo_cb.isChecked()
        equirect = self._projection() == _PROJECTION_EQUIRECT
        if not equirect and self.run_stitch_cb.isChecked():
            self.run_stitch_cb.setChecked(False)
            return
        stitch_enabled = equirect and self.run_stitch_cb.isChecked()
        overexp_enabled = self.run_overexp_cb.isChecked()
        custom_enabled = self.run_custom_cb.isChecked()

        self.external_images_panel.setVisible(not equirect)
        self.yolo_section.content_widget.setEnabled(yolo_enabled)
        self.yolo_bottom_enhance_label.setEnabled(yolo_enabled and equirect)
        self.yolo_bottom_enhance_combo.setEnabled(yolo_enabled and equirect)
        self.run_stitch_cb.setEnabled(equirect)
        self.run_stitch_cb.setToolTip(
            i18n.tip("MASK_TASK_STITCH") if equirect else i18n.tip("MASK_TASK_STITCH_DISABLED_NORMAL")
        )
        self.stitch_boundary_width_edit.setEnabled(stitch_enabled)
        self.stitch_workers_edit.setEnabled(stitch_enabled or overexp_enabled)
        self.overexp_threshold_edit.setEnabled(overexp_enabled)
        self.overexp_dilate_edit.setEnabled(overexp_enabled)
        self.custom_mask_clear_btn.setEnabled(bool(self._custom_mask_path_text()))
        self.custom_mask_path_label.setEnabled(custom_enabled or bool(self._custom_mask_path_text()))
        self._render_mask_preview()
        self._update_ready_status()

    def _custom_mask_path_text(self) -> str:
        return self._custom_mask_path.strip()

    def _set_custom_mask_path(self, path: str | Path, *, activate: bool = True) -> None:
        self._custom_mask_path = str(path).strip()
        self.custom_mask_path_label.setText(self._custom_mask_path or i18n.t("CUSTOM_MASK_NOT_SELECTED"))
        self.custom_mask_path_label.setToolTip(self._custom_mask_path or i18n.tip("CUSTOM_MASK_FILE"))
        if activate and self._custom_mask_path and not self.run_custom_cb.isChecked():
            self.run_custom_cb.blockSignals(True)
            try:
                self.run_custom_cb.setChecked(True)
            finally:
                self.run_custom_cb.blockSignals(False)
        self._update_task_controls()

    def _clear_custom_mask_path(self) -> None:
        self._custom_mask_path = ""
        self.custom_mask_path_label.setText(i18n.t("CUSTOM_MASK_NOT_SELECTED"))
        self.custom_mask_path_label.setToolTip(i18n.tip("CUSTOM_MASK_FILE"))
        if self.run_custom_cb.isChecked():
            self.run_custom_cb.blockSignals(True)
            try:
                self.run_custom_cb.setChecked(False)
            finally:
                self.run_custom_cb.blockSignals(False)
        self._update_task_controls()

    def _browse_custom_mask(self, *, activate: bool = True) -> bool:
        start_dir = self.scene_dir or str(Path.home())
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            i18n.t("CUSTOM_MASK_SELECT_FILE"),
            start_dir,
            i18n.t("CUSTOM_MASK_FILE_FILTER"),
        )
        if not selected:
            return False
        self._set_custom_mask_path(selected, activate=activate)
        return True

    def _on_custom_mask_toggled(self, checked: bool) -> None:
        if checked and not self._custom_mask_path_text():
            if not self._browse_custom_mask(activate=True):
                self.run_custom_cb.blockSignals(True)
                try:
                    self.run_custom_cb.setChecked(False)
                finally:
                    self.run_custom_cb.blockSignals(False)
        self._update_task_controls()

    def _on_images_dir_changed(self, path: str) -> None:
        self.mask_preview.set_images_dir(path)
        self._render_mask_preview()

    def _ensure_scene_for_external_images(self) -> bool:
        if self.scene_dir:
            return True

        result = QMessageBox.question(
            self,
            i18n.t("EXTERNAL_IMAGES_SCENE_REQUIRED_TITLE"),
            i18n.t("EXTERNAL_IMAGES_SCENE_REQUIRED_MESSAGE"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if result != QMessageBox.Yes:
            return False

        scene = QFileDialog.getExistingDirectory(self, i18n.t("EXTERNAL_IMAGES_SELECT_SCENE"))
        if not scene:
            return False
        self.scene_dir_suggested.emit(scene)
        if not self.scene_dir:
            self.set_scene_dir(scene)
        return bool(self.scene_dir)

    def _open_images_dir(self) -> None:
        if not self._ensure_scene_for_external_images():
            return
        images_dir = Path(self._images_dir_text())
        images_dir.mkdir(parents=True, exist_ok=True)
        self._on_images_dir_changed(str(images_dir))
        self._update_ready_status()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(images_dir)))

    def _add_external_images_from_folder(self) -> None:
        if not self._ensure_scene_for_external_images():
            return
        source = QFileDialog.getExistingDirectory(
            self,
            i18n.t("EXTERNAL_IMAGES_SELECT_FOLDER"),
            self.scene_dir,
        )
        if not source:
            return

        source_dir = Path(source)
        images_dir = Path(self._images_dir_text())
        try:
            if source_dir.resolve() == images_dir.resolve():
                QMessageBox.information(
                    self,
                    i18n.t("EXTERNAL_IMAGES_RESULT_TITLE"),
                    i18n.t("EXTERNAL_IMAGES_SOURCE_IS_TARGET"),
                )
                return
        except OSError:
            pass

        added, skipped = self._import_external_images_from_dir(source_dir)
        QMessageBox.information(
            self,
            i18n.t("EXTERNAL_IMAGES_RESULT_TITLE"),
            i18n.t("EXTERNAL_IMAGES_RESULT").format(added=added, skipped=skipped),
        )

    def _import_external_images_from_dir(self, source_dir: Path) -> tuple[int, int]:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        if not source_dir.is_dir():
            raise ValueError(i18n.t("EXTERNAL_IMAGES_SOURCE_NOT_FOUND").format(path=source_dir))

        images_dir = Path(self._images_dir_text())
        images_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        skipped = 0
        for src in sorted(source_dir.iterdir(), key=lambda p: p.name.lower()):
            if not src.is_file() or src.suffix.lower() not in _IMAGE_EXTS:
                continue
            dst = images_dir / src.name
            try:
                if src.resolve() == dst.resolve():
                    skipped += 1
                    continue
            except OSError:
                pass
            if dst.exists():
                skipped += 1
                continue
            shutil.copy2(src, dst)
            added += 1

        self.images_path_label.setText(str(images_dir))
        self._on_images_dir_changed(str(images_dir))
        self.mask_preview.refresh_image_list(prefer_current=True)
        self._render_mask_preview()
        self._update_ready_status()
        return added, skipped

    def _stitch_boundary_width(self) -> float:
        value = self._clamp_stitch_boundary_width(float(self.stitch_boundary_width_edit.value()))
        if value != self.stitch_boundary_width_edit.value():
            self.stitch_boundary_width_edit.setValue(value)
        return value

    @staticmethod
    def _clamp_stitch_boundary_width(value: float) -> float:
        if not math.isfinite(value):
            return _STITCH_BOUNDARY_DEFAULT
        return max(_STITCH_BOUNDARY_MIN, min(_STITCH_BOUNDARY_MAX, value))

    def _schedule_render_mask_preview(self) -> None:
        self._mask_preview_render_pending = True
        self._mask_preview_render_timer.start()

    def _flush_scheduled_mask_preview(self) -> None:
        if not self._mask_preview_render_pending:
            return
        self._mask_preview_render_pending = False
        self._render_mask_preview()

    def _render_mask_preview(self) -> None:
        try:
            width = self._stitch_boundary_width()
        except ValueError:
            width = None
        config = MaskPreviewConfig(
            use_yolo=self.run_yolo_cb.isChecked(),
            use_stitch=self.run_stitch_cb.isChecked(),
            use_overexposure=self.run_overexp_cb.isChecked(),
            stitch_boundary_width_deg=width,
            overexposure_threshold=int(self.overexp_threshold_edit.value()),
            overexposure_dilate=int(self.overexp_dilate_edit.value()),
            masks_dir=self._masks_dir_text(),
            use_custom=self.run_custom_cb.isChecked(),
            custom_mask_path=self._custom_mask_path_text(),
        )
        self.mask_preview.render(config)

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        ready, reason = self._readiness()
        if not ready:
            raise ValueError(reason)
        requested_steps = self._selected_mask_tasks()
        if not requested_steps:
            raise ValueError(i18n.t("MASK_TASK_REQUIRED"))

        self._ensure_no_pending_drop_images()
        self._ensure_no_untracked_images()

        steps = []
        if "yolo" in requested_steps:
            steps.append(("yolo", self._build_yolo_cmd()))
        if "stitch" in requested_steps:
            steps.append(("stitch", self._build_stitch_cmd()))
        if "overexposure" in requested_steps:
            steps.append(("overexposure", self._build_overexposure_cmd()))
        if "custom" in requested_steps:
            steps.append(("custom", self._build_custom_cmd()))
        return steps

    def _ensure_no_pending_drop_images(self) -> None:
        if not self.scene_dir:
            return
        images = self._images_dir_text()
        if not images:
            return
        scene_dir = Path(self.scene_dir)
        csv_path = scene_dir / "selected_frames.csv"
        if not csv_path.exists():
            return

        pending = pending_drop_image_paths(scene_dir, images_dir=Path(images))
        if not pending:
            return

        preview = "\n".join(f"- {p.name}" for p in pending[:5])
        if len(pending) > 5:
            preview += f"\n- ... +{len(pending) - 5}"
        raise ValueError(i18n.t("MASK_PENDING_DROPS_ERROR").format(n=len(pending), files=preview))

    def _ensure_no_untracked_images(self) -> None:
        if not self.scene_dir:
            return
        images = self._images_dir_text()
        if not images:
            return
        scene_dir = Path(self.scene_dir)
        csv_path = scene_dir / "selected_frames.csv"
        if not csv_path.exists():
            return

        untracked = untracked_image_paths(scene_dir, images_dir=Path(images))
        if not untracked:
            return

        preview = "\n".join(f"- {p.name}" for p in untracked[:5])
        if len(untracked) > 5:
            preview += f"\n- ... +{len(untracked) - 5}"
        raise ValueError(i18n.t("MASK_UNTRACKED_IMAGES_ERROR").format(n=len(untracked), files=preview))

    def _build_yolo_cmd(self) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        if not images:
            raise ValueError("画像フォルダが指定されていません")
        if not masks:
            raise ValueError("マスクフォルダが指定されていません")

        script = self.base_dir / "yolo_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"yolo_mask.py が見つかりません: {script}")

        # レベルはコンボのインデックスで取得 (テキストに説明が入っているため)
        level = str(self.yolo_level_combo.currentIndex())
        classes = self._selected_classes()
        cmd = [
            sys.executable, "-u", str(script),
            images, masks,
            "--level", level,
            "--expand", self._yolo_expand_arg(),
            "--projection", self._projection(),
        ]
        if classes:
            cmd.extend(["--classes", ",".join(str(c) for c in classes)])
        cmd.extend(self._bottom_enhance_args())
        return cmd

    def _build_yolo_preview_cmd(self, image_path: Path, output_dir: Path) -> list[str]:
        script = self.base_dir / "yolo_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"yolo_mask.py が見つかりません: {script}")

        level = str(self.yolo_level_combo.currentIndex())
        classes = self._selected_classes()
        cmd = [
            sys.executable, "-u", str(script),
            str(image_path), str(output_dir),
            "--level", level,
            "--expand", self._yolo_expand_arg(),
            "--projection", self._projection(),
        ]
        if classes:
            cmd.extend(["--classes", ",".join(str(c) for c in classes)])
        cmd.extend(self._bottom_enhance_args())
        return cmd

    def _mask_output_dir_for_image(self, image_path: Path) -> Path:
        masks_root = Path(self._masks_dir_text())
        try:
            rel_parent = image_path.resolve().relative_to(Path(self._images_dir_text()).resolve()).parent
        except Exception:
            rel_parent = Path()
        return masks_root / rel_parent

    def _mask_output_path_for_image(self, image_path: Path) -> Path:
        return self._mask_output_dir_for_image(image_path) / f"{image_path.stem}.png"

    def _build_yolo_current_cmd(self, image_path: Path) -> list[str]:
        output_dir = self._mask_output_dir_for_image(image_path)
        return self._build_yolo_preview_cmd(image_path, output_dir)

    def _run_yolo_preview(self) -> None:
        if self._yolo_preview_proc is not None and self._yolo_preview_proc.state() != QProcess.NotRunning:
            return
        if self._current_reprocess_active:
            return

        image_path = self.mask_preview.current_image_path()
        if image_path is None:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_YOLO_NO_IMAGE"))
            return

        self._cleanup_yolo_preview_temp()
        self._yolo_preview_temp = tempfile.TemporaryDirectory(prefix="stechdrive_yolo_preview_")
        output_dir = Path(self._yolo_preview_temp.name)
        output_path = output_dir / _yolo_preview_output_name(image_path)

        try:
            cmd = self._build_yolo_preview_cmd(image_path, output_dir)
        except (ValueError, FileNotFoundError) as e:
            self.mask_preview.set_status_text(str(e))
            self._cleanup_yolo_preview_temp()
            return

        self._yolo_preview_image = image_path
        self._yolo_preview_output = output_path
        self.mask_preview.clear_yolo_preview_mask(image_path)
        self.mask_preview.set_yolo_preview_running(True)
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_YOLO_RUNNING"))

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain_yolo_preview_output)
        proc.errorOccurred.connect(self._on_yolo_preview_error)
        proc.finished.connect(self._on_yolo_preview_finished)
        self._yolo_preview_proc = proc
        proc.start()

    def _drain_yolo_preview_output(self) -> None:
        if self._yolo_preview_proc is not None:
            self._yolo_preview_proc.readAllStandardOutput()

    def _on_yolo_preview_error(self, _error: QProcess.ProcessError) -> None:
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_YOLO_FAILED"))

    def _on_yolo_preview_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        image_path = self._yolo_preview_image
        output_path = self._yolo_preview_output
        ok = (
            exit_code == 0
            and image_path is not None
            and output_path is not None
            and output_path.is_file()
            and self.mask_preview.set_yolo_preview_mask(image_path, output_path)
        )
        self.mask_preview.set_yolo_preview_running(False)
        if ok:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_YOLO_TEMP"))
        else:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_YOLO_FAILED"))
        self._cleanup_yolo_preview_temp()
        self._yolo_preview_proc = None
        self._yolo_preview_image = None
        self._yolo_preview_output = None
        self._render_mask_preview()

    def _cleanup_yolo_preview_temp(self) -> None:
        if self._yolo_preview_temp is None:
            return
        try:
            self._yolo_preview_temp.cleanup()
        except Exception:
            pass
        self._yolo_preview_temp = None

    def _run_current_image_reprocess(self) -> None:
        if self._current_reprocess_active:
            return
        if self._yolo_preview_proc is not None and self._yolo_preview_proc.state() != QProcess.NotRunning:
            return

        image_paths = self._selected_reprocess_image_paths()
        if not image_paths:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_YOLO_NO_IMAGE"))
            return
        if not self._selected_mask_tasks():
            self.mask_preview.set_status_text(i18n.t("MASK_TASK_REQUIRED"))
            return

        self._current_reprocess_active = True
        self._current_reprocess_queue = image_paths
        self._current_reprocess_total = len(image_paths)
        self._current_reprocess_completed = 0
        self._current_reprocess_failed = []
        self._current_reprocess_succeeded = []
        self._current_reprocess_last_success = None
        self.mask_preview.set_current_reprocess_running(True)
        self._start_next_current_reprocess()

    def _selected_reprocess_image_paths(self) -> list[Path]:
        paths = self.mask_preview.selected_reprocess_image_paths()
        result: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            try:
                key = str(path.resolve()).lower()
            except Exception:
                key = str(path).lower()
            if key in seen:
                continue
            if not path.exists() or not path.is_file():
                continue
            seen.add(key)
            result.append(path)
        return result

    def _start_next_current_reprocess(self) -> None:
        if not self._current_reprocess_active:
            return
        if not self._current_reprocess_queue:
            self._finish_reprocess_batch()
            return

        image_path = self._current_reprocess_queue.pop(0)
        self._set_current_reprocess_progress(image_path)
        mask_path = self._mask_output_path_for_image(image_path)
        try:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            self.mask_preview.clear_yolo_preview_mask(image_path)
            if not self.run_yolo_cb.isChecked():
                self._apply_current_image_postprocess(image_path, mask_path)
                self._record_current_reprocess_result(success=True, image_path=image_path)
                self._queue_next_current_reprocess()
                return

            cmd = self._build_yolo_current_cmd(image_path)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            self.mask_preview.set_status_text(str(e))
            self._record_current_reprocess_result(success=False, image_path=image_path)
            self._queue_next_current_reprocess()
            return

        self._current_reprocess_image = image_path
        self._current_reprocess_mask = mask_path

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain_current_reprocess_output)
        proc.errorOccurred.connect(self._on_current_reprocess_error)
        proc.finished.connect(self._on_current_reprocess_finished)
        self._current_reprocess_proc = proc
        proc.start()

    def _drain_current_reprocess_output(self) -> None:
        if self._current_reprocess_proc is not None:
            self._current_reprocess_proc.readAllStandardOutput()

    def _on_current_reprocess_error(self, _error: QProcess.ProcessError) -> None:
        self.mask_preview.set_status_text(i18n.t("MASK_REPROCESS_CURRENT_FAILED"))

    def _on_current_reprocess_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        image_path = self._current_reprocess_image
        mask_path = self._current_reprocess_mask
        success = exit_code == 0 and image_path is not None and mask_path is not None and mask_path.is_file()
        if success and image_path is not None and mask_path is not None:
            try:
                self._apply_current_image_postprocess(image_path, mask_path)
            except Exception as e:
                success = False
                self.mask_preview.set_status_text(str(e))

        self._record_current_reprocess_result(success=success, image_path=image_path)

        self._current_reprocess_proc = None
        self._current_reprocess_image = None
        self._current_reprocess_mask = None
        self._queue_next_current_reprocess()

    def _record_current_reprocess_result(self, *, success: bool, image_path: Path | None) -> None:
        self._current_reprocess_completed += 1
        if success and image_path is not None:
            self._current_reprocess_last_success = image_path
            self._current_reprocess_succeeded.append(image_path)
        if not success and image_path is not None:
            self._current_reprocess_failed.append(image_path)

    def _set_current_reprocess_progress(self, image_path: Path) -> None:
        self.mask_preview.set_status_text(
            i18n.t("MASK_REPROCESS_SELECTED_PROGRESS").format(
                done=self._current_reprocess_completed + 1,
                total=self._current_reprocess_total,
                name=image_path.name,
            )
        )

    def _queue_next_current_reprocess(self) -> None:
        QTimer.singleShot(0, self._start_next_current_reprocess)

    def _finish_reprocess_batch(self) -> None:
        total = self._current_reprocess_total
        completed = self._current_reprocess_completed
        failed = len(self._current_reprocess_failed)
        succeeded_images = list(self._current_reprocess_succeeded)
        last_success = total == 1 and completed == 1 and failed == 0
        last_image = self._current_reprocess_last_success

        self._current_reprocess_active = False
        self._current_reprocess_queue = []
        self._current_reprocess_total = 0
        self._current_reprocess_completed = 0
        self._current_reprocess_last_success = None
        self.mask_preview.set_current_reprocess_running(False)
        self.mask_preview.refresh_image_list(prefer_current=True)
        self.mask_preview.invalidate_thumbnail_images(succeeded_images)
        self._render_mask_preview()
        self._mask_preview_render_timer.stop()
        self._mask_preview_render_pending = False
        self._update_ready_status()

        if last_success and last_image is not None:
            self.mask_preview.set_status_text(
                i18n.t("MASK_REPROCESS_CURRENT_DONE").format(name=last_image.name)
            )
        elif failed == 0:
            self.mask_preview.set_status_text(
                i18n.t("MASK_REPROCESS_SELECTED_DONE").format(done=completed, total=total)
            )
        else:
            self.mask_preview.set_status_text(
                i18n.t("MASK_REPROCESS_SELECTED_FAILED").format(failed=failed, total=total)
            )
        self._current_reprocess_failed = []
        self._current_reprocess_succeeded = []

    def _apply_current_image_postprocess(self, image_path: Path, mask_path: Path) -> None:
        if self._projection() == _PROJECTION_EQUIRECT and self.run_stitch_cb.isChecked():
            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(i18n.t("MASK_REPROCESS_NO_BASE_MASK"))
            h, w = mask.shape[:2]
            stitch = create_angular_stitched_mask(
                w,
                h,
                boundary_width_to_limit_angle(self._stitch_boundary_width()),
            )
            mask = cv2.bitwise_and(mask, stitch)
            if not imwrite_unicode(mask_path, mask):
                raise RuntimeError(i18n.t("MASK_REPROCESS_CURRENT_FAILED"))

        if self.run_overexp_cb.isChecked():
            source_img = read_image_preserve_depth(str(image_path))
            if source_img is None:
                raise RuntimeError(i18n.t("PREVIEW_LOAD_FAIL"))
            overexp = detect_overexposure(
                source_img,
                threshold=int(self.overexp_threshold_edit.value()),
                dilate_px=int(self.overexp_dilate_edit.value()),
            )
            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != overexp.shape:
                    mask = cv2.resize(mask, (overexp.shape[1], overexp.shape[0]), interpolation=cv2.INTER_NEAREST)
                overexp = cv2.bitwise_and(mask, overexp)
            if not imwrite_unicode(mask_path, overexp):
                raise RuntimeError(i18n.t("MASK_REPROCESS_CURRENT_FAILED"))

        if self.run_custom_cb.isChecked():
            custom_mask = self._custom_mask_path_text()
            if not custom_mask:
                raise RuntimeError(i18n.t("CUSTOM_MASK_REQUIRED"))
            loaded_custom, load_error = load_custom_mask(custom_mask)
            if loaded_custom is None:
                raise RuntimeError(load_error or i18n.t("CUSTOM_MASK_NOT_FOUND").format(path=custom_mask))
            error = merge_custom_mask_for_image(
                image_path,
                Path(self._images_dir_text()),
                Path(self._masks_dir_text()),
                loaded_custom.mask,
            )
            if not error.applied:
                raise RuntimeError(error.message or i18n.t("MASK_REPROCESS_CURRENT_FAILED"))

    def _build_stitch_cmd(self) -> list[str]:
        masks = self._masks_dir_text()
        if not masks:
            raise ValueError("マスクフォルダが指定されていません")

        script = self.base_dir / "stitch_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"stitch_mask.py が見つかりません: {script}")

        return [
            sys.executable, "-u", str(script),
            masks, masks,
            "--boundary-width", f"{self._stitch_boundary_width():g}",
            "--workers", str(self.stitch_workers_edit.value()),
        ]

    def _build_overexposure_cmd(self) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        if not images:
            raise ValueError("画像フォルダが指定されていません")
        if not masks:
            raise ValueError("マスクフォルダが指定されていません")

        script = self.base_dir / "overexposure_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"overexposure_mask.py が見つかりません: {script}")

        return [
            sys.executable, "-u", str(script),
            images, masks,
            "--threshold", str(self.overexp_threshold_edit.value()),
            "--dilate", str(self.overexp_dilate_edit.value()),
            "--workers", str(self.stitch_workers_edit.value()),
        ]

    def _build_custom_cmd(self) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        custom_mask = self._custom_mask_path_text()
        if not images:
            raise ValueError("画像フォルダが指定されていません")
        if not masks:
            raise ValueError("マスクフォルダが指定されていません")
        if not custom_mask:
            raise ValueError(i18n.t("CUSTOM_MASK_REQUIRED"))

        script = self.base_dir / "custom_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"custom_mask.py が見つかりません: {script}")

        return [
            sys.executable, "-u", str(script),
            images, masks, custom_mask,
        ]

    # -- プログレス解析 --

    def on_line(self, line: str) -> tuple[int, int] | None:
        m = _MASK_PROGRESS_RE.search(line)
        if m:
            self._phase_done = int(m.group(1))
            self._phase_total = int(m.group(2))
            return self._phase_done, self._phase_total

        if _YOLO_PROCESSED_RE.match(line):
            self._phase_done += 1
            return self._phase_done, self._phase_total

        m = _STITCH_TASK_RE.match(line)
        if m:
            self._phase_total = int(m.group(1))
            self._phase_done = 0
            self._stitch_chunk_total = 0
            self._stitch_chunk_done = 0
            self._stitch_done_before = 0
            return 0, self._phase_total

        m = _STITCH_TQDM_RE.search(line)
        if m:
            done = int(m.group(1))
            total = int(m.group(2))
            self._stitch_chunk_done = done
            self._stitch_chunk_total = total
            overall = self._stitch_done_before + done
            return overall, self._phase_total if self._phase_total > 0 else total

        return None

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        if phase == "yolo" and exit_code == 0:
            self._phase_total = 0
            self._phase_done = 0

    def on_queue_finished(self, success: bool) -> None:
        if not success:
            return
        self.mask_preview.clear_yolo_preview_mask()
        self.mask_preview.refresh_image_list(prefer_current=True, force_thumbnails=True)
        self._render_mask_preview()
        self._update_ready_status()

    def shutdown(self) -> None:
        self.mask_preview.shutdown()


def _yolo_preview_output_name(image_path: Path) -> str:
    return f"{image_path.stem}.png"
