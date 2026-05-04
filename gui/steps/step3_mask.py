"""Step 3: マスク生成 (YOLO + スティッチ + 白飛び + 空 + カスタム)"""
from __future__ import annotations

import os
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from apply_frame_decisions import pending_drop_image_paths, untracked_image_paths
from custom_mask import load_custom_mask
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
    QTabWidget,
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
from gui.user_settings import load_user_settings_section, update_user_settings_section
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
_SKY_EXPAND_MIN = -16
_SKY_EXPAND_MAX = 64
_SKY_EXPAND_DEFAULT = 0
_SKY_MIN_SCORE_MIN = 0.0
_SKY_MIN_SCORE_MAX = 2.0
_SKY_MIN_SCORE_DEFAULT = 0.0
_SKY_MIN_AREA_PERCENT_MIN = 0.0
_SKY_MIN_AREA_PERCENT_MAX = 5.0
_SKY_MIN_AREA_PERCENT_DEFAULT = 0.05
_SKY_INFERENCE_SIZES = ("512", "768", "1008", "1024")
_SKY_INFERENCE_SIZE_DEFAULT_INDEX = 1
_SKY_SAM31_INFERENCE_SIZE = "1008"
_SKY_BACKENDS = ("mask2former", "sam31")
_SKY_SAM31_CHECKPOINT = Path("models") / "sam3.1" / "sam3.1_multiplex.pt"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_PROJECTION_EQUIRECT = "equirect"
_PROJECTION_NORMAL = "normal"
_LICENSE_NOTICE_SECTION = "license_notices"
_YOLO_SAM_NOTICE_VERSION = 1
_YOLO_SAM_NOTICE_KEY = "yolo_sam_models_ack_version"
_SKY_NOTICE_VERSION = 2
_SKY_NOTICE_KEY = "sky_models_ack_version"


class MaskStep(BaseStepWidget):
    scene_dir_suggested = Signal(str)

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._phase_total = 0
        self._phase_done = 0
        self._stitch_chunk_total = 0
        self._stitch_chunk_done = 0
        self._stitch_done_before = 0
        self._mask_preview_proc: QProcess | None = None
        self._mask_preview_temp: tempfile.TemporaryDirectory[str] | None = None
        self._mask_preview_image: Path | None = None
        self._mask_preview_output: Path | None = None
        self._mask_preview_config: MaskPreviewConfig | None = None
        self._mask_preview_commands: list[tuple[str, list[str]]] = []
        self._custom_mask_path = ""
        self._current_reprocess_proc: QProcess | None = None
        self._current_reprocess_image: Path | None = None
        self._current_reprocess_mask: Path | None = None
        self._current_reprocess_commands: list[tuple[str, list[str]]] = []
        self._current_reprocess_phase = ""
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
        task_row.setSpacing(6)
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

        self.run_sky_cb = QCheckBox(i18n.t("MASK_TASK_SKY"))
        self.run_sky_cb.setToolTip(i18n.tip("MASK_TASK_SKY"))
        self.run_sky_cb.setChecked(False)
        task_row.addWidget(self.run_sky_cb)

        self.run_custom_cb = QCheckBox(i18n.t("MASK_TASK_CUSTOM"))
        self.run_custom_cb.setToolTip(i18n.tip("MASK_TASK_CUSTOM"))
        self.run_custom_cb.setChecked(False)
        task_row.addWidget(self.run_custom_cb)

        task_row.addStretch()

        layout.addLayout(task_row)

        # --- マスク設定タブ ---
        self.mask_settings_tabs = QTabWidget()
        self.mask_settings_tabs.setObjectName("maskSettingsTabs")
        layout.addWidget(self.mask_settings_tabs)

        # --- YOLO設定 ---
        self.yolo_section = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_section)
        yolo_layout.setContentsMargins(8, 8, 8, 8)
        yolo_layout.setSpacing(6)
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
        yolo_layout.addWidget(yolo_settings_row_widget)

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
        yolo_layout.addWidget(bottom_settings_row_widget)

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
        yolo_layout.addWidget(class_list_section)
        yolo_layout.addStretch()
        self.mask_settings_tabs.addTab(self.yolo_section, i18n.t("MASK_TAB_YOLO"))

        # --- スティッチ+白飛び設定 ---
        self.other_section = QWidget()
        other_layout = QVBoxLayout(self.other_section)
        other_layout.setContentsMargins(8, 8, 8, 8)
        other_layout.setSpacing(6)
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

        other_layout.addLayout(other_form)
        other_layout.addStretch()
        self.mask_settings_tabs.addTab(self.other_section, i18n.t("MASK_TAB_STITCH_OVEREXP"))

        self.sky_section = QWidget()
        sky_layout = QVBoxLayout(self.sky_section)
        sky_layout.setContentsMargins(8, 8, 8, 8)
        sky_layout.setSpacing(6)
        sky_form = QFormLayout()
        sky_form.setSpacing(6)

        self.sky_backend_combo = QComboBox()
        self.sky_backend_combo.addItems(
            [
                i18n.t("SKY_MODEL_MASK2FORMER"),
                i18n.t("SKY_MODEL_SAM31"),
            ]
        )
        self.sky_backend_combo.setToolTip(i18n.tip("SKY_MODEL"))
        self.sky_backend_combo.setFixedWidth(132)
        add_tooltip_row(sky_form, i18n.t("SKY_MODEL"), self.sky_backend_combo, i18n.tip("SKY_MODEL"))

        self.sky_mode_combo = QComboBox()
        self.sky_mode_combo.addItems(
            [
                i18n.t("SKY_MODE_HYBRID"),
                i18n.t("SKY_MODE_DIRECT"),
                i18n.t("SKY_MODE_TOP"),
            ]
        )
        self.sky_mode_combo.setToolTip(i18n.tip("SKY_MODE"))
        self.sky_mode_combo.setFixedWidth(132)
        self.sky_mode_label = QLabel(i18n.t("SKY_MODE"))
        self.sky_mode_label.setToolTip(i18n.tip("SKY_MODE"))
        add_tooltip_row(sky_form, i18n.t("SKY_MODE"), self.sky_mode_combo, i18n.tip("SKY_MODE"))

        self.sky_inference_size_combo = QComboBox()
        self.sky_inference_size_combo.addItems(_SKY_INFERENCE_SIZES)
        self.sky_inference_size_combo.setCurrentIndex(_SKY_INFERENCE_SIZE_DEFAULT_INDEX)
        self.sky_inference_size_combo.setToolTip(i18n.tip("SKY_INFERENCE_SIZE"))
        self.sky_inference_size_combo.setFixedWidth(86)
        add_tooltip_row(
            sky_form,
            i18n.t("SKY_INFERENCE_SIZE"),
            self.sky_inference_size_combo,
            i18n.tip("SKY_INFERENCE_SIZE"),
        )

        self.sky_expand_edit = DragSpinBox(
            minimum=_SKY_EXPAND_MIN,
            maximum=_SKY_EXPAND_MAX,
            step=1,
            value=_SKY_EXPAND_DEFAULT,
            suffix=" px",
            drag_pixels_per_step=6.0,
        )
        self.sky_expand_edit.setToolTip(i18n.tip("SKY_EXPAND"))
        self.sky_expand_edit.setFixedWidth(80)
        add_tooltip_row(sky_form, i18n.t("SKY_EXPAND"), self.sky_expand_edit, i18n.tip("SKY_EXPAND"))

        self.sky_min_score_edit = DragDoubleSpinBox(
            minimum=_SKY_MIN_SCORE_MIN,
            maximum=_SKY_MIN_SCORE_MAX,
            step=0.05,
            decimals=2,
            value=_SKY_MIN_SCORE_DEFAULT,
            drag_pixels_per_step=8.0,
        )
        self.sky_min_score_edit.setToolTip(i18n.tip("SKY_MIN_SCORE"))
        self.sky_min_score_edit.setFixedWidth(80)
        add_tooltip_row(
            sky_form,
            i18n.t("SKY_MIN_SCORE"),
            self.sky_min_score_edit,
            i18n.tip("SKY_MIN_SCORE"),
        )

        self.sky_min_area_edit = DragDoubleSpinBox(
            minimum=_SKY_MIN_AREA_PERCENT_MIN,
            maximum=_SKY_MIN_AREA_PERCENT_MAX,
            step=0.05,
            decimals=2,
            value=_SKY_MIN_AREA_PERCENT_DEFAULT,
            suffix=" %",
            drag_pixels_per_step=8.0,
        )
        self.sky_min_area_edit.setToolTip(i18n.tip("SKY_MIN_AREA"))
        self.sky_min_area_edit.setFixedWidth(86)
        add_tooltip_row(
            sky_form,
            i18n.t("SKY_MIN_AREA"),
            self.sky_min_area_edit,
            i18n.tip("SKY_MIN_AREA"),
        )

        self.sky_top_connected_cb = QCheckBox(i18n.t("SKY_TOP_CONNECTED"))
        self.sky_top_connected_cb.setToolTip(i18n.tip("SKY_TOP_CONNECTED"))
        self.sky_top_connected_cb.setChecked(True)
        sky_form.addRow("", self.sky_top_connected_cb)

        sky_layout.addLayout(sky_form)
        sky_layout.addStretch()
        self.mask_settings_tabs.addTab(self.sky_section, i18n.t("MASK_TAB_SKY"))

        self.custom_section = QWidget()
        custom_layout = QVBoxLayout(self.custom_section)
        custom_layout.setContentsMargins(8, 8, 8, 8)
        custom_layout.setSpacing(6)
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
        custom_layout.addLayout(custom_form)

        custom_button_row = QHBoxLayout()
        custom_button_row.setSpacing(6)
        self.custom_mask_browse_btn = QPushButton(i18n.t("CUSTOM_MASK_BROWSE"))
        self.custom_mask_browse_btn.setToolTip(i18n.tip("CUSTOM_MASK_BROWSE"))
        custom_button_row.addWidget(self.custom_mask_browse_btn, stretch=1)
        self.custom_mask_clear_btn = QPushButton(i18n.t("CUSTOM_MASK_CLEAR"))
        self.custom_mask_clear_btn.setToolTip(i18n.tip("CUSTOM_MASK_CLEAR"))
        custom_button_row.addWidget(self.custom_mask_clear_btn, stretch=1)
        custom_layout.addLayout(custom_button_row)
        custom_layout.addStretch()
        self.mask_settings_tabs.addTab(self.custom_section, i18n.t("MASK_TAB_CUSTOM"))

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

        for cb in (self.run_yolo_cb, self.run_stitch_cb, self.run_overexp_cb, self.run_sky_cb):
            cb.toggled.connect(self._update_task_controls)
        self.run_custom_cb.toggled.connect(self._on_custom_mask_toggled)
        self.custom_mask_browse_btn.clicked.connect(lambda _checked=False: self._browse_custom_mask(activate=True))
        self.custom_mask_clear_btn.clicked.connect(lambda _checked=False: self._clear_custom_mask_path())
        self.stitch_boundary_width_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.yolo_level_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.yolo_expand_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.yolo_bottom_enhance_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
        for cb in self.class_cbs:
            cb.toggled.connect(lambda _checked=False: self._schedule_render_mask_preview())
        self.overexp_threshold_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.overexp_dilate_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_backend_combo.currentIndexChanged.connect(lambda _: self._on_sky_backend_changed())
        self.sky_mode_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_inference_size_combo.currentIndexChanged.connect(lambda _: self._on_sky_inference_size_changed())
        self.sky_expand_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_min_score_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_min_area_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_top_connected_cb.toggled.connect(lambda _: self._schedule_render_mask_preview())
        self.mask_preview.current_image_changed.connect(lambda: self._schedule_render_mask_preview())
        self.mask_preview.mask_preview_requested.connect(self._run_mask_preview)
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
        if self.run_sky_cb.isChecked():
            requested_steps.append("sky")
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

    def _sky_backend_arg(self) -> str:
        idx = self.sky_backend_combo.currentIndex()
        return _SKY_BACKENDS[max(0, min(idx, len(_SKY_BACKENDS) - 1))]

    def _sam31_checkpoint_path(self) -> Path:
        return self.base_dir / _SKY_SAM31_CHECKPOINT

    def _on_sky_backend_changed(self) -> None:
        if self._sky_backend_arg() == "sam31" and self.sky_inference_size_combo.currentText() != _SKY_SAM31_INFERENCE_SIZE:
            idx = self.sky_inference_size_combo.findText(_SKY_SAM31_INFERENCE_SIZE)
            if idx >= 0:
                self.sky_inference_size_combo.setCurrentIndex(idx)
        self._schedule_render_mask_preview()

    def _on_sky_inference_size_changed(self) -> None:
        if self._sky_backend_arg() == "sam31" and self.sky_inference_size_combo.currentText() != _SKY_SAM31_INFERENCE_SIZE:
            idx = self.sky_inference_size_combo.findText(_SKY_SAM31_INFERENCE_SIZE)
            if idx >= 0:
                self.sky_inference_size_combo.blockSignals(True)
                try:
                    self.sky_inference_size_combo.setCurrentIndex(idx)
                finally:
                    self.sky_inference_size_combo.blockSignals(False)
        self._schedule_render_mask_preview()

    def _sky_mode_arg(self) -> str:
        idx = self.sky_mode_combo.currentIndex()
        return ("hybrid", "direct", "top")[max(0, min(idx, 2))]

    def _sky_inference_size_arg(self) -> str:
        text = self.sky_inference_size_combo.currentText().strip()
        return text if text else str(_SKY_INFERENCE_SIZES[_SKY_INFERENCE_SIZE_DEFAULT_INDEX])

    def _sky_min_area_ratio_arg(self) -> str:
        return f"{float(self.sky_min_area_edit.value()) / 100.0:g}"

    def _sky_common_args(self) -> list[str]:
        args = [
            "--backend", self._sky_backend_arg(),
            "--projection", self._projection(),
            "--mode", self._sky_mode_arg(),
            "--inference-size", self._sky_inference_size_arg(),
            "--expand", str(self.sky_expand_edit.value()),
            "--min-score", f"{float(self.sky_min_score_edit.value()):g}",
            "--min-area-ratio", self._sky_min_area_ratio_arg(),
        ]
        if not self.sky_top_connected_cb.isChecked():
            args.append("--no-top-connected")
        return args

    def _update_task_controls(self) -> None:
        yolo_enabled = self.run_yolo_cb.isChecked()
        equirect = self._projection() == _PROJECTION_EQUIRECT
        if not equirect and self.run_stitch_cb.isChecked():
            self.run_stitch_cb.setChecked(False)
            return
        stitch_enabled = equirect and self.run_stitch_cb.isChecked()
        overexp_enabled = self.run_overexp_cb.isChecked()
        sky_enabled = self.run_sky_cb.isChecked()
        custom_enabled = self.run_custom_cb.isChecked()

        self.external_images_panel.setVisible(not equirect)
        self.yolo_section.setEnabled(yolo_enabled)
        self.sky_section.setEnabled(sky_enabled)
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
        self._update_sky_backend_availability()
        self._render_mask_preview()
        self._update_ready_status()

    def _update_sky_backend_availability(self) -> None:
        sam_available = self._sam31_checkpoint_path().is_file()
        model = self.sky_backend_combo.model()
        item = model.item(1) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(sam_available)
            item.setToolTip(
                i18n.tip("SKY_MODEL_SAM31") if sam_available else i18n.t("SKY_MODEL_SAM31_MISSING")
            )
        if not sam_available and self._sky_backend_arg() == "sam31":
            self.sky_backend_combo.blockSignals(True)
            try:
                self.sky_backend_combo.setCurrentIndex(0)
            finally:
                self.sky_backend_combo.blockSignals(False)

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
        self.mask_preview.render(self._mask_preview_config_from_controls())

    def _mask_preview_config_from_controls(self) -> MaskPreviewConfig:
        try:
            width = self._stitch_boundary_width()
        except ValueError:
            width = None
        return MaskPreviewConfig(
            use_yolo=self.run_yolo_cb.isChecked(),
            use_stitch=self.run_stitch_cb.isChecked(),
            use_overexposure=self.run_overexp_cb.isChecked(),
            use_sky=self.run_sky_cb.isChecked(),
            stitch_boundary_width_deg=width,
            overexposure_threshold=int(self.overexp_threshold_edit.value()),
            overexposure_dilate=int(self.overexp_dilate_edit.value()),
            masks_dir=self._masks_dir_text(),
            use_custom=self.run_custom_cb.isChecked(),
            custom_mask_path=self._custom_mask_path_text(),
            settings_key=self._mask_generation_settings_key(),
        )

    def _mask_generation_settings_key(self) -> tuple:
        return (
            self._projection(),
            self.yolo_level_combo.currentIndex(),
            int(self.yolo_expand_edit.value()),
            tuple(self._selected_classes()),
            self.yolo_bottom_enhance_combo.currentIndex(),
            self._sky_backend_arg(),
            self._sky_mode_arg(),
            self._sky_inference_size_arg(),
            int(self.sky_expand_edit.value()),
            float(self.sky_min_score_edit.value()),
            float(self.sky_min_area_edit.value()),
            bool(self.sky_top_connected_cb.isChecked()),
        )

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

        steps: list[tuple[str, list[str]]] = []
        fresh_base_needed = True
        if "yolo" in requested_steps:
            steps.append(("yolo", self._build_yolo_cmd()))
            fresh_base_needed = False
        if "stitch" in requested_steps:
            if fresh_base_needed:
                steps.append(("init_masks", self._build_init_masks_cmd()))
                fresh_base_needed = False
            steps.append(("stitch", self._build_stitch_cmd()))
        if "overexposure" in requested_steps:
            steps.append(("overexposure", self._build_overexposure_cmd(replace=fresh_base_needed)))
            fresh_base_needed = False
        if "sky" in requested_steps:
            steps.append(("sky", self._build_sky_cmd(replace=fresh_base_needed)))
            fresh_base_needed = False
        if "custom" in requested_steps:
            steps.append(("custom", self._build_custom_cmd(replace=fresh_base_needed)))
            fresh_base_needed = False
        return steps

    def confirm_commands(self, commands: list[tuple[str, list[str]]]) -> bool:
        if any(phase == "yolo" for phase, _cmd in commands):
            if not self._confirm_yolo_sam_license_notice():
                return False
        if any(phase == "sky" for phase, _cmd in commands):
            if not self._confirm_sky_license_notice():
                return False
        return True

    def _confirm_yolo_sam_license_notice(self) -> bool:
        if self._yolo_sam_notice_acknowledged():
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(i18n.t("YOLO_SAM_LICENSE_NOTICE_TITLE"))
        box.setText(i18n.t("YOLO_SAM_LICENSE_NOTICE_BODY"))
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)

        remember_cb = QCheckBox(i18n.t("YOLO_SAM_LICENSE_NOTICE_DONT_SHOW_AGAIN"))
        remember_cb.setChecked(True)
        box.setCheckBox(remember_cb)

        continue_btn = box.addButton(
            i18n.t("YOLO_SAM_LICENSE_NOTICE_CONTINUE"),
            QMessageBox.AcceptRole,
        )
        box.addButton(i18n.CANCEL, QMessageBox.RejectRole)
        box.setDefaultButton(continue_btn)

        box.exec()
        if box.clickedButton() != continue_btn:
            return False
        if remember_cb.isChecked():
            self._set_yolo_sam_notice_acknowledged()
        return True

    def _yolo_sam_notice_acknowledged(self) -> bool:
        settings = load_user_settings_section(_LICENSE_NOTICE_SECTION)
        try:
            version = int(settings.get(_YOLO_SAM_NOTICE_KEY, 0))
        except (TypeError, ValueError):
            version = 0
        return version >= _YOLO_SAM_NOTICE_VERSION

    @staticmethod
    def _set_yolo_sam_notice_acknowledged() -> None:
        update_user_settings_section(
            _LICENSE_NOTICE_SECTION,
            {_YOLO_SAM_NOTICE_KEY: _YOLO_SAM_NOTICE_VERSION},
        )

    def _confirm_sky_license_notice(self) -> bool:
        if self._sky_notice_acknowledged():
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(i18n.t("SKY_LICENSE_NOTICE_TITLE"))
        box.setText(i18n.t("SKY_LICENSE_NOTICE_BODY"))
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)

        remember_cb = QCheckBox(i18n.t("YOLO_SAM_LICENSE_NOTICE_DONT_SHOW_AGAIN"))
        remember_cb.setChecked(True)
        box.setCheckBox(remember_cb)

        continue_btn = box.addButton(
            i18n.t("YOLO_SAM_LICENSE_NOTICE_CONTINUE"),
            QMessageBox.AcceptRole,
        )
        box.addButton(i18n.CANCEL, QMessageBox.RejectRole)
        box.setDefaultButton(continue_btn)

        box.exec()
        if box.clickedButton() != continue_btn:
            return False
        if remember_cb.isChecked():
            self._set_sky_notice_acknowledged()
        return True

    def _sky_notice_acknowledged(self) -> bool:
        settings = load_user_settings_section(_LICENSE_NOTICE_SECTION)
        try:
            version = int(settings.get(_SKY_NOTICE_KEY, 0))
        except (TypeError, ValueError):
            version = 0
        return version >= _SKY_NOTICE_VERSION

    @staticmethod
    def _set_sky_notice_acknowledged() -> None:
        update_user_settings_section(
            _LICENSE_NOTICE_SECTION,
            {_SKY_NOTICE_KEY: _SKY_NOTICE_VERSION},
        )

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

    def _mask_output_dir_for_image(self, image_path: Path, masks_root: Path | None = None) -> Path:
        masks_root = masks_root or Path(self._masks_dir_text())
        try:
            rel_parent = image_path.resolve().relative_to(Path(self._images_dir_text()).resolve()).parent
        except Exception:
            rel_parent = Path()
        return masks_root / rel_parent

    def _mask_output_path_for_image(self, image_path: Path, masks_root: Path | None = None) -> Path:
        return self._mask_output_dir_for_image(image_path, masks_root=masks_root) / f"{image_path.stem}.png"

    def _build_yolo_current_cmd(self, image_path: Path, masks_root: Path | None = None) -> list[str]:
        output_dir = self._mask_output_dir_for_image(image_path, masks_root=masks_root)
        return self._build_yolo_preview_cmd(image_path, output_dir)

    def _build_init_masks_cmd(self) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        if not images:
            raise ValueError("画像フォルダが指定されていません")
        if not masks:
            raise ValueError("マスクフォルダが指定されていません")

        script = self.base_dir / "init_masks.py"
        if not script.exists():
            raise FileNotFoundError(f"init_masks.py が見つかりません: {script}")

        return [sys.executable, "-u", str(script), images, masks]

    def _build_sky_cmd(self, *, replace: bool = False) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        if not images:
            raise ValueError("画像フォルダが指定されていません")
        if not masks:
            raise ValueError("マスクフォルダが指定されていません")

        script = self.base_dir / "sky_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"sky_mask.py が見つかりません: {script}")

        cmd = [
            sys.executable, "-u", str(script),
            images, masks,
            *self._sky_common_args(),
        ]
        if replace:
            cmd.append("--replace")
        return cmd

    def _build_sky_current_cmd(
        self,
        image_path: Path,
        *,
        replace: bool = False,
        masks_root: Path | None = None,
    ) -> list[str]:
        output_dir = self._mask_output_dir_for_image(image_path, masks_root=masks_root)
        script = self.base_dir / "sky_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"sky_mask.py が見つかりません: {script}")
        cmd = [
            sys.executable, "-u", str(script),
            str(image_path), str(output_dir),
            *self._sky_common_args(),
        ]
        if replace:
            cmd.append("--replace")
        return cmd

    def _run_mask_preview(self) -> None:
        if self._mask_preview_proc is not None and self._mask_preview_proc.state() != QProcess.NotRunning:
            return
        if self._current_reprocess_active:
            return

        image_path = self.mask_preview.current_image_path()
        if image_path is None:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_NO_IMAGE"))
            return
        config = self._mask_preview_config_from_controls()
        if self.mask_preview.has_active_temporary_preview(config):
            self.mask_preview.clear_temporary_preview_mask(image_path)
            self._render_mask_preview()
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_CLEARED"))
            return
        if not self._selected_mask_tasks():
            self.mask_preview.set_status_text(i18n.t("MASK_TASK_REQUIRED"))
            return

        if self.mask_preview.preview_mode() == "thumbnails":
            self.mask_preview.set_preview_mode("single")
            if self.mask_preview.has_active_temporary_preview(config):
                self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_TEMP"))
                return

        if self.run_yolo_cb.isChecked() and not self._confirm_yolo_sam_license_notice():
            self.mask_preview.set_status_text(i18n.t("YOLO_SAM_LICENSE_NOTICE_CANCELED"))
            return
        if self.run_sky_cb.isChecked() and not self._confirm_sky_license_notice():
            self.mask_preview.set_status_text(i18n.t("SKY_LICENSE_NOTICE_CANCELED"))
            return

        self._cleanup_mask_preview_temp()
        self._mask_preview_temp = tempfile.TemporaryDirectory(prefix="stechdrive_mask_preview_")
        masks_root = Path(self._mask_preview_temp.name)
        output_path = self._mask_output_path_for_image(image_path, masks_root=masks_root)

        try:
            commands = self._build_image_external_commands(image_path, masks_root=masks_root)
        except (ValueError, FileNotFoundError) as e:
            self.mask_preview.set_status_text(str(e))
            self._cleanup_mask_preview_temp()
            return

        self._mask_preview_image = image_path
        self._mask_preview_output = output_path
        self._mask_preview_config = config
        self._mask_preview_commands = commands
        self.mask_preview.clear_temporary_preview_mask(image_path)
        self.mask_preview.set_mask_preview_running(True)
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_RUNNING"))
        self._start_next_mask_preview_command()

    def _start_next_mask_preview_command(self) -> None:
        if not self._mask_preview_commands:
            image_path = self._mask_preview_image
            output_path = self._mask_preview_output
            config = self._mask_preview_config
            ok = image_path is not None and output_path is not None and config is not None
            if ok and image_path is not None and output_path is not None:
                try:
                    self._apply_current_image_postprocess(
                        image_path,
                        output_path,
                        replace=not output_path.is_file(),
                    )
                except Exception as e:
                    ok = False
                    self.mask_preview.set_status_text(str(e))
            if ok and image_path is not None and output_path is not None and config is not None:
                ok = output_path.is_file() and self.mask_preview.set_temporary_preview_mask(image_path, output_path, config)
            self.mask_preview.set_mask_preview_running(False)
            if ok:
                self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_TEMP"))
            else:
                self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_FAILED"))
            self._cleanup_mask_preview_temp()
            self._mask_preview_proc = None
            self._mask_preview_image = None
            self._mask_preview_output = None
            self._mask_preview_config = None
            self._render_mask_preview()
            return

        _phase, cmd = self._mask_preview_commands.pop(0)
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain_mask_preview_output)
        proc.errorOccurred.connect(self._on_mask_preview_error)
        proc.finished.connect(self._on_mask_preview_finished)
        self._mask_preview_proc = proc
        proc.start()

    def _drain_mask_preview_output(self) -> None:
        if self._mask_preview_proc is not None:
            self._mask_preview_proc.readAllStandardOutput()

    def _on_mask_preview_error(self, _error: QProcess.ProcessError) -> None:
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_FAILED"))

    def _on_mask_preview_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._mask_preview_proc = None
        if exit_code == 0:
            self._start_next_mask_preview_command()
            return
        self.mask_preview.set_mask_preview_running(False)
        self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_FAILED"))
        self._cleanup_mask_preview_temp()
        self._mask_preview_image = None
        self._mask_preview_output = None
        self._mask_preview_config = None
        self._mask_preview_commands = []
        self._render_mask_preview()

    def _cleanup_mask_preview_temp(self) -> None:
        if self._mask_preview_temp is None:
            return
        try:
            self._mask_preview_temp.cleanup()
        except Exception:
            pass
        self._mask_preview_temp = None

    def _run_current_image_reprocess(self) -> None:
        if self._current_reprocess_active:
            return
        if self._mask_preview_proc is not None and self._mask_preview_proc.state() != QProcess.NotRunning:
            return

        image_paths = self._selected_reprocess_image_paths()
        if not image_paths:
            self.mask_preview.set_status_text(i18n.t("MASK_PREVIEW_NO_IMAGE"))
            return
        if not self._selected_mask_tasks():
            self.mask_preview.set_status_text(i18n.t("MASK_TASK_REQUIRED"))
            return
        if self.run_yolo_cb.isChecked() and not self._confirm_yolo_sam_license_notice():
            self.mask_preview.set_status_text(i18n.t("YOLO_SAM_LICENSE_NOTICE_CANCELED"))
            return
        if self.run_sky_cb.isChecked() and not self._confirm_sky_license_notice():
            self.mask_preview.set_status_text(i18n.t("SKY_LICENSE_NOTICE_CANCELED"))
            return

        self._current_reprocess_active = True
        self._current_reprocess_queue = image_paths
        self._current_reprocess_total = len(image_paths)
        self._current_reprocess_completed = 0
        self._current_reprocess_failed = []
        self._current_reprocess_succeeded = []
        self._current_reprocess_last_success = None
        self.mask_preview.set_current_reprocess_running(True)
        self.mask_preview.wait_for_thumbnail_rendering()
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
            commands = self._build_current_reprocess_external_commands(image_path)
            if not commands:
                self._apply_current_image_postprocess(image_path, mask_path, replace=True)
                self._record_current_reprocess_result(success=True, image_path=image_path)
                self._queue_next_current_reprocess()
                return
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            self.mask_preview.set_status_text(str(e))
            self._record_current_reprocess_result(success=False, image_path=image_path)
            self._queue_next_current_reprocess()
            return

        self._current_reprocess_image = image_path
        self._current_reprocess_mask = mask_path
        self._current_reprocess_commands = commands
        self._start_next_current_reprocess_external_command()

    def _build_current_reprocess_external_commands(self, image_path: Path) -> list[tuple[str, list[str]]]:
        return self._build_image_external_commands(image_path, masks_root=None)

    def _build_image_external_commands(
        self,
        image_path: Path,
        *,
        masks_root: Path | None,
    ) -> list[tuple[str, list[str]]]:
        commands: list[tuple[str, list[str]]] = []
        fresh_base_needed = True
        if self.run_yolo_cb.isChecked():
            commands.append(("yolo", self._build_yolo_current_cmd(image_path, masks_root=masks_root)))
            fresh_base_needed = False
        if self.run_sky_cb.isChecked():
            commands.append((
                "sky",
                self._build_sky_current_cmd(image_path, replace=fresh_base_needed, masks_root=masks_root),
            ))
            fresh_base_needed = False
        return commands

    def _start_next_current_reprocess_external_command(self) -> None:
        if not self._current_reprocess_commands:
            image_path = self._current_reprocess_image
            mask_path = self._current_reprocess_mask
            success = image_path is not None and mask_path is not None and mask_path.is_file()
            if success and image_path is not None and mask_path is not None:
                try:
                    self._apply_current_image_postprocess(image_path, mask_path)
                except Exception as e:
                    success = False
                    self.mask_preview.set_status_text(str(e))
            self._record_current_reprocess_result(success=bool(success), image_path=image_path)
            self._current_reprocess_image = None
            self._current_reprocess_mask = None
            self._current_reprocess_phase = ""
            self._queue_next_current_reprocess()
            return

        phase, cmd = self._current_reprocess_commands.pop(0)
        self._current_reprocess_phase = phase
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
        self._current_reprocess_proc = None
        if success:
            self._start_next_current_reprocess_external_command()
            return

        self._record_current_reprocess_result(success=False, image_path=image_path)
        self._current_reprocess_image = None
        self._current_reprocess_mask = None
        self._current_reprocess_commands = []
        self._current_reprocess_phase = ""
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

    def _apply_current_image_postprocess(
        self,
        image_path: Path,
        mask_path: Path,
        *,
        replace: bool = False,
    ) -> None:
        source_img: np.ndarray | None = None
        mask: np.ndarray | None = None

        def load_source() -> np.ndarray:
            nonlocal source_img
            if source_img is None:
                source_img = read_image_preserve_depth(str(image_path))
                if source_img is None:
                    raise RuntimeError(i18n.t("PREVIEW_LOAD_FAIL"))
            return source_img

        def current_mask(target_shape: tuple[int, int]) -> np.ndarray:
            nonlocal mask
            if mask is None:
                existing = None if replace else imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
                if existing is None:
                    mask = np.full(target_shape, 255, dtype=np.uint8)
                else:
                    mask = existing
            if mask.shape != target_shape:
                mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            return mask

        if self._projection() == _PROJECTION_EQUIRECT and self.run_stitch_cb.isChecked():
            if mask is None and not replace and mask_path.is_file():
                existing = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
                target_shape = existing.shape[:2] if existing is not None else load_source().shape[:2]
            else:
                target_shape = load_source().shape[:2]
            base = current_mask(target_shape)
            h, w = base.shape[:2]
            stitch = create_angular_stitched_mask(
                w,
                h,
                boundary_width_to_limit_angle(self._stitch_boundary_width()),
            )
            mask = cv2.bitwise_and(base, stitch)

        if self.run_overexp_cb.isChecked():
            source = load_source()
            overexp = detect_overexposure(
                source,
                threshold=int(self.overexp_threshold_edit.value()),
                dilate_px=int(self.overexp_dilate_edit.value()),
            )
            mask = cv2.bitwise_and(current_mask(overexp.shape), overexp)

        if self.run_custom_cb.isChecked():
            custom_mask = self._custom_mask_path_text()
            if not custom_mask:
                raise RuntimeError(i18n.t("CUSTOM_MASK_REQUIRED"))
            loaded_custom, load_error = load_custom_mask(custom_mask)
            if loaded_custom is None:
                raise RuntimeError(load_error or i18n.t("CUSTOM_MASK_NOT_FOUND").format(path=custom_mask))
            source_shape = load_source().shape[:2]
            if loaded_custom.mask.shape != source_shape:
                raise RuntimeError(
                    f"Skipped (size mismatch): {image_path.name} "
                    f"image={source_shape[1]}x{source_shape[0]} "
                    f"custom={loaded_custom.mask.shape[1]}x{loaded_custom.mask.shape[0]}"
                )
            mask = cv2.bitwise_and(current_mask(loaded_custom.mask.shape), loaded_custom.mask)

        if mask is not None:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not imwrite_unicode(mask_path, mask):
                raise RuntimeError(i18n.t("MASK_REPROCESS_CURRENT_FAILED"))

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

    def _build_overexposure_cmd(self, *, replace: bool = False) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        if not images:
            raise ValueError("画像フォルダが指定されていません")
        if not masks:
            raise ValueError("マスクフォルダが指定されていません")

        script = self.base_dir / "overexposure_mask.py"
        if not script.exists():
            raise FileNotFoundError(f"overexposure_mask.py が見つかりません: {script}")

        cmd = [
            sys.executable, "-u", str(script),
            images, masks,
            "--threshold", str(self.overexp_threshold_edit.value()),
            "--dilate", str(self.overexp_dilate_edit.value()),
            "--workers", str(self.stitch_workers_edit.value()),
        ]
        if replace:
            cmd.append("--replace")
        return cmd

    def _build_custom_cmd(self, *, replace: bool = False) -> list[str]:
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

        cmd = [
            sys.executable, "-u", str(script),
            images, masks, custom_mask,
        ]
        if replace:
            cmd.append("--replace")
        return cmd

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
