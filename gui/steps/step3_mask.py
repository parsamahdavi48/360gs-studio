"""Step 3: マスク生成 (人物 + スティッチ + 白飛び + 空 + カスタム)"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from apply_frame_decisions import pending_drop_image_paths, untracked_image_paths
from custom_mask import load_custom_mask
from gui import i18n
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.common.icons import delete_icon, file_picker_icon, minus_icon, plus_icon
from gui.mask.mask_preview import MaskPreviewConfig, MaskPreviewWidget
from gui.steps import mask_commands as mask_command_defs
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)
from gui.steps.mask_commands import (
    MaskCommandContext,
    build_custom_cmd,
    build_init_masks_cmd,
    build_mask2former_cmd,
    build_overexposure_cmd,
    build_primary_mask_cmd,
    build_sam31_prompt_cmd,
    build_stitch_cmd,
)
from gui.steps.mask_image_import import IMAGE_EXTENSIONS as _IMAGE_EXTS
from gui.steps.mask_image_import import import_external_images
from gui.steps.sam31_setup import ensure_sam31_checkpoint_available
from gui.user_settings import load_user_settings_section, update_user_settings_section
from image_io import imread_unicode, imwrite_unicode
from mask_view_recipes import QUALITY_CHOICES
from overexposure_mask import detect_overexposure, read_image_preserve_depth
from scene_layout import selected_frames_path
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
_YOLO_EXPAND_DEFAULT = 0
_PERSON_BACKENDS = (
    mask_command_defs.PERSON_BACKEND_YOLO_SAM,
    mask_command_defs.PERSON_BACKEND_MASK2FORMER,
    mask_command_defs.PERSON_BACKEND_SAM31,
)
_PERSON_SAM31_PROMPT = "person"
_SAM31_MERGE_REPLACE = mask_command_defs.SAM31_MERGE_REPLACE
_SAM31_MERGE_ADD = mask_command_defs.SAM31_MERGE_ADD
_SAM31_MERGE_SUBTRACT = mask_command_defs.SAM31_MERGE_SUBTRACT
_SAM31_MERGE_MODES = mask_command_defs.SAM31_MERGE_MODES
_SAM31_PROMPT_PRESETS: tuple[tuple[str, str], ...] = (
    ("person", "人物"),
    ("sky", "空"),
    ("tripod", "三脚"),
    ("hand", "手"),
    ("camera", "カメラ"),
    ("selfie stick", "自撮り棒"),
    ("cell phone", "スマホ"),
    ("car", "車"),
)
_ADE20K_FALLBACK_CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain",
    "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house",
    "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper",
    "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar",
    "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier",
    "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag",
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
_SKY_MIN_SCORE_MAX = 1.0
_SKY_MIN_SCORE_DEFAULT = 0.0
_SKY_MIN_AREA_PERCENT_MIN = 0.0
_SKY_MIN_AREA_PERCENT_MAX = 5.0
_SKY_MIN_AREA_PERCENT_DEFAULT = 0.0
_SKY_INFERENCE_SIZES = ("512", "768", "1008", "1024")
_SKY_INFERENCE_SIZE_DEFAULT_INDEX = 1
_SKY_SAM31_INFERENCE_SIZE = "1008"
_SKY_BACKENDS = ("mask2former", "sam31")
_SKY_SAM31_CHECKPOINT = Path("models") / "sam3.1" / "sam3.1_multiplex.pt"
_PROJECTION_EQUIRECT = "equirect"
_PROJECTION_NORMAL = "normal"
_LICENSE_NOTICE_SECTION = "license_notices"
_YOLO_SAM_NOTICE_VERSION = 3
_YOLO_SAM_NOTICE_KEY = "yolo_sam_models_ack_version"
_SKY_NOTICE_VERSION = 2
_SKY_NOTICE_KEY = "sky_models_ack_version"


def _ade20k_class_names(base_dir: Path) -> tuple[str, ...]:
    config_path = base_dir / "models" / "mask2former-swin-large-ade-semantic" / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            labels = config.get("id2label", {})
            if labels:
                return tuple(str(labels[str(idx)]) for idx in range(len(labels)))
        except Exception:
            pass
    return _ADE20K_FALLBACK_CLASSES


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
        self.images_path_row = QWidget()
        images_path_row = QHBoxLayout(self.images_path_row)
        images_path_row.setContentsMargins(0, 0, 0, 0)
        images_path_row.setSpacing(6)
        self.images_path_label = QLabel("-")
        self.images_path_label.setToolTip(i18n.tip("IMAGES_DIR"))
        self.images_path_label.setWordWrap(True)
        self.images_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        images_path_row.addWidget(self.images_path_label, stretch=1)

        self.add_external_images_btn = QToolButton()
        self.add_external_images_btn.setObjectName("iconToolButton")
        self.add_external_images_btn.setIcon(plus_icon())
        self.add_external_images_btn.setToolTip(i18n.tip("EXTERNAL_IMAGES_ADD"))
        self.add_external_images_btn.setAccessibleName(i18n.t("EXTERNAL_IMAGES_ADD"))
        self.add_external_images_btn.setFixedSize(32, 32)
        images_path_row.addWidget(self.add_external_images_btn)

        add_tooltip_row(path_form, i18n.IMAGES_DIR, self.images_path_row, i18n.tip("IMAGES_DIR"))
        self.masks_path_label = QLabel("-")
        self.masks_path_label.setToolTip(i18n.tip("MASKS_DIR"))
        self.masks_path_label.setWordWrap(True)
        self.masks_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add_tooltip_row(path_form, i18n.MASKS_DIR, self.masks_path_label, i18n.tip("MASKS_DIR"))
        layout.addLayout(path_form)

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

        # --- 追加マスク ---
        task_row = QHBoxLayout()
        task_row.setSpacing(6)
        task_row.addWidget(QLabel(i18n.t("ADDITIONAL_MASKS_LABEL")))

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

        # --- マスク設定タブ ---
        self.mask_settings_tabs = QTabWidget()
        self.mask_settings_tabs.setObjectName("maskSettingsTabs")
        layout.addWidget(self.mask_settings_tabs)

        # --- マスク設定 ---
        self.yolo_section = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_section)
        yolo_layout.setContentsMargins(8, 8, 8, 8)
        yolo_layout.setSpacing(6)

        person_backend_row_widget = QWidget()
        person_backend_row = QHBoxLayout(person_backend_row_widget)
        person_backend_row.setContentsMargins(0, 0, 0, 0)
        person_backend_row.setSpacing(6)
        self.person_backend_label = QLabel(i18n.t("MASK_MODEL"))
        self.person_backend_label.setToolTip(i18n.tip("PERSON_MODEL"))
        person_backend_row.addWidget(self.person_backend_label)
        self.person_backend_combo = QComboBox()
        self.person_backend_combo.setToolTip(i18n.tip("PERSON_MODEL"))
        self.person_backend_combo.addItems(
            [
                i18n.t("PERSON_MODEL_YOLO_SAM"),
                i18n.t("SKY_MODEL_MASK2FORMER"),
                i18n.t("PERSON_MODEL_SAM31"),
            ]
        )
        self.person_backend_combo.setItemData(0, i18n.tip("PERSON_MODEL_YOLO_SAM"), Qt.ToolTipRole)
        self.person_backend_combo.setItemData(1, i18n.tip("SKY_MODEL_MASK2FORMER"), Qt.ToolTipRole)
        self.person_backend_combo.setItemData(2, i18n.tip("PERSON_MODEL_SAM31"), Qt.ToolTipRole)
        self.person_backend_combo.setFixedWidth(132)
        person_backend_row.addWidget(self.person_backend_combo)

        self.sky_inference_size_label = QLabel(i18n.t("SKY_INFERENCE_SIZE"))
        self.sky_inference_size_label.setToolTip(i18n.tip("SKY_INFERENCE_SIZE"))
        person_backend_row.addWidget(self.sky_inference_size_label)

        self.sky_inference_size_combo = QComboBox()
        self.sky_inference_size_combo.addItems(_SKY_INFERENCE_SIZES)
        self.sky_inference_size_combo.setCurrentIndex(_SKY_INFERENCE_SIZE_DEFAULT_INDEX)
        self.sky_inference_size_combo.setToolTip(i18n.tip("SKY_INFERENCE_SIZE"))
        self.sky_inference_size_combo.setFixedWidth(78)
        person_backend_row.addWidget(self.sky_inference_size_combo)
        person_backend_row.addStretch()
        self.person_backend_row = person_backend_row_widget
        yolo_layout.addWidget(person_backend_row_widget)

        yolo_settings_row_widget = QWidget()
        yolo_settings_row = QHBoxLayout(yolo_settings_row_widget)
        yolo_settings_row.setContentsMargins(0, 0, 0, 0)
        yolo_settings_row.setSpacing(6)

        self.yolo_level_label = QLabel(i18n.t("MASK_QUALITY"))
        self.yolo_level_label.setToolTip(i18n.tip("MASK_QUALITY"))
        yolo_settings_row.addWidget(self.yolo_level_label)

        self.yolo_level_combo = QComboBox()
        self.yolo_level_combo.setToolTip(i18n.tip("MASK_QUALITY"))
        self.yolo_level_combo.addItems(
            [
                i18n.t("MASK_QUALITY_STANDARD"),
                i18n.t("MASK_QUALITY_HIGH"),
                i18n.t("MASK_QUALITY_BEST"),
            ]
        )
        self.yolo_level_combo.setCurrentIndex(1)
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

        self.sam_apply_mode_label = QLabel(i18n.t("SAM31_APPLY_MODE"))
        self.sam_apply_mode_label.setToolTip(i18n.tip("SAM31_APPLY_MODE"))
        yolo_settings_row.addWidget(self.sam_apply_mode_label)

        self.sam_apply_mode_combo = QComboBox()
        self.sam_apply_mode_combo.setToolTip(i18n.tip("SAM31_APPLY_MODE"))
        self.sam_apply_mode_combo.addItem(i18n.t("SAM31_APPLY_REPLACE"), _SAM31_MERGE_REPLACE)
        self.sam_apply_mode_combo.addItem(i18n.t("SAM31_APPLY_ADD"), _SAM31_MERGE_ADD)
        self.sam_apply_mode_combo.addItem(i18n.t("SAM31_APPLY_SUBTRACT"), _SAM31_MERGE_SUBTRACT)
        self.sam_apply_mode_combo.setFixedWidth(106)
        yolo_settings_row.addWidget(self.sam_apply_mode_combo)
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
        self.yolo_bottom_settings_row.setVisible(False)
        yolo_layout.addWidget(bottom_settings_row_widget)

        class_list_section = CollapsibleSection(i18n.t("DETECTION_TARGET_SECTION"), expanded=False)
        class_list_section.setToolTip(i18n.tip("YOLO_CLASS_LIST_SECTION"))
        class_list_section.toggle_button.setToolTip(i18n.tip("YOLO_CLASS_LIST_SECTION"))
        self.yolo_class_list_section = class_list_section
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
        self.mask_settings_tabs.addTab(self.yolo_section, i18n.t("MASK_TAB_YOLO"))

        self.ade_class_list_section = CollapsibleSection(i18n.t("DETECTION_TARGET_SECTION"), expanded=False)
        self.ade_class_list_section.setToolTip(i18n.tip("ADE20K_CLASS_LIST_SECTION"))
        self.ade_class_list_section.toggle_button.setToolTip(i18n.tip("ADE20K_CLASS_LIST_SECTION"))
        ade_scroll = QScrollArea()
        ade_scroll.setWidgetResizable(True)
        ade_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ade_scroll.setMaximumHeight(180)
        ade_grid_widget = QWidget()
        ade_grid = QGridLayout(ade_grid_widget)
        ade_grid.setSpacing(2)
        self.ade_class_names = _ade20k_class_names(self.base_dir)
        self.ade_class_cbs: list[QCheckBox] = []
        for idx, name in enumerate(self.ade_class_names):
            cb = QCheckBox(f"{idx}: {name.strip()}")
            if name.strip().lower() in {"person", "sky"}:
                cb.setChecked(True)
            self.ade_class_cbs.append(cb)
            ade_grid.addWidget(cb, idx // cols, idx % cols)
        ade_scroll.setWidget(ade_grid_widget)
        self.ade_class_list_section.content_layout.addWidget(ade_scroll)
        yolo_layout.addWidget(self.ade_class_list_section)

        self.sam_prompt_section = CollapsibleSection(i18n.t("DETECTION_TARGET_SECTION"), expanded=False)
        self.sam_prompt_section.setToolTip(i18n.tip("SAM31_PROMPT_SECTION"))
        self.sam_prompt_section.toggle_button.setToolTip(i18n.tip("SAM31_PROMPT_SECTION"))

        self.sam_custom_prompt_row = QHBoxLayout()
        self.sam_custom_prompt_row.setContentsMargins(0, 0, 0, 0)
        self.sam_custom_prompt_row.setSpacing(6)
        self.sam_custom_prompt_icon = QLabel()
        self.sam_custom_prompt_icon.setPixmap(plus_icon(16).pixmap(QSize(16, 16)))
        self.sam_custom_prompt_icon.setToolTip(i18n.tip("SAM31_CUSTOM_PROMPT"))
        self.sam_custom_prompt_icon.setFixedSize(20, 20)
        self.sam_custom_prompt_edit = QLineEdit()
        self.sam_custom_prompt_edit.setPlaceholderText(i18n.t("SAM31_CUSTOM_PROMPT_PLACEHOLDER"))
        self.sam_custom_prompt_edit.setToolTip(i18n.tip("SAM31_CUSTOM_PROMPT"))
        self.sam_custom_prompt_row.addWidget(self.sam_custom_prompt_icon)
        self.sam_custom_prompt_row.addWidget(self.sam_custom_prompt_edit)
        self.sam_prompt_section.content_layout.addLayout(self.sam_custom_prompt_row)

        self.sam_subtract_prompt_row = QHBoxLayout()
        self.sam_subtract_prompt_row.setContentsMargins(0, 0, 0, 0)
        self.sam_subtract_prompt_row.setSpacing(6)
        self.sam_subtract_prompt_icon = QLabel()
        self.sam_subtract_prompt_icon.setPixmap(minus_icon(16).pixmap(QSize(16, 16)))
        self.sam_subtract_prompt_icon.setToolTip(i18n.tip("SAM31_SUBTRACT_PROMPT"))
        self.sam_subtract_prompt_icon.setFixedSize(20, 20)
        self.sam_subtract_prompt_edit = QLineEdit()
        self.sam_subtract_prompt_edit.setPlaceholderText(i18n.t("SAM31_SUBTRACT_PROMPT_PLACEHOLDER"))
        self.sam_subtract_prompt_edit.setToolTip(i18n.tip("SAM31_SUBTRACT_PROMPT"))
        self.sam_subtract_prompt_row.addWidget(self.sam_subtract_prompt_icon)
        self.sam_subtract_prompt_row.addWidget(self.sam_subtract_prompt_edit)
        self.sam_prompt_section.content_layout.addLayout(self.sam_subtract_prompt_row)

        self.sam_prompt_grid_widget = QWidget()
        sam_grid = QGridLayout(self.sam_prompt_grid_widget)
        sam_grid.setSpacing(2)
        self.sam_prompt_cbs: list[tuple[str, QCheckBox]] = []
        for idx, (prompt, label) in enumerate(_SAM31_PROMPT_PRESETS):
            cb = QCheckBox(f"{label} ({prompt})")
            if prompt in {"person", "sky"}:
                cb.setChecked(True)
            self.sam_prompt_cbs.append((prompt, cb))
            sam_grid.addWidget(cb, idx // cols, idx % cols)
        self.sam_prompt_section.content_layout.addWidget(self.sam_prompt_grid_widget)
        yolo_layout.addWidget(self.sam_prompt_section)

        # --- スティッチ+白飛び設定 ---
        self.other_section = QWidget()
        other_layout = QVBoxLayout(self.other_section)
        other_layout.setContentsMargins(8, 8, 8, 8)
        other_layout.setSpacing(6)

        stitch_row = QHBoxLayout()
        stitch_row.setContentsMargins(0, 0, 0, 0)
        stitch_row.setSpacing(6)

        self.stitch_boundary_width_edit = DragDoubleSpinBox(
            minimum=_STITCH_BOUNDARY_MIN,
            maximum=_STITCH_BOUNDARY_MAX,
            step=0.5,
            decimals=1,
            value=_STITCH_BOUNDARY_DEFAULT,
        )
        self.stitch_boundary_width_edit.setToolTip(i18n.tip("STITCH_BOUNDARY_WIDTH"))
        self.stitch_boundary_width_edit.setFixedWidth(80)
        self.stitch_boundary_width_label = QLabel(i18n.t("STITCH_BOUNDARY_WIDTH_COMPACT"))
        self.stitch_boundary_width_label.setToolTip(i18n.tip("STITCH_BOUNDARY_WIDTH"))
        stitch_row.addWidget(self.stitch_boundary_width_label)
        stitch_row.addWidget(self.stitch_boundary_width_edit)

        worker_default = os.cpu_count() or 4
        self.stitch_workers_edit = DragSpinBox(
            minimum=1,
            maximum=max(1, worker_default * 2),
            step=1,
            value=worker_default,
        )
        self.stitch_workers_edit.setToolTip(i18n.tip("STITCH_WORKERS"))
        self.stitch_workers_edit.setFixedWidth(70)
        self.stitch_workers_label = QLabel(i18n.t("STITCH_WORKERS_COMPACT"))
        self.stitch_workers_label.setToolTip(i18n.tip("STITCH_WORKERS"))
        stitch_row.addWidget(self.stitch_workers_label)
        stitch_row.addWidget(self.stitch_workers_edit)
        stitch_row.addStretch()
        other_layout.addLayout(stitch_row)

        overexp_row = QHBoxLayout()
        overexp_row.setContentsMargins(0, 0, 0, 0)
        overexp_row.setSpacing(6)

        self.overexp_threshold_edit = DragSpinBox(
            minimum=_OVEREXP_THRESHOLD_MIN,
            maximum=_OVEREXP_THRESHOLD_MAX,
            step=1,
            value=_OVEREXP_THRESHOLD_DEFAULT,
            drag_pixels_per_step=4.0,
        )
        self.overexp_threshold_edit.setToolTip(i18n.tip("OVEREXPOSURE_THRESHOLD"))
        self.overexp_threshold_edit.setFixedWidth(74)
        self.overexp_threshold_label = QLabel(i18n.t("OVEREXPOSURE_THRESHOLD_COMPACT"))
        self.overexp_threshold_label.setToolTip(i18n.tip("OVEREXPOSURE_THRESHOLD"))
        overexp_row.addWidget(self.overexp_threshold_label)
        overexp_row.addWidget(self.overexp_threshold_edit)

        self.overexp_dilate_edit = DragSpinBox(
            minimum=_OVEREXP_DILATE_MIN,
            maximum=_OVEREXP_DILATE_MAX,
            step=1,
            value=_OVEREXP_DILATE_DEFAULT,
            drag_pixels_per_step=6.0,
        )
        self.overexp_dilate_edit.setToolTip(i18n.tip("OVEREXPOSURE_DILATE"))
        self.overexp_dilate_edit.setFixedWidth(74)
        self.overexp_dilate_label = QLabel(i18n.t("OVEREXPOSURE_DILATE_COMPACT"))
        self.overexp_dilate_label.setToolTip(i18n.tip("OVEREXPOSURE_DILATE"))
        overexp_row.addWidget(self.overexp_dilate_label)
        overexp_row.addWidget(self.overexp_dilate_edit)
        overexp_row.addStretch()
        other_layout.addLayout(overexp_row)

        custom_form = QFormLayout()
        custom_form.setSpacing(6)
        self.custom_mask_path_row = QWidget()
        custom_mask_path_row = QHBoxLayout(self.custom_mask_path_row)
        custom_mask_path_row.setContentsMargins(0, 0, 0, 0)
        custom_mask_path_row.setSpacing(6)
        self.custom_mask_path_label = QLabel(i18n.t("CUSTOM_MASK_NOT_SELECTED"))
        self.custom_mask_path_label.setToolTip(i18n.tip("CUSTOM_MASK_FILE"))
        self.custom_mask_path_label.setWordWrap(True)
        self.custom_mask_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        custom_mask_path_row.addWidget(self.custom_mask_path_label, stretch=1)

        self.custom_mask_browse_btn = QToolButton()
        self.custom_mask_browse_btn.setObjectName("iconToolButton")
        self.custom_mask_browse_btn.setIcon(file_picker_icon())
        self.custom_mask_browse_btn.setToolTip(i18n.tip("CUSTOM_MASK_BROWSE"))
        self.custom_mask_browse_btn.setAccessibleName(i18n.t("CUSTOM_MASK_BROWSE"))
        self.custom_mask_browse_btn.setFixedSize(32, 32)
        custom_mask_path_row.addWidget(self.custom_mask_browse_btn)

        self.custom_mask_clear_btn = QToolButton()
        self.custom_mask_clear_btn.setObjectName("iconToolButton")
        self.custom_mask_clear_btn.setIcon(delete_icon())
        self.custom_mask_clear_btn.setToolTip(i18n.tip("CUSTOM_MASK_CLEAR"))
        self.custom_mask_clear_btn.setAccessibleName(i18n.t("CUSTOM_MASK_CLEAR"))
        self.custom_mask_clear_btn.setFixedSize(32, 32)
        custom_mask_path_row.addWidget(self.custom_mask_clear_btn)

        add_tooltip_row(
            custom_form,
            i18n.t("CUSTOM_MASK_FILE"),
            self.custom_mask_path_row,
            i18n.tip("CUSTOM_MASK_FILE"),
        )
        other_layout.addLayout(custom_form)
        other_layout.addStretch()
        self.custom_section = self.other_section
        self.mask_settings_tabs.addTab(self.other_section, i18n.t("MASK_TAB_OPTIONS"))

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
        self.sky_backend_combo.setItemData(0, i18n.tip("SKY_MODEL_MASK2FORMER"), Qt.ToolTipRole)
        self.sky_backend_combo.setItemData(1, i18n.tip("SKY_MODEL_SAM31"), Qt.ToolTipRole)
        self.sky_backend_combo.setToolTip(i18n.tip("SKY_MODEL"))
        self.sky_backend_combo.setFixedWidth(132)
        add_tooltip_row(sky_form, i18n.t("SKY_MODEL"), self.sky_backend_combo, i18n.tip("SKY_MODEL"))
        self.sky_backend_label = sky_form.labelForField(self.sky_backend_combo)
        self.sky_backend_combo.setVisible(False)
        if self.sky_backend_label is not None:
            self.sky_backend_label.setVisible(False)

        self.sky_mode_combo = QComboBox()
        self.sky_mode_combo.addItems(
            [
                i18n.t("SKY_MODE_FULL"),
                i18n.t("SKY_MODE_DIRECT"),
                i18n.t("SKY_MODE_TOP"),
                i18n.t("SKY_MODE_BOTTOM"),
                i18n.t("SKY_MODE_HYBRID"),
            ]
        )
        self.sky_mode_combo.setToolTip(i18n.tip("SKY_MODE"))
        self.sky_mode_combo.setFixedWidth(132)
        add_tooltip_row(sky_form, i18n.t("SKY_MODE"), self.sky_mode_combo, i18n.tip("SKY_MODE"))
        self.sky_mode_field_label = sky_form.labelForField(self.sky_mode_combo)
        self.sky_mode_combo.setVisible(False)
        if self.sky_mode_field_label is not None:
            self.sky_mode_field_label.setVisible(False)

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
        self.sky_expand_label = sky_form.labelForField(self.sky_expand_edit)
        self.sky_expand_edit.setVisible(False)
        if self.sky_expand_label is not None:
            self.sky_expand_label.setVisible(False)

        sky_layout.addLayout(sky_form)

        self.sky_model_details_section = CollapsibleSection(i18n.t("SKY_MODEL_DETAILS_SECTION"), expanded=False)
        sky_model_details_form = QFormLayout()
        sky_model_details_form.setSpacing(6)

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
            sky_model_details_form,
            i18n.t("SKY_MIN_SCORE"),
            self.sky_min_score_edit,
            i18n.tip("SKY_MIN_SCORE"),
        )
        self.sky_min_score_label = sky_model_details_form.labelForField(self.sky_min_score_edit)
        self.sky_model_details_section.content_layout.addLayout(sky_model_details_form)
        sky_layout.addWidget(self.sky_model_details_section)

        self.sky_postprocess_section = CollapsibleSection(i18n.t("SKY_POSTPROCESS_SECTION"), expanded=False)
        sky_postprocess_row = QHBoxLayout()
        sky_postprocess_row.setContentsMargins(0, 0, 0, 0)
        sky_postprocess_row.setSpacing(6)

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
        self.sky_min_area_edit.setFixedWidth(74)
        self.sky_min_area_label = QLabel(i18n.t("SKY_MIN_AREA"))
        self.sky_min_area_label.setToolTip(i18n.tip("SKY_MIN_AREA"))
        sky_postprocess_row.addWidget(self.sky_min_area_label)
        sky_postprocess_row.addWidget(self.sky_min_area_edit)

        self.sky_top_connected_cb = QCheckBox(i18n.t("SKY_TOP_CONNECTED"))
        self.sky_top_connected_cb.setToolTip(i18n.tip("SKY_TOP_CONNECTED"))
        self.sky_top_connected_cb.setChecked(False)
        sky_postprocess_row.addWidget(self.sky_top_connected_cb)
        sky_postprocess_row.addStretch()

        self.sky_postprocess_section.content_layout.addLayout(sky_postprocess_row)
        sky_layout.addWidget(self.sky_postprocess_section)
        yolo_layout.addWidget(self.sky_section)
        yolo_layout.addStretch()

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

        for cb in (self.run_stitch_cb, self.run_overexp_cb):
            cb.toggled.connect(self._update_task_controls)
        self.run_custom_cb.toggled.connect(self._on_custom_mask_toggled)
        self.custom_mask_browse_btn.clicked.connect(lambda _checked=False: self._browse_custom_mask(activate=True))
        self.custom_mask_clear_btn.clicked.connect(lambda _checked=False: self._clear_custom_mask_path())
        self.stitch_boundary_width_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.person_backend_combo.currentIndexChanged.connect(lambda _: self._on_person_backend_changed())
        self.yolo_level_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.yolo_expand_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.yolo_bottom_enhance_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
        for cb in self.class_cbs:
            cb.toggled.connect(lambda _checked=False: self._schedule_render_mask_preview())
        for cb in self.ade_class_cbs:
            cb.toggled.connect(lambda _checked=False: self._schedule_render_mask_preview())
        for _prompt, cb in self.sam_prompt_cbs:
            cb.toggled.connect(lambda _checked=False: self._schedule_render_mask_preview())
        self.sam_custom_prompt_edit.textChanged.connect(lambda _text: self._schedule_render_mask_preview())
        self.sam_subtract_prompt_edit.textChanged.connect(lambda _text: self._schedule_render_mask_preview())
        self.sam_apply_mode_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
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
        requested_steps = ["yolo"]
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
        self.yolo_level_combo.setCurrentIndex(0 if projection == _PROJECTION_NORMAL else 1)
        self._update_task_controls()

    def _scene_csv_path(self) -> Path:
        return selected_frames_path(Path(self.scene_dir))

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
        self._readiness()
        self.primary_action_state_changed.emit()

    def _selected_classes(self) -> list[int]:
        return [i for i, cb in enumerate(self.class_cbs) if cb.isChecked()]

    def _selected_ade_labels(self) -> list[str]:
        labels = [name.strip() for name, cb in zip(self.ade_class_names, self.ade_class_cbs, strict=True) if cb.isChecked()]
        return [label for label in labels if label] or ["person", "sky"]

    @staticmethod
    def _split_sam_prompt_text(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"[,;\n]", text) if part.strip()]

    def _selected_sam_prompts(self) -> list[str]:
        prompts = [prompt for prompt, cb in self.sam_prompt_cbs if cb.isChecked()]
        prompts.extend(self._split_sam_prompt_text(self.sam_custom_prompt_edit.text()))
        return list(dict.fromkeys(prompts)) or [_PERSON_SAM31_PROMPT]

    def _selected_sam_subtract_prompts(self) -> list[str]:
        return list(dict.fromkeys(self._split_sam_prompt_text(self.sam_subtract_prompt_edit.text())))

    def _sam31_merge_mode_arg(self) -> str:
        data = self.sam_apply_mode_combo.currentData()
        mode = str(data or _SAM31_MERGE_REPLACE)
        return mode if mode in _SAM31_MERGE_MODES else _SAM31_MERGE_REPLACE

    def _person_backend_arg(self) -> str:
        idx = self.person_backend_combo.currentIndex()
        return _PERSON_BACKENDS[max(0, min(idx, len(_PERSON_BACKENDS) - 1))]

    def _person_uses_sam31(self) -> bool:
        return self._person_backend_arg() == "sam31"

    def _person_uses_mask2former(self) -> bool:
        return self._person_backend_arg() == "mask2former"

    def _on_person_backend_changed(self) -> None:
        if self._person_uses_mask2former():
            self.sky_backend_combo.setCurrentIndex(0)
        elif self._person_uses_sam31():
            self.sky_backend_combo.setCurrentIndex(1)
            if self.sky_inference_size_combo.currentText() != _SKY_SAM31_INFERENCE_SIZE:
                idx = self.sky_inference_size_combo.findText(_SKY_SAM31_INFERENCE_SIZE)
                if idx >= 0:
                    self.sky_inference_size_combo.setCurrentIndex(idx)
        else:
            self.sky_backend_combo.setCurrentIndex(0)
        self._update_task_controls()

    def _yolo_expand_arg(self) -> str:
        return str(self.yolo_expand_edit.value())

    def _quality_arg(self) -> str:
        idx = max(0, min(self.yolo_level_combo.currentIndex(), len(QUALITY_CHOICES) - 1))
        return QUALITY_CHOICES[idx]

    def _bottom_enhance_args(self) -> list[str]:
        return []

    def _sky_backend_arg(self) -> str:
        idx = self.sky_backend_combo.currentIndex()
        return _SKY_BACKENDS[max(0, min(idx, len(_SKY_BACKENDS) - 1))]

    def _sam31_checkpoint_path(self) -> Path:
        return self.base_dir / _SKY_SAM31_CHECKPOINT

    def _sam31_available(self) -> bool:
        return self._sam31_checkpoint_path().is_file()

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
        modes = ("full", "direct", "top", "bottom", "hybrid")
        return modes[max(0, min(idx, len(modes) - 1))]

    def _sky_inference_size_arg(self) -> str:
        text = self.sky_inference_size_combo.currentText().strip()
        return text if text else str(_SKY_INFERENCE_SIZES[_SKY_INFERENCE_SIZE_DEFAULT_INDEX])

    def _sky_min_area_ratio_arg(self) -> str:
        return f"{float(self.sky_min_area_edit.value()) / 100.0:g}"

    def _sky_postprocess_args(self) -> list[str]:
        args = ["--min-area-ratio", self._sky_min_area_ratio_arg()]
        if self.sky_top_connected_cb.isChecked():
            args.append("--top-connected")
        return args

    def _update_task_controls(self) -> None:
        equirect = self._projection() == _PROJECTION_EQUIRECT
        if not equirect and self.run_stitch_cb.isChecked():
            self.run_stitch_cb.setChecked(False)
            return
        stitch_enabled = equirect and self.run_stitch_cb.isChecked()
        overexp_enabled = self.run_overexp_cb.isChecked()
        custom_enabled = self.run_custom_cb.isChecked()

        self._update_person_backend_availability()
        model = self._person_backend_arg()
        person_sam31 = self._person_uses_sam31()
        person_mask2former = self._person_uses_mask2former()
        yolo_sam_enabled = model == "yolo_sam"
        semantic_enabled = model in {"mask2former", "sam31"}
        self.yolo_section.setEnabled(True)
        self.sky_section.setVisible(semantic_enabled)
        self.ade_class_list_section.setVisible(person_mask2former)
        self.sam_prompt_section.setVisible(person_sam31)
        self.yolo_class_list_section.setVisible(yolo_sam_enabled)
        self.yolo_level_label.setVisible(True)
        self.yolo_level_combo.setVisible(True)
        self.yolo_level_label.setEnabled(True)
        self.yolo_level_combo.setEnabled(True)
        self.yolo_expand_label.setEnabled(True)
        self.yolo_expand_edit.setEnabled(True)
        self.sam_apply_mode_label.setVisible(person_sam31)
        self.sam_apply_mode_combo.setVisible(person_sam31)
        self.yolo_bottom_settings_row.setVisible(False)
        self.yolo_bottom_settings_row.setEnabled(False)
        self.yolo_bottom_enhance_label.setEnabled(False)
        self.yolo_bottom_enhance_combo.setEnabled(False)
        self.yolo_class_list_section.setEnabled(yolo_sam_enabled)
        self.sky_inference_size_combo.setEnabled(person_mask2former)
        self.sky_inference_size_label.setEnabled(person_mask2former)
        self.sky_model_details_section.setVisible(person_mask2former)
        self.sky_postprocess_section.setVisible(semantic_enabled)
        self.sky_min_score_edit.setVisible(person_mask2former)
        if self.sky_min_score_label is not None:
            self.sky_min_score_label.setVisible(person_mask2former)
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

    def _update_person_backend_availability(self) -> None:
        sam_available = self._sam31_available()
        model = self.person_backend_combo.model()
        item = model.item(2) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(True)
            item.setToolTip(
                i18n.tip("PERSON_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD")
            )
            self.person_backend_combo.setItemData(
                2,
                i18n.tip("PERSON_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD"),
                Qt.ToolTipRole,
            )

    def _update_sky_backend_availability(self) -> None:
        sam_available = self._sam31_available()
        model = self.sky_backend_combo.model()
        item = model.item(1) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(True)
            item.setToolTip(
                i18n.tip("SKY_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD")
            )
            self.sky_backend_combo.setItemData(
                1,
                i18n.tip("SKY_MODEL_SAM31") if sam_available else i18n.tip("SAM31_CHECKPOINT_DOWNLOAD"),
                Qt.ToolTipRole,
            )

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
        added, skipped = import_external_images(source_dir, images_dir)

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
            use_yolo=True,
            use_stitch=self.run_stitch_cb.isChecked(),
            use_overexposure=self.run_overexp_cb.isChecked(),
            use_sky=False,
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
            self._person_backend_arg(),
            self.yolo_level_combo.currentIndex(),
            int(self.yolo_expand_edit.value()),
            tuple(self._selected_classes()),
            tuple(self._selected_ade_labels()),
            tuple(self._selected_sam_prompts()),
            tuple(self._selected_sam_subtract_prompts()),
            self._sam31_merge_mode_arg(),
            self._sky_backend_arg(),
            self._sky_inference_size_arg(),
            int(self.sky_expand_edit.value()),
            float(self.sky_min_score_edit.value()),
            float(self.sky_min_area_edit.value()),
            bool(self.sky_top_connected_cb.isChecked()),
        )

    # -- コマンド構築 --

    def phase_display_name(self, phase: str) -> str:
        labels = {
            "yolo": "MASK_PHASE_PRIMARY",
            "stitch": "MASK_PHASE_STITCH",
            "overexposure": "MASK_PHASE_OVEREXPOSURE",
            "custom": "MASK_PHASE_CUSTOM",
            "init_masks": "MASK_PHASE_INIT",
        }
        key = labels.get(phase)
        return i18n.t(key) if key else phase

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
        if "custom" in requested_steps:
            steps.append(("custom", self._build_custom_cmd(replace=fresh_base_needed)))
            fresh_base_needed = False
        return steps

    def confirm_commands(self, commands: list[tuple[str, list[str]]]) -> bool:
        if any(phase == "yolo" for phase, _cmd in commands):
            if self._person_backend_arg() == "yolo_sam":
                if not self._confirm_yolo_sam_license_notice():
                    return False
            else:
                if not self._confirm_sky_license_notice():
                    return False
                if self._uses_sam31_for_primary_mask() and not self._ensure_sam31_checkpoint_available():
                    return False
        return True

    def _uses_sam31_for_primary_mask(self) -> bool:
        return self._person_uses_sam31() or self._sky_backend_arg() == "sam31"

    def _ensure_sam31_checkpoint_available(self) -> bool:
        return ensure_sam31_checkpoint_available(
            self,
            self._sam31_checkpoint_path(),
            on_available=self._refresh_sam31_backend_availability,
        )

    def _refresh_sam31_backend_availability(self) -> None:
        self._update_person_backend_availability()
        self._update_sky_backend_availability()

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
        csv_path = selected_frames_path(scene_dir)
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
        csv_path = selected_frames_path(scene_dir)
        if not csv_path.exists():
            return

        untracked = untracked_image_paths(scene_dir, images_dir=Path(images))
        if not untracked:
            return

        preview = "\n".join(f"- {p.name}" for p in untracked[:5])
        if len(untracked) > 5:
            preview += f"\n- ... +{len(untracked) - 5}"
        raise ValueError(i18n.t("MASK_UNTRACKED_IMAGES_ERROR").format(n=len(untracked), files=preview))

    def _mask_command_context(self) -> MaskCommandContext:
        return MaskCommandContext(
            python_executable=sys.executable,
            base_dir=self.base_dir,
            projection=self._projection(),
            quality=self._quality_arg(),
            yolo_expand=self._yolo_expand_arg(),
            sky_inference_size=self._sky_inference_size_arg(),
            sky_min_score=f"{float(self.sky_min_score_edit.value()):g}",
            sky_min_area_ratio=self._sky_min_area_ratio_arg(),
            sky_top_connected=self.sky_top_connected_cb.isChecked(),
            stitch_boundary_width=self._stitch_boundary_width(),
            stitch_workers=str(self.stitch_workers_edit.value()),
            overexposure_threshold=str(self.overexp_threshold_edit.value()),
            overexposure_dilate=str(self.overexp_dilate_edit.value()),
            custom_mask=self._custom_mask_path_text(),
            yolo_classes=tuple(self._selected_classes()),
            yolo_extra_args=tuple(self._bottom_enhance_args()),
            ade_labels=tuple(self._selected_ade_labels()),
            sam_prompts=tuple(self._selected_sam_prompts()),
            sam_subtract_prompts=tuple(self._selected_sam_subtract_prompts()),
            sam31_merge_mode=self._sam31_merge_mode_arg(),
        )

    def _build_yolo_cmd(self) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        return build_primary_mask_cmd(
            self._mask_command_context(),
            images,
            masks,
            backend=self._person_backend_arg(),
        )

    def _build_yolo_preview_cmd(self, image_path: Path, output_dir: Path) -> list[str]:
        return build_primary_mask_cmd(
            self._mask_command_context(),
            str(image_path),
            str(output_dir),
            backend=self._person_backend_arg(),
        )

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

    def _build_sam31_prompt_cmd(
        self,
        images: str | Path,
        masks: str | Path,
        *,
        prompts: list[str],
        subtract_prompts: list[str] | None = None,
        merge_mode: str | None = None,
        replace: bool = False,
    ) -> list[str]:
        return build_sam31_prompt_cmd(
            self._mask_command_context(),
            images,
            masks,
            prompts=prompts,
            subtract_prompts=subtract_prompts,
            merge_mode=merge_mode,
            replace=replace,
        )

    def _build_mask2former_cmd(
        self,
        images: str | Path,
        masks: str | Path,
        *,
        replace: bool = False,
    ) -> list[str]:
        return build_mask2former_cmd(
            self._mask_command_context(),
            images,
            masks,
            replace=replace,
        )

    def _build_init_masks_cmd(self) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        return build_init_masks_cmd(self._mask_command_context(), images, masks)

    def _seed_sam31_preview_base_mask(self, image_path: Path, output_path: Path) -> None:
        if not self._person_uses_sam31() or self._sam31_merge_mode_arg() == _SAM31_MERGE_REPLACE:
            return
        existing_path = self._mask_output_path_for_image(image_path)
        if not existing_path.is_file():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing_path, output_path)

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
        if not self._selected_mask_tasks():
            self.mask_preview.set_status_text(i18n.t("MASK_TASK_REQUIRED"))
            return

        if self.mask_preview.preview_mode() == "thumbnails":
            self.mask_preview.set_preview_mode("single")

        if self._person_backend_arg() == "yolo_sam":
            if not self._confirm_yolo_sam_license_notice():
                self.mask_preview.set_status_text(i18n.t("YOLO_SAM_LICENSE_NOTICE_CANCELED"))
                return
        elif not self._confirm_sky_license_notice():
            self.mask_preview.set_status_text(i18n.t("SKY_LICENSE_NOTICE_CANCELED"))
            return
        elif self._uses_sam31_for_primary_mask() and not self._ensure_sam31_checkpoint_available():
            self.mask_preview.set_status_text(i18n.t("SAM31_DOWNLOAD_CANCELED"))
            return

        self._cleanup_mask_preview_temp()
        self._mask_preview_temp = tempfile.TemporaryDirectory(prefix="stechdrive_mask_preview_")
        masks_root = Path(self._mask_preview_temp.name)
        output_path = self._mask_output_path_for_image(image_path, masks_root=masks_root)
        self._seed_sam31_preview_base_mask(image_path, output_path)

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
        if self._person_backend_arg() == "yolo_sam":
            if not self._confirm_yolo_sam_license_notice():
                self.mask_preview.set_status_text(i18n.t("YOLO_SAM_LICENSE_NOTICE_CANCELED"))
                return
        elif not self._confirm_sky_license_notice():
            self.mask_preview.set_status_text(i18n.t("SKY_LICENSE_NOTICE_CANCELED"))
            return
        elif self._uses_sam31_for_primary_mask() and not self._ensure_sam31_checkpoint_available():
            self.mask_preview.set_status_text(i18n.t("SAM31_DOWNLOAD_CANCELED"))
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
        commands.append(("yolo", self._build_yolo_current_cmd(image_path, masks_root=masks_root)))
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
        return build_stitch_cmd(self._mask_command_context(), masks)

    def _build_overexposure_cmd(self, *, replace: bool = False) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        return build_overexposure_cmd(
            self._mask_command_context(),
            images,
            masks,
            replace=replace,
        )

    def _build_custom_cmd(self, *, replace: bool = False) -> list[str]:
        images = self._images_dir_text()
        masks = self._masks_dir_text()
        custom_mask = self._custom_mask_path_text()
        if not custom_mask:
            raise ValueError(i18n.t("CUSTOM_MASK_REQUIRED"))
        return build_custom_cmd(
            self._mask_command_context(),
            images,
            masks,
            replace=replace,
        )

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
