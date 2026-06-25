"""Step 3: マスク生成 (人物 + スティッチ + 白飛び + 空 + カスタム)"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QProcess, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.mask_refresh_plan import (
    MASK_SCOPE_ALL,
    MASK_SCOPE_MISSING,
    MASK_SCOPE_STALE,
)
from core.sky_mask import CITYSCAPES_CLASS_NAMES, DEFAULT_SEMANTIC_LABELS
from gui import i18n
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.common.icons import delete_icon, file_picker_icon, minus_icon, plus_icon
from gui.common.runner_types import ExternalCommandQueue
from gui.mask.mask_preview import MaskPreviewConfig, MaskPreviewWidget
from gui.steps import mask_commands as mask_command_defs
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)
from gui.steps.step3_mask_actions import Step3MaskActionsMixin
from gui.steps.step3_mask_batch import Step3MaskBatchMixin
from gui.steps.step3_mask_controls import (
    _SKY_INFERENCE_SIZE_DEFAULT_INDEX,
    _SKY_INFERENCE_SIZES,
    Step3MaskControlsMixin,
)
from gui.steps.step3_mask_license import Step3MaskLicenseMixin
from gui.steps.step3_mask_progress import MaskProgressParser
from gui.steps.step3_mask_scene import Step3MaskSceneMixin

__all__ = ["MaskStep", "_yolo_preview_output_name"]

_COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

_STITCH_BOUNDARY_MIN = 0.0
_STITCH_BOUNDARY_MAX = 30.0
_STITCH_BOUNDARY_DEFAULT = 5.0
_YOLO_EXPAND_MIN = 0
_YOLO_EXPAND_MAX = 32
_YOLO_EXPAND_DEFAULT = 0
_SAM31_MERGE_REPLACE = mask_command_defs.SAM31_MERGE_REPLACE
_SAM31_MERGE_ADD = mask_command_defs.SAM31_MERGE_ADD
_SAM31_MERGE_SUBTRACT = mask_command_defs.SAM31_MERGE_SUBTRACT
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
_SEMANTIC_DEFAULT_SELECTED_CLASSES = frozenset(label.lower() for label in DEFAULT_SEMANTIC_LABELS)
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
_PROJECTION_EQUIRECT = "equirect"
_PROJECTION_NORMAL = "normal"
_MASK_SCOPE_COMBO_MIN_WIDTH = 180
_MASK_SCOPE_COMBO_MAX_WIDTH = 260
_COMBO_TEXT_PADDING = 48


def _fit_combo_width_to_items(combo: QComboBox, *, min_width: int, max_width: int) -> None:
    metrics = combo.fontMetrics()
    widest = max((metrics.horizontalAdvance(combo.itemText(index)) for index in range(combo.count())), default=0)
    combo.setFixedWidth(min(max_width, max(min_width, widest + _COMBO_TEXT_PADDING)))


class MaskStep(
    Step3MaskControlsMixin,
    Step3MaskActionsMixin,
    Step3MaskSceneMixin,
    Step3MaskBatchMixin,
    Step3MaskLicenseMixin,
    BaseStepWidget,
):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._progress_parser = MaskProgressParser()
        self._mask_preview_proc: QProcess | None = None
        self._mask_preview_temp: tempfile.TemporaryDirectory[str] | None = None
        self._mask_preview_image: Path | None = None
        self._mask_preview_output: Path | None = None
        self._mask_preview_config: MaskPreviewConfig | None = None
        self._mask_preview_commands: ExternalCommandQueue = []
        self._custom_mask_path = ""
        self._current_reprocess_proc: QProcess | None = None
        self._current_reprocess_image: Path | None = None
        self._current_reprocess_mask: Path | None = None
        self._current_reprocess_commands: ExternalCommandQueue = []
        self._current_reprocess_phase = ""
        self._current_reprocess_active = False
        self._current_reprocess_queue: list[Path] = []
        self._current_reprocess_total = 0
        self._current_reprocess_completed = 0
        self._current_reprocess_failed: list[Path] = []
        self._current_reprocess_succeeded: list[Path] = []
        self._current_reprocess_last_success: Path | None = None
        self._current_reprocess_run_id = ""
        self._current_reprocess_settings: dict | None = None
        self._mask_batch_settings: dict | None = None
        self._mask_batch_phases: list[str] = []
        self._mask_batch_targets: list[Path] = []
        self._syncing_mask_source_combo = False
        self._project_projection = _PROJECTION_EQUIRECT
        self._projection_mixed = False
        self._projection_source = "default"
        self._image_projection_map: dict[str, str] = {}
        self._scene_inventory_cache = None
        self._scene_inventory_cache_key: tuple[str, str, str, str] | None = None
        self._scene_inventory_refresh_token: tuple | None = None
        self._scene_inventory_synced_on_scene_change = False
        self._readiness_cache: tuple[bool, str] | None = None
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
        self.settings_scroll = settings_scroll
        self.settings_content = settings
        self.settings_layout = layout

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

        add_tooltip_row(path_form, i18n.IMAGES_DIR, self.images_path_row, i18n.tip("IMAGES_DIR"))
        self.images_path_row_label = path_form.labelForField(self.images_path_row)
        self.masks_path_label = QLabel("-")
        self.masks_path_label.setToolTip(i18n.tip("MASKS_DIR"))
        self.masks_path_label.setWordWrap(True)
        self.masks_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add_tooltip_row(path_form, i18n.MASKS_DIR, self.masks_path_label, i18n.tip("MASKS_DIR"))
        self.masks_path_row_label = path_form.labelForField(self.masks_path_label)

        self.mask_source_combo = QComboBox()
        self.mask_source_combo.setToolTip(i18n.tip("MASK_SOURCE"))
        self.mask_source_combo.setFixedWidth(180)
        self.mask_source_combo.currentIndexChanged.connect(lambda _: self._on_mask_source_changed())
        add_tooltip_row(path_form, i18n.t("MASK_SOURCE"), self.mask_source_combo, i18n.tip("MASK_SOURCE"))
        self.mask_source_row_label = path_form.labelForField(self.mask_source_combo)

        self.mask_scope_combo = QComboBox()
        self.mask_scope_combo.setToolTip(i18n.tip("MASK_SCOPE"))
        self.mask_scope_combo.addItem(i18n.t("MASK_SCOPE_MISSING"), MASK_SCOPE_MISSING)
        self.mask_scope_combo.addItem(i18n.t("MASK_SCOPE_STALE"), MASK_SCOPE_STALE)
        self.mask_scope_combo.addItem(i18n.t("MASK_SCOPE_ALL"), MASK_SCOPE_ALL)
        self.mask_scope_combo.setItemData(0, i18n.tip("MASK_SCOPE_MISSING"), Qt.ToolTipRole)
        self.mask_scope_combo.setItemData(1, i18n.tip("MASK_SCOPE_STALE"), Qt.ToolTipRole)
        self.mask_scope_combo.setItemData(2, i18n.tip("MASK_SCOPE_ALL"), Qt.ToolTipRole)
        _fit_combo_width_to_items(
            self.mask_scope_combo,
            min_width=_MASK_SCOPE_COMBO_MIN_WIDTH,
            max_width=_MASK_SCOPE_COMBO_MAX_WIDTH,
        )
        self.mask_scope_combo.currentIndexChanged.connect(lambda _: self._update_ready_status())
        add_tooltip_row(path_form, i18n.t("MASK_SCOPE"), self.mask_scope_combo, i18n.tip("MASK_SCOPE"))
        layout.addLayout(path_form)

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
                i18n.t("SKY_MODEL_YOLO26_SEM"),
                i18n.t("PERSON_MODEL_SAM31"),
            ]
        )
        self.person_backend_combo.setItemData(0, i18n.tip("PERSON_MODEL_YOLO_SAM"), Qt.ToolTipRole)
        self.person_backend_combo.setItemData(1, i18n.tip("SKY_MODEL_YOLO26_SEM"), Qt.ToolTipRole)
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

        yolo_settings_row.addStretch()
        self.yolo_settings_row = yolo_settings_row_widget
        yolo_layout.addWidget(yolo_settings_row_widget)

        mask_apply_row_widget = QWidget()
        mask_apply_row = QHBoxLayout(mask_apply_row_widget)
        mask_apply_row.setContentsMargins(0, 0, 0, 0)
        mask_apply_row.setSpacing(6)
        self.mask_apply_mode_label = QLabel(i18n.t("MASK_APPLY_MODE"))
        self.mask_apply_mode_label.setToolTip(i18n.tip("MASK_APPLY_MODE"))
        mask_apply_row.addWidget(self.mask_apply_mode_label)
        self.mask_apply_mode_combo = QComboBox()
        self.mask_apply_mode_combo.setToolTip(i18n.tip("MASK_APPLY_MODE"))
        self.mask_apply_mode_combo.addItem(i18n.t("MASK_APPLY_REPLACE"), _SAM31_MERGE_REPLACE)
        self.mask_apply_mode_combo.addItem(i18n.t("MASK_APPLY_ADD"), _SAM31_MERGE_ADD)
        self.mask_apply_mode_combo.addItem(i18n.t("MASK_APPLY_SUBTRACT"), _SAM31_MERGE_SUBTRACT)
        self.mask_apply_mode_combo.setFixedWidth(118)
        mask_apply_row.addWidget(self.mask_apply_mode_combo)
        mask_apply_row.addStretch()
        self.mask_apply_row = mask_apply_row_widget
        self.sam_apply_mode_label = self.mask_apply_mode_label
        self.sam_apply_mode_combo = self.mask_apply_mode_combo
        yolo_layout.addWidget(mask_apply_row_widget)

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

        self.semantic_class_list_section = CollapsibleSection(i18n.t("DETECTION_TARGET_SECTION"), expanded=False)
        self.semantic_class_list_section.setToolTip(i18n.tip("CITYSCAPES_CLASS_LIST_SECTION"))
        self.semantic_class_list_section.toggle_button.setToolTip(i18n.tip("CITYSCAPES_CLASS_LIST_SECTION"))
        semantic_scroll = QScrollArea()
        semantic_scroll.setWidgetResizable(True)
        semantic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        semantic_scroll.setMaximumHeight(180)
        semantic_grid_widget = QWidget()
        semantic_grid = QGridLayout(semantic_grid_widget)
        semantic_grid.setSpacing(2)
        self.semantic_class_names = CITYSCAPES_CLASS_NAMES
        self.semantic_class_cbs: list[QCheckBox] = []
        for idx, name in enumerate(self.semantic_class_names):
            cb = QCheckBox(f"{idx}: {name.strip()}")
            if name.strip().lower() in _SEMANTIC_DEFAULT_SELECTED_CLASSES:
                cb.setChecked(True)
            self.semantic_class_cbs.append(cb)
            semantic_grid.addWidget(cb, idx // cols, idx % cols)
        semantic_scroll.setWidget(semantic_grid_widget)
        self.semantic_class_list_section.content_layout.addWidget(semantic_scroll)
        yolo_layout.addWidget(self.semantic_class_list_section)

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
                i18n.t("SKY_MODEL_YOLO26_SEM"),
                i18n.t("SKY_MODEL_SAM31"),
            ]
        )
        self.sky_backend_combo.setItemData(0, i18n.tip("SKY_MODEL_YOLO26_SEM"), Qt.ToolTipRole)
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
        for cb in self.semantic_class_cbs:
            cb.toggled.connect(lambda _checked=False: self._schedule_render_mask_preview())
        for _prompt, cb in self.sam_prompt_cbs:
            cb.toggled.connect(lambda _checked=False: self._schedule_render_mask_preview())
        self.sam_custom_prompt_edit.textChanged.connect(lambda _text: self._schedule_render_mask_preview())
        self.sam_subtract_prompt_edit.textChanged.connect(lambda _text: self._schedule_render_mask_preview())
        self.mask_apply_mode_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.overexp_threshold_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.overexp_dilate_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_backend_combo.currentIndexChanged.connect(lambda _: self._on_sky_backend_changed())
        self.sky_mode_combo.currentIndexChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_inference_size_combo.currentIndexChanged.connect(lambda _: self._on_sky_inference_size_changed())
        self.sky_expand_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_min_score_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_min_area_edit.valueChanged.connect(lambda _: self._schedule_render_mask_preview())
        self.sky_top_connected_cb.toggled.connect(lambda _: self._schedule_render_mask_preview())
        self.mask_preview.current_image_changed.connect(self._on_preview_current_image_changed)
        self.mask_preview.mask_preview_requested.connect(self._run_mask_preview)
        self.mask_preview.current_reprocess_requested.connect(self._run_current_image_reprocess)
        self._set_projection(_PROJECTION_EQUIRECT)
        self._refresh_mask_source_options()
        self._update_task_controls()
        self._on_images_dir_changed(self._images_dir_text())
        self._update_ready_status()

    def shutdown(self) -> None:
        self.mask_preview.shutdown()


def _yolo_preview_output_name(image_path: Path) -> str:
    return f"{image_path.stem}.png"
