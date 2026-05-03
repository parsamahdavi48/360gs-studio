"""Step 3: マスク生成 (YOLO + スティッチ + 白飛び)"""
from __future__ import annotations

import os
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

from apply_frame_decisions import pending_drop_image_paths, untracked_image_paths
from PySide6.QtCore import QProcess, Qt, QUrl, Signal
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
    ("strong", ["--bottom-conf", "0.15", "--bottom-tta-rotations", "4", "--bottom-temporal-window", "2"]),
    (
        "max",
        [
            "--bottom-conf", "0.10",
            "--bottom-tta-rotations", "4",
            "--bottom-model", "x",
            "--bottom-temporal-window", "4",
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
        preview_title = QLabel(i18n.t("MASK_PREVIEW_SECTION"))
        preview_title.setObjectName("paneTitle")
        preview_layout.addWidget(preview_title)
        self.mask_preview = MaskPreviewWidget()
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
        self.stitch_boundary_width_edit.valueChanged.connect(lambda _: self._render_mask_preview())
        self.overexp_threshold_edit.valueChanged.connect(lambda _: self._render_mask_preview())
        self.overexp_dilate_edit.valueChanged.connect(lambda _: self._render_mask_preview())
        self.mask_preview.current_image_changed.connect(lambda: self._render_mask_preview())
        self.mask_preview.opacity_slider.valueChanged.connect(lambda _: self._render_mask_preview())
        self.mask_preview.yolo_preview_requested.connect(self._run_yolo_preview)
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
        self._render_mask_preview()
        self._update_ready_status()

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

    def _run_yolo_preview(self) -> None:
        if self._yolo_preview_proc is not None and self._yolo_preview_proc.state() != QProcess.NotRunning:
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


def _yolo_preview_output_name(image_path: Path) -> str:
    return f"{image_path.stem}.png"
