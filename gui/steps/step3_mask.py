"""Step 3: マスク生成 (YOLO + スティッチ + 白飛び)"""
from __future__ import annotations

import os
import math
import re
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.mask.stitch_preview import StitchPreviewWidget
from gui.steps.base_step import BaseStepWidget

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

_YOLO_LINE_RE = re.compile(r"^Processing:\s+")
_STITCH_TASK_RE = re.compile(r"^Processing\s+(\d+)\s+images\s+with\s+\d+\s+workers\.\.\.$")
_STITCH_TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
_STITCH_BOUNDARY_MIN = 0.0
_STITCH_BOUNDARY_MAX = 30.0
_STITCH_BOUNDARY_DEFAULT = 5.0
_YOLO_EXPAND_MIN = -64
_YOLO_EXPAND_MAX = 256
_YOLO_EXPAND_DEFAULT = 12
_OVEREXP_THRESHOLD_MIN = 0
_OVEREXP_THRESHOLD_MAX = 255
_OVEREXP_THRESHOLD_DEFAULT = 250
_OVEREXP_DILATE_MIN = 0
_OVEREXP_DILATE_MAX = 128
_OVEREXP_DILATE_DEFAULT = 8


class MaskStep(BaseStepWidget):
    run_requested = Signal()

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._phase_total = 0
        self._phase_done = 0
        self._stitch_chunk_total = 0
        self._stitch_chunk_done = 0
        self._stitch_done_before = 0
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        top_scroll = QScrollArea()
        top_scroll.setWidgetResizable(True)
        top_scroll.setFrameShape(QScrollArea.NoFrame)
        top = QWidget()
        layout = QVBoxLayout(top)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(8)

        # --- パス設定 ---
        path_form = QFormLayout()
        path_form.setSpacing(6)
        self.images_browse = BrowseWidget(mode="dir", placeholder="元画像のフォルダ")
        self.images_browse.setToolTip(i18n.tip("IMAGES_DIR"))
        path_form.addRow(i18n.IMAGES_DIR, self.images_browse)
        self.masks_browse = BrowseWidget(mode="dir", placeholder="マスク出力先")
        self.masks_browse.setToolTip(i18n.tip("MASKS_DIR"))
        path_form.addRow(i18n.MASKS_DIR, self.masks_browse)
        layout.addLayout(path_form)

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

        self.run_masks_btn = QPushButton(f"  {i18n.t('RUN_MASKS')}")
        self.run_masks_btn.setToolTip(i18n.tip("RUN_MASKS"))
        self.run_masks_btn.setObjectName("primary")
        self.run_masks_btn.setFixedHeight(34)
        self.run_masks_btn.setMinimumWidth(180)
        self.run_masks_btn.clicked.connect(self.run_requested.emit)
        task_row.addWidget(self.run_masks_btn)

        layout.addLayout(task_row)

        # --- YOLO設定 (折りたたみ) ---
        self.yolo_section = CollapsibleSection(i18n.t("YOLO_SECTION"), expanded=True)
        yolo_form = QFormLayout()
        yolo_form.setSpacing(6)

        self.yolo_level_combo = QComboBox()
        self.yolo_level_combo.setToolTip(i18n.tip("YOLO_LEVEL"))
        self.yolo_level_combo.addItems(["0 (高速)", "1 (標準)", "2 (高品質)", "3 (最高品質)"])
        self.yolo_level_combo.setCurrentIndex(1)
        yolo_form.addRow(i18n.YOLO_LEVEL, self.yolo_level_combo)

        self.yolo_expand_edit = DragSpinBox(
            minimum=_YOLO_EXPAND_MIN,
            maximum=_YOLO_EXPAND_MAX,
            step=1,
            value=_YOLO_EXPAND_DEFAULT,
            drag_pixels_per_step=6.0,
        )
        self.yolo_expand_edit.setToolTip(i18n.tip("YOLO_EXPAND"))
        self.yolo_expand_edit.setFixedWidth(80)
        yolo_form.addRow(i18n.YOLO_EXPAND, self.yolo_expand_edit)

        self.yolo_add_ext_cb = QCheckBox(i18n.t("ADD_EXT_LABEL"))
        self.yolo_add_ext_cb.setToolTip(i18n.tip("YOLO_ADD_EXT"))
        yolo_form.addRow("", self.yolo_add_ext_cb)

        # クラス選択
        class_inner = QWidget()
        class_layout = QVBoxLayout(class_inner)
        class_layout.setContentsMargins(0, 4, 0, 0)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for label, fn in [
            (i18n.CLASS_PRESET_PERSON, lambda: self._set_classes([0])),
            (i18n.CLASS_PRESET_VEHICLES, lambda: self._set_classes([0, 1, 2, 3, 5, 7])),
            (i18n.CLASS_PRESET_ALL, lambda: self._set_classes(list(range(80)))),
            (i18n.CLASS_PRESET_CLEAR, lambda: self._set_classes([])),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.clicked.connect(fn)
            preset_row.addWidget(btn)
        preset_row.addStretch()
        class_layout.addLayout(preset_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(160)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(2)
        self.class_cbs: list[QCheckBox] = []
        cols = 4
        for idx, name in enumerate(_COCO_CLASS_NAMES):
            cb = QCheckBox(f"{idx}: {name}")
            if idx == 0:
                cb.setChecked(True)
            self.class_cbs.append(cb)
            grid.addWidget(cb, idx // cols, idx % cols)
        scroll.setWidget(grid_widget)
        class_layout.addWidget(scroll)

        yolo_form.addRow(i18n.YOLO_CLASSES, class_inner)
        self.yolo_section.content_layout.addLayout(yolo_form)
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
        other_form.addRow(i18n.STITCH_BOUNDARY_WIDTH, self.stitch_boundary_width_edit)

        worker_default = os.cpu_count() or 4
        self.stitch_workers_edit = DragSpinBox(
            minimum=1,
            maximum=max(1, worker_default * 2),
            step=1,
            value=worker_default,
        )
        self.stitch_workers_edit.setToolTip(i18n.tip("STITCH_WORKERS"))
        self.stitch_workers_edit.setFixedWidth(80)
        other_form.addRow(i18n.STITCH_WORKERS, self.stitch_workers_edit)

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
        other_form.addRow(i18n.OVEREXPOSURE_THRESHOLD, self.overexp_threshold_edit)

        self.overexp_dilate_edit = DragSpinBox(
            minimum=_OVEREXP_DILATE_MIN,
            maximum=_OVEREXP_DILATE_MAX,
            step=1,
            value=_OVEREXP_DILATE_DEFAULT,
            drag_pixels_per_step=6.0,
        )
        self.overexp_dilate_edit.setToolTip(i18n.tip("OVEREXPOSURE_DILATE"))
        self.overexp_dilate_edit.setFixedWidth(80)
        other_form.addRow(i18n.OVEREXPOSURE_DILATE, self.overexp_dilate_edit)

        self.other_section.content_layout.addLayout(other_form)
        layout.addWidget(self.other_section)

        # Metashape SfM 案内（マスクと一緒に SfM すると精度向上）
        notice = QLabel(i18n.METASHAPE_NOTICE)
        notice.setWordWrap(True)
        notice.setAlignment(Qt.AlignCenter)
        notice.setStyleSheet(
            "padding: 16px; background: #1a1a2e; border: 1px solid #f59e0b; "
            "border-radius: 8px; color: #fbbf24; font-size: 10pt;"
        )
        layout.addWidget(notice)

        layout.addStretch()

        self.stitch_preview_section = CollapsibleSection(i18n.t("STITCH_PREVIEW_SECTION"), expanded=True)
        self.stitch_preview = StitchPreviewWidget()
        self.stitch_preview_section.content_layout.addWidget(self.stitch_preview)

        top_scroll.setWidget(top)
        splitter.addWidget(top_scroll)
        splitter.addWidget(self.stitch_preview_section)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 360])
        root_layout.addWidget(splitter)

        for cb in (self.run_yolo_cb, self.run_stitch_cb, self.run_overexp_cb):
            cb.toggled.connect(self._update_task_controls)
        self.images_browse.path_changed.connect(self._on_images_dir_changed)
        self.stitch_boundary_width_edit.valueChanged.connect(lambda _: self._render_stitch_preview())
        self.stitch_preview.sample_edit.textChanged.connect(lambda _: self._render_stitch_preview())
        self.stitch_preview.opacity_slider.valueChanged.connect(lambda _: self._render_stitch_preview())
        self.stitch_preview.opacity_spin.valueChanged.connect(lambda _: self._render_stitch_preview())
        self._update_task_controls()
        self._on_images_dir_changed(self.images_browse.text())

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if path:
            p = Path(path)
            self.images_browse.set_text(str(p / "images"))
            self.masks_browse.set_text(str(p / "masks"))
            self._on_images_dir_changed(str(p / "images"))

    def _set_classes(self, indices: list[int]) -> None:
        for i, cb in enumerate(self.class_cbs):
            cb.setChecked(i in indices)

    def _selected_classes(self) -> list[int]:
        return [i for i, cb in enumerate(self.class_cbs) if cb.isChecked()]

    def _update_task_controls(self) -> None:
        yolo_enabled = self.run_yolo_cb.isChecked()
        stitch_enabled = self.run_stitch_cb.isChecked()
        overexp_enabled = self.run_overexp_cb.isChecked()

        self.yolo_section.content_widget.setEnabled(yolo_enabled)
        if stitch_enabled or overexp_enabled:
            self.other_section.toggle_button.setChecked(True)
        self.stitch_boundary_width_edit.setEnabled(True)
        self.stitch_workers_edit.setEnabled(stitch_enabled or overexp_enabled)
        self.overexp_threshold_edit.setEnabled(overexp_enabled)
        self.overexp_dilate_edit.setEnabled(overexp_enabled)
        self._render_stitch_preview()

    def _on_images_dir_changed(self, path: str) -> None:
        self.stitch_preview.set_images_dir(path)
        self._render_stitch_preview()

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

    def _render_stitch_preview(self) -> None:
        try:
            width = self._stitch_boundary_width()
        except ValueError:
            width = None
        self.stitch_preview.render(width)

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        steps = []
        if self.run_yolo_cb.isChecked():
            steps.append(("yolo", self._build_yolo_cmd()))
        if self.run_stitch_cb.isChecked():
            steps.append(("stitch", self._build_stitch_cmd()))
        if self.run_overexp_cb.isChecked():
            steps.append(("overexposure", self._build_overexposure_cmd()))
        if not steps:
            raise ValueError(i18n.t("MASK_TASK_REQUIRED"))
        return steps

    def _build_yolo_cmd(self) -> list[str]:
        images = self.images_browse.text()
        masks = self.masks_browse.text()
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
            "--expand", str(self.yolo_expand_edit.value()),
        ]
        if classes:
            cmd.extend(["--classes", ",".join(str(c) for c in classes)])
        if self.yolo_add_ext_cb.isChecked():
            cmd.append("--add-ext")
        return cmd

    def _build_stitch_cmd(self) -> list[str]:
        masks = self.masks_browse.text()
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
        images = self.images_browse.text()
        masks = self.masks_browse.text()
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
        if _YOLO_LINE_RE.match(line):
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
