"""Step 3: マスク生成 (YOLO + スティッチ + 白飛び)"""
from __future__ import annotations

import os
import math
import re
import sys
import tempfile
from pathlib import Path

from apply_frame_decisions import pending_drop_image_paths, untracked_image_paths
from PySide6.QtCore import QProcess, Qt
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

_YOLO_LINE_RE = re.compile(r"^Processing:\s+")
_STITCH_TASK_RE = re.compile(r"^Processing\s+(\d+)\s+images\s+with\s+\d+\s+workers\.\.\.$")
_STITCH_TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
_STITCH_BOUNDARY_MIN = 0.0
_STITCH_BOUNDARY_MAX = 30.0
_STITCH_BOUNDARY_DEFAULT = 5.0
_YOLO_EXPAND_MIN = -16
_YOLO_EXPAND_MAX = 32
_YOLO_EXPAND_DEFAULT = 2
_OVEREXP_THRESHOLD_MIN = 1
_OVEREXP_THRESHOLD_MAX = 254
_OVEREXP_THRESHOLD_DEFAULT = 254
_OVEREXP_DILATE_MIN = 0
_OVEREXP_DILATE_MAX = 128
_OVEREXP_DILATE_DEFAULT = 1


class MaskStep(BaseStepWidget):
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

        # --- パス設定 ---
        path_form = QFormLayout()
        path_form.setSpacing(6)
        self.images_browse = BrowseWidget(mode="dir", placeholder="元画像のフォルダ")
        self.images_browse.setToolTip(i18n.tip("IMAGES_DIR"))
        add_tooltip_row(path_form, i18n.IMAGES_DIR, self.images_browse, i18n.tip("IMAGES_DIR"))
        self.masks_browse = BrowseWidget(mode="dir", placeholder="マスク出力先")
        self.masks_browse.setToolTip(i18n.tip("MASKS_DIR"))
        add_tooltip_row(path_form, i18n.MASKS_DIR, self.masks_browse, i18n.tip("MASKS_DIR"))
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

        layout.addLayout(task_row)

        # --- YOLO設定 (折りたたみ) ---
        self.yolo_section = CollapsibleSection(i18n.t("YOLO_SECTION"), expanded=True)
        yolo_form = QFormLayout()
        yolo_form.setSpacing(6)

        self.yolo_level_combo = QComboBox()
        self.yolo_level_combo.setToolTip(i18n.tip("YOLO_LEVEL"))
        self.yolo_level_combo.addItems(["0 (高速)", "1 (標準)", "2 (高品質)", "3 (最高品質)"])
        self.yolo_level_combo.setCurrentIndex(1)
        add_tooltip_row(yolo_form, i18n.YOLO_LEVEL, self.yolo_level_combo, i18n.tip("YOLO_LEVEL"))

        self.yolo_expand_edit = DragSpinBox(
            minimum=_YOLO_EXPAND_MIN,
            maximum=_YOLO_EXPAND_MAX,
            step=1,
            value=_YOLO_EXPAND_DEFAULT,
            suffix=" px",
            drag_pixels_per_step=6.0,
        )
        self.yolo_expand_edit.setToolTip(i18n.tip("YOLO_EXPAND"))
        self.yolo_expand_edit.setFixedWidth(88)
        add_tooltip_row(yolo_form, i18n.YOLO_EXPAND, self.yolo_expand_edit, i18n.tip("YOLO_EXPAND"))

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
        class_layout.addWidget(class_list_section)

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

        preview_pane = QWidget()
        preview_pane.setObjectName("workPane")
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)
        preview_title = QLabel(i18n.t("MASK_PREVIEW_SECTION"))
        preview_title.setObjectName("paneTitle")
        preview_layout.addWidget(preview_title)
        self.mask_preview = MaskPreviewWidget()
        preview_layout.addWidget(self.mask_preview, stretch=1)
        notice = QLabel(i18n.METASHAPE_NOTICE)
        notice.setObjectName("workflowNote")
        notice.setWordWrap(True)
        preview_layout.addWidget(notice)

        settings_scroll.setWidget(settings)
        splitter.addWidget(settings_scroll)
        splitter.addWidget(preview_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        root_layout.addWidget(splitter)

        for cb in (self.run_yolo_cb, self.run_stitch_cb, self.run_overexp_cb):
            cb.toggled.connect(self._update_task_controls)
        self.images_browse.path_changed.connect(self._on_images_dir_changed)
        self.masks_browse.path_changed.connect(lambda _: self._render_mask_preview())
        self.yolo_add_ext_cb.toggled.connect(lambda _: self._render_mask_preview())
        self.stitch_boundary_width_edit.valueChanged.connect(lambda _: self._render_mask_preview())
        self.overexp_threshold_edit.valueChanged.connect(lambda _: self._render_mask_preview())
        self.overexp_dilate_edit.valueChanged.connect(lambda _: self._render_mask_preview())
        self.mask_preview.sample_edit.textChanged.connect(lambda _: self._render_mask_preview())
        self.mask_preview.opacity_slider.valueChanged.connect(lambda _: self._render_mask_preview())
        self.mask_preview.opacity_spin.valueChanged.connect(lambda _: self._render_mask_preview())
        self.mask_preview.yolo_preview_requested.connect(self._run_yolo_preview)
        self._update_task_controls()
        self._on_images_dir_changed(self.images_browse.text())

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if path:
            p = Path(path)
            self.images_browse.set_text(str(p / "images"))
            self.masks_browse.set_text(str(p / "masks"))
            self._on_images_dir_changed(str(p / "images"))

    def primary_action_text(self) -> str:
        return i18n.t("GENERATE")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("RUN_MASKS")

    def _set_classes(self, indices: list[int]) -> None:
        for i, cb in enumerate(self.class_cbs):
            cb.setChecked(i in indices)

    def _selected_classes(self) -> list[int]:
        return [i for i, cb in enumerate(self.class_cbs) if cb.isChecked()]

    def _yolo_expand_arg(self) -> str:
        return str(self.yolo_expand_edit.value())

    def _update_task_controls(self) -> None:
        yolo_enabled = self.run_yolo_cb.isChecked()
        stitch_enabled = self.run_stitch_cb.isChecked()
        overexp_enabled = self.run_overexp_cb.isChecked()

        self.yolo_section.content_widget.setEnabled(yolo_enabled)
        self.stitch_boundary_width_edit.setEnabled(stitch_enabled)
        self.stitch_workers_edit.setEnabled(stitch_enabled or overexp_enabled)
        self.overexp_threshold_edit.setEnabled(overexp_enabled)
        self.overexp_dilate_edit.setEnabled(overexp_enabled)
        self._render_mask_preview()

    def _on_images_dir_changed(self, path: str) -> None:
        self.mask_preview.set_images_dir(path)
        self._render_mask_preview()

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
            masks_dir=self.masks_browse.text(),
            yolo_add_ext=self.yolo_add_ext_cb.isChecked(),
        )
        self.mask_preview.render(config)

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        requested_steps = []
        if self.run_yolo_cb.isChecked():
            requested_steps.append("yolo")
        if self.run_stitch_cb.isChecked():
            requested_steps.append("stitch")
        if self.run_overexp_cb.isChecked():
            requested_steps.append("overexposure")
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
        images = self.images_browse.text()
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
        images = self.images_browse.text()
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
            "--expand", self._yolo_expand_arg(),
        ]
        if classes:
            cmd.extend(["--classes", ",".join(str(c) for c in classes)])
        if self.yolo_add_ext_cb.isChecked():
            cmd.append("--add-ext")
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
        ]
        if classes:
            cmd.extend(["--classes", ",".join(str(c) for c in classes)])
        if self.yolo_add_ext_cb.isChecked():
            cmd.append("--add-ext")
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
        output_path = output_dir / _yolo_preview_output_name(image_path, self.yolo_add_ext_cb.isChecked())

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


def _yolo_preview_output_name(image_path: Path, add_ext: bool) -> str:
    return f"{image_path.name}.png" if add_ext else f"{image_path.stem}.png"
