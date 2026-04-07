"""Step 3: マスク生成 (YOLO + スティッチ + 白飛び)"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
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


class MaskStep(BaseStepWidget):
    run_requested = Signal()

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._run_mode = "all"
        self._phase_total = 0
        self._phase_done = 0
        self._stitch_chunk_total = 0
        self._stitch_chunk_done = 0
        self._stitch_done_before = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- パス設定 ---
        path_form = QFormLayout()
        path_form.setSpacing(6)
        self.images_browse = BrowseWidget(mode="dir", placeholder="元画像のフォルダ")
        path_form.addRow(i18n.IMAGES_DIR, self.images_browse)
        self.masks_browse = BrowseWidget(mode="dir", placeholder="マスク出力先")
        path_form.addRow(i18n.MASKS_DIR, self.masks_browse)
        layout.addLayout(path_form)

        # --- メイン実行ボタン (目立つ) ---
        main_btn_row = QHBoxLayout()
        main_btn_row.setSpacing(8)

        self.all_btn = QPushButton(f"  {i18n.RUN_ALL}")
        self.all_btn.setObjectName("primary")
        self.all_btn.setFixedHeight(34)
        self.all_btn.clicked.connect(lambda: self._set_run_mode("all"))
        main_btn_row.addWidget(self.all_btn, stretch=1)

        self.both_btn = QPushButton(i18n.RUN_YOLO_STITCH)
        self.both_btn.setFixedHeight(34)
        self.both_btn.clicked.connect(lambda: self._set_run_mode("both"))
        main_btn_row.addWidget(self.both_btn)

        layout.addLayout(main_btn_row)

        # --- 個別実行 (小さめ) ---
        sub_btn_row = QHBoxLayout()
        sub_btn_row.setSpacing(6)
        for label, mode in [
            (i18n.RUN_YOLO, "yolo"),
            (i18n.RUN_STITCH, "stitch"),
            (i18n.RUN_OVEREXPOSURE, "overexposure"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, m=mode: self._set_run_mode(m))
            sub_btn_row.addWidget(btn)
        sub_btn_row.addStretch()
        layout.addLayout(sub_btn_row)

        # --- YOLO設定 (折りたたみ) ---
        yolo_section = CollapsibleSection("YOLO 人物検出", expanded=True)
        yolo_form = QFormLayout()
        yolo_form.setSpacing(6)

        self.yolo_level_combo = QComboBox()
        self.yolo_level_combo.addItems(["0 (高速)", "1 (標準)", "2 (高品質)", "3 (最高品質)"])
        self.yolo_level_combo.setCurrentIndex(1)
        yolo_form.addRow(i18n.YOLO_LEVEL, self.yolo_level_combo)

        self.yolo_expand_edit = QLineEdit("12")
        self.yolo_expand_edit.setFixedWidth(80)
        yolo_form.addRow(i18n.YOLO_EXPAND, self.yolo_expand_edit)

        self.yolo_add_ext_cb = QCheckBox("マスクに拡張子を付加")
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
        yolo_section.content_layout.addLayout(yolo_form)
        layout.addWidget(yolo_section)

        # --- スティッチ+白飛び設定 (折りたたみ) ---
        other_section = CollapsibleSection("スティッチ / 白飛び設定", expanded=False)
        other_form = QFormLayout()
        other_form.setSpacing(6)

        self.stitch_fov_edit = QLineEdit("190")
        self.stitch_fov_edit.setFixedWidth(80)
        other_form.addRow(i18n.STITCH_FOV, self.stitch_fov_edit)

        self.stitch_workers_edit = QLineEdit(str(os.cpu_count() or 4))
        self.stitch_workers_edit.setFixedWidth(80)
        other_form.addRow(i18n.STITCH_WORKERS, self.stitch_workers_edit)

        sep = QLabel("")
        sep.setFixedHeight(8)
        other_form.addRow(sep)

        self.overexp_threshold_edit = QLineEdit("250")
        self.overexp_threshold_edit.setFixedWidth(80)
        other_form.addRow(i18n.OVEREXPOSURE_THRESHOLD, self.overexp_threshold_edit)

        self.overexp_dilate_edit = QLineEdit("8")
        self.overexp_dilate_edit.setFixedWidth(80)
        other_form.addRow(i18n.OVEREXPOSURE_DILATE, self.overexp_dilate_edit)

        other_section.content_layout.addLayout(other_form)
        layout.addWidget(other_section)

        layout.addStretch()

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if path:
            p = Path(path)
            self.images_browse.set_text(str(p / "images"))
            self.masks_browse.set_text(str(p / "masks"))

    def _set_run_mode(self, mode: str) -> None:
        self._run_mode = mode
        self.run_requested.emit()

    def _set_classes(self, indices: list[int]) -> None:
        for i, cb in enumerate(self.class_cbs):
            cb.setChecked(i in indices)

    def _selected_classes(self) -> list[int]:
        return [i for i, cb in enumerate(self.class_cbs) if cb.isChecked()]

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        steps = []
        if self._run_mode in ("yolo", "both", "all"):
            steps.append(("yolo", self._build_yolo_cmd()))
        if self._run_mode in ("stitch", "both", "all"):
            steps.append(("stitch", self._build_stitch_cmd()))
        if self._run_mode in ("overexposure", "all"):
            steps.append(("overexposure", self._build_overexposure_cmd()))
        if not steps:
            raise ValueError("実行モードを選択してください")
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
            "--expand", self.yolo_expand_edit.text().strip(),
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
            "--fov", self.stitch_fov_edit.text().strip(),
            "--workers", self.stitch_workers_edit.text().strip(),
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
            "--threshold", self.overexp_threshold_edit.text().strip(),
            "--dilate", self.overexp_dilate_edit.text().strip(),
            "--workers", self.stitch_workers_edit.text().strip(),
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
