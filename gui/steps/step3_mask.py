"""Step 3: マスク生成 (YOLO + スティッチ)"""
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
        self._run_mode = "both"  # "yolo", "stitch", "both"
        self._phase_total = 0
        self._phase_done = 0
        self._stitch_chunk_total = 0
        self._stitch_chunk_done = 0
        self._stitch_done_before = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.images_browse = BrowseWidget(mode="dir")
        form.addRow(i18n.IMAGES_DIR, self.images_browse)

        self.masks_browse = BrowseWidget(mode="dir")
        form.addRow(i18n.MASKS_DIR, self.masks_browse)

        # YOLO パラメータ
        self.yolo_level_combo = QComboBox()
        self.yolo_level_combo.addItems(["0", "1", "2", "3"])
        self.yolo_level_combo.setCurrentIndex(1)
        form.addRow(i18n.YOLO_LEVEL, self.yolo_level_combo)

        self.yolo_expand_edit = QLineEdit("12")
        form.addRow(i18n.YOLO_EXPAND, self.yolo_expand_edit)

        self.yolo_add_ext_cb = QCheckBox()
        form.addRow(i18n.YOLO_ADD_EXT, self.yolo_add_ext_cb)

        # スティッチ パラメータ
        self.stitch_fov_edit = QLineEdit("190")
        form.addRow(i18n.STITCH_FOV, self.stitch_fov_edit)

        self.stitch_workers_edit = QLineEdit(str(os.cpu_count() or 4))
        form.addRow(i18n.STITCH_WORKERS, self.stitch_workers_edit)

        # 白飛びマスク パラメータ
        self.overexp_threshold_edit = QLineEdit("250")
        form.addRow(i18n.OVEREXPOSURE_THRESHOLD, self.overexp_threshold_edit)

        self.overexp_dilate_edit = QLineEdit("8")
        form.addRow(i18n.OVEREXPOSURE_DILATE, self.overexp_dilate_edit)

        layout.addLayout(form)

        # YOLO クラス選択（折りたたみ）
        class_section = CollapsibleSection(i18n.YOLO_CLASSES, expanded=False)
        class_inner = QWidget()
        class_layout = QVBoxLayout(class_inner)
        class_layout.setContentsMargins(0, 0, 0, 0)

        preset_row = QHBoxLayout()
        for label, fn in [
            (i18n.CLASS_PRESET_PERSON, lambda: self._set_classes([0])),
            (i18n.CLASS_PRESET_VEHICLES, lambda: self._set_classes([0, 1, 2, 3, 5, 7])),
            (i18n.CLASS_PRESET_ALL, lambda: self._set_classes(list(range(80)))),
            (i18n.CLASS_PRESET_CLEAR, lambda: self._set_classes([])),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(fn)
            preset_row.addWidget(btn)
        preset_row.addStretch()
        class_layout.addLayout(preset_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
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

        class_section.content_layout.addWidget(class_inner)
        layout.addWidget(class_section)

        # 実行モード選択ボタン
        mode_row = QHBoxLayout()
        self.yolo_btn = QPushButton(i18n.RUN_YOLO)
        self.yolo_btn.clicked.connect(lambda: self._set_run_mode("yolo"))
        mode_row.addWidget(self.yolo_btn)

        self.stitch_btn = QPushButton(i18n.RUN_STITCH)
        self.stitch_btn.clicked.connect(lambda: self._set_run_mode("stitch"))
        mode_row.addWidget(self.stitch_btn)

        self.both_btn = QPushButton(i18n.RUN_YOLO_STITCH)
        self.both_btn.clicked.connect(lambda: self._set_run_mode("both"))
        mode_row.addWidget(self.both_btn)

        self.overexp_btn = QPushButton(i18n.RUN_OVEREXPOSURE)
        self.overexp_btn.clicked.connect(lambda: self._set_run_mode("overexposure"))
        mode_row.addWidget(self.overexp_btn)

        self.all_btn = QPushButton(i18n.RUN_ALL)
        self.all_btn.clicked.connect(lambda: self._set_run_mode("all"))
        mode_row.addWidget(self.all_btn)

        mode_row.addStretch()
        layout.addLayout(mode_row)

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

        classes = self._selected_classes()
        cmd = [
            sys.executable, "-u", str(script),
            images, masks,
            "--level", self.yolo_level_combo.currentText(),
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
