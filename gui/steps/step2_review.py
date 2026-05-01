"""Step 2: フレーム確認 + 選別確定"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.steps.base_step import BaseStepWidget


_CARD_STYLE = """
    background-color: #252a31;
    border: 1px solid #3a424d;
    border-radius: 6px;
    padding: 16px;
"""

_WORKFLOW_STYLE = """
    background-color: #101316;
    border: 1px solid #2dd4bf;
    border-radius: 6px;
    padding: 20px;
    color: #ccfbf1;
    font-size: 11pt;
    line-height: 1.6;
"""


class ReviewStep(BaseStepWidget):
    run_requested = Signal()

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._last_commands: list[tuple[str, list[str]]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ワークフロー案内カード
        workflow = QLabel(i18n.t("STEP2_WORKFLOW"))
        workflow.setAlignment(Qt.AlignCenter)
        workflow.setStyleSheet(_WORKFLOW_STYLE)
        workflow.setWordWrap(True)
        layout.addWidget(workflow)

        # 設定カード
        card = QWidget()
        card.setStyleSheet(_CARD_STYLE)
        card_layout = QVBoxLayout(card)

        form = QFormLayout()
        form.setSpacing(6)

        self.csv_edit = QLineEdit("selected_frames.csv")
        self.csv_edit.setToolTip(i18n.tip("CSV_FILE"))
        form.addRow(i18n.CSV_FILE, self.csv_edit)

        self.prefix_edit = QLineEdit("")
        self.prefix_edit.setToolTip(i18n.tip("FILENAME_PREFIX"))
        self.prefix_edit.setPlaceholderText(i18n.t("AUTO_PREFIX_HINT"))
        form.addRow(i18n.FILENAME_PREFIX, self.prefix_edit)

        self.backup_cb = QCheckBox(i18n.t("BACKUP_BEFORE_FINALIZE"))
        self.backup_cb.setToolTip(i18n.t("BACKUP_BEFORE_FINALIZE_HINT"))
        self.backup_cb.setChecked(False)
        form.addRow("", self.backup_cb)

        card_layout.addLayout(form)
        layout.addWidget(card)

        # アクションボタン
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.review_btn = QPushButton(f"  {i18n.OPEN_REVIEW}")
        self.review_btn.setToolTip(i18n.tip("OPEN_REVIEW"))
        self.review_btn.setObjectName("primary")
        self.review_btn.setFixedHeight(34)
        self.review_btn.clicked.connect(self._open_review)
        btn_row.addWidget(self.review_btn)

        self.finalize_btn = QPushButton(i18n.t("FINALIZE_BUTTON"))
        self.finalize_btn.setToolTip(i18n.t("FINALIZE_BUTTON_HINT"))
        self.finalize_btn.setFixedHeight(34)
        self.finalize_btn.clicked.connect(self._finalize_inplace)
        btn_row.addWidget(self.finalize_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 次ステップ案内
        notice = QLabel(i18n.NEXT_STEP_MASK_NOTICE)
        notice.setWordWrap(True)
        notice.setAlignment(Qt.AlignCenter)
        notice.setStyleSheet(
            "padding: 16px; background: #101316; border: 1px solid #2dd4bf; "
            "border-radius: 6px; color: #ccfbf1; font-size: 10pt;"
        )
        layout.addWidget(notice)

        layout.addStretch()

    def _csv_path(self) -> Path:
        return Path(self.scene_dir) / self.csv_edit.text().strip()

    def _has_csv(self) -> bool:
        return self._csv_path().exists()

    def _open_review(self) -> None:
        script = self.base_dir / "review_frames.py"
        if not script.exists():
            QMessageBox.critical(self, i18n.INVALID_INPUT, f"review_frames.py が見つかりません: {script}")
            return
        if not self._has_csv():
            QMessageBox.critical(self, i18n.INVALID_INPUT, f"CSVが見つかりません: {self._csv_path()}")
            return
        cmd = [sys.executable, str(script), self.scene_dir]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))

    def _finalize_inplace(self) -> None:
        if not self._has_csv():
            QMessageBox.critical(self, i18n.INVALID_INPUT, f"CSVが見つかりません: {self._csv_path()}")
            return
        backup_on = self.backup_cb.isChecked()
        if backup_on:
            confirm_text = (
                "除外にしたフレームを images/ から削除し、採用フレームを連番で再採番します。\n\n"
                "実行前に images/ を images_backup/ にコピーします（既存バックアップは上書き）。\n"
                "selected_frames.csv のバックアップも自動作成されます。\n\n"
                "実行してよいですか？"
            )
        else:
            confirm_text = (
                "除外にしたフレームを images/ から削除し、採用フレームを連番で再採番します。\n\n"
                "⚠ 画像のバックアップは作成されません。削除された画像は復元できません。\n"
                "selected_frames.csv のみ自動バックアップされます。\n\n"
                "実行してよいですか？"
            )
        result = QMessageBox.question(
            self, "確定",
            confirm_text,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if result != QMessageBox.Yes:
            return
        extra = ["--finalize-in-place"]
        if backup_on:
            extra.extend(["--backup-dir", "images_backup"])
        self._last_commands = [("finalize", self._build_apply_cmd(extra))]
        self.run_requested.emit()

    def _build_apply_cmd(self, extra_args: list[str] | None = None) -> list[str]:
        script = self.base_dir / "apply_frame_decisions.py"
        if not script.exists():
            raise FileNotFoundError(f"apply_frame_decisions.py が見つかりません: {script}")
        cmd = [sys.executable, "-u", str(script), self.scene_dir]
        prefix = self.prefix_edit.text().strip()
        if prefix:
            cmd.extend(["--filename-prefix", prefix])
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def build_commands(self) -> list[tuple[str, list[str]]]:
        if hasattr(self, "_last_commands") and self._last_commands:
            cmds = self._last_commands
            self._last_commands = []
            return cmds
        raise ValueError("確定ボタンを押してください")
