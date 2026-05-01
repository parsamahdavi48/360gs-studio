"""Step 2: フレーム確認 + 選別確定"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.steps.base_step import BaseStepWidget


class ReviewStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._review_widget: QWidget | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QScrollArea.NoFrame)
        settings_scroll.setFixedWidth(380)

        settings = QWidget()
        settings.setObjectName("settingsPane")
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 8, 4)
        settings_layout.setSpacing(10)

        workflow = QLabel(i18n.t("STEP2_WORKFLOW"))
        workflow.setObjectName("workflowNote")
        workflow.setAlignment(Qt.AlignCenter)
        workflow.setWordWrap(True)
        settings_layout.addWidget(workflow)

        form = QFormLayout()
        form.setSpacing(6)

        self.csv_edit = QLineEdit("selected_frames.csv")
        self.csv_edit.setToolTip(i18n.tip("CSV_FILE"))
        self.csv_edit.editingFinished.connect(lambda: self._load_embedded_review(show_error=False))
        form.addRow(i18n.CSV_FILE, self.csv_edit)

        self.prefix_edit = QLineEdit("")
        self.prefix_edit.setToolTip(i18n.tip("FILENAME_PREFIX"))
        self.prefix_edit.setPlaceholderText(i18n.t("AUTO_PREFIX_HINT"))
        form.addRow(i18n.FILENAME_PREFIX, self.prefix_edit)

        self.backup_cb = QCheckBox(i18n.t("BACKUP_BEFORE_FINALIZE"))
        self.backup_cb.setToolTip(i18n.t("BACKUP_BEFORE_FINALIZE_HINT"))
        self.backup_cb.setChecked(False)
        form.addRow("", self.backup_cb)

        settings_layout.addLayout(form)

        self.reload_review_btn = QPushButton(i18n.t("REVIEW_LOAD_EMBEDDED"))
        self.reload_review_btn.setToolTip(i18n.t("REVIEW_LOAD_EMBEDDED_HINT"))
        self.reload_review_btn.clicked.connect(lambda: self._load_embedded_review(show_error=True))
        settings_layout.addWidget(self.reload_review_btn)

        self.review_window_btn = QPushButton(i18n.t("OPEN_REVIEW_EXTERNAL"))
        self.review_window_btn.setToolTip(i18n.tip("OPEN_REVIEW"))
        self.review_window_btn.clicked.connect(self._open_review_window)
        settings_layout.addWidget(self.review_window_btn)

        notice = QLabel(i18n.NEXT_STEP_MASK_NOTICE)
        notice.setObjectName("workflowNote")
        notice.setWordWrap(True)
        notice.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(notice)
        settings_layout.addStretch()

        self.review_pane = QWidget()
        self.review_pane.setObjectName("workPane")
        self.review_layout = QVBoxLayout(self.review_pane)
        self.review_layout.setContentsMargins(12, 12, 12, 12)
        self.review_layout.setSpacing(8)
        self._set_review_placeholder(i18n.t("REVIEW_EMBED_EMPTY"))

        settings_scroll.setWidget(settings)
        splitter.addWidget(settings_scroll)
        splitter.addWidget(self.review_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 860])
        root_layout.addWidget(splitter)

    def _csv_path(self) -> Path:
        return Path(self.scene_dir) / self.csv_edit.text().strip()

    def _has_csv(self) -> bool:
        return self._csv_path().exists()

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        self._load_embedded_review(show_error=False)

    def primary_action_text(self) -> str:
        return i18n.t("ACTION_FINALIZE_REVIEW")

    def primary_action_tooltip(self) -> str:
        return i18n.t("FINALIZE_BUTTON_HINT")

    def _clear_review_pane(self) -> None:
        while self.review_layout.count():
            item = self.review_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._review_widget = None

    def _set_review_placeholder(self, text: str) -> None:
        self._clear_review_pane()
        label = QLabel(text)
        label.setObjectName("emptyPaneMessage")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        self.review_layout.addWidget(label, stretch=1)

    def _load_embedded_review(self, show_error: bool = True) -> None:
        if not self.scene_dir:
            self._set_review_placeholder(i18n.t("REVIEW_EMBED_EMPTY"))
            return
        if not self._has_csv():
            self._set_review_placeholder(i18n.t("REVIEW_EMBED_MISSING").format(path=str(self._csv_path())))
            return
        try:
            from review_frames import ReviewWidget

            widget = ReviewWidget(Path(self.scene_dir), self._csv_path())
        except Exception as e:
            self._set_review_placeholder(str(e))
            if show_error:
                QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))
            return

        self._clear_review_pane()
        self._review_widget = widget
        self.review_layout.addWidget(widget, stretch=1)

    def _open_review_window(self) -> None:
        script = self.base_dir / "review_frames.py"
        if not script.exists():
            QMessageBox.critical(self, i18n.INVALID_INPUT, f"review_frames.py が見つかりません: {script}")
            return
        if not self._has_csv():
            QMessageBox.critical(self, i18n.INVALID_INPUT, f"CSVが見つかりません: {self._csv_path()}")
            return
        cmd = [sys.executable, str(script), self.scene_dir, "--csv", self.csv_edit.text().strip()]
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))

    def _confirm_finalize(self) -> bool:
        if self.backup_cb.isChecked():
            confirm_text = (
                "除外にしたフレームを images/ から削除し、採用フレームを連番で再採番します。\n\n"
                "実行前に images/ を images_backup/ にコピーします（既存バックアップは上書き）。\n"
                "selected_frames.csv のバックアップも自動作成されます。\n\n"
                "実行してよいですか？"
            )
        else:
            confirm_text = (
                "除外にしたフレームを images/ から削除し、採用フレームを連番で再採番します。\n\n"
                "画像のバックアップは作成されません。削除された画像は復元できません。\n"
                "selected_frames.csv のみ自動バックアップされます。\n\n"
                "実行してよいですか？"
            )
        result = QMessageBox.question(
            self,
            i18n.t("ACTION_FINALIZE_REVIEW"),
            confirm_text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

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
        if not self._has_csv():
            raise ValueError(f"CSVが見つかりません: {self._csv_path()}")
        if not self._confirm_finalize():
            return []
        extra = ["--finalize-in-place"]
        if self.backup_cb.isChecked():
            extra.extend(["--backup-dir", "images_backup"])
        return [("finalize", self._build_apply_cmd(extra))]

    def on_queue_finished(self, success: bool) -> None:
        if success:
            self._load_embedded_review(show_error=False)
