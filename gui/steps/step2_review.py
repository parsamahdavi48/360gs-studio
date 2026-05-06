"""Step 2: フレーム確認 + 選別確定"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)
from scene_layout import frame_backups_dir, selected_frames_path


class ReviewStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._review_widget: QWidget | None = None
        self._loaded_csv_signature: tuple[Path, int, int] | None = None
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
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(*SETTINGS_PANE_MARGINS)
        settings_layout.setSpacing(10)

        self.backup_cb = QCheckBox(i18n.t("BACKUP_BEFORE_FINALIZE"))
        self.backup_cb.setToolTip(i18n.t("BACKUP_BEFORE_FINALIZE_HINT"))
        self.backup_cb.setChecked(False)
        settings_layout.addWidget(self.backup_cb)

        notice = QLabel(i18n.NEXT_STEP_MASK_NOTICE)
        notice.setObjectName("workflowNote")
        notice.setWordWrap(True)
        notice.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        settings_layout.addWidget(notice)
        settings_layout.addStretch()

        self.review_pane = QWidget()
        self.review_pane.setObjectName("workPane")
        self.review_layout = QVBoxLayout(self.review_pane)
        self.review_layout.setContentsMargins(12, 12, 12, 12)
        self.review_layout.setSpacing(8)
        self._set_review_placeholder(i18n.t("REVIEW_EMBED_NO_SCENE"))

        settings_scroll.setWidget(settings)
        splitter.addWidget(settings_scroll)
        splitter.addWidget(self.review_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 860])
        root_layout.addWidget(splitter)

    def _csv_path(self) -> Path:
        return selected_frames_path(Path(self.scene_dir))

    def _has_csv(self) -> bool:
        if not self.scene_dir:
            return False
        return self._csv_path().exists()

    def _csv_signature(self) -> tuple[Path, int, int] | None:
        if not self.scene_dir:
            return None
        csv_path = self._csv_path()
        try:
            st = csv_path.stat()
        except OSError:
            return None
        return csv_path.resolve(), int(st.st_mtime_ns), int(st.st_size)

    def set_scene_dir(self, path: str) -> None:
        changed = path != self.scene_dir
        super().set_scene_dir(path)
        if changed:
            self._loaded_csv_signature = None
            if path:
                self._set_review_placeholder(i18n.t("REVIEW_EMBED_EMPTY"))
            else:
                self._set_review_placeholder(i18n.t("REVIEW_EMBED_NO_SCENE"))

    def on_activated(self) -> None:
        self._refresh_embedded_review(force=False, show_error=False)

    def primary_action_text(self) -> str:
        return i18n.t("ACTION_FINALIZE_REVIEW")

    def primary_action_tooltip(self) -> str:
        return i18n.t("FINALIZE_BUTTON_HINT")

    def primary_action_enabled(self) -> bool:
        widget = self._review_widget
        if widget is None:
            return False
        has_pending = getattr(widget, "has_pending_finalize", None)
        if not callable(has_pending):
            return False
        return bool(has_pending())

    def _clear_review_pane(self) -> None:
        while self.review_layout.count():
            item = self.review_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                shutdown = getattr(widget, "shutdown", None)
                if callable(shutdown):
                    shutdown()
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
        self.primary_action_state_changed.emit()

    def _on_review_decisions_changed(self) -> None:
        self.primary_action_state_changed.emit()

    def _refresh_embedded_review(self, force: bool = False, show_error: bool = True) -> None:
        if not self.scene_dir:
            self._loaded_csv_signature = None
            self._set_review_placeholder(i18n.t("REVIEW_EMBED_NO_SCENE"))
            return

        signature = self._csv_signature()
        if signature is None:
            self._loaded_csv_signature = None
            if force or self._review_widget is not None:
                self._set_review_placeholder(i18n.t("REVIEW_EMBED_MISSING").format(path=str(self._csv_path())))
            else:
                self._set_review_placeholder(i18n.t("REVIEW_EMBED_EMPTY"))
            return

        if not force and self._review_widget is not None and signature == self._loaded_csv_signature:
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
        self._loaded_csv_signature = signature
        decisions_changed = getattr(widget, "decisions_changed", None)
        if decisions_changed is not None:
            decisions_changed.connect(self._on_review_decisions_changed)
        self.review_layout.addWidget(widget, stretch=1)
        self.primary_action_state_changed.emit()

    def _confirm_finalize(self) -> bool:
        if self.backup_cb.isChecked():
            confirm_text = (
                "除外にしたフレームを images/ から削除し、採用フレームのファイル名は維持します。\n\n"
                "適用前に images/ を _stechdrive/frames/backups/images/ にコピーします（既存バックアップは上書き）。\n"
                "_stechdrive/frames/selected_frames.csv のバックアップも自動作成されます。\n\n"
                "適用してよいですか？"
            )
        else:
            confirm_text = (
                "除外にしたフレームを images/ から削除し、採用フレームのファイル名は維持します。\n\n"
                "画像のバックアップは作成されません。削除された画像は復元できません。\n"
                "_stechdrive/frames/selected_frames.csv のみ自動バックアップされます。\n\n"
                "適用してよいですか？"
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
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def build_commands(self) -> list[tuple[str, list[str]]]:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        if not self._has_csv():
            raise ValueError(f"CSVが見つかりません: {self._csv_path()}")
        if not self._confirm_finalize():
            return []
        extra = ["--finalize-in-place"]
        if self.backup_cb.isChecked():
            extra.extend(["--backup-dir", str(frame_backups_dir(Path(self.scene_dir)) / "images")])
        return [("finalize", self._build_apply_cmd(extra))]

    def on_queue_finished(self, success: bool) -> None:
        if success:
            self._refresh_embedded_review(force=True, show_error=False)

    def shutdown(self) -> None:
        widget = self._review_widget
        if widget is None:
            return
        shutdown = getattr(widget, "shutdown", None)
        if callable(shutdown):
            shutdown()
