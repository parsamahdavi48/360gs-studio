"""Step 2: フレーム確認 + 選別確定"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.apply_frame_decisions import load_rows, normalize_decision, pending_drop_image_paths
from core.scene_layout import review_dir, selected_frames_path
from core.scene_project import append_review_run, file_identity, scene_relative, utc_now_iso
from gui import i18n
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)


class ReviewStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._review_widget: QWidget | None = None
        self._loaded_csv_signature: tuple[Path, int, int] | None = None
        self._pending_review_run: dict | None = None
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

        self.filter_label = QLabel(i18n.t("REVIEW_SOURCE_FILTER"))
        self.filter_label.setToolTip(i18n.tip("REVIEW_SOURCE_FILTER"))
        settings_layout.addWidget(self.filter_label)
        self.source_filter_combo = QComboBox()
        self.source_filter_combo.setToolTip(i18n.tip("REVIEW_SOURCE_FILTER"))
        self.source_filter_combo.currentIndexChanged.connect(self._on_source_filter_changed)
        settings_layout.addWidget(self.source_filter_combo)

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
        self._sync_source_filter_combo([])

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
            from core.review_frames import ReviewWidget

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
        options = []
        source_filter_options = getattr(widget, "source_filter_options", None)
        if callable(source_filter_options):
            options = source_filter_options()
        self._sync_source_filter_combo(options)
        self.primary_action_state_changed.emit()

    def _sync_source_filter_combo(self, options: list[dict]) -> None:
        self.source_filter_combo.blockSignals(True)
        try:
            self.source_filter_combo.clear()
            if not options:
                self.source_filter_combo.addItem(i18n.t("REVIEW_SOURCE_FILTER_ALL").format(n=0), "all")
            else:
                for option in options:
                    self.source_filter_combo.addItem(str(option.get("label") or ""), str(option.get("key") or "all"))
            self.source_filter_combo.setEnabled(len(options) > 1)
        finally:
            self.source_filter_combo.blockSignals(False)

    def _on_source_filter_changed(self, _index: int) -> None:
        widget = self._review_widget
        if widget is None:
            return
        set_source_filter = getattr(widget, "set_source_filter", None)
        if not callable(set_source_filter):
            return
        set_source_filter(str(self.source_filter_combo.currentData() or "all"))

    def _confirm_finalize(self) -> bool:
        confirm_text = i18n.t("FINALIZE_CONFIRM_MESSAGE")
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
        self._pending_review_run = self._review_run_snapshot()
        self._prepare_review_backup(self._pending_review_run)
        return [("finalize", self._build_apply_cmd(extra))]

    def on_queue_finished(self, success: bool) -> None:
        if success:
            self._finish_review_run_snapshot()
            self._refresh_embedded_review(force=True, show_error=False)
        else:
            self._pending_review_run = None

    def _review_counts(self) -> dict:
        csv_path = self._csv_path()
        try:
            rows = load_rows(csv_path)
        except Exception:
            rows = []
        keep = 0
        drop = 0
        for row in rows:
            if normalize_decision(row) == "drop":
                drop += 1
            else:
                keep += 1
        return {"rows": len(rows), "keep": keep, "drop": drop}

    def _review_run_snapshot(self) -> dict:
        scene = Path(self.scene_dir)
        dropped = pending_drop_image_paths(scene)
        run_id = f"review_{utc_now_iso().replace(':', '').replace('-', '')}"
        backup_root = review_dir(scene) / "backups" / run_id
        return {
            "id": run_id,
            "created_at": utc_now_iso(),
            "mode": "finalize_in_place",
            "backup_mode": "dropped_images",
            "backup_dir": scene_relative(scene, backup_root),
            "csv_before": file_identity(self._csv_path()),
            "counts_before": self._review_counts(),
            "pending_drop_images": [scene_relative(scene, p) for p in dropped],
            "backed_up_drop_images": [],
        }

    def _prepare_review_backup(self, record: dict) -> None:
        if not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        backup_root = scene / str(record.get("backup_dir", ""))
        backup_root.mkdir(parents=True, exist_ok=True)
        csv_before = backup_root / "selected_frames.before.csv"
        shutil.copy2(self._csv_path(), csv_before)
        record["csv_before_copy"] = scene_relative(scene, csv_before)

        backed_up: list[str] = []
        dropped = pending_drop_image_paths(scene)
        for path in dropped:
            try:
                rel = path.resolve().relative_to(scene.resolve())
            except (OSError, ValueError):
                rel = Path(path.name)
            dest = backup_root / "dropped_images" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            backed_up.append(scene_relative(scene, dest))
        record["backed_up_drop_images"] = backed_up

    def _finish_review_run_snapshot(self) -> None:
        if not self.scene_dir or self._pending_review_run is None:
            return
        scene = Path(self.scene_dir)
        record = dict(self._pending_review_run)
        record["completed_at"] = utc_now_iso()
        record["csv_after"] = file_identity(self._csv_path())
        record["counts_after"] = self._review_counts()
        backup_dir = scene / str(record.get("backup_dir", ""))
        if backup_dir.is_dir():
            csv_after = backup_dir / "selected_frames.after.csv"
            shutil.copy2(self._csv_path(), csv_after)
            record["csv_after_copy"] = scene_relative(scene, csv_after)
        append_review_run(scene, record)
        self._pending_review_run = None

    def shutdown(self) -> None:
        widget = self._review_widget
        if widget is None:
            return
        shutdown = getattr(widget, "shutdown", None)
        if callable(shutdown):
            shutdown()
