"""AprilTag scale UI wiring for Step 4."""

from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.apriltag_markers import (
    DEFAULT_APRILTAG_FAMILY,
    DEFAULT_APRILTAG_ID,
    DEFAULT_APRILTAG_SIZE_M,
    MAX_APRILTAG_IDS_PER_RUN,
    available_families,
    clamp_tag_id,
    marker_tooltip_html,
    parse_tag_ids,
    tag_id_range,
)
from core.apriltag_printable import available_pages, create_printable_target
from core.apriltag_scale_apply import apply_scene_output_scale, validate_scale_output_dataset
from core.scene_layout import scene_output_dir, step4_meta_dir
from gui import i18n
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.common.form_rows import add_tooltip_row

_APRILTAG_PROGRESS_RE = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)")


class Step4AprilTagMixin:
    def _init_apriltag_state(self) -> None:
        self._apriltag_scale_ui_enabled = True
        self._apriltag_estimate_process: QProcess | None = None
        self._apriltag_cancel_requested = False
        self._apriltag_output_buffer = ""
        self._apriltag_output_lines: list[str] = []
        self._apriltag_last_scale: float | None = None
        self._apriltag_scale_applied = False

    def _build_apriltag_scale_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(6)

        self.apriltag_tag_size_edit = DragDoubleSpinBox(
            minimum=0.001,
            maximum=10.0,
            step=0.005,
            decimals=3,
            value=DEFAULT_APRILTAG_SIZE_M,
            suffix=" m",
        )
        self.apriltag_tag_size_edit.setFixedWidth(96)
        self.apriltag_tag_size_edit.setToolTip(i18n.tip("APRILTAG_TAG_SIZE"))
        add_tooltip_row(form, i18n.t("APRILTAG_TAG_SIZE"), self.apriltag_tag_size_edit, i18n.tip("APRILTAG_TAG_SIZE"))

        self.apriltag_family_combo = QComboBox()
        self.apriltag_family_combo.setFixedWidth(120)
        self.apriltag_family_combo.setToolTip(i18n.tip("APRILTAG_FAMILY"))
        for family in available_families():
            self.apriltag_family_combo.addItem(family, family)
        self.apriltag_family_combo.setCurrentText(DEFAULT_APRILTAG_FAMILY)
        self.apriltag_family_combo.currentIndexChanged.connect(self._on_apriltag_family_changed)
        add_tooltip_row(form, i18n.t("APRILTAG_FAMILY"), self.apriltag_family_combo, i18n.tip("APRILTAG_FAMILY"))

        self.apriltag_id_edit = QLineEdit(str(DEFAULT_APRILTAG_ID))
        self.apriltag_id_edit.setFixedWidth(160)
        self.apriltag_id_edit.setToolTip(i18n.tip("APRILTAG_TAG_IDS"))
        self.apriltag_id_edit.editingFinished.connect(self._normalize_apriltag_tag_ids)
        self.apriltag_id_edit.textChanged.connect(self._update_apriltag_marker_tooltips)
        add_tooltip_row(form, i18n.t("APRILTAG_TAG_IDS"), self.apriltag_id_edit, i18n.tip("APRILTAG_TAG_IDS"))

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.apriltag_estimate_btn = QPushButton(i18n.t("APRILTAG_ESTIMATE"))
        self.apriltag_estimate_btn.setToolTip(i18n.tip("APRILTAG_ESTIMATE"))
        self.apriltag_estimate_btn.clicked.connect(self._run_apriltag_scale_estimate)
        action_layout.addWidget(self.apriltag_estimate_btn)
        self.apriltag_apply_btn = QPushButton(i18n.t("APRILTAG_APPLY_SCALE"))
        self.apriltag_apply_btn.setToolTip(i18n.tip("APRILTAG_APPLY_SCALE"))
        self.apriltag_apply_btn.clicked.connect(self._apply_apriltag_scale)
        action_layout.addWidget(self.apriltag_apply_btn)
        action_layout.addStretch()
        form.addRow("", action_row)

        self.apriltag_result_label = QLabel(i18n.t("APRILTAG_RESULT_EMPTY"))
        self.apriltag_result_label.setWordWrap(True)
        self.apriltag_result_label.setToolTip(i18n.tip("APRILTAG_RESULT"))
        form.addRow(i18n.t("APRILTAG_RESULT"), self.apriltag_result_label)

        layout.addLayout(form)
        layout.addWidget(self._build_apriltag_print_section())
        layout.addStretch()
        self._on_apriltag_family_changed()
        self._sync_apriltag_controls()
        return tab

    def _build_apriltag_print_section(self) -> QWidget:
        self.apriltag_print_section = CollapsibleSection(i18n.t("APRILTAG_PRINT_SECTION"), expanded=False)
        form = QFormLayout()
        form.setSpacing(6)

        self.apriltag_print_family_combo = QComboBox()
        self.apriltag_print_family_combo.setFixedWidth(120)
        for family in available_families():
            self.apriltag_print_family_combo.addItem(family, family)
        self.apriltag_print_family_combo.setCurrentText(DEFAULT_APRILTAG_FAMILY)
        self.apriltag_print_family_combo.currentIndexChanged.connect(self._on_apriltag_print_family_changed)
        add_tooltip_row(
            form,
            i18n.t("APRILTAG_FAMILY"),
            self.apriltag_print_family_combo,
            i18n.tip("APRILTAG_PRINT_FAMILY"),
        )

        self.apriltag_print_id_edit = DragSpinBox(minimum=0, maximum=586, step=1, value=DEFAULT_APRILTAG_ID)
        self.apriltag_print_id_edit.setFixedWidth(86)
        self.apriltag_print_id_edit.valueChanged.connect(self._update_apriltag_marker_tooltips)
        add_tooltip_row(
            form,
            i18n.t("APRILTAG_PRINT_TAG_ID"),
            self.apriltag_print_id_edit,
            i18n.tip("APRILTAG_PRINT_TAG_ID"),
        )

        self.apriltag_print_page_combo = QComboBox()
        self.apriltag_print_page_combo.setFixedWidth(96)
        for page in available_pages():
            self.apriltag_print_page_combo.addItem(page, page)
        self.apriltag_print_page_combo.setToolTip(i18n.tip("APRILTAG_PRINT_PAGE"))
        add_tooltip_row(
            form,
            i18n.t("APRILTAG_PRINT_PAGE"),
            self.apriltag_print_page_combo,
            i18n.tip("APRILTAG_PRINT_PAGE"),
        )

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.apriltag_print_btn = QPushButton(i18n.t("APRILTAG_PRINT_EXPORT"))
        self.apriltag_print_btn.setToolTip(i18n.tip("APRILTAG_PRINT_EXPORT"))
        self.apriltag_print_btn.clicked.connect(self._export_apriltag_pdf)
        action_layout.addWidget(self.apriltag_print_btn)
        action_layout.addStretch()
        form.addRow("", action_row)

        self.apriltag_print_status_label = QLabel("")
        self.apriltag_print_status_label.setWordWrap(True)
        form.addRow("", self.apriltag_print_status_label)

        self.apriltag_print_section.content_layout.addLayout(form)
        return self.apriltag_print_section

    def _apriltag_current_family(self) -> str:
        return str(self.apriltag_family_combo.currentData() or DEFAULT_APRILTAG_FAMILY)

    def _apriltag_print_family(self) -> str:
        return str(self.apriltag_print_family_combo.currentData() or DEFAULT_APRILTAG_FAMILY)

    def _selected_apriltag_ids(self) -> tuple[int, ...]:
        family = self._apriltag_current_family()
        text = self.apriltag_id_edit.text().strip()
        if not text:
            return (clamp_tag_id(family, DEFAULT_APRILTAG_ID),)
        return parse_tag_ids(text, family=family, max_ids=MAX_APRILTAG_IDS_PER_RUN) or (
            clamp_tag_id(family, DEFAULT_APRILTAG_ID),
        )

    def _normalize_apriltag_tag_ids(self) -> None:
        family = self._apriltag_current_family()
        raw_ids: list[int] = []
        for raw in self.apriltag_id_edit.text().replace(",", " ").split():
            try:
                raw_ids.append(int(raw))
            except ValueError:
                pass
        if not raw_ids:
            raw_ids = [DEFAULT_APRILTAG_ID]
        ids: list[int] = []
        seen: set[int] = set()
        for tag_id in raw_ids:
            clamped = clamp_tag_id(family, tag_id)
            if clamped not in seen:
                ids.append(clamped)
                seen.add(clamped)
            if len(ids) >= MAX_APRILTAG_IDS_PER_RUN:
                break
        self.apriltag_id_edit.setText(", ".join(str(tag_id) for tag_id in ids))
        self._update_apriltag_marker_tooltips()

    def _on_apriltag_family_changed(self) -> None:
        self._normalize_apriltag_tag_ids()
        self._update_apriltag_marker_tooltips()

    def _on_apriltag_print_family_changed(self) -> None:
        family = self._apriltag_print_family()
        low, high = tag_id_range(family)
        with QSignalBlocker(self.apriltag_print_id_edit):
            self.apriltag_print_id_edit.setRange(low, high)
            self.apriltag_print_id_edit.setValue(clamp_tag_id(family, self.apriltag_print_id_edit.value()))
        self._update_apriltag_marker_tooltips()

    @staticmethod
    def _rich_tooltip(base: str, preview_html: str) -> str:
        escaped = html.escape(base).replace("\n", "<br>")
        return f"<qt>{escaped}<br><br>{preview_html}</qt>"

    def _update_apriltag_marker_tooltips(self) -> None:
        if not hasattr(self, "apriltag_id_edit"):
            return
        family = self._apriltag_current_family()
        try:
            tag_id = self._selected_apriltag_ids()[0]
            preview = marker_tooltip_html(family, tag_id)
            self.apriltag_id_edit.setToolTip(self._rich_tooltip(i18n.tip("APRILTAG_TAG_IDS"), preview))
            self.apriltag_family_combo.setToolTip(self._rich_tooltip(i18n.tip("APRILTAG_FAMILY"), preview))
        except Exception:
            self.apriltag_id_edit.setToolTip(i18n.tip("APRILTAG_TAG_IDS"))
            self.apriltag_family_combo.setToolTip(i18n.tip("APRILTAG_FAMILY"))

        if hasattr(self, "apriltag_print_id_edit"):
            print_family = self._apriltag_print_family()
            print_id = int(self.apriltag_print_id_edit.value())
            preview = marker_tooltip_html(print_family, print_id)
            self.apriltag_print_id_edit.setToolTip(self._rich_tooltip(i18n.tip("APRILTAG_PRINT_TAG_ID"), preview))
            self.apriltag_print_family_combo.setToolTip(
                self._rich_tooltip(i18n.tip("APRILTAG_PRINT_FAMILY"), preview)
            )

    def _sync_apriltag_controls(self) -> None:
        if not hasattr(self, "apriltag_estimate_btn"):
            return
        running = self._apriltag_estimate_process is not None
        self.apriltag_estimate_btn.setEnabled(not running)
        self.apriltag_apply_btn.setEnabled(
            not running and self._apriltag_last_scale is not None and not self._apriltag_scale_applied
        )
        if hasattr(self, "apriltag_print_btn"):
            self.apriltag_print_btn.setEnabled(not running)
        for widget in (
            self.apriltag_tag_size_edit,
            self.apriltag_family_combo,
            self.apriltag_id_edit,
        ):
            widget.setEnabled(not running)

    def _apriltag_tag_size_m(self) -> float:
        value = float(self.apriltag_tag_size_edit.value())
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(i18n.t("APRILTAG_TAG_SIZE_INVALID"))
        return value

    def _build_apriltag_scale_cmd(self, report_path: Path) -> list[str]:
        script = self.base_dir / "scripts" / "estimate_apriltag_scale.py"
        if not script.is_file():
            raise ValueError(f"estimate_apriltag_scale.py not found: {script}")
        dataset = validate_scale_output_dataset(Path(self.scene_dir))
        tag_size = self._apriltag_tag_size_m()
        tag_ids = self._selected_apriltag_ids()

        cmd = [
            sys.executable,
            "-u",
            str(script),
            str(dataset.transforms_json),
            "--tag-size-m",
            f"{tag_size:.9g}",
            "--family",
            self._apriltag_current_family(),
            "--report-json",
            str(report_path),
        ]
        for tag_id in tag_ids:
            cmd.extend(["--tag-id", str(tag_id)])
        return cmd

    def _warn_apriltag(self, message: str) -> None:
        self.apriltag_result_label.setText(message)
        QMessageBox.warning(self, i18n.t("STEP4_TAB_APRILTAG_SCALE"), message)

    def _run_apriltag_scale_estimate(self) -> None:
        if self._apriltag_estimate_process is not None:
            return
        if not self.scene_dir:
            self._warn_apriltag(i18n.t("APRILTAG_SCENE_REQUIRED"))
            return
        report_path = step4_meta_dir(Path(self.scene_dir)) / "apriltag_scale_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = self._build_apriltag_scale_cmd(report_path)
        except Exception as exc:
            self._warn_apriltag(str(exc))
            return

        self._apriltag_last_scale = None
        self._apriltag_scale_applied = False
        self._apriltag_cancel_requested = False
        self._apriltag_output_buffer = ""
        self._apriltag_output_lines = []
        self.apriltag_result_label.setText(i18n.t("APRILTAG_RUNNING"))
        process = QProcess(self)
        self._apriltag_estimate_process = process
        self._sync_apriltag_controls()
        self.background_task_started.emit(f"{i18n.STATUS_RUNNING}: {i18n.t('STEP4_TAB_APRILTAG_SCALE')}")
        self.background_line_received.emit("$ " + " ".join(cmd))
        process.setProgram(cmd[0])
        process.setArguments(cmd[1:])
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_apriltag_scale_output)
        process.finished.connect(
            lambda exit_code, status, path=report_path: self._on_apriltag_scale_finished(exit_code, status, path)
        )
        process.start()
        if not process.waitForStarted(3000):
            detail = process.errorString().strip() or "-"
            self._apriltag_estimate_process = None
            process.deleteLater()
            self.apriltag_result_label.setText(i18n.t("APRILTAG_FAILED").format(detail=detail))
            self.background_line_received.emit(f"[apriltag_scale] start failed: {detail}")
            self.background_task_finished.emit(False, False)
            self._sync_apriltag_controls()

    def _on_apriltag_scale_output(self) -> None:
        process = self._apriltag_estimate_process
        if process is None:
            return
        self._flush_apriltag_scale_output(process)

    def _flush_apriltag_scale_output(self, process: QProcess) -> None:
        data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self._apriltag_output_buffer += data.replace("\r", "\n")
        while "\n" in self._apriltag_output_buffer:
            line, self._apriltag_output_buffer = self._apriltag_output_buffer.split("\n", 1)
            self._handle_apriltag_output_line(line.rstrip("\r"))

    def _handle_apriltag_output_line(self, line: str) -> None:
        if not line:
            return
        self._apriltag_output_lines.append(line)
        self.background_line_received.emit(line)
        progress = _APRILTAG_PROGRESS_RE.match(line)
        if progress:
            self.background_progress_changed.emit(int(progress.group(1)), int(progress.group(2)))
            return
        if line.startswith("[apriltag] "):
            self.background_status_changed.emit(line[len("[apriltag] ") :])

    def _on_apriltag_scale_finished(
        self,
        exit_code: int,
        _status: QProcess.ExitStatus,
        report_path: Path,
    ) -> None:
        process = self._apriltag_estimate_process
        if process is not None:
            self._flush_apriltag_scale_output(process)
        if self._apriltag_output_buffer.strip():
            for line in self._apriltag_output_buffer.strip().splitlines():
                self._handle_apriltag_output_line(line.rstrip("\r"))
            self._apriltag_output_buffer = ""
        detail = "\n".join(self._apriltag_output_lines)
        if process is not None:
            process.deleteLater()
        self._apriltag_estimate_process = None
        canceled = self._apriltag_cancel_requested
        self._apriltag_cancel_requested = False
        self.background_line_received.emit(f"[apriltag_scale] exit_code={exit_code} canceled={int(canceled)}")
        self.background_task_finished.emit(exit_code == 0 and not canceled, canceled)

        if canceled:
            self.apriltag_result_label.setText(i18n.STATUS_CANCELED)
            self._sync_apriltag_controls()
            return
        if exit_code != 0:
            self.apriltag_result_label.setText(
                i18n.t("APRILTAG_FAILED").format(detail=self._message_detail_tail(detail))
            )
            self._sync_apriltag_controls()
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            estimate = report["estimate"]
            scale = float(estimate["scale"])
        except Exception as exc:
            self.apriltag_result_label.setText(i18n.t("APRILTAG_FAILED").format(detail=str(exc)))
            self._sync_apriltag_controls()
            return

        self._apriltag_last_scale = scale
        self._apriltag_scale_applied = False
        self.apriltag_result_label.setText(
            i18n.t("APRILTAG_RESULT_FORMAT").format(
                scale=f"{scale:.9g}",
                observations=int(estimate.get("observation_count", 0)),
                pairs=int(estimate.get("pair_count", 0)),
                inliers=int(estimate.get("inlier_count", 0)),
                rms=float(estimate.get("rms_residual_m", 0.0)),
            )
        )
        self._sync_apriltag_controls()

    def has_background_task(self) -> bool:
        return self._apriltag_estimate_process is not None

    def cancel_background_task(self) -> None:
        process = self._apriltag_estimate_process
        if process is None or process.state() == QProcess.NotRunning:
            return
        self._apriltag_cancel_requested = True
        self.background_status_changed.emit(i18n.STATUS_CANCELED)
        self.background_line_received.emit("[apriltag_scale] cancel requested")
        process.kill()

    def _apply_apriltag_scale(self) -> None:
        if self._apriltag_last_scale is None:
            return
        if not self.scene_dir:
            self._warn_apriltag(i18n.t("APRILTAG_SCENE_REQUIRED"))
            return
        try:
            result = apply_scene_output_scale(Path(self.scene_dir), self._apriltag_last_scale)
        except Exception as exc:
            self._warn_apriltag(str(exc))
            return
        self._apriltag_scale_applied = True
        self.apriltag_result_label.setText(
            i18n.t("APRILTAG_APPLIED_FORMAT").format(
                scale=f"{result.scale:.9g}",
                frames=result.frames_scaled,
                points=result.points_scaled,
                backup=str(result.transforms_backup.parent),
            )
        )
        self._sync_apriltag_controls()

    @staticmethod
    def _wrapped_apriltag_path(path: Path, *, max_chars: int = 62) -> str:
        text = str(path)
        if len(text) <= max_chars:
            return text
        normalized = text.replace("\\", "/")
        parts = normalized.split("/")
        lines: list[str] = []
        line = parts[0] if parts else ""
        for part in parts[1:]:
            piece = f"/{part}"
            if line and len(line) + len(piece) > max_chars:
                lines.append(line)
                line = part
            else:
                line += piece
        if line:
            lines.append(line)
        return "\n".join(lines)

    def _export_apriltag_pdf(self) -> None:
        if not self.scene_dir:
            self._warn_apriltag(i18n.t("APRILTAG_SCENE_REQUIRED"))
            return
        output_dir = scene_output_dir(Path(self.scene_dir))
        try:
            target = create_printable_target(
                output_dir,
                family=self._apriltag_print_family(),
                tag_id=int(self.apriltag_print_id_edit.value()),
                tag_size_m=self._apriltag_tag_size_m(),
                page=str(self.apriltag_print_page_combo.currentData() or "A4"),
            )
        except Exception as exc:
            self.apriltag_print_status_label.setText(str(exc))
            QMessageBox.warning(self, i18n.t("APRILTAG_PRINT_SECTION"), str(exc))
            return
        self.apriltag_print_status_label.setToolTip(str(target.page_pdf))
        self.apriltag_print_status_label.setText(
            i18n.t("APRILTAG_PRINT_SAVED").format(path=self._wrapped_apriltag_path(target.page_pdf))
        )

    def _apriltag_tab_selected(self) -> bool:
        return (
            hasattr(self, "settings_tabs")
            and getattr(self, "apriltag_tab_index", None) is not None
            and self.settings_tabs.currentIndex() == self.apriltag_tab_index
        )

    def shutdown(self) -> None:
        process = self._apriltag_estimate_process
        if process is not None:
            self._apriltag_cancel_requested = True
            process.kill()
            process.waitForFinished(3000)
            process.deleteLater()
            self._apriltag_estimate_process = None
