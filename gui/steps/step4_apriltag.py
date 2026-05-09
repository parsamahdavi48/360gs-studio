"""Experimental AprilTag scale UI wiring for Step 4."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from core.scene_layout import step4_meta_dir
from gui import i18n
from gui.common.form_rows import add_tooltip_row
from gui.feature_flags import apriltag_scale_enabled


class Step4AprilTagMixin:
    def _init_apriltag_state(self) -> None:
        self._apriltag_scale_ui_enabled = apriltag_scale_enabled()
        self._apriltag_estimate_process: QProcess | None = None
        self._apriltag_last_scale: float | None = None

    def _build_apriltag_scale_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(6)

        self.apriltag_enable_cb = QCheckBox(i18n.t("APRILTAG_SCALE_ENABLE"))
        self.apriltag_enable_cb.setToolTip(i18n.tip("APRILTAG_SCALE_ENABLE"))
        self.apriltag_enable_cb.toggled.connect(self._sync_apriltag_controls)
        form.addRow("", self.apriltag_enable_cb)

        self.apriltag_tag_size_edit = QLineEdit("0.160")
        self.apriltag_tag_size_edit.setFixedWidth(86)
        self.apriltag_tag_size_edit.setToolTip(i18n.tip("APRILTAG_TAG_SIZE"))
        add_tooltip_row(form, i18n.t("APRILTAG_TAG_SIZE"), self.apriltag_tag_size_edit, i18n.tip("APRILTAG_TAG_SIZE"))

        self.apriltag_family_combo = QComboBox()
        self.apriltag_family_combo.setFixedWidth(120)
        self.apriltag_family_combo.setToolTip(i18n.tip("APRILTAG_FAMILY"))
        for family in ("tag36h11", "tag25h9", "tag16h5"):
            self.apriltag_family_combo.addItem(family, family)
        add_tooltip_row(form, i18n.t("APRILTAG_FAMILY"), self.apriltag_family_combo, i18n.tip("APRILTAG_FAMILY"))

        self.apriltag_id_edit = QLineEdit("")
        self.apriltag_id_edit.setFixedWidth(86)
        self.apriltag_id_edit.setToolTip(i18n.tip("APRILTAG_TAG_ID"))
        add_tooltip_row(form, i18n.t("APRILTAG_TAG_ID"), self.apriltag_id_edit, i18n.tip("APRILTAG_TAG_ID"))

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

        self.apriltag_status_label = QLabel(i18n.t("APRILTAG_DEV_STATUS"))
        self.apriltag_status_label.setWordWrap(True)
        self.apriltag_status_label.setStyleSheet("color: #8888aa; font-size: 9pt;")
        form.addRow("", self.apriltag_status_label)

        layout.addLayout(form)
        layout.addStretch()
        self._sync_apriltag_controls()
        return tab

    def _sync_apriltag_controls(self) -> None:
        if not self._apriltag_scale_ui_enabled or not hasattr(self, "apriltag_estimate_btn"):
            return
        running = self._apriltag_estimate_process is not None
        enabled = self.apriltag_enable_cb.isChecked() and not running
        self.apriltag_estimate_btn.setEnabled(enabled)
        self.apriltag_apply_btn.setEnabled(enabled and self._apriltag_last_scale is not None)

    def _build_apriltag_scale_cmd(self, report_path: Path) -> list[str]:
        script = self.base_dir / "scripts" / "estimate_apriltag_scale.py"
        if not script.is_file():
            raise ValueError(f"estimate_apriltag_scale.py not found: {script}")
        transforms = self._output_dir() / "transforms.json"
        if not transforms.is_file():
            raise ValueError(i18n.t("APRILTAG_TRANSFORMS_MISSING").format(path=str(transforms)))

        tag_size_text = self.apriltag_tag_size_edit.text().strip()
        try:
            tag_size = float(tag_size_text)
        except ValueError as exc:
            raise ValueError(i18n.t("APRILTAG_TAG_SIZE_INVALID")) from exc
        if not math.isfinite(tag_size) or tag_size <= 0.0:
            raise ValueError(i18n.t("APRILTAG_TAG_SIZE_INVALID"))

        cmd = [
            sys.executable,
            "-u",
            str(script),
            str(transforms),
            "--tag-size-m",
            tag_size_text,
            "--family",
            str(self.apriltag_family_combo.currentData() or "tag36h11"),
            "--report-json",
            str(report_path),
            "--equirect-temp-dir",
            str(step4_meta_dir(Path(self.scene_dir)) / "apriltag_projection"),
        ]
        tag_id = self.apriltag_id_edit.text().strip()
        if tag_id:
            cmd.extend(["--tag-id", tag_id])
        return cmd

    def _run_apriltag_scale_estimate(self) -> None:
        if self._apriltag_estimate_process is not None:
            return
        if not self.scene_dir:
            self.apriltag_result_label.setText(i18n.t("APRILTAG_SCENE_REQUIRED"))
            return
        report_path = step4_meta_dir(Path(self.scene_dir)) / "apriltag_scale_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = self._build_apriltag_scale_cmd(report_path)
        except Exception as exc:
            self.apriltag_result_label.setText(str(exc))
            return

        self._apriltag_last_scale = None
        self.apriltag_result_label.setText(i18n.t("APRILTAG_RUNNING"))
        self._sync_apriltag_controls()
        process = QProcess(self)
        self._apriltag_estimate_process = process
        process.setProgram(cmd[0])
        process.setArguments(cmd[1:])
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.finished.connect(
            lambda exit_code, status, path=report_path: self._on_apriltag_scale_finished(exit_code, status, path)
        )
        process.start()
        if not process.waitForStarted(3000):
            detail = process.errorString().strip() or "-"
            self._apriltag_estimate_process = None
            self.apriltag_result_label.setText(i18n.t("APRILTAG_FAILED").format(detail=detail))
            self._sync_apriltag_controls()

    def _on_apriltag_scale_finished(
        self,
        exit_code: int,
        _status: QProcess.ExitStatus,
        report_path: Path,
    ) -> None:
        process = self._apriltag_estimate_process
        detail = self._qprocess_output_text(process) if process is not None else ""
        if process is not None:
            process.deleteLater()
        self._apriltag_estimate_process = None

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

    def _apply_apriltag_scale(self) -> None:
        if self._apriltag_last_scale is None:
            return
        self.ms_scale_edit.setText(f"{self._apriltag_last_scale:.9g}")
        self._on_profile_option_changed()
        self.apriltag_status_label.setText(i18n.t("APRILTAG_APPLIED"))
