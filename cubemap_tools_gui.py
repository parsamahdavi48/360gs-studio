#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from PySide6.QtCore import QProcess, Qt, QTimer
    from PySide6.QtGui import QCloseEvent, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    QProcess = None
    Qt = None
    QTimer = None
    QCloseEvent = None
    QImage = None
    QPixmap = None
    QApplication = None
    QCheckBox = None
    QComboBox = None
    QFileDialog = None
    QFormLayout = None
    QGridLayout = None
    QHBoxLayout = None
    QLabel = None
    QLineEdit = None
    QMainWindow = None
    QMessageBox = None
    QPlainTextEdit = None
    QPushButton = None
    QSlider = None
    QVBoxLayout = None
    QWidget = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None


_CONVERT_RE = re.compile(r"^Converting\s+(\d+)\s+images\.\.\.$")
_MIN_YAW_SLOTS = 4
_MAX_YAW_SLOTS = 8
_DEFAULT_YAW_SLOTS = 6
_MAX_PITCH_ROWS = 9
_WARN_ENABLED_VIEWS = 24
_BLOCK_ENABLED_VIEWS = 40


def _normalize_angle(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    r = ry @ rx
    r[np.abs(r) < 1e-10] = 0.0
    return r


if QMainWindow is not None:
    class CubemapToolsWindow(QMainWindow):
        def __init__(self, initial_scene_dir: str | None = None) -> None:
            super().__init__()
            self.base_dir = Path(__file__).resolve().parent
            self.proc: QProcess | None = None
            self.current_phase = ""
            self._process_buffer = ""
            self._converted_total = 0
            self._processed_sources = 0
            self._preview_pixmap: QPixmap | None = None
            self.preview_images: list[Path] = []
            self._preview_slider_sync = False
            self.cancel_requested = False

            self.pitch_rows: list[dict] = []
            self.yaw_slot_labels: list[QLabel] = []

            self.setWindowTitle("Cubemap Tools")
            self.resize(1380, 940)

            self._build_ui(initial_scene_dir or ".")
            self._apply_scene_paths()
            self._apply_pitch_rows()
            self._refresh_action_buttons()
            self._render_preview()

        def _build_ui(self, initial_scene_dir: str) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            form = QFormLayout()

            self.scene_dir_edit = QLineEdit(initial_scene_dir)
            self.scene_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            self.scene_dir_edit.textChanged.connect(lambda _: self._update_output_estimate())
            browse_scene_btn = QPushButton("Browse")
            browse_scene_btn.clicked.connect(self._browse_scene_dir)
            apply_scene_btn = QPushButton("Apply Scene Paths")
            apply_scene_btn.clicked.connect(self._apply_scene_paths)
            row = QHBoxLayout()
            row.addWidget(self.scene_dir_edit)
            row.addWidget(browse_scene_btn)
            row.addWidget(apply_scene_btn)
            form.addRow("Scene Directory", row)

            self.output_dir_edit = QLineEdit()
            self.output_dir_edit.textChanged.connect(lambda _: self._refresh_action_buttons())
            browse_output_btn = QPushButton("Browse")
            browse_output_btn.clicked.connect(self._browse_output_dir)
            row = QHBoxLayout()
            row.addWidget(self.output_dir_edit)
            row.addWidget(browse_output_btn)
            form.addRow("Output Directory", row)

            self.json_name_edit = QLineEdit("transforms.json")
            form.addRow("Transforms JSON", self.json_name_edit)

            self.mask_dir_edit = QLineEdit()
            self.mask_dir_edit.textChanged.connect(lambda _: self._render_preview())
            browse_mask_btn = QPushButton("Browse")
            browse_mask_btn.clicked.connect(self._browse_mask_dir)
            row = QHBoxLayout()
            row.addWidget(self.mask_dir_edit)
            row.addWidget(browse_mask_btn)
            form.addRow("Mask Directory", row)

            self.yaw_offset_edit = QLineEdit("45.0")
            self.yaw_offset_edit.textChanged.connect(self._on_yaw_or_pitch_changed)
            form.addRow("Yaw Offset (deg)", self.yaw_offset_edit)

            self.yaw_slots_combo = QComboBox()
            self.yaw_slots_combo.addItems([str(v) for v in range(_MIN_YAW_SLOTS, _MAX_YAW_SLOTS + 1)])
            self.yaw_slots_combo.setCurrentText(str(_DEFAULT_YAW_SLOTS))
            self.yaw_slots_combo.currentTextChanged.connect(lambda _: self._on_yaw_slots_changed())
            form.addRow("Yaw Slots", self.yaw_slots_combo)

            self.pitch_list_edit = QLineEdit("-30,0,30")
            form.addRow("Pitch Rows (deg CSV)", self.pitch_list_edit)

            self.fov_label = QLabel("90.0 (fixed)")
            form.addRow("FOV", self.fov_label)

            self.sample_image_edit = QLineEdit()
            self.sample_image_edit.textChanged.connect(self._on_sample_image_changed)
            browse_sample_btn = QPushButton("Browse")
            browse_sample_btn.clicked.connect(self._browse_sample_image)
            auto_sample_btn = QPushButton("Auto")
            auto_sample_btn.clicked.connect(self._auto_select_sample_image)
            reload_sample_btn = QPushButton("Reload")
            reload_sample_btn.clicked.connect(self._refresh_preview_image_list)
            row = QHBoxLayout()
            row.addWidget(self.sample_image_edit)
            row.addWidget(browse_sample_btn)
            row.addWidget(auto_sample_btn)
            row.addWidget(reload_sample_btn)
            form.addRow("Preview Image", row)

            self.preview_timeline_slider = QSlider(Qt.Horizontal)
            self.preview_timeline_slider.setRange(0, 0)
            self.preview_timeline_slider.setValue(0)
            self.preview_timeline_slider.setEnabled(False)
            self.preview_timeline_slider.valueChanged.connect(self._on_preview_timeline_changed)
            self.preview_timeline_label = QLabel("0 / 0")
            timeline_row = QHBoxLayout()
            timeline_row.addWidget(self.preview_timeline_slider, stretch=1)
            timeline_row.addWidget(self.preview_timeline_label)
            form.addRow("Preview Timeline", timeline_row)

            self.mask_overlay_slider = QSlider(Qt.Horizontal)
            self.mask_overlay_slider.setRange(0, 100)
            self.mask_overlay_slider.setValue(35)
            self.mask_overlay_slider.valueChanged.connect(lambda _: self._render_preview())
            form.addRow("Mask Overlay (%)", self.mask_overlay_slider)

            self.preview_mask_edit = QLineEdit()
            self.preview_mask_edit.textChanged.connect(lambda _: self._render_preview())
            browse_preview_mask_btn = QPushButton("Browse")
            browse_preview_mask_btn.clicked.connect(self._browse_preview_mask_image)
            clear_preview_mask_btn = QPushButton("Clear")
            clear_preview_mask_btn.clicked.connect(self._clear_preview_mask_image)
            row = QHBoxLayout()
            row.addWidget(self.preview_mask_edit)
            row.addWidget(browse_preview_mask_btn)
            row.addWidget(clear_preview_mask_btn)
            form.addRow("Preview Mask Image", row)

            options_row = QHBoxLayout()
            self.mask_from_alpha_check = QCheckBox("Extract mask from alpha (--mask_from_alpha)")
            self.mask_from_alpha_check.setChecked(False)
            options_row.addWidget(self.mask_from_alpha_check)

            self.no_image_check = QCheckBox("Transforms only (--no_image)")
            self.no_image_check.setChecked(False)
            options_row.addWidget(self.no_image_check)

            self.no_transform_check = QCheckBox("No axis transform (--no_transform)")
            self.no_transform_check.setChecked(False)
            options_row.addWidget(self.no_transform_check)

            self.duplicate_check = QCheckBox("Allow duplicate (--duplicate)")
            self.duplicate_check.setChecked(False)
            options_row.addWidget(self.duplicate_check)
            options_row.addStretch(1)
            form.addRow("Options", options_row)

            layout.addLayout(form)

            view_ctrl_row = QHBoxLayout()
            apply_pitch_btn = QPushButton("Apply Pitch Rows")
            apply_pitch_btn.clicked.connect(self._apply_pitch_rows)
            view_ctrl_row.addWidget(apply_pitch_btn)

            all_on_btn = QPushButton("All On")
            all_on_btn.clicked.connect(self._all_on)
            view_ctrl_row.addWidget(all_on_btn)

            all_off_btn = QPushButton("All Off")
            all_off_btn.clicked.connect(self._all_off)
            view_ctrl_row.addWidget(all_off_btn)

            refresh_preview_btn = QPushButton("Refresh Preview")
            refresh_preview_btn.clicked.connect(self._render_preview)
            view_ctrl_row.addWidget(refresh_preview_btn)

            view_ctrl_row.addStretch(1)

            self.selected_views_label = QLabel("Selected views: 0")
            self.selected_views_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            view_ctrl_row.addWidget(self.selected_views_label)
            layout.addLayout(view_ctrl_row)

            self.estimate_label = QLabel("Estimated output images: -")
            self.estimate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(self.estimate_label)

            self.view_grid_widget = QWidget()
            self.view_grid_layout = QGridLayout(self.view_grid_widget)
            layout.addWidget(self.view_grid_widget)

            self.preview_label = QLabel("Preview unavailable")
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setMinimumSize(960, 440)
            self.preview_label.setStyleSheet("border: 1px solid palette(mid);")
            layout.addWidget(self.preview_label, stretch=1)

            btn_row = QHBoxLayout()
            self.run_button = QPushButton("Run Cubemap Convert")
            self.run_button.clicked.connect(self._run_convert)
            btn_row.addWidget(self.run_button)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._cancel_running_process)
            self.cancel_button.setEnabled(False)
            btn_row.addWidget(self.cancel_button)
            btn_row.addStretch(1)

            self.status_label = QLabel("Idle")
            self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            btn_row.addWidget(self.status_label)
            layout.addLayout(btn_row)

            self.log_text = QPlainTextEdit()
            self.log_text.setReadOnly(True)
            layout.addWidget(self.log_text, stretch=1)

        def _is_running(self) -> bool:
            return self.proc is not None and self.proc.state() != QProcess.NotRunning

        def _append_log(self, text: str) -> None:
            self.log_text.appendPlainText(text)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _set_running_state(self, running: bool, status_text: str) -> None:
            self.status_label.setText(status_text)
            self.run_button.setEnabled(not running and Path(self.scene_dir_edit.text().strip()).is_dir())
            self.cancel_button.setEnabled(running)

        def _refresh_action_buttons(self) -> None:
            if self._is_running():
                self.run_button.setEnabled(False)
                self.cancel_button.setEnabled(True)
                return
            self.run_button.setEnabled(Path(self.scene_dir_edit.text().strip()).is_dir())
            self.cancel_button.setEnabled(False)

        def _browse_scene_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select scene directory")
            if path:
                self.scene_dir_edit.setText(path)
                self._refresh_preview_image_list(prefer_current=False)

        def _browse_output_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select output directory")
            if path:
                self.output_dir_edit.setText(path)

        def _browse_mask_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select mask directory")
            if path:
                self.mask_dir_edit.setText(path)

        def _browse_sample_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select preview image",
                "",
                "Image files (*.jpg *.jpeg *.png);;All files (*.*)",
            )
            if path:
                self.sample_image_edit.setText(path)

        def _browse_preview_mask_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select preview mask image",
                "",
                "Image files (*.png *.jpg *.jpeg);;All files (*.*)",
            )
            if path:
                self.preview_mask_edit.setText(path)

        def _clear_preview_mask_image(self) -> None:
            self.preview_mask_edit.setText("")

        def _apply_scene_paths(self) -> None:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            self.output_dir_edit.setText(str(scene_dir / "cubic"))
            self.mask_dir_edit.setText(str(scene_dir / "masks"))
            self._refresh_preview_image_list(prefer_current=False)
            self._refresh_action_buttons()
            self._update_output_estimate()

        def _auto_select_sample_image(self) -> None:
            if not self.preview_images:
                self._refresh_preview_image_list(prefer_current=False)
                return
            self._set_preview_index(0)

        def _iter_scene_preview_images(self) -> list[Path]:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            images_dir = scene_dir / "images"
            roots = [images_dir] if images_dir.is_dir() else [scene_dir]
            exts = {".jpg", ".jpeg", ".png"}

            candidates: list[Path] = []
            seen: set[str] = set()
            for root in roots:
                if not root.is_dir():
                    continue
                for p in root.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in exts:
                        continue
                    key = str(p.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(p)
            candidates.sort(key=lambda x: str(x).lower())
            return candidates

        def _update_preview_timeline_label(self, idx: int = -1) -> None:
            total = len(self.preview_images)
            if total <= 0:
                self.preview_timeline_label.setText("0 / 0")
                return
            if idx < 0:
                try:
                    idx = self.preview_timeline_slider.value()
                except Exception:
                    idx = 0
            idx = max(0, min(idx, total - 1))
            name = self.preview_images[idx].name
            self.preview_timeline_label.setText(f"{idx + 1} / {total} : {name}")

        def _set_preview_index(self, idx: int) -> None:
            if not self.preview_images:
                self.sample_image_edit.setText("")
                self._update_preview_timeline_label(-1)
                return
            idx = max(0, min(idx, len(self.preview_images) - 1))
            self._preview_slider_sync = True
            self.preview_timeline_slider.setValue(idx)
            self._preview_slider_sync = False
            self.sample_image_edit.setText(str(self.preview_images[idx]))
            self._update_preview_timeline_label(idx)

        def _sync_preview_slider_from_sample(self) -> None:
            sample_text = self.sample_image_edit.text().strip()
            if not sample_text:
                self._update_preview_timeline_label(-1)
                return
            try:
                sample_key = str(Path(sample_text).resolve()).lower()
            except Exception:
                self._update_preview_timeline_label(-1)
                return

            for idx, path in enumerate(self.preview_images):
                if str(path.resolve()).lower() == sample_key:
                    self._preview_slider_sync = True
                    self.preview_timeline_slider.setValue(idx)
                    self._preview_slider_sync = False
                    self._update_preview_timeline_label(idx)
                    return

            total = len(self.preview_images)
            if total > 0:
                self.preview_timeline_label.setText(f"custom / {total} : {Path(sample_text).name}")
            else:
                self.preview_timeline_label.setText("custom / 0")

        def _refresh_preview_image_list(self, prefer_current: bool = True) -> None:
            current_text = self.sample_image_edit.text().strip()
            self.preview_images = self._iter_scene_preview_images()
            total = len(self.preview_images)

            self.preview_timeline_slider.setEnabled(total > 0)
            self.preview_timeline_slider.setRange(0, max(0, total - 1))

            if total <= 0:
                self.preview_timeline_slider.setValue(0)
                self._update_preview_timeline_label(-1)
                if not current_text:
                    self.sample_image_edit.setText("")
                self._render_preview()
                return

            target_idx = 0
            if prefer_current and current_text:
                try:
                    current_key = str(Path(current_text).resolve()).lower()
                    for idx, p in enumerate(self.preview_images):
                        if str(p.resolve()).lower() == current_key:
                            target_idx = idx
                            break
                except Exception:
                    pass

            self._set_preview_index(target_idx)
            self._render_preview()

        def _on_preview_timeline_changed(self, idx: int) -> None:
            if self._preview_slider_sync:
                return
            if idx < 0 or idx >= len(self.preview_images):
                self._update_preview_timeline_label(-1)
                return
            self._set_preview_index(idx)

        def _on_sample_image_changed(self, _text: str) -> None:
            self._sync_preview_slider_from_sample()
            self._render_preview()

        @staticmethod
        def _parse_float(text: str, label: str) -> float:
            try:
                return float(text.strip())
            except Exception as e:
                raise ValueError(f"{label} is invalid: {e}") from e

        def _yaw_slot_count(self) -> int:
            text = self.yaw_slots_combo.currentText().strip()
            try:
                value = int(text)
            except Exception as e:
                raise ValueError(f"Yaw Slots is invalid: {e}") from e
            if value < _MIN_YAW_SLOTS or value > _MAX_YAW_SLOTS:
                raise ValueError(f"Yaw Slots must be in [{_MIN_YAW_SLOTS}, {_MAX_YAW_SLOTS}]")
            return value

        def _yaw_step_deg(self) -> float:
            return 360.0 / float(self._yaw_slot_count())

        def _count_input_images(self) -> int:
            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            images_dir = scene_dir / "images"

            roots: list[Path] = [images_dir] if images_dir.is_dir() else [scene_dir]
            exts = {".jpg", ".jpeg", ".png"}
            seen: set[str] = set()
            count = 0
            for root in roots:
                if not root.is_dir():
                    continue
                for p in root.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in exts:
                        continue
                    key = str(p.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    count += 1
            return count

        def _update_output_estimate(self) -> None:
            try:
                views = self._collect_views(include_disabled=True)
            except Exception:
                self.estimate_label.setText("Estimated output images: -")
                return

            enabled = sum(1 for v in views if v["enabled"])
            source_count = self._count_input_images()
            estimated = source_count * enabled
            warn = ""
            if enabled > _BLOCK_ENABLED_VIEWS:
                warn = " [too many selected]"
            elif enabled > _WARN_ENABLED_VIEWS:
                warn = " [high]"
            self.estimate_label.setText(
                f"Estimated output images: {estimated} ({source_count} x {enabled}){warn}"
            )

        def _parse_pitches(self) -> list[float]:
            raw = self.pitch_list_edit.text().strip()
            if not raw:
                raise ValueError("Pitch rows is empty")

            tokens = [t for t in re.split(r"[,\s]+", raw) if t]
            pitches: list[float] = []
            seen: set[str] = set()
            for token in tokens:
                p = float(token)
                if p < -90.0 or p > 90.0:
                    raise ValueError(f"Pitch out of range [-90, 90]: {p}")
                key = f"{p:.6f}"
                if key in seen:
                    continue
                seen.add(key)
                pitches.append(p)

            if not pitches:
                raise ValueError("No valid pitches")
            if len(pitches) > _MAX_PITCH_ROWS:
                raise ValueError(f"Pitch rows are limited to {_MAX_PITCH_ROWS}")
            return pitches

        def _on_yaw_slots_changed(self) -> None:
            self._apply_pitch_rows()

        def _clear_view_grid(self) -> None:
            while self.view_grid_layout.count():
                item = self.view_grid_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def _on_yaw_or_pitch_changed(self) -> None:
            self._update_yaw_slot_labels()
            self._update_selected_views_label()
            self._render_preview()

        def _apply_pitch_rows(self) -> None:
            old_state: dict[str, list[bool]] = {}
            for row in self.pitch_rows:
                key = f"{row['pitch']:.6f}"
                old_state[key] = [cb.isChecked() for cb in row["checks"]]

            try:
                pitches = self._parse_pitches()
            except Exception as e:
                QMessageBox.critical(self, "Pitch Error", str(e))
                return

            slot_count = self._yaw_slot_count()
            self.pitch_rows = []
            self.yaw_slot_labels = []
            self._clear_view_grid()

            head_pitch = QLabel("Pitch / Slot")
            self.view_grid_layout.addWidget(head_pitch, 0, 0)

            for slot in range(slot_count):
                lab = QLabel(f"S{slot}")
                lab.setAlignment(Qt.AlignCenter)
                self.yaw_slot_labels.append(lab)
                self.view_grid_layout.addWidget(lab, 0, slot + 1)

            for row_idx, pitch in enumerate(pitches, start=1):
                pitch_label = QLabel(f"{pitch:g} deg")
                pitch_label.setAlignment(Qt.AlignCenter)
                self.view_grid_layout.addWidget(pitch_label, row_idx, 0)

                checks: list[QCheckBox] = []
                key = f"{pitch:.6f}"
                restored = old_state.get(key)
                for slot in range(slot_count):
                    cb = QCheckBox()
                    if restored is not None and slot < len(restored):
                        checked = restored[slot]
                    else:
                        checked = abs(pitch) < 1e-6
                    cb.setChecked(checked)
                    cb.toggled.connect(self._on_view_selection_changed)
                    self.view_grid_layout.addWidget(cb, row_idx, slot + 1, alignment=Qt.AlignCenter)
                    checks.append(cb)

                self.pitch_rows.append({"pitch": pitch, "checks": checks})

            self._update_yaw_slot_labels()
            self._update_selected_views_label()
            self._render_preview()

        def _all_on(self) -> None:
            for row in self.pitch_rows:
                for cb in row["checks"]:
                    cb.setChecked(True)
            self._on_view_selection_changed()

        def _all_off(self) -> None:
            for row in self.pitch_rows:
                for cb in row["checks"]:
                    cb.setChecked(False)
            self._on_view_selection_changed()

        def _on_view_selection_changed(self, *_args) -> None:
            self._update_selected_views_label()
            self._render_preview()

        def _update_yaw_slot_labels(self) -> None:
            try:
                yaw_offset = self._parse_float(self.yaw_offset_edit.text(), "Yaw Offset")
                yaw_step = self._yaw_step_deg()
            except Exception:
                for idx, lab in enumerate(self.yaw_slot_labels):
                    lab.setText(f"S{idx}")
                return

            for idx, lab in enumerate(self.yaw_slot_labels):
                yaw = _normalize_angle(yaw_offset + idx * yaw_step)
                lab.setText(f"S{idx}\n{yaw:.1f}deg")

        def _angle_token(self, angle: float) -> str:
            sign = "p" if angle >= 0 else "m"
            value = abs(angle)
            token = f"{value:g}".replace(".", "d")
            return f"{sign}{token}"

        def _collect_views(self, include_disabled: bool) -> list[dict]:
            yaw_offset = self._parse_float(self.yaw_offset_edit.text(), "Yaw Offset")
            yaw_step = self._yaw_step_deg()
            views: list[dict] = []

            for row in self.pitch_rows:
                pitch = float(row["pitch"])
                for slot, cb in enumerate(row["checks"]):
                    enabled = cb.isChecked()
                    if not include_disabled and not enabled:
                        continue
                    views.append(
                        {
                            "name": f"pit{self._angle_token(pitch)}_s{slot}",
                            "yaw": float(yaw_offset + slot * yaw_step),
                            "pitch": pitch,
                            "enabled": enabled,
                            "slot": slot,
                        }
                    )
            return views

        def _update_selected_views_label(self) -> None:
            try:
                views = self._collect_views(include_disabled=True)
                slot_count = self._yaw_slot_count()
            except Exception:
                self.selected_views_label.setText("Selected views: -")
                self._update_output_estimate()
                return

            selected = sum(1 for v in views if v["enabled"])
            warn = ""
            if selected > _BLOCK_ENABLED_VIEWS:
                warn = " [limit exceeded]"
            elif selected > _WARN_ENABLED_VIEWS:
                warn = " [high]"
            self.selected_views_label.setText(
                f"Selected views: {selected} / {len(views)} (slots={slot_count}){warn}"
            )
            self._update_output_estimate()

        def _find_mask_for_sample(self, sample_path: Path) -> Path | None:
            mask_dir = Path(self.mask_dir_edit.text().strip())
            if not mask_dir.is_dir():
                return None

            candidates: list[Path] = [
                mask_dir / sample_path.name,
                mask_dir / f"{sample_path.name}.png",
                mask_dir / f"{sample_path.stem}.png",
            ]

            scene_dir = Path(self.scene_dir_edit.text().strip() or ".")
            images_root = scene_dir / "images"
            if sample_path.is_absolute() and images_root.is_dir():
                try:
                    rel = sample_path.relative_to(images_root)
                    candidates.extend(
                        [
                            mask_dir / rel,
                            mask_dir / rel.with_suffix(".png"),
                            mask_dir / f"{rel.name}.png",
                        ]
                    )
                except Exception:
                    pass

            for c in candidates:
                if c.exists() and c.is_file():
                    return c
            return None

        def _resolve_preview_mask_path(self, sample_path: Path) -> Path | None:
            manual = self.preview_mask_edit.text().strip()
            if manual:
                manual_path = Path(manual)
                if manual_path.exists() and manual_path.is_file():
                    return manual_path
            return self._find_mask_for_sample(sample_path)

        def _view_boundary_segments(
            self,
            width: int,
            height: int,
            yaw_deg: float,
            pitch_deg: float,
            fov_deg: float,
            samples_per_edge: int = 120,
        ) -> list[np.ndarray]:
            t = np.linspace(-1.0, 1.0, samples_per_edge)
            u = np.concatenate([
                t,
                np.ones_like(t),
                t[::-1],
                -np.ones_like(t),
                np.array([-1.0]),
            ])
            v = np.concatenate([
                np.ones_like(t),
                t[::-1],
                -np.ones_like(t),
                t,
                np.array([1.0]),
            ])

            focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
            rays = np.stack([u, v, np.full_like(u, focal)], axis=-1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

            r = _rotation_matrix(yaw_deg, pitch_deg)
            rays = rays @ r.T

            lon = np.arctan2(rays[:, 0], rays[:, 2])
            lat = np.arcsin(np.clip(rays[:, 1], -1.0, 1.0))

            x = (lon / np.pi + 1.0) * 0.5 * width
            y = (0.5 - lat / np.pi) * height
            points = np.stack([x, y], axis=-1)

            segments: list[np.ndarray] = []
            current: list[np.ndarray] = [points[0]]
            threshold = width * 0.45
            for p in points[1:]:
                if abs(float(p[0] - current[-1][0])) > threshold:
                    if len(current) >= 2:
                        segments.append(np.array(current, dtype=np.float32))
                    current = [p]
                else:
                    current.append(p)
            if len(current) >= 2:
                segments.append(np.array(current, dtype=np.float32))
            return segments

        def _render_preview(self) -> None:
            sample_path = Path(self.sample_image_edit.text().strip())
            if not sample_path.exists():
                self.preview_label.setText("Preview image not found")
                self._preview_pixmap = None
                return

            img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
            if img is None:
                self.preview_label.setText("Failed to load preview image")
                self._preview_pixmap = None
                return

            max_width = 1900
            if img.shape[1] > max_width:
                scale = max_width / float(img.shape[1])
                img = cv2.resize(
                    img,
                    (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            mask_path = self._resolve_preview_mask_path(sample_path)
            if mask_path is not None:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    alpha = float(self.mask_overlay_slider.value()) / 100.0
                    if alpha > 0:
                        masked = mask < 128
                        overlay = np.zeros_like(img)
                        overlay[:, :, 2] = 255
                        img[masked] = (
                            (1.0 - alpha) * img[masked].astype(np.float32)
                            + alpha * overlay[masked].astype(np.float32)
                        ).astype(np.uint8)

            try:
                views = self._collect_views(include_disabled=True)
            except Exception:
                views = []

            h, w = img.shape[:2]
            for view in views:
                color = (0, 220, 255) if view["enabled"] else (128, 128, 128)
                thickness = 2 if view["enabled"] else 1
                segments = self._view_boundary_segments(w, h, view["yaw"], view["pitch"], 90.0)
                all_points: list[np.ndarray] = []
                for seg in segments:
                    if len(seg) < 2:
                        continue
                    pts = np.round(seg).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)
                    all_points.append(seg)

                if view["enabled"] and all_points:
                    merged = np.concatenate(all_points, axis=0)
                    cx = int(np.clip(np.mean(merged[:, 0]), 0, w - 1))
                    cy = int(np.clip(np.mean(merged[:, 1]), 0, h - 1))
                    label = f"p{view['pitch']:g}/s{view['slot']}"
                    cv2.putText(
                        img,
                        label,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        lineType=cv2.LINE_AA,
                    )

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
            self._preview_pixmap = QPixmap.fromImage(qimg)
            self._update_preview_pixmap()

        def _update_preview_pixmap(self) -> None:
            if self._preview_pixmap is None:
                return
            scaled = self._preview_pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)

        def resizeEvent(self, event) -> None:  # pragma: no cover - UI event
            super().resizeEvent(event)
            self._update_preview_pixmap()

        def _write_views_config(self, output_dir: Path, views: list[dict]) -> Path:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "views_config.json"
            payload = {
                "fov": 90.0,
                "views": [
                    {
                        "name": v["name"],
                        "yaw": float(v["yaw"]),
                        "pitch": float(v["pitch"]),
                        "enabled": bool(v["enabled"]),
                    }
                    for v in views
                ],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path

        def _build_cmd(self) -> list[str]:
            script = self.base_dir / "cubemap_transforms_json.py"
            if not script.exists():
                raise FileNotFoundError(f"cubemap_transforms_json.py not found: {script}")

            input_dir = Path(self.scene_dir_edit.text().strip() or ".")
            if not input_dir.is_dir():
                raise ValueError(f"Scene directory not found: {input_dir}")

            output_dir = Path(self.output_dir_edit.text().strip() or (input_dir / "cubic"))
            json_name = self.json_name_edit.text().strip() or "transforms.json"

            all_views = self._collect_views(include_disabled=True)
            enabled_count = sum(1 for v in all_views if v["enabled"])
            if enabled_count <= 0:
                raise ValueError("At least one view must be enabled")
            if enabled_count > _BLOCK_ENABLED_VIEWS:
                raise ValueError(
                    f"Too many enabled views ({enabled_count}). "
                    f"Reduce to <= {_BLOCK_ENABLED_VIEWS}."
                )

            views_json = self._write_views_config(output_dir, all_views)

            cmd = [
                sys.executable,
                str(script),
                str(input_dir),
                str(output_dir),
                "--json",
                json_name,
                "--fov",
                "90",
                "--views-json",
                str(views_json),
            ]

            mask_dir = Path(self.mask_dir_edit.text().strip())
            if mask_dir.is_dir():
                cmd.extend(["--mask_dir", str(mask_dir)])

            if self.mask_from_alpha_check.isChecked():
                cmd.append("--mask_from_alpha")
            if self.no_image_check.isChecked():
                cmd.append("--no_image")
            if self.no_transform_check.isChecked():
                cmd.append("--no_transform")
            if self.duplicate_check.isChecked():
                cmd.append("--duplicate")

            return cmd

        def _start_process(self, cmd: list[str], phase: str) -> None:
            if self._is_running():
                QMessageBox.warning(self, "Busy", "Another process is running.")
                return

            self.cancel_requested = False
            self.current_phase = phase
            self._process_buffer = ""
            self._converted_total = 0
            self._processed_sources = 0
            self._append_log("$ " + " ".join(cmd))

            proc = QProcess(self)
            proc.setProgram(cmd[0])
            proc.setArguments(cmd[1:])
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(self._on_process_output)
            proc.errorOccurred.connect(self._on_process_error)
            proc.finished.connect(self._on_process_finished)
            self.proc = proc

            self._set_running_state(True, f"Running: {phase}")
            proc.start()

        def _terminate_process_gracefully(self, proc: QProcess, phase: str, timeout_ms: int = 3000) -> None:
            if proc.state() == QProcess.NotRunning:
                return
            self._append_log(f"[{phase}] cancel requested; sending terminate")
            proc.terminate()
            QTimer.singleShot(timeout_ms, lambda p=proc, ph=phase: self._force_kill_if_running(p, ph))

        def _force_kill_if_running(self, proc: QProcess, phase: str) -> None:
            if proc.state() == QProcess.NotRunning:
                return
            self._append_log(f"[{phase}] terminate timeout; killing process")
            proc.kill()

        def _cancel_running_process(self) -> None:
            if not self._is_running() or self.proc is None:
                return
            self.cancel_requested = True
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Canceling...")
            self._terminate_process_gracefully(self.proc, self.current_phase)

        def _handle_process_line(self, line: str) -> None:
            if not line:
                return
            self._append_log(line)

            m = _CONVERT_RE.match(line)
            if m:
                self._converted_total = int(m.group(1))
                self._processed_sources = 0
                self.status_label.setText(f"Running: convert (0/{self._converted_total})")
                return

            if line.startswith("Processing:"):
                self._processed_sources += 1
                if self._converted_total > 0:
                    self.status_label.setText(
                        f"Running: convert ({min(self._processed_sources, self._converted_total)}/{self._converted_total})"
                    )
                else:
                    self.status_label.setText(f"Running: convert ({self._processed_sources})")

        def _on_process_output(self) -> None:
            if self.proc is None:
                return
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            data = data.replace("\r", "\n")
            self._process_buffer += data
            while "\n" in self._process_buffer:
                line, self._process_buffer = self._process_buffer.split("\n", 1)
                self._handle_process_line(line.rstrip("\r"))

        def _on_process_error(self, _error) -> None:
            if self.proc is None:
                return
            self._append_log(f"[{self.current_phase}] process error occurred")

        def _on_process_finished(self, exit_code: int, _status) -> None:
            if self._process_buffer:
                tail = self._process_buffer.replace("\r", "\n").strip()
                if tail:
                    for line in tail.splitlines():
                        self._handle_process_line(line)
                self._process_buffer = ""

            phase = self.current_phase
            was_canceled = self.cancel_requested
            self.cancel_requested = False
            if exit_code == 0:
                if was_canceled:
                    self._append_log(f"[{phase}] canceled by user")
                    self._set_running_state(False, f"Canceled: {phase}")
                else:
                    self._append_log(f"[{phase}] completed successfully")
                    self._set_running_state(False, f"Done: {phase}")
            else:
                if was_canceled:
                    self._append_log(f"[{phase}] canceled by user")
                    self._set_running_state(False, f"Canceled: {phase}")
                else:
                    self._append_log(f"[{phase}] failed (exit={exit_code})")
                    self._set_running_state(False, f"Failed: {phase}")
            self.proc = None

        def _run_convert(self) -> None:
            try:
                all_views = self._collect_views(include_disabled=True)
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return

            enabled_count = sum(1 for v in all_views if v["enabled"])
            if enabled_count > _WARN_ENABLED_VIEWS and enabled_count <= _BLOCK_ENABLED_VIEWS:
                reply = QMessageBox.question(
                    self,
                    "Large View Count",
                    (
                        f"{enabled_count} views are enabled. "
                        "This may increase processing time and output size.\n\n"
                        "Continue?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            try:
                cmd = self._build_cmd()
            except Exception as e:
                QMessageBox.critical(self, "Invalid Input", str(e))
                return
            self._start_process(cmd, "cubemap")

        def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - UI event
            if self._is_running() and self.proc is not None:
                self.proc.terminate()
                if not self.proc.waitForFinished(2000):
                    self.proc.kill()
                    self.proc.waitForFinished(2000)
            super().closeEvent(event)

else:
    class CubemapToolsWindow:  # pragma: no cover - placeholder when PySide6 missing
        pass


def ensure_gui_deps() -> None:
    if QApplication is None:
        raise RuntimeError(f"PySide6 is required to run this GUI: {_PYSIDE_IMPORT_ERROR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI wrapper for cubemap_transforms_json.py")
    parser.add_argument("--scene-dir", default=".", help="Initial scene directory shown in GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_gui_deps()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = CubemapToolsWindow(initial_scene_dir=args.scene_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
