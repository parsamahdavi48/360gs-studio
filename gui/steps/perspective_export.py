"""Perspective Export workspace."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.app_job import perspective_app_job
from core.image_io import imread_unicode
from gs360studio.domain.models import ViewSpec, atomic_write_json, cubemap_view_specs, grid_view_specs
from gs360studio.engine.perspective_export import estimate_batch_size
from gs360studio.engine.projection import ProjectionMapCache, project_equirectangular
from gs360studio.platform.job_store import recover_interrupted_jobs
from gs360studio.platform.project_store import migrate_legacy_project
from gui.common import dialogs
from gui.common.browse_widget import BrowseWidget
from gui.common.runner_types import StepCommandQueue
from gui.perspective.globe_widget import ViewGlobeWidget
from gui.steps.base_step import BaseStepWidget

_PROGRESS_RE = re.compile(r"^PROGRESS:(\d+)/(\d+)$")
_COLUMNS = ("On", "Name", "Yaw", "Pitch", "Roll", "H-FOV", "V-FOV", "Width", "Height")


class PerspectiveExportStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._views = cubemap_view_specs(1600)
        self._updating_table = False
        self._preview_source = None
        self._preview_cache = ProjectionMapCache(8)
        self._source_duration_sec = 0.0
        self._source_frame_count = 0
        self._build_ui()
        self._populate_table()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        paths = QWidget()
        form = QFormLayout(paths)
        self.input_browse = BrowseWidget(mode="file", filter_str="Media (*.mp4 *.mov *.mkv *.avi *.png *.jpg *.jpeg *.tif *.tiff *.webp);;All files (*.*)")
        self.output_browse = BrowseWidget(mode="dir")
        form.addRow("Input media", self.input_browse)
        form.addRow("Output folder", self.output_browse)
        root.addWidget(paths)

        options = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItem("PNG sequence", "png")
        self.format_combo.addItem("JPEG sequence", "jpeg")
        self.format_combo.addItem("H.265 / HEVC video", "video")
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.01, 3600.0)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" s")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(0, 32)
        self.batch_spin.setSpecialValueText("Auto")
        self.nvenc_check = QCheckBox("NVENC")
        self.rig_check = QCheckBox("COLMAP rig")
        self.overwrite_check = QCheckBox("Replace existing export")
        for label, widget in (("Format", self.format_combo), ("Interval", self.interval_spin), ("Batch", self.batch_spin)):
            options.addWidget(QLabel(label))
            options.addWidget(widget)
        options.addWidget(self.nvenc_check)
        options.addWidget(self.rig_check)
        options.addWidget(self.overwrite_check)
        options.addStretch()
        root.addLayout(options)

        preset_row = QHBoxLayout()
        for label, callback in (
            ("Cubemap", self._apply_cubemap),
            ("8-view ring", self._apply_ring),
            ("Add view", self._add_view),
            ("Remove view", self._remove_view),
            ("Save profile", self._save_profile),
            ("Load profile", self._load_profile),
            ("Refresh preview", self._refresh_preview),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            preset_row.addWidget(button)
        preset_row.addStretch()
        root.addLayout(preset_row)

        split = QSplitter(Qt.Horizontal)
        designer = QWidget()
        designer_layout = QVBoxLayout(designer)
        designer_layout.setContentsMargins(0, 0, 0, 0)
        self.globe = ViewGlobeWidget()
        self.globe.set_views(self._views, 0)
        self.globe.view_selected.connect(self._select_row)
        self.globe.views_changed.connect(self._adopt_globe_views)
        designer_layout.addWidget(self.globe, stretch=1)
        self.estimate_label = QLabel("")
        self.estimate_label.setObjectName("workflowNote")
        self.estimate_label.setWordWrap(True)
        designer_layout.addWidget(self.estimate_label)
        split.addWidget(designer)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel("Choose an image or video, then refresh the preview.")
        self.preview_label.setObjectName("workPane")
        self.preview_label.setMinimumSize(360, 240)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setScaledContents(False)
        right_layout.addWidget(self.preview_label, stretch=2)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        right_layout.addWidget(self.table, stretch=3)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        root.addWidget(split, stretch=1)

        self.input_browse.path_changed.connect(self._on_input_changed)
        self.output_browse.path_changed.connect(lambda _value: self.primary_action_state_changed.emit())
        self.format_combo.currentIndexChanged.connect(self._update_options)
        self.rig_check.toggled.connect(self._update_options)
        self.batch_spin.valueChanged.connect(lambda _value: self._update_estimate())
        self.interval_spin.valueChanged.connect(lambda _value: self._update_estimate())
        self._update_options()

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if not path:
            return
        scene = Path(path)
        if not self.output_browse.text():
            self.output_browse.set_text(str(scene / "output" / "perspective"))
        try:
            _manifest, report = migrate_legacy_project(scene)
            recovered = recover_interrupted_jobs(scene)
            if report.migrated or recovered:
                self.estimate_label.setText(
                    f"Project metadata ready. Migrated legacy scene: {report.migrated}; recovered jobs: {len(recovered)}."
                )
        except (OSError, ValueError) as exc:
            self.estimate_label.setText(f"Project metadata warning: {exc}")

    def primary_action_text(self) -> str:
        return "Export Perspective Views"

    def primary_action_tooltip(self) -> str:
        return "Export all enabled views with a bounded shared decode pipeline."

    def primary_action_enabled(self) -> bool:
        return bool(self.input_browse.text() and self.output_browse.text() and any(view.enabled for view in self._views))

    def process_log_dir(self) -> Path | None:
        return Path(self.scene_dir) / "_360gs" / "logs" if self.scene_dir else None

    def build_commands(self) -> StepCommandQueue:
        input_path = Path(self.input_browse.text())
        if not input_path.exists():
            raise ValueError(f"Input media does not exist: {input_path}")
        output_path = Path(self.output_browse.text())
        output_format = str(self.format_combo.currentData())
        if self.rig_check.isChecked() and output_format == "video":
            raise ValueError("COLMAP rig mode cannot export video")
        enabled_views = [view for view in self._views if view.enabled]
        if self.rig_check.isChecked():
            rig_shapes = {(view.width, view.height, view.hfov_deg, view.effective_vfov_deg) for view in enabled_views}
            if len(rig_shapes) != 1:
                raise ValueError("COLMAP rig export currently requires matching view dimensions and FOV values")
        payload = {
            "kind": "perspective_export",
            "scene_dir": self.scene_dir,
            "input_path": str(input_path),
            "output_dir": str(output_path),
            "views": [view.to_dict() for view in enabled_views],
            "output_format": output_format,
            "frame_interval_sec": self.interval_spin.value(),
            "use_nvenc": self.nvenc_check.isChecked(),
            "batch_size": self.batch_spin.value(),
            "colmap_rig": self.rig_check.isChecked(),
            "overwrite": self.overwrite_check.isChecked(),
        }
        return [("perspective_export", perspective_app_job(payload))]

    def on_line(self, line: str) -> tuple[int, int] | None:
        match = _PROGRESS_RE.match(line.strip())
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def phase_display_name(self, _phase: str) -> str:
        return "Perspective export"

    def _populate_table(self, selected: int | None = None) -> None:
        self._updating_table = True
        try:
            self.table.setRowCount(len(self._views))
            for row, view in enumerate(self._views):
                enabled = QTableWidgetItem()
                enabled.setCheckState(Qt.Checked if view.enabled else Qt.Unchecked)
                enabled.setData(Qt.UserRole, view.id)
                values = (
                    enabled,
                    QTableWidgetItem(view.name),
                    QTableWidgetItem(f"{view.yaw_deg:g}"),
                    QTableWidgetItem(f"{view.pitch_deg:g}"),
                    QTableWidgetItem(f"{view.roll_deg:g}"),
                    QTableWidgetItem(f"{view.hfov_deg:g}"),
                    QTableWidgetItem("" if view.vfov_deg is None else f"{view.vfov_deg:g}"),
                    QTableWidgetItem(str(view.width)),
                    QTableWidgetItem(str(view.height)),
                )
                for column, item in enumerate(values):
                    self.table.setItem(row, column, item)
            self.table.resizeColumnsToContents()
            if self._views:
                target = max(0, min(selected if selected is not None else self.table.currentRow(), len(self._views) - 1))
                self.table.selectRow(target)
                self.globe.set_views(self._views, target)
        finally:
            self._updating_table = False
        self._update_estimate()
        self.primary_action_state_changed.emit()

    def _view_from_row(self, row: int) -> ViewSpec:
        previous = self._views[row]

        def text(column: int) -> str:
            item = self.table.item(row, column)
            return item.text().strip() if item is not None else ""

        return ViewSpec(
            id=previous.id,
            name=text(1) or previous.name,
            enabled=self.table.item(row, 0).checkState() == Qt.Checked,
            yaw_deg=float(text(2)),
            pitch_deg=float(text(3)),
            roll_deg=float(text(4)),
            hfov_deg=float(text(5)),
            vfov_deg=float(text(6)) if text(6) else None,
            width=int(text(7)),
            height=int(text(8)),
            interpolation=previous.interpolation,
        )

    def _on_cell_changed(self, row: int, _column: int) -> None:
        if self._updating_table or not 0 <= row < len(self._views):
            return
        try:
            self._views[row] = self._view_from_row(row)
        except ValueError as exc:
            self.estimate_label.setText(f"Invalid view: {exc}")
            return
        self.globe.set_views(self._views, row)
        self._update_estimate()
        self.primary_action_state_changed.emit()

    def _on_current_cell_changed(self, row: int, _column: int, _previous_row: int, _previous_column: int) -> None:
        if row >= 0:
            self.globe.set_views(self._views, row)
            self._refresh_preview()

    def _select_row(self, index: int) -> None:
        if 0 <= index < len(self._views):
            self.table.selectRow(index)

    def _adopt_globe_views(self) -> None:
        self._views = self.globe.views()
        self._populate_table(self.globe.selected_index())

    def _apply_cubemap(self) -> None:
        self._views = cubemap_view_specs(1600)
        self._populate_table(0)

    def _apply_ring(self) -> None:
        self._views = grid_view_specs(yaw_count=8, pitches=(0.0,), hfov_deg=100.0, size=1600)
        self._populate_table(0)

    def _add_view(self) -> None:
        index = len(self._views) + 1
        self._views.append(ViewSpec(id=f"view_{index:02d}", name=f"View {index}", yaw_deg=(index - 1) * 45.0))
        self._populate_table(len(self._views) - 1)

    def _remove_view(self) -> None:
        row = self.table.currentRow()
        if len(self._views) <= 1 or row < 0:
            return
        self._views.pop(row)
        self._populate_table(min(row, len(self._views) - 1))

    def _update_options(self) -> None:
        video = self.format_combo.currentData() == "video"
        self.interval_spin.setEnabled(not video)
        self.rig_check.setEnabled(not video)
        if video:
            self.rig_check.setChecked(False)
        self._update_estimate()

    def _update_estimate(self) -> None:
        enabled = [view for view in self._views if view.enabled]
        batch = self.batch_spin.value() or estimate_batch_size(enabled)
        batches = (len(enabled) + max(1, batch) - 1) // max(1, batch)
        megapixels = sum(view.width * view.height for view in enabled) / 1_000_000.0
        output_format = str(self.format_combo.currentData())
        if output_format == "video":
            source_frames = self._source_frame_count
            output_count = len(enabled)
            bytes_per_pixel = 0.10
        else:
            source_frames = (
                max(1, round(self._source_duration_sec / self.interval_spin.value()))
                if self._source_duration_sec > 0
                else self._source_frame_count
            )
            output_count = source_frames * len(enabled)
            bytes_per_pixel = 0.45 if output_format == "jpeg" else 1.8
        storage_bytes = megapixels * 1_000_000 * max(0, source_frames) * bytes_per_pixel
        processing_seconds = megapixels * max(0, source_frames) / 120.0
        storage_text = self._format_bytes(storage_bytes) if source_frames else "unknown storage"
        time_text = self._format_seconds(processing_seconds) if source_frames else "unknown time"
        self.estimate_label.setText(
            f"{len(enabled)} enabled view(s) · {output_count or 'unknown'} output item(s) · about {storage_text} · "
            f"about {time_text} · {megapixels:.1f} MP/source frame · {batches} shared decode batch(es). "
            "Right-click a globe marker to enable or disable it; drag empty globe space to orbit."
        )

    @staticmethod
    def _format_bytes(value: float) -> str:
        units = ("B", "KB", "MB", "GB", "TB")
        size = max(0.0, value)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return "unknown storage"

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0, round(value))
        if seconds < 60:
            return f"{seconds} sec"
        if seconds < 3600:
            return f"{seconds // 60} min {seconds % 60:02d} sec"
        return f"{seconds // 3600} hr {(seconds % 3600) // 60:02d} min"

    def _probe_source_count(self, path: Path) -> None:
        self._source_duration_sec = 0.0
        self._source_frame_count = 0
        if path.is_dir():
            suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
            self._source_frame_count = sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)
            return
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
            self._source_frame_count = 1
            return
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = max(0, round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
            self._source_frame_count = frames
            self._source_duration_sec = frames / fps if fps > 0 else 0.0
        finally:
            capture.release()

    def _on_input_changed(self, value: str) -> None:
        if value and not self.output_browse.text():
            source = Path(value)
            self.output_browse.set_text(str(source.parent / f"{source.stem}_perspective"))
        self._preview_source = None
        if value:
            try:
                self._probe_source_count(Path(value))
            except OSError:
                self._source_duration_sec = 0.0
                self._source_frame_count = 0
        else:
            self._source_duration_sec = 0.0
            self._source_frame_count = 0
        self._update_estimate()
        self.primary_action_state_changed.emit()

    def _load_preview_source(self):
        path = Path(self.input_browse.text())
        if not path.is_file():
            return None
        image = imread_unicode(path, cv2.IMREAD_COLOR)
        if image is None:
            capture = cv2.VideoCapture(str(path))
            try:
                ok, image = capture.read()
                if not ok:
                    return None
            finally:
                capture.release()
        if image.shape[1] > 2048:
            scale = 2048.0 / image.shape[1]
            image = cv2.resize(image, (2048, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        return image

    def _refresh_preview(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._views):
            return
        if self._preview_source is None:
            self._preview_source = self._load_preview_source()
        if self._preview_source is None:
            self.preview_label.setText("Preview unavailable for this source.")
            return
        view = self._views[row]
        max_width, max_height = 900, 520
        scale = min(max_width / view.width, max_height / view.height, 1.0)
        preview_view = replace(view, width=max(16, round(view.width * scale)), height=max(16, round(view.height * scale)))
        projected = project_equirectangular(self._preview_source, preview_view, cache=self._preview_cache)
        rgb = cv2.cvtColor(projected, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self.preview_label.setPixmap(QPixmap.fromImage(image))

    def _profile_payload(self) -> dict:
        return {
            "schema_version": 1,
            "input_path": self.input_browse.text(),
            "output_dir": self.output_browse.text(),
            "output_format": self.format_combo.currentData(),
            "frame_interval_sec": self.interval_spin.value(),
            "batch_size": self.batch_spin.value(),
            "use_nvenc": self.nvenc_check.isChecked(),
            "colmap_rig": self.rig_check.isChecked(),
            "overwrite": self.overwrite_check.isChecked(),
            "views": [view.to_dict() for view in self._views],
        }

    def _save_profile(self) -> None:
        start = str(Path(self.scene_dir) / "_360gs" / "profiles") if self.scene_dir else ""
        path, _ = dialogs.get_save_file_name(self, "Save view profile", start, "JSON (*.json)")
        if path:
            atomic_write_json(Path(path), self._profile_payload())

    def _load_profile(self) -> None:
        start = str(Path(self.scene_dir) / "_360gs" / "profiles") if self.scene_dir else ""
        path, _ = dialogs.get_open_file_name(self, "Load view profile", start, "JSON (*.json)")
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) != 1 or not isinstance(payload.get("views"), list):
                raise ValueError("unsupported profile schema")
            self._views = [ViewSpec.from_dict(item, index=index) for index, item in enumerate(payload["views"])]
            self.input_browse.set_text(str(payload.get("input_path") or ""))
            self.output_browse.set_text(str(payload.get("output_dir") or ""))
            index = self.format_combo.findData(str(payload.get("output_format") or "png"))
            if index >= 0:
                self.format_combo.setCurrentIndex(index)
            self.interval_spin.setValue(float(payload.get("frame_interval_sec") or 1.0))
            self.batch_spin.setValue(int(payload.get("batch_size") or 0))
            self.nvenc_check.setChecked(bool(payload.get("use_nvenc", False)))
            self.rig_check.setChecked(bool(payload.get("colmap_rig", False)))
            self.overwrite_check.setChecked(bool(payload.get("overwrite", False)))
            self._populate_table(0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Invalid profile", str(exc))
