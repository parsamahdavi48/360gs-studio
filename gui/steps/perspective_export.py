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
from gs360studio.platform.job_store import list_jobs, recover_interrupted_jobs
from gs360studio.platform.project_store import migrate_legacy_project
from gui import i18n
from gui.common import dialogs
from gui.common.browse_widget import BrowseWidget
from gui.common.icons import folder_icon
from gui.common.runner_types import StepCommandQueue
from gui.perspective.globe_widget import ViewGlobeWidget
from gui.steps.base_step import BaseStepWidget

_PROGRESS_RE = re.compile(r"^PROGRESS:(\d+)/(\d+)$")
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})
_COLUMNS = tuple(
    i18n.t(key)
    for key in (
        "PERSPECTIVE_COLUMN_ON",
        "PERSPECTIVE_COLUMN_NAME",
        "PERSPECTIVE_COLUMN_YAW",
        "PERSPECTIVE_COLUMN_PITCH",
        "PERSPECTIVE_COLUMN_ROLL",
        "PERSPECTIVE_COLUMN_HFOV",
        "PERSPECTIVE_COLUMN_VFOV",
        "PERSPECTIVE_COLUMN_WIDTH",
        "PERSPECTIVE_COLUMN_HEIGHT",
        "PERSPECTIVE_COLUMN_INTERPOLATION",
    )
)


class PerspectiveExportStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._views = cubemap_view_specs(1600)
        self._updating_table = False
        self._preview_source = None
        self._preview_cache = ProjectionMapCache(8)
        self._source_duration_sec = 0.0
        self._source_frame_count = 0
        self._auto_input_path = ""
        self._auto_output_path = ""
        self._resume_job_id = ""
        self._recoverable_job = None
        self._build_ui()
        self._populate_table()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        paths = QWidget()
        form = QFormLayout(paths)
        self.input_browse = BrowseWidget(mode="file", filter_str="Media (*.mp4 *.mov *.mkv *.avi *.png *.jpg *.jpeg *.tif *.tiff *.webp);;All files (*.*)")
        self.input_folder_button = self.input_browse.add_icon_button(
            folder_icon(),
            i18n.t("PERSPECTIVE_CHOOSE_IMAGE_FOLDER"),
            self._choose_input_folder,
            accessible_name=i18n.t("PERSPECTIVE_CHOOSE_IMAGE_FOLDER"),
        )
        self.output_browse = BrowseWidget(mode="dir")
        form.addRow(i18n.t("PERSPECTIVE_INPUT_MEDIA"), self.input_browse)
        form.addRow(i18n.t("PERSPECTIVE_OUTPUT_FOLDER"), self.output_browse)
        root.addWidget(paths)

        options = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItem(i18n.t("PERSPECTIVE_FORMAT_PNG"), "png")
        self.format_combo.addItem(i18n.t("PERSPECTIVE_FORMAT_JPEG"), "jpeg")
        self.format_combo.addItem(i18n.t("PERSPECTIVE_FORMAT_HEVC"), "video")
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.01, 3600.0)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" s")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(0, 32)
        self.batch_spin.setSpecialValueText(i18n.t("AUTO"))
        self.nvenc_check = QCheckBox("NVENC")
        self.rig_check = QCheckBox(i18n.t("PERSPECTIVE_COLMAP_RIG"))
        self.overwrite_check = QCheckBox(i18n.t("PERSPECTIVE_REPLACE_EXPORT"))
        for label, widget in (
            (i18n.t("PERSPECTIVE_FORMAT"), self.format_combo),
            (i18n.t("PERSPECTIVE_INTERVAL"), self.interval_spin),
            (i18n.t("PERSPECTIVE_BATCH"), self.batch_spin),
        ):
            options.addWidget(QLabel(label))
            options.addWidget(widget)
        options.addWidget(self.nvenc_check)
        options.addWidget(self.rig_check)
        options.addWidget(self.overwrite_check)
        options.addStretch()
        root.addLayout(options)

        preset_row = QHBoxLayout()
        for label, callback in (
            (i18n.t("PERSPECTIVE_CUBEMAP"), self._apply_cubemap),
            (i18n.t("PERSPECTIVE_RING"), self._apply_ring),
            (i18n.t("PERSPECTIVE_ADD_VIEW"), self._add_view),
            (i18n.t("PERSPECTIVE_REMOVE_VIEW"), self._remove_view),
            (i18n.t("PERSPECTIVE_SAVE_PROFILE"), self._save_profile),
            (i18n.t("PERSPECTIVE_LOAD_PROFILE"), self._load_profile),
            (i18n.t("PERSPECTIVE_REFRESH_PREVIEW"), self._refresh_preview),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            preset_row.addWidget(button)
        preset_row.addStretch()
        self.restore_job_button = QPushButton(i18n.t("PERSPECTIVE_RESTORE_JOB"))
        self.restore_job_button.setVisible(False)
        self.restore_job_button.clicked.connect(self._restore_interrupted_job)
        preset_row.addWidget(self.restore_job_button)
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
        self.preview_label = QLabel(i18n.t("PERSPECTIVE_PREVIEW_PROMPT"))
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
        self._sync_scene_state()

    def on_activated(self) -> None:
        self._sync_scene_state()

    def _sync_scene_state(self) -> None:
        previous_auto_input = self._auto_input_path
        previous_auto_output = self._auto_output_path
        path = self.scene_dir
        if not path:
            self._recoverable_job = None
            self.restore_job_button.setVisible(False)
            return
        scene = Path(path)
        output = str(scene / "output" / "perspective")
        if not self.output_browse.text() or self.output_browse.text() == previous_auto_output:
            self._auto_output_path = output
            self.output_browse.set_text(output)
        try:
            manifest, report = migrate_legacy_project(scene)
            recovered = recover_interrupted_jobs(scene)
            candidate = self._default_project_source(scene, manifest.sources)
            if candidate and (not self.input_browse.text() or self.input_browse.text() == previous_auto_input):
                self._auto_input_path = str(candidate)
                self.input_browse.set_text(self._auto_input_path)
            jobs = list_jobs(scene)
            resolved_job_ids = {
                dependency_id
                for job in jobs
                if job.job_type == "perspective-export" and job.status == "completed"
                for dependency_id in job.dependency_ids
            }
            interrupted = [
                job
                for job in jobs
                if job.job_type == "perspective-export"
                and job.status == "interrupted"
                and job.job_id not in resolved_job_ids
            ]
            self._recoverable_job = max(interrupted, key=lambda job: job.created_at) if interrupted else None
            self.restore_job_button.setVisible(self._recoverable_job is not None)
            if report.migrated or recovered:
                self.estimate_label.setText(
                    i18n.t("PERSPECTIVE_PROJECT_READY").format(
                        migrated=report.migrated,
                        recovered=len(recovered),
                    )
                )
        except (OSError, ValueError) as exc:
            self.estimate_label.setText(i18n.t("PERSPECTIVE_PROJECT_WARNING").format(error=exc))

    def primary_action_text(self) -> str:
        return i18n.t("PERSPECTIVE_EXPORT_ACTION")

    def primary_action_tooltip(self) -> str:
        return i18n.t("PERSPECTIVE_EXPORT_TOOLTIP")

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
        if self._resume_job_id:
            payload["resume_job_id"] = self._resume_job_id
        return [("perspective_export", perspective_app_job(payload))]

    def on_line(self, line: str) -> tuple[int, int] | None:
        match = _PROGRESS_RE.match(line.strip())
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def phase_display_name(self, _phase: str) -> str:
        return i18n.t("PERSPECTIVE_TITLE")

    def on_queue_finished(self, success: bool) -> None:
        if success:
            self._resume_job_id = ""
            self._sync_scene_state()

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
                    QTableWidgetItem(view.interpolation),
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
            interpolation=text(9) or previous.interpolation,
        )

    def _on_cell_changed(self, row: int, _column: int) -> None:
        if self._updating_table or not 0 <= row < len(self._views):
            return
        try:
            self._views[row] = self._view_from_row(row)
        except ValueError as exc:
            self.estimate_label.setText(i18n.t("PERSPECTIVE_INVALID_VIEW").format(error=exc))
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
        self._views.append(
            ViewSpec(
                id=f"view_{index:02d}",
                name=i18n.t("PERSPECTIVE_VIEW_NAME").format(index=index),
                yaw_deg=(index - 1) * 45.0,
            )
        )
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
        storage_text = self._format_bytes(storage_bytes) if source_frames else i18n.t("PERSPECTIVE_UNKNOWN_STORAGE")
        time_text = self._format_seconds(processing_seconds) if source_frames else i18n.t("PERSPECTIVE_UNKNOWN_TIME")
        self.estimate_label.setText(
            i18n.t("PERSPECTIVE_ESTIMATE").format(
                views=len(enabled),
                outputs=output_count or i18n.t("PERSPECTIVE_UNKNOWN"),
                storage=storage_text,
                time=time_text,
                megapixels=f"{megapixels:.1f}",
                batches=batches,
            )
        )

    @staticmethod
    def _format_bytes(value: float) -> str:
        units = ("B", "KB", "MB", "GB", "TB")
        size = max(0.0, value)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return i18n.t("PERSPECTIVE_UNKNOWN_STORAGE")

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
            self._source_frame_count = sum(
                1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
            )
            return
        if path.suffix.lower() in _IMAGE_SUFFIXES:
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
        if value:
            self._refresh_preview()

    def _load_preview_source(self):
        path = Path(self.input_browse.text())
        if path.is_dir():
            path = next(
                (item for item in sorted(path.iterdir()) if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES),
                Path(),
            )
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
            self.preview_label.clear()
            self.preview_label.setText(i18n.t("PERSPECTIVE_PREVIEW_UNAVAILABLE"))
            return
        view = self._views[row]
        max_width, max_height = 900, 520
        scale = min(max_width / view.width, max_height / view.height, 1.0)
        preview_view = replace(view, width=max(16, round(view.width * scale)), height=max(16, round(view.height * scale)))
        projected = project_equirectangular(self._preview_source, preview_view, cache=self._preview_cache)
        rgb = cv2.cvtColor(projected, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self.preview_label.setPixmap(QPixmap.fromImage(image))

    @staticmethod
    def _default_project_source(scene: Path, sources: list[dict]) -> Path | None:
        images = scene / "images"
        if images.is_dir() and any(
            item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES for item in images.iterdir()
        ):
            return images
        for source in sources:
            value = str(source.get("path") or "").strip()
            if not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = scene / candidate
            if candidate.exists():
                return candidate
        return None

    def _choose_input_folder(self) -> None:
        start = self.input_browse.text() or self.scene_dir
        path = dialogs.get_existing_directory(self, i18n.t("PERSPECTIVE_CHOOSE_IMAGE_FOLDER"), start)
        if path:
            self._auto_input_path = ""
            self.input_browse.set_text(path)

    def _restore_interrupted_job(self) -> None:
        job = self._recoverable_job
        if job is None:
            return
        try:
            self._apply_configuration(job.configuration)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, i18n.t("PERSPECTIVE_INVALID_PROFILE"), str(exc))
            return
        self._resume_job_id = job.job_id
        self.restore_job_button.setVisible(False)
        self.estimate_label.setText(i18n.t("PERSPECTIVE_JOB_RESTORED"))
        self.primary_action_state_changed.emit()

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
        path, _ = dialogs.get_save_file_name(
            self,
            i18n.t("PERSPECTIVE_SAVE_PROFILE"),
            start,
            "JSON (*.json)",
        )
        if path:
            atomic_write_json(Path(path), self._profile_payload())

    def _load_profile(self) -> None:
        start = str(Path(self.scene_dir) / "_360gs" / "profiles") if self.scene_dir else ""
        path, _ = dialogs.get_open_file_name(
            self,
            i18n.t("PERSPECTIVE_LOAD_PROFILE"),
            start,
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) != 1 or not isinstance(payload.get("views"), list):
                raise ValueError("unsupported profile schema")
            self._apply_configuration(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, i18n.t("PERSPECTIVE_INVALID_PROFILE"), str(exc))

    def _apply_configuration(self, payload: dict) -> None:
        views = payload.get("views")
        if not isinstance(views, list):
            raise ValueError("configuration requires a views list")
        self._resume_job_id = ""
        self._views = [ViewSpec.from_dict(item, index=index) for index, item in enumerate(views)]
        self._auto_input_path = ""
        self._auto_output_path = ""
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
