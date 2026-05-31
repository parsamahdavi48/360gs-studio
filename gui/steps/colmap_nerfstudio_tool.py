"""COLMAP sparse result to Nerfstudio JSON/PLY dataset tool."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.app_job import dataset_app_job
from core.colmap_sparse_model import resolve_model_dir
from core.dataset_job_spec import colmap_nerfstudio_job, write_dataset_job
from core.scene_layout import jobs_dir, scene_output_dir
from core.workflow_artifacts import (
    DATASET_KIND_NERF_JSON_PLY,
    SFM_KIND_COLMAP_SPARSE,
    register_dataset_artifact,
    register_sfm_artifact,
)
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.form_rows import add_tooltip_row
from gui.common.runner_types import StepCommandQueue
from gui.steps.base_step import BaseStepWidget
from gui.steps.output_reset import clear_path, path_has_contents

_PROGRESS_RE = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)")


class ColmapNerfstudioTool(BaseStepWidget):
    """Create a Nerfstudio dataset from this app's COLMAP SfM output."""

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._syncing_defaults = False
        self._source_user_edited = False
        self._output_user_edited = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(7)

        self.source_browse = BrowseWidget(
            mode="dir",
            placeholder=i18n.t("COLMAP_NERF_SOURCE_PLACEHOLDER"),
        )
        self.source_browse.setToolTip(i18n.tip("COLMAP_NERF_SOURCE"))
        self.source_browse.path_changed.connect(self._on_source_path_changed)
        add_tooltip_row(form, i18n.t("COLMAP_NERF_SOURCE"), self.source_browse, i18n.tip("COLMAP_NERF_SOURCE"))

        self.output_browse = BrowseWidget(
            mode="dir",
            placeholder=i18n.t("COLMAP_NERF_OUTPUT_PLACEHOLDER"),
        )
        self.output_browse.setToolTip(i18n.tip("COLMAP_NERF_OUTPUT"))
        self.output_browse.path_changed.connect(self._on_output_path_changed)
        add_tooltip_row(form, i18n.t("COLMAP_NERF_OUTPUT"), self.output_browse, i18n.tip("COLMAP_NERF_OUTPUT"))

        layout.addLayout(form)
        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

    def set_scene_dir(self, path: str) -> None:
        previous_scene = self.scene_dir
        super().set_scene_dir(path)
        if path != previous_scene:
            self._source_user_edited = False
            self._output_user_edited = False
        self._apply_scene_defaults()

    def on_activated(self) -> None:
        self._apply_scene_defaults()

    def primary_action_text(self) -> str:
        return i18n.t("DATASET_RUN_COLMAP_NERF")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("DATASET_RUN_COLMAP_NERF")

    def primary_action_enabled(self) -> bool:
        try:
            self._validate_inputs()
        except ValueError:
            return False
        return True

    def build_commands(self) -> StepCommandQueue:
        self._validate_inputs()
        output = self._output_dir()
        if not self._prepare_output_dir(output):
            return []

        scene = Path(self.scene_dir)
        source = self._source_dir()
        job_path = jobs_dir(scene) / "colmap_nerfstudio_job.json"
        payload = colmap_nerfstudio_job(
            scene_dir=scene,
            colmap_root=source,
            output_dir=output,
            require_complete_masks=True,
        )
        write_dataset_job(job_path, payload)
        return [("colmap_nerfstudio", dataset_app_job(payload, job_path))]

    def phase_display_name(self, phase: str) -> str:
        if phase == "colmap_nerfstudio":
            return i18n.t("PHASE_COLMAP_NERFSTUDIO")
        return super().phase_display_name(phase)

    def on_line(self, line: str) -> tuple[int, int] | None:
        progress = _PROGRESS_RE.match(line)
        if progress:
            return int(progress.group(1)), int(progress.group(2))
        return None

    def on_queue_finished(self, success: bool) -> None:
        if not success or not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        source = self._source_dir()
        output = self._output_dir()
        sparse_dir = self._resolved_sparse_dir(source)
        settings = {
            "coordinate_contract": "nerfstudio_colmap_to_json_default",
            "require_complete_masks": True,
        }
        sfm_record = register_sfm_artifact(
            scene,
            artifact_id=self._artifact_id("colmap_sfm"),
            kind=SFM_KIND_COLMAP_SPARSE,
            root=source,
            files={
                "images_dir": source / "images",
                "masks_dir": source / "masks",
                "colmap_sparse_dir": sparse_dir,
            },
            source_inputs=[source],
            settings=settings,
            metadata={"source_kind": "colmap_sparse"},
        )
        register_dataset_artifact(
            scene,
            artifact_id=self._artifact_id("colmap_nerfstudio"),
            root=output,
            kind=DATASET_KIND_NERF_JSON_PLY,
            source_artifact_id=sfm_record.id,
            source_inputs=[output],
            settings=settings,
            metadata={"source_kind": "colmap_sparse"},
        )

    def _validate_inputs(self) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        source = self._source_dir()
        if not source.is_dir():
            raise ValueError(i18n.t("COLMAP_NERF_SOURCE_NOT_FOUND").format(path=source))
        images = source / "images"
        if not images.is_dir():
            raise ValueError(i18n.t("COLMAP_NERF_IMAGES_NOT_FOUND").format(path=images))
        try:
            self._resolved_sparse_dir(source)
        except FileNotFoundError as exc:
            raise ValueError(i18n.t("COLMAP_NERF_SPARSE_NOT_FOUND").format(path=source / "sparse")) from exc
        output = self._output_dir()
        if not str(output):
            raise ValueError(i18n.t("COLMAP_NERF_OUTPUT_REQUIRED"))
        output_root = scene_output_dir(Path(self.scene_dir)).resolve()
        try:
            resolved_output = output.resolve()
        except OSError:
            resolved_output = output.absolute()
        if resolved_output == output_root:
            raise ValueError(i18n.t("COLMAP_NERF_OUTPUT_ROOT_ERROR").format(path=output))
        try:
            resolved_output.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(i18n.t("COLMAP_NERF_OUTPUT_OUTSIDE_ERROR").format(path=output, root=output_root)) from exc
        if self._paths_overlap(source, output):
            raise ValueError(i18n.t("COLMAP_NERF_OUTPUT_OVERLAPS_SOURCE").format(output=output, source=source))

    def _prepare_output_dir(self, output: Path) -> bool:
        if self.scene_dir is None:
            return False
        output_root = scene_output_dir(Path(self.scene_dir))
        if path_has_contents(output):
            result = QMessageBox.question(
                self,
                i18n.t("OUTPUT_RESET_TITLE"),
                i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(output)),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return False
            clear_path(output, allowed_roots=[output_root])
        output.mkdir(parents=True, exist_ok=True)
        return True

    def _apply_scene_defaults(self) -> None:
        self._syncing_defaults = True
        try:
            if not self.scene_dir:
                self._set_browse_text(self.source_browse, "")
                self._set_browse_text(self.output_browse, "")
                return
            scene = Path(self.scene_dir)
            if not self._source_user_edited:
                self._set_browse_text(self.source_browse, str(scene_output_dir(scene) / "colmap_rig"))
            if not self._output_user_edited:
                self._set_browse_text(self.output_browse, str(scene_output_dir(scene) / "colmap_nerfstudio"))
        finally:
            self._syncing_defaults = False
        self.primary_action_state_changed.emit()

    def _on_source_path_changed(self, _path: str) -> None:
        if not self._syncing_defaults:
            self._source_user_edited = True
        self.primary_action_state_changed.emit()

    def _on_output_path_changed(self, _path: str) -> None:
        if not self._syncing_defaults:
            self._output_user_edited = True
        self.primary_action_state_changed.emit()

    def _source_dir(self) -> Path:
        text = self.source_browse.text()
        return Path(text) if text else Path()

    def _output_dir(self) -> Path:
        text = self.output_browse.text()
        return Path(text) if text else Path()

    @staticmethod
    def _resolved_sparse_dir(source: Path) -> Path:
        for candidate in (source / "sparse" / "0", source / "sparse", source):
            try:
                return resolve_model_dir(candidate)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"No COLMAP sparse model found under: {source}")

    @staticmethod
    def _set_browse_text(browse: BrowseWidget, text: str) -> None:
        blocker = QSignalBlocker(browse.line_edit)
        try:
            browse.set_text(text)
        finally:
            del blocker

    @staticmethod
    def _paths_overlap(left: Path, right: Path) -> bool:
        try:
            left_resolved = left.resolve()
            right_resolved = right.resolve()
        except OSError:
            left_resolved = left.absolute()
            right_resolved = right.absolute()
        return (
            left_resolved == right_resolved
            or _is_relative_to(left_resolved, right_resolved)
            or _is_relative_to(right_resolved, left_resolved)
        )

    @staticmethod
    def _artifact_id(prefix: str) -> str:
        from core.scene_project import utc_now_iso

        return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '')}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
