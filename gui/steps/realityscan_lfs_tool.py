"""RealityScan CSV/PLY to LichtFeld COLMAP dataset GUI tool."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.app_job import dataset_app_job
from core.dataset_job_spec import realityscan_lfs_colmap_job, write_dataset_job
from core.realityscan_to_lfs_colmap import DEFAULT_UNDISTORT_ALPHA
from core.realityscan_to_transforms import (
    REALITYSCAN_IMAGE_DIR_NAMES,
    REALITYSCAN_MASK_DIR_NAMES,
    read_realityscan_csv,
    related_realityscan_asset_roots,
)
from core.scene_layout import jobs_dir, scene_images_dir, scene_masks_dir, scene_output_dir
from core.scene_preview_diagnostics import analyze_named_camera_images
from core.workflow_artifacts import (
    DATASET_KIND_LICHTFELD_COLMAP,
    SFM_KIND_REALITYSCAN_CSV_PLY,
    register_dataset_artifact,
    register_sfm_artifact,
)
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.form_rows import add_tooltip_row
from gui.common.runner_types import StepCommandQueue
from gui.steps.base_step import BaseStepWidget

_PROGRESS_RE = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)")


class RealityScanLfsTool(BaseStepWidget):
    """Build a LichtFeld-compatible COLMAP dataset from RealityScan exports."""

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._syncing_defaults = False
        self._csv_user_edited = False
        self._ply_user_edited = False
        self._images_user_edited = False
        self._masks_user_edited = False
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

        self.summary_note = QLabel()
        self.summary_note.setObjectName("workflowNote")
        self.summary_note.setWordWrap(True)
        layout.addWidget(self.summary_note)

        form = QFormLayout()
        form.setSpacing(7)

        csv_filter = "CSV (*.csv);;All (*.*)"
        ply_filter = "PLY (*.ply);;All (*.*)"
        self.csv_browse = BrowseWidget(mode="file", filter_str=csv_filter)
        self.csv_browse.setToolTip(i18n.tip("RS_LFS_CSV"))
        add_tooltip_row(form, i18n.t("RS_LFS_CSV"), self.csv_browse, i18n.tip("RS_LFS_CSV"))

        self.ply_browse = BrowseWidget(mode="file", filter_str=ply_filter)
        self.ply_browse.setToolTip(i18n.tip("RS_LFS_PLY"))
        add_tooltip_row(form, i18n.t("RS_LFS_PLY"), self.ply_browse, i18n.tip("RS_LFS_PLY"))

        self.images_browse = BrowseWidget(mode="dir")
        self.images_browse.setToolTip(i18n.tip("RS_LFS_IMAGES"))
        self.images_extra_hint = self._build_extra_folder_hint()
        self.images_field = self._build_asset_field(self.images_browse, self.images_extra_hint)
        add_tooltip_row(form, i18n.t("RS_LFS_IMAGES"), self.images_field, i18n.tip("RS_LFS_IMAGES"))

        self.masks_browse = BrowseWidget(mode="dir")
        self.masks_browse.setToolTip(i18n.tip("RS_LFS_MASKS"))
        self.masks_extra_hint = self._build_extra_folder_hint()
        self.masks_field = self._build_asset_field(self.masks_browse, self.masks_extra_hint)
        add_tooltip_row(form, i18n.t("RS_LFS_MASKS"), self.masks_field, i18n.tip("RS_LFS_MASKS"))

        self.output_browse = BrowseWidget(mode="dir")
        self.output_browse.setToolTip(i18n.tip("RS_LFS_OUTPUT"))
        self.output_browse.path_changed.connect(self._on_output_path_changed)
        add_tooltip_row(form, i18n.t("RS_LFS_OUTPUT"), self.output_browse, i18n.tip("RS_LFS_OUTPUT"))

        self.pre_undistort_cb = QCheckBox(i18n.t("RS_LFS_PRE_UNDISTORT"))
        self.pre_undistort_cb.setToolTip(i18n.tip("RS_LFS_PRE_UNDISTORT"))
        self.pre_undistort_cb.toggled.connect(self._on_pre_undistort_toggled)
        add_tooltip_row(
            form,
            "",
            self.pre_undistort_cb,
            i18n.tip("RS_LFS_PRE_UNDISTORT"),
        )

        self.skip_missing_cb = QCheckBox(i18n.t("RS_LFS_SKIP_MISSING"))
        self.skip_missing_cb.setToolTip(i18n.tip("RS_LFS_SKIP_MISSING"))
        add_tooltip_row(form, "", self.skip_missing_cb, i18n.tip("RS_LFS_SKIP_MISSING"))

        layout.addLayout(form)
        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

        self.csv_browse.path_changed.connect(lambda _path: self._on_input_path_changed("csv"))
        self.ply_browse.path_changed.connect(lambda _path: self._on_input_path_changed("ply"))
        self.images_browse.path_changed.connect(lambda _path: self._on_input_path_changed("images"))
        self.masks_browse.path_changed.connect(lambda _path: self._on_input_path_changed("masks"))
        self.output_browse.path_changed.connect(lambda _path: self.primary_action_state_changed.emit())
        self.pre_undistort_cb.toggled.connect(lambda _checked: self.primary_action_state_changed.emit())
        self.skip_missing_cb.toggled.connect(lambda _checked: self.primary_action_state_changed.emit())
        self._update_extra_folder_hints()
        self._update_data_quality_note()

    def set_scene_dir(self, path: str) -> None:
        previous_scene = self.scene_dir
        super().set_scene_dir(path)
        if path != previous_scene:
            self._csv_user_edited = False
            self._ply_user_edited = False
            self._images_user_edited = False
            self._masks_user_edited = False
            self._output_user_edited = False
        self._apply_scene_defaults()

    def on_activated(self) -> None:
        self._apply_scene_defaults()

    def primary_action_text(self) -> str:
        return i18n.t("RS_LFS_CREATE")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("RS_LFS_CREATE")

    def primary_action_enabled(self) -> bool:
        try:
            self._validate_inputs()
        except ValueError:
            return False
        return True

    def phase_display_name(self, phase: str) -> str:
        return i18n.t("PHASE_RS_LFS_COLMAP") if phase == "realityscan_lfs_colmap" else phase

    def on_line(self, line: str) -> tuple[int, int] | None:
        progress = _PROGRESS_RE.match(line)
        if progress:
            return int(progress.group(1)), int(progress.group(2))
        return None

    def build_commands(self) -> StepCommandQueue:
        self._validate_inputs()
        job_path = jobs_dir(Path(self.scene_dir)) / "realityscan_lfs_colmap_job.json"
        payload = realityscan_lfs_colmap_job(
            csv_path=self.csv_browse.text(),
            output_dir=self.output_browse.text(),
            images_dir=self.images_browse.text(),
            masks_dir=self.masks_browse.text() or None,
            ply_path=self.ply_browse.text(),
            skip_missing_images=self.skip_missing_cb.isChecked(),
            pre_undistort_distorted_images=self.pre_undistort_cb.isChecked(),
            undistort_alpha=DEFAULT_UNDISTORT_ALPHA,
        )
        write_dataset_job(job_path, payload)

        return [("realityscan_lfs_colmap", dataset_app_job(payload, job_path))]

    def on_queue_finished(self, success: bool) -> None:
        if not success or not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        settings = {
            "pre_undistort_distorted_images": self.pre_undistort_cb.isChecked(),
            "skip_missing_images": self.skip_missing_cb.isChecked(),
        }
        sfm_record = register_sfm_artifact(
            scene,
            artifact_id=self._artifact_id("rs_sfm"),
            kind=SFM_KIND_REALITYSCAN_CSV_PLY,
            root=Path(self.csv_browse.text()).parent,
            files={
                "csv": self.csv_browse.text(),
                "ply": self.ply_browse.text(),
                "images_dir": self.images_browse.text(),
                "masks_dir": self.masks_browse.text(),
            },
            source_inputs=[
                self.csv_browse.text(),
                self.ply_browse.text(),
                self.images_browse.text(),
                self.masks_browse.text(),
            ],
            settings=settings,
        )
        register_dataset_artifact(
            scene,
            artifact_id=self._artifact_id("rs_lfs_colmap"),
            root=self.output_browse.text(),
            kind=DATASET_KIND_LICHTFELD_COLMAP,
            source_artifact_id=sfm_record.id,
            source_inputs=[self.output_browse.text()],
            settings=settings,
        )

    def _validate_inputs(self) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        csv = Path(self.csv_browse.text())
        if not csv.is_file():
            raise ValueError(i18n.t("RS_LFS_CSV_NOT_FOUND").format(path=self.csv_browse.text() or "-"))
        ply = Path(self.ply_browse.text())
        if not ply.is_file():
            raise ValueError(i18n.t("RS_LFS_PLY_NOT_FOUND").format(path=self.ply_browse.text() or "-"))
        images = Path(self.images_browse.text())
        if not images.is_dir():
            raise ValueError(i18n.t("RS_LFS_IMAGES_NOT_FOUND").format(path=self.images_browse.text() or "-"))
        masks_text = self.masks_browse.text()
        if masks_text and not Path(masks_text).is_dir():
            raise ValueError(i18n.t("RS_LFS_MASKS_NOT_FOUND").format(path=masks_text))
        output = self.output_browse.text()
        if not output:
            raise ValueError(i18n.t("RS_LFS_OUTPUT_REQUIRED"))

    def _on_output_path_changed(self, _path: str) -> None:
        if not self._syncing_defaults:
            self._output_user_edited = True

    def _on_input_path_changed(self, field: str) -> None:
        if not self._syncing_defaults:
            if field == "csv":
                self._csv_user_edited = True
            elif field == "ply":
                self._ply_user_edited = True
            elif field == "images":
                self._images_user_edited = True
            elif field == "masks":
                self._masks_user_edited = True
        self.primary_action_state_changed.emit()
        self._update_extra_folder_hints()
        self._update_data_quality_note()

    def _on_pre_undistort_toggled(self, _checked: bool) -> None:
        if not self._output_user_edited:
            self._set_browse_text(self.output_browse, str(self._default_output_dir()))

    def _apply_scene_defaults(self) -> None:
        if not self.scene_dir:
            self._syncing_defaults = True
            try:
                self.csv_browse.set_text("")
                self.ply_browse.set_text("")
                self.images_browse.set_text("")
                self.masks_browse.set_text("")
                self.output_browse.set_text("")
            finally:
                self._syncing_defaults = False
            self._update_extra_folder_hints()
            self._update_data_quality_note()
            return
        root = self._realityscan_dir()
        self._syncing_defaults = True
        try:
            if not self._csv_user_edited:
                csv = self._first_existing(root, ("rs_*.csv", "*.csv"))
                self.csv_browse.set_text(str(csv) if csv else "")
            if not self._ply_user_edited:
                ply = self._first_existing(root, ("rs_*.ply", "*.ply"))
                self.ply_browse.set_text(str(ply) if ply else "")
            if not self._images_user_edited:
                images = root / "images"
                self.images_browse.set_text(str(images if images.is_dir() else scene_images_dir(Path(self.scene_dir))))
            if not self._masks_user_edited:
                masks = root / "masks"
                fallback = scene_masks_dir(Path(self.scene_dir))
                self.masks_browse.set_text(str(masks if masks.is_dir() else fallback if fallback.is_dir() else ""))
            if not self.output_browse.text() or not self._output_user_edited:
                self.output_browse.set_text(str(self._default_output_dir()))
        finally:
            self._syncing_defaults = False
        self._update_extra_folder_hints()
        self._update_data_quality_note()

    def _realityscan_dir(self) -> Path:
        scene = Path(self.scene_dir)
        output_realityscan = scene_output_dir(scene) / "realityscan"
        return output_realityscan if output_realityscan.is_dir() else scene

    def _default_output_dir(self) -> Path:
        name = "lfs_colmap_undistorted" if self.pre_undistort_cb.isChecked() else "lfs_colmap"
        return self._realityscan_dir() / name

    def _update_data_quality_note(self) -> None:
        description = i18n.t("DATASET_TOOL_RS_LFS_DESC")
        csv_text = self.csv_browse.text()
        images_text = self.images_browse.text()
        csv_path = Path(csv_text) if csv_text else None
        images_dir = Path(images_text) if images_text else None
        if csv_path is None or not csv_path.is_file() or images_dir is None or not images_dir.is_dir():
            self.summary_note.setText(description)
            return
        try:
            camera_names = [row.name for row in read_realityscan_csv(csv_path)]
            extra_images_dir = images_dir.parent / "extra_images" if images_dir.name.casefold() == "images" else images_dir / "extra_images"
            diagnostics = analyze_named_camera_images(
                camera_names,
                images_dir,
                additional_image_roots=(extra_images_dir,),
            )
        except Exception:
            self.summary_note.setText(description)
            return

        parts = [
            i18n.t("RS_LFS_INPUT_SUMMARY").format(
                camera_count=diagnostics.camera_image_count,
                image_count=diagnostics.image_count,
            )
        ]
        if diagnostics.camera_images_missing_on_disk:
            parts.append(
                i18n.t("RS_LFS_DATA_QUALITY_CAMERA_IMAGES_MISSING").format(
                    count=len(diagnostics.camera_images_missing_on_disk)
                )
            )
        self.summary_note.setText(f"{description}\n\n" + "\n".join(parts))

    def _update_extra_folder_hints(self) -> None:
        self._update_extra_folder_hint(
            self.images_browse.text(),
            self.images_extra_hint,
            REALITYSCAN_IMAGE_DIR_NAMES,
            i18n.t("RS_LFS_ADDITIONAL_IMAGES_USED"),
        )
        self._update_extra_folder_hint(
            self.masks_browse.text(),
            self.masks_extra_hint,
            REALITYSCAN_MASK_DIR_NAMES,
            i18n.t("RS_LFS_ADDITIONAL_MASKS_USED"),
        )

    @staticmethod
    def _build_asset_field(browse: BrowseWidget, hint: QLabel) -> QWidget:
        field = QWidget()
        layout = QVBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(browse)
        layout.addWidget(hint)
        return field

    @staticmethod
    def _build_extra_folder_hint() -> QLabel:
        hint = QLabel()
        hint.setObjectName("assetFolderHint")
        hint.setWordWrap(True)
        hint.hide()
        return hint

    @staticmethod
    def _update_extra_folder_hint(
        text: str,
        label: QLabel,
        dir_names: tuple[str, ...],
        message_template: str,
    ) -> None:
        base = Path(text) if text else None
        if base is None or not base.is_dir():
            label.hide()
            label.setText("")
            label.setToolTip("")
            return
        roots = [
            root
            for root in related_realityscan_asset_roots(base, dir_names)
            if root.is_dir() and not RealityScanLfsTool._paths_equivalent(root, base)
        ]
        if not roots:
            label.hide()
            label.setText("")
            label.setToolTip("")
            return
        folders = " / ".join(root.name for root in roots)
        label.setText(message_template.format(folders=folders))
        label.setToolTip("\n".join(str(root) for root in roots))
        label.show()

    @staticmethod
    def _first_existing(root: Path, patterns: tuple[str, ...]) -> Path | None:
        for pattern in patterns:
            matches = sorted(root.glob(pattern), key=lambda p: p.name.lower())
            for match in matches:
                if match.is_file():
                    return match
        return None

    @staticmethod
    def _set_browse_text(browse: BrowseWidget, text: str) -> None:
        blocker = QSignalBlocker(browse.line_edit)
        try:
            browse.set_text(text)
        finally:
            del blocker

    @staticmethod
    def _artifact_id(prefix: str) -> str:
        from core.scene_project import utc_now_iso

        return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '')}"

    @staticmethod
    def _paths_equivalent(a: Path, b: Path) -> bool:
        try:
            return a.resolve() == b.resolve()
        except OSError:
            return a.absolute() == b.absolute()
