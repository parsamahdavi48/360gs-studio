"""RealityScan CSV/PLY to LichtFeld COLMAP dataset GUI tool."""

from __future__ import annotations

import sys
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

from core.realityscan_to_lfs_colmap import DEFAULT_UNDISTORT_ALPHA
from core.scene_layout import scene_images_dir, scene_masks_dir, scene_output_dir
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.form_rows import add_tooltip_row
from gui.steps.base_step import BaseStepWidget


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

        description = QLabel(i18n.t("DATASET_TOOL_RS_LFS_DESC"))
        description.setObjectName("workflowNote")
        description.setWordWrap(True)
        layout.addWidget(description)

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
        add_tooltip_row(form, i18n.t("RS_LFS_IMAGES"), self.images_browse, i18n.tip("RS_LFS_IMAGES"))

        self.masks_browse = BrowseWidget(mode="dir")
        self.masks_browse.setToolTip(i18n.tip("RS_LFS_MASKS"))
        add_tooltip_row(form, i18n.t("RS_LFS_MASKS"), self.masks_browse, i18n.tip("RS_LFS_MASKS"))

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

    def build_commands(self) -> list[tuple[str, list[str]]]:
        self._validate_inputs()
        script = self.base_dir / "realityscan_to_lfs_colmap.py"
        if not script.is_file():
            raise FileNotFoundError(f"realityscan_to_lfs_colmap.py not found: {script}")

        cmd = [
            sys.executable,
            "-u",
            str(script),
            self.csv_browse.text(),
            self.output_browse.text(),
            "--images-dir",
            self.images_browse.text(),
            "--ply",
            self.ply_browse.text(),
        ]
        masks = self.masks_browse.text()
        if masks:
            cmd.extend(["--masks-dir", masks])
        if self.skip_missing_cb.isChecked():
            cmd.append("--skip-missing-images")
        if self.pre_undistort_cb.isChecked():
            cmd.append("--pre-undistort-distorted-images")
            cmd.extend(["--undistort-alpha", f"{DEFAULT_UNDISTORT_ALPHA:g}"])
        return [("realityscan_lfs_colmap", cmd)]

    def _validate_inputs(self) -> None:
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

    def _realityscan_dir(self) -> Path:
        scene = Path(self.scene_dir)
        output_realityscan = scene_output_dir(scene) / "realityscan"
        return output_realityscan if output_realityscan.is_dir() else scene

    def _default_output_dir(self) -> Path:
        name = "lfs_colmap_undistorted" if self.pre_undistort_cb.isChecked() else "lfs_colmap"
        return self._realityscan_dir() / name

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
