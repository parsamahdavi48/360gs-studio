"""Standalone AprilTag scale adjustment tool."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QFormLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from core.scene_layout import scene_metashape_cubemap_dir, step4_export_settings_path
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.form_rows import add_tooltip_row
from gui.steps.base_step import BaseStepWidget
from gui.steps.step4_apriltag import Step4AprilTagMixin


class AprilTagScaleTool(Step4AprilTagMixin, BaseStepWidget):
    """Estimate and apply scene scale for an existing projected dataset."""

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._scale_output_user_edited = False
        self._syncing_scale_output = False
        self._init_apriltag_state()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel(i18n.t("DATASET_TOOL_SCALE_TITLE"))
        title.setObjectName("paneTitle")
        layout.addWidget(title)

        description = QLabel(i18n.t("DATASET_TOOL_SCALE_DESC"))
        description.setObjectName("workflowNote")
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        form.setSpacing(7)
        self.scale_output_browse = BrowseWidget(mode="dir")
        self.scale_output_browse.setToolTip(i18n.tip("APRILTAG_OUTPUT_DATASET"))
        self.scale_output_browse.path_changed.connect(self._on_scale_output_changed)
        add_tooltip_row(
            form,
            i18n.t("APRILTAG_OUTPUT_DATASET"),
            self.scale_output_browse,
            i18n.tip("APRILTAG_OUTPUT_DATASET"),
        )
        layout.addLayout(form)

        layout.addWidget(self._build_apriltag_scale_tab(), stretch=1)
        scroll.setWidget(content)
        root.addWidget(scroll)

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        self._sync_default_scale_output()

    def on_activated(self) -> None:
        self._sync_default_scale_output()
        self._sync_apriltag_controls()

    def primary_action_text(self) -> str:
        return i18n.t("SCALE_TOOL_INTERNAL_ACTION")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("APRILTAG_TAB_PRIMARY_ACTION")

    def primary_action_enabled(self) -> bool:
        return False

    def _display_output_dir(self) -> Path:
        text = self.scale_output_browse.text().strip()
        if text:
            return Path(text)
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return self._configured_scale_output(Path(self.scene_dir))

    def _on_scale_output_changed(self, _path: str) -> None:
        if not self._syncing_scale_output:
            self._scale_output_user_edited = True

    def _sync_default_scale_output(self) -> None:
        if not self.scene_dir or self._scale_output_user_edited:
            return
        output = self._configured_scale_output(Path(self.scene_dir))
        self._syncing_scale_output = True
        blocker = QSignalBlocker(self.scale_output_browse.line_edit)
        try:
            self.scale_output_browse.set_text(str(output))
        finally:
            del blocker
            self._syncing_scale_output = False

    @staticmethod
    def _configured_scale_output(scene: Path) -> Path:
        settings = step4_export_settings_path(scene)
        if settings.is_file():
            try:
                data = json.loads(settings.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict):
                output_dir = str(data.get("output_dir") or "").strip()
                if output_dir:
                    path = Path(output_dir)
                    return path if path.is_absolute() else scene / path
                portable = data.get("portable_output")
                if isinstance(portable, dict):
                    root = str(portable.get("root") or "").strip()
                    if root:
                        return scene / root
        return scene_metashape_cubemap_dir(scene)
