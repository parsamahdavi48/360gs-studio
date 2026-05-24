"""Dataset creation step with route cards and standalone utilities."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from core.app_job import AppJob
from gui import i18n
from gui.steps.apriltag_scale_tool import AprilTagScaleTool
from gui.steps.base_step import BaseStepWidget
from gui.steps.colmap_text_model_tool import ColmapTextModelTool
from gui.steps.realityscan_lfs_tool import RealityScanLfsTool
from gui.steps.sfm_route_specs import SFM_ROUTE_COLMAP, SFM_ROUTE_METASHAPE, SFM_ROUTE_SPHERESFM, normalize_sfm_route
from gui.steps.step4_contracts import (
    _PIPELINE_STAGE_CONVERSION,
    _PIPELINE_STAGE_SFM,
    _PROFILE_CUSTOM,
    _PROFILE_LICHTFELD,
    _PROFILE_REALITYSCAN,
)
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.workflow_cards import WorkflowCardGrid, WorkflowCardSpec

_PAGE_MENU = "menu"
_PAGE_METASHAPE = "metashape_dataset"
_PAGE_SPHERESFM = "spheresfm_dataset"
_PAGE_COLMAP_READY = "colmap_ready"
_PAGE_CUBEMAP_LEGACY = "cubemap"
_PAGE_REALITYSCAN = "realityscan_lfs"
_PAGE_SCALE = "scale"
_PAGE_COLMAP_TEXT = "colmap_text_model"
_CUBEMAP_PAGES = {_PAGE_METASHAPE, _PAGE_SPHERESFM}


class DatasetStep(BaseStepWidget):
    """Owns dataset-generation tools while preserving existing command contracts."""

    tool_changed = Signal(str)

    def __init__(self, base_dir: Path, cubemap_step: CubemapStep, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self.cubemap_step = cubemap_step
        self.realityscan_tool = RealityScanLfsTool(base_dir)
        self.scale_tool = AprilTagScaleTool(base_dir)
        self.colmap_text_tool = ColmapTextModelTool(base_dir)
        self._page = _PAGE_MENU
        self._page_indices: dict[str, int] = {}
        self._build_ui()
        self._connect_child_signals()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stack = QStackedWidget()
        self._page_indices[_PAGE_MENU] = self.stack.addWidget(self._build_menu_page())
        cubemap_page_index = self.stack.addWidget(
            self._wrap_cubemap_detail_page(i18n.t("DATASET_TOOL_METASHAPE_TITLE"), i18n.t("DATASET_TOOL_METASHAPE_DESC"))
        )
        self._page_indices[_PAGE_METASHAPE] = cubemap_page_index
        self._page_indices[_PAGE_SPHERESFM] = cubemap_page_index
        self._page_indices[_PAGE_CUBEMAP_LEGACY] = cubemap_page_index
        self._page_indices[_PAGE_COLMAP_READY] = self.stack.addWidget(self._build_colmap_ready_page())
        self._page_indices[_PAGE_REALITYSCAN] = self.stack.addWidget(
            self._wrap_detail_page(i18n.t("DATASET_TOOL_RS_LFS_TITLE"), self.realityscan_tool)
        )
        self._page_indices[_PAGE_SCALE] = self.stack.addWidget(
            self._wrap_detail_page(i18n.t("DATASET_TOOL_SCALE_TITLE"), self.scale_tool)
        )
        self._page_indices[_PAGE_COLMAP_TEXT] = self.stack.addWidget(
            self._wrap_detail_page(
                i18n.t("DATASET_TOOL_COLMAP_TEXT_TITLE"),
                self.colmap_text_tool,
                header_extra=self._make_detail_note(i18n.t("DATASET_TOOL_COLMAP_TEXT_DESC")),
            )
        )
        root.addWidget(self.stack)

    def _build_menu_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        self.dataset_menu_note = QLabel(i18n.t("DATASET_MENU_DESC"))
        self.dataset_menu_note.setObjectName("workflowNote")
        self.dataset_menu_note.setWordWrap(True)
        layout.addWidget(self.dataset_menu_note)

        specs = (
            WorkflowCardSpec(
                _PAGE_METASHAPE,
                i18n.t("DATASET_TOOL_METASHAPE_TITLE"),
                i18n.t("DATASET_TOOL_METASHAPE_CARD_BODY"),
                i18n.t("DATASET_TOOL_METASHAPE_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_METASHAPE"),
            ),
            WorkflowCardSpec(
                _PAGE_REALITYSCAN,
                i18n.t("DATASET_TOOL_RS_LFS_TITLE"),
                i18n.t("DATASET_TOOL_RS_LFS_CARD_BODY"),
                i18n.t("DATASET_TOOL_RS_LFS_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_RS_LFS"),
            ),
            WorkflowCardSpec(
                _PAGE_SPHERESFM,
                i18n.t("DATASET_TOOL_SPHERESFM_TITLE"),
                i18n.t("DATASET_TOOL_SPHERESFM_CARD_BODY"),
                i18n.t("DATASET_TOOL_SPHERESFM_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_SPHERESFM"),
            ),
            WorkflowCardSpec(
                _PAGE_SCALE,
                i18n.t("DATASET_TOOL_SCALE_TITLE"),
                i18n.t("DATASET_TOOL_SCALE_CARD_BODY"),
                i18n.t("DATASET_TOOL_SCALE_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_SCALE"),
            ),
            WorkflowCardSpec(
                _PAGE_COLMAP_TEXT,
                i18n.t("DATASET_TOOL_COLMAP_TEXT_TITLE"),
                i18n.t("DATASET_TOOL_COLMAP_TEXT_CARD_BODY"),
                i18n.t("DATASET_TOOL_COLMAP_TEXT_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_COLMAP_TEXT"),
            ),
        )
        self.card_grid = WorkflowCardGrid(specs)
        for page_id, button in self.card_grid.buttons.items():
            button.clicked.connect(lambda _checked=False, p=page_id: self.show_tool(p))
        layout.addWidget(self.card_grid)
        layout.addStretch()
        return page

    def _wrap_cubemap_detail_page(self, title: str, description: str) -> QWidget:
        page = QWidget()
        page.setAccessibleName(title)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.cubemap_detail_note = QLabel(description)
        self.cubemap_detail_note.setObjectName("workflowNote")
        self.cubemap_detail_note.setWordWrap(True)
        layout.addWidget(self.cubemap_detail_note)
        self.cubemap_detail_layout = layout
        layout.addWidget(self.cubemap_step, stretch=1)
        return page

    def _make_detail_note(self, text: str) -> QLabel:
        note = QLabel(text)
        note.setObjectName("workflowNote")
        note.setWordWrap(True)
        note.setContentsMargins(8, 8, 8, 0)
        return note

    def _wrap_detail_page(self, title: str, child: QWidget, *, header_extra: QWidget | None = None) -> QWidget:
        page = QWidget()
        page.setAccessibleName(title)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        if header_extra is not None:
            layout.addWidget(header_extra)
        layout.addWidget(child, stretch=1)
        return page

    def _build_colmap_ready_page(self) -> QWidget:
        page = QWidget()
        page.setAccessibleName(i18n.t("DATASET_TOOL_COLMAP_READY_TITLE"))
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        description = QLabel(i18n.t("DATASET_TOOL_COLMAP_READY_DESC"))
        description.setObjectName("workflowNote")
        description.setWordWrap(True)
        layout.addWidget(description)

        viewer_btn = QPushButton(i18n.t("SFM_OPEN_VIEWER"))
        viewer_btn.setObjectName("secondary")
        viewer_btn.setToolTip(i18n.tip("SFM_OPEN_VIEWER"))
        viewer_btn.clicked.connect(lambda _checked=False: self.cubemap_step.open_scene_preview())
        layout.addWidget(viewer_btn, alignment=Qt.AlignLeft)
        layout.addStretch()
        return page

    def _connect_child_signals(self) -> None:
        for child in (self.cubemap_step, self.realityscan_tool, self.scale_tool, self.colmap_text_tool):
            child.primary_action_state_changed.connect(self.primary_action_state_changed)
            child.background_task_started.connect(self.background_task_started)
            child.background_line_received.connect(self.background_line_received)
            child.background_progress_changed.connect(self.background_progress_changed)
            child.background_status_changed.connect(self.background_status_changed)
            child.background_task_finished.connect(self.background_task_finished)

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if self.cubemap_step.scene_dir != path:
            self.cubemap_step.set_scene_dir(path)
        self.realityscan_tool.set_scene_dir(path)
        self.scale_tool.set_scene_dir(path)
        self.colmap_text_tool.set_scene_dir(path)

    def on_activated(self) -> None:
        if self._page in _CUBEMAP_PAGES:
            self._configure_cubemap_tool(self._page)
        if self._page not in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            self._active_step().on_activated()
        self.primary_action_state_changed.emit()

    def show_tool(self, page: str) -> None:
        if page == _PAGE_CUBEMAP_LEGACY:
            page = _PAGE_METASHAPE
        if page not in self._page_indices:
            page = _PAGE_MENU
        self._page = page
        if page in _CUBEMAP_PAGES:
            self._configure_cubemap_tool(page)
        self.stack.setCurrentIndex(self._page_indices[page])
        if page not in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            self._active_step().on_activated()
        self._focus_default_detail_tab(page)
        self.tool_changed.emit(page)
        self.primary_action_state_changed.emit()

    def show_cubemap_route(self, route_id: str) -> None:
        route = normalize_sfm_route(route_id)
        if route == SFM_ROUTE_SPHERESFM:
            self.show_tool(_PAGE_SPHERESFM)
        elif route == SFM_ROUTE_COLMAP:
            self.cubemap_step._set_export_method(SFM_ROUTE_COLMAP)
            self.show_tool(_PAGE_COLMAP_READY)
        else:
            self.show_tool(_PAGE_METASHAPE)

    def _configure_cubemap_tool(self, page: str) -> None:
        self._attach_cubemap_step()
        self.cubemap_step.export_method_row.setVisible(False)
        self._set_metashape_training_profile_controls_visible()
        if page == _PAGE_SPHERESFM:
            self.cubemap_step._set_export_method(SFM_ROUTE_SPHERESFM)
            self.cubemap_step.set_pipeline_stage_intent(_PIPELINE_STAGE_SFM, False)
            self.cubemap_step.set_pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION, True)
            self.cubemap_step.activate_pipeline_stage(_PIPELINE_STAGE_SFM)
            self.cubemap_detail_note.setText(i18n.t("DATASET_TOOL_SPHERESFM_DESC"))
            return

        self.cubemap_step._set_export_method(SFM_ROUTE_METASHAPE)
        self.cubemap_step.set_pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION, True)
        self.cubemap_step.activate_pipeline_stage(_PIPELINE_STAGE_SFM)
        self.cubemap_detail_note.setText(i18n.t("DATASET_TOOL_METASHAPE_DESC"))

    def _attach_cubemap_step(self) -> None:
        if self.cubemap_detail_layout.indexOf(self.cubemap_step) >= 0:
            return
        self.cubemap_detail_layout.addWidget(self.cubemap_step, stretch=1)

    def _set_metashape_training_profile_controls_visible(self) -> None:
        self.cubemap_step.profile_combo.setVisible(True)
        if self.cubemap_step.profile_label is not None:
            self.cubemap_step.profile_label.setVisible(True)
        for profile in (_PROFILE_REALITYSCAN, _PROFILE_CUSTOM):
            index = self.cubemap_step.profile_combo.findData(profile)
            if index < 0:
                continue
            if self.cubemap_step.profile_combo.currentData() == _PROFILE_REALITYSCAN:
                self.cubemap_step._set_combo_data(self.cubemap_step.profile_combo, _PROFILE_LICHTFELD)
            if self.cubemap_step.profile_combo.currentData() == _PROFILE_CUSTOM:
                self.cubemap_step._set_combo_data(self.cubemap_step.profile_combo, _PROFILE_LICHTFELD)
            self.cubemap_step.profile_combo.view().setRowHidden(index, True)

        spheresfm_custom_index = self.cubemap_step.spheresfm_profile_combo.findData(_PROFILE_CUSTOM)
        if spheresfm_custom_index >= 0:
            if self.cubemap_step.spheresfm_profile_combo.currentData() == _PROFILE_CUSTOM:
                self.cubemap_step._set_combo_data(self.cubemap_step.spheresfm_profile_combo, _PROFILE_LICHTFELD)
            self.cubemap_step.spheresfm_profile_combo.view().setRowHidden(spheresfm_custom_index, True)

    def _focus_default_detail_tab(self, page: str) -> None:
        if page in _CUBEMAP_PAGES:
            if self.cubemap_step.settings_tabs.isTabEnabled(self.cubemap_step.output_tab_index):
                self.cubemap_step.settings_tabs.setCurrentIndex(self.cubemap_step.output_tab_index)
            return
        if page == _PAGE_COLMAP_TEXT:
            self.colmap_text_tool.focus_output_tab()

    def current_tool(self) -> str:
        return self._page

    def header_title(self) -> str:
        if self._page == _PAGE_METASHAPE:
            return i18n.t("DATASET_TOOL_METASHAPE_TITLE")
        if self._page == _PAGE_SPHERESFM:
            return i18n.t("DATASET_TOOL_SPHERESFM_TITLE")
        if self._page == _PAGE_REALITYSCAN:
            return i18n.t("DATASET_TOOL_RS_LFS_TITLE")
        if self._page == _PAGE_SCALE:
            return i18n.t("DATASET_TOOL_SCALE_TITLE")
        if self._page == _PAGE_COLMAP_TEXT:
            return i18n.t("DATASET_TOOL_COLMAP_TEXT_TITLE")
        if self._page == _PAGE_COLMAP_READY:
            return i18n.t("DATASET_TOOL_COLMAP_READY_TITLE")
        return i18n.t("DATASET_MENU_TITLE")

    def header_back_enabled(self) -> bool:
        return self._page != _PAGE_MENU

    def header_back_tooltip(self) -> str:
        return i18n.t("DATASET_BACK_TO_MENU")

    def header_back(self) -> None:
        self.show_tool(_PAGE_MENU)

    def _active_step(self) -> BaseStepWidget:
        if self._page in _CUBEMAP_PAGES:
            return self.cubemap_step
        if self._page == _PAGE_REALITYSCAN:
            return self.realityscan_tool
        if self._page == _PAGE_SCALE:
            return self.scale_tool
        if self._page == _PAGE_COLMAP_TEXT:
            return self.colmap_text_tool
        return self

    def primary_action_text(self) -> str:
        if self._page == _PAGE_MENU:
            return i18n.t("DATASET_SELECT_TOOL")
        if self._page == _PAGE_COLMAP_READY:
            return i18n.t("DATASET_COLMAP_READY_ACTION")
        if self._page == _PAGE_METASHAPE:
            return i18n.t("DATASET_RUN_METASHAPE")
        if self._page == _PAGE_SPHERESFM:
            return i18n.t("DATASET_RUN_SPHERESFM")
        return self._active_step().primary_action_text()

    def primary_action_tooltip(self) -> str:
        if self._page == _PAGE_MENU:
            return i18n.tip("DATASET_SELECT_TOOL")
        if self._page == _PAGE_COLMAP_READY:
            return i18n.tip("DATASET_COLMAP_READY_ACTION")
        if self._page == _PAGE_METASHAPE:
            return i18n.tip("DATASET_RUN_METASHAPE")
        if self._page == _PAGE_SPHERESFM:
            return i18n.tip("DATASET_RUN_SPHERESFM")
        return self._active_step().primary_action_tooltip()

    def primary_action_enabled(self) -> bool:
        if self._page in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            return False
        return self._active_step().primary_action_enabled()

    def run_primary_action(self) -> bool:
        if self._page in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            return False
        if self._page in _CUBEMAP_PAGES:
            self._configure_cubemap_tool(self._page)
        return self._active_step().run_primary_action()

    def build_commands(self) -> list[tuple[str, list[str] | AppJob]]:
        if self._page in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            return []
        if self._page in _CUBEMAP_PAGES:
            self._configure_cubemap_tool(self._page)
        return self._active_step().build_commands()

    def confirm_commands(self, commands: list[tuple[str, list[str] | AppJob]]) -> bool:
        if self._page in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            return True
        return self._active_step().confirm_commands(commands)

    def process_log_dir(self) -> Path | None:
        if self._page in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            return None
        return self._active_step().process_log_dir()

    def phase_display_name(self, phase: str) -> str:
        if self._page == _PAGE_MENU:
            return phase
        return self._active_step().phase_display_name(phase)

    def phase_status_text(self, phase: str, queue_index: int, queue_total: int) -> str:
        if self._page == _PAGE_MENU:
            return super().phase_status_text(phase, queue_index, queue_total)
        return self._active_step().phase_status_text(phase, queue_index, queue_total)

    def on_line(self, line: str) -> tuple[int, int] | None:
        if self._page == _PAGE_MENU:
            return None
        return self._active_step().on_line(line)

    def on_phase_started(self, phase: str) -> tuple[int, int] | None:
        if self._page == _PAGE_MENU:
            return None
        return self._active_step().on_phase_started(phase)

    def on_phase_log_started(self, phase: str, path: str) -> None:
        if self._page == _PAGE_MENU:
            return
        self._active_step().on_phase_log_started(phase, path)

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        if self._page == _PAGE_MENU:
            return
        self._active_step().on_phase_finished(phase, exit_code, canceled)

    def on_queue_finished(self, success: bool) -> None:
        if self._page == _PAGE_MENU:
            return
        self._active_step().on_queue_finished(success)

    def has_background_task(self) -> bool:
        return self.cubemap_step.has_background_task() or self.scale_tool.has_background_task()

    def cancel_background_task(self) -> None:
        if self.cubemap_step.has_background_task():
            self.cubemap_step.cancel_background_task()
        if self.scale_tool.has_background_task():
            self.scale_tool.cancel_background_task()

    def shutdown(self) -> None:
        self.cubemap_step.shutdown()
        self.scale_tool.shutdown()
