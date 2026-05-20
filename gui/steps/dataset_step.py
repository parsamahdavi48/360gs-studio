"""Dataset creation step with route cards and standalone utilities."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from gui import i18n
from gui.steps.apriltag_scale_tool import AprilTagScaleTool
from gui.steps.base_step import BaseStepWidget
from gui.steps.realityscan_lfs_tool import RealityScanLfsTool
from gui.steps.sfm_route_specs import SFM_ROUTE_COLMAP, SFM_ROUTE_METASHAPE, SFM_ROUTE_SPHERESFM, normalize_sfm_route
from gui.steps.step4_contracts import _PIPELINE_STAGE_CONVERSION, _PIPELINE_STAGE_SFM
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.workflow_cards import WorkflowCardGrid, WorkflowCardSpec

_PAGE_MENU = "menu"
_PAGE_METASHAPE = "metashape_dataset"
_PAGE_SPHERESFM = "spheresfm_dataset"
_PAGE_COLMAP_READY = "colmap_ready"
_PAGE_CUBEMAP_LEGACY = "cubemap"
_PAGE_REALITYSCAN = "realityscan_lfs"
_PAGE_SCALE = "scale"
_CUBEMAP_PAGES = {_PAGE_METASHAPE, _PAGE_SPHERESFM}


class DatasetStep(BaseStepWidget):
    """Owns dataset-generation tools while preserving existing command contracts."""

    tool_changed = Signal(str)

    def __init__(self, base_dir: Path, cubemap_step: CubemapStep, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self.cubemap_step = cubemap_step
        self.realityscan_tool = RealityScanLfsTool(base_dir)
        self.scale_tool = AprilTagScaleTool(base_dir)
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
        root.addWidget(self.stack)

    def _build_menu_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        title = QLabel(i18n.t("DATASET_MENU_TITLE"))
        title.setObjectName("paneTitle")
        layout.addWidget(title)

        description = QLabel(i18n.t("DATASET_MENU_DESC"))
        description.setObjectName("workflowNote")
        description.setWordWrap(True)
        layout.addWidget(description)

        specs = (
            WorkflowCardSpec(
                _PAGE_METASHAPE,
                i18n.t("DATASET_TOOL_METASHAPE_TITLE"),
                i18n.t("DATASET_TOOL_METASHAPE_CARD_BODY"),
                i18n.t("DATASET_TOOL_METASHAPE_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_METASHAPE"),
            ),
            WorkflowCardSpec(
                _PAGE_SPHERESFM,
                i18n.t("DATASET_TOOL_SPHERESFM_TITLE"),
                i18n.t("DATASET_TOOL_SPHERESFM_CARD_BODY"),
                i18n.t("DATASET_TOOL_SPHERESFM_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_SPHERESFM"),
            ),
            WorkflowCardSpec(
                _PAGE_REALITYSCAN,
                i18n.t("DATASET_TOOL_RS_LFS_TITLE"),
                i18n.t("DATASET_TOOL_RS_LFS_CARD_BODY"),
                i18n.t("DATASET_TOOL_RS_LFS_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_RS_LFS"),
            ),
            WorkflowCardSpec(
                _PAGE_SCALE,
                i18n.t("DATASET_TOOL_SCALE_TITLE"),
                i18n.t("DATASET_TOOL_SCALE_CARD_BODY"),
                i18n.t("DATASET_TOOL_SCALE_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_SCALE"),
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
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QWidget()
        header.setObjectName("toolDetailHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 8, 8, 0)
        header_layout.setSpacing(8)
        back_btn = QPushButton(i18n.t("DATASET_BACK_TO_MENU"))
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(lambda _checked=False: self.show_tool(_PAGE_MENU))
        header_layout.addWidget(back_btn)
        self.cubemap_detail_title = QLabel(title)
        self.cubemap_detail_title.setObjectName("paneTitle")
        header_layout.addWidget(self.cubemap_detail_title)
        header_layout.addStretch()
        layout.addWidget(header)

        self.cubemap_detail_note = QLabel(description)
        self.cubemap_detail_note.setObjectName("workflowNote")
        self.cubemap_detail_note.setWordWrap(True)
        self.cubemap_detail_note.setContentsMargins(8, 0, 8, 0)
        layout.addWidget(self.cubemap_detail_note)
        layout.addWidget(self.cubemap_step, stretch=1)
        return page

    def _wrap_detail_page(self, title: str, child: QWidget, *, header_extra: QWidget | None = None) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QWidget()
        header.setObjectName("toolDetailHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 8, 8, 0)
        header_layout.setSpacing(8)
        back_btn = QPushButton(i18n.t("DATASET_BACK_TO_MENU"))
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(lambda _checked=False: self.show_tool(_PAGE_MENU))
        header_layout.addWidget(back_btn)
        label = QLabel(title)
        label.setObjectName("paneTitle")
        header_layout.addWidget(label)
        header_layout.addStretch()
        layout.addWidget(header)
        if header_extra is not None:
            layout.addWidget(header_extra)
        layout.addWidget(child, stretch=1)
        return page

    def _build_colmap_ready_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        header = QWidget()
        header.setObjectName("toolDetailHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        back_btn = QPushButton(i18n.t("DATASET_BACK_TO_MENU"))
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(lambda _checked=False: self.show_tool(_PAGE_MENU))
        header_layout.addWidget(back_btn)
        title = QLabel(i18n.t("DATASET_TOOL_COLMAP_READY_TITLE"))
        title.setObjectName("paneTitle")
        header_layout.addWidget(title)
        header_layout.addStretch()
        layout.addWidget(header)

        description = QLabel(i18n.t("DATASET_TOOL_COLMAP_READY_DESC"))
        description.setObjectName("workflowNote")
        description.setWordWrap(True)
        layout.addWidget(description)

        viewer_btn = QPushButton(i18n.t("SFM_OPEN_VIEWER"))
        viewer_btn.setObjectName("secondary")
        viewer_btn.setToolTip(i18n.tip("SFM_OPEN_VIEWER"))
        viewer_btn.clicked.connect(lambda _checked=False: self.cubemap_step.open_scene_preview())
        layout.addWidget(viewer_btn)
        layout.addStretch()
        return page

    def _connect_child_signals(self) -> None:
        for child in (self.cubemap_step, self.realityscan_tool, self.scale_tool):
            child.primary_action_state_changed.connect(self.primary_action_state_changed)
            child.background_task_started.connect(self.background_task_started)
            child.background_line_received.connect(self.background_line_received)
            child.background_progress_changed.connect(self.background_progress_changed)
            child.background_status_changed.connect(self.background_status_changed)
            child.background_task_finished.connect(self.background_task_finished)

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        self.cubemap_step.set_scene_dir(path)
        self.realityscan_tool.set_scene_dir(path)
        self.scale_tool.set_scene_dir(path)

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
        self.cubemap_step.export_method_row.setVisible(False)
        if page == _PAGE_SPHERESFM:
            self.cubemap_step._set_export_method(SFM_ROUTE_SPHERESFM)
            self.cubemap_step.set_pipeline_stage_intent(_PIPELINE_STAGE_SFM, False)
            self.cubemap_step.set_pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION, True)
            self.cubemap_step.activate_pipeline_stage(_PIPELINE_STAGE_SFM)
            self.cubemap_detail_title.setText(i18n.t("DATASET_TOOL_SPHERESFM_TITLE"))
            self.cubemap_detail_note.setText(i18n.t("DATASET_TOOL_SPHERESFM_DESC"))
            return

        self.cubemap_step._set_export_method(SFM_ROUTE_METASHAPE)
        self.cubemap_step.set_pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION, True)
        self.cubemap_step.activate_pipeline_stage(_PIPELINE_STAGE_SFM)
        self.cubemap_detail_title.setText(i18n.t("DATASET_TOOL_METASHAPE_TITLE"))
        self.cubemap_detail_note.setText(i18n.t("DATASET_TOOL_METASHAPE_DESC"))

    def current_tool(self) -> str:
        return self._page

    def _active_step(self) -> BaseStepWidget:
        if self._page in _CUBEMAP_PAGES:
            return self.cubemap_step
        if self._page == _PAGE_REALITYSCAN:
            return self.realityscan_tool
        if self._page == _PAGE_SCALE:
            return self.scale_tool
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

    def build_commands(self) -> list[tuple[str, list[str]]]:
        if self._page in {_PAGE_MENU, _PAGE_COLMAP_READY}:
            return []
        if self._page in _CUBEMAP_PAGES:
            self._configure_cubemap_tool(self._page)
        return self._active_step().build_commands()

    def confirm_commands(self, commands: list[tuple[str, list[str]]]) -> bool:
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
