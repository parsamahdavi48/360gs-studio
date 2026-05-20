"""Dataset creation step with route cards and standalone utilities."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from gui import i18n
from gui.steps.apriltag_scale_tool import AprilTagScaleTool
from gui.steps.base_step import BaseStepWidget
from gui.steps.realityscan_lfs_tool import RealityScanLfsTool
from gui.steps.sfm_route_specs import normalize_sfm_route
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.workflow_cards import WorkflowCardGrid, WorkflowCardSpec

_PAGE_MENU = "menu"
_PAGE_CUBEMAP = "cubemap"
_PAGE_REALITYSCAN = "realityscan_lfs"
_PAGE_SCALE = "scale"


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
        self._page_indices[_PAGE_CUBEMAP] = self.stack.addWidget(
            self._wrap_detail_page(i18n.t("DATASET_TOOL_CUBEMAP_TITLE"), self.cubemap_step)
        )
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
                _PAGE_CUBEMAP,
                i18n.t("DATASET_TOOL_CUBEMAP_TITLE"),
                i18n.t("DATASET_TOOL_CUBEMAP_CARD_BODY"),
                i18n.t("DATASET_TOOL_CUBEMAP_CARD_FOOTER"),
                i18n.tip("DATASET_TOOL_CUBEMAP"),
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

    def _wrap_detail_page(self, title: str, child: QWidget) -> QWidget:
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
        layout.addWidget(child, stretch=1)
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
        if self._page != _PAGE_MENU:
            self._active_step().on_activated()
        self.primary_action_state_changed.emit()

    def show_tool(self, page: str) -> None:
        if page not in self._page_indices:
            page = _PAGE_MENU
        self._page = page
        self.stack.setCurrentIndex(self._page_indices[page])
        if page != _PAGE_MENU:
            self._active_step().on_activated()
        self.tool_changed.emit(page)
        self.primary_action_state_changed.emit()

    def show_cubemap_route(self, route_id: str) -> None:
        self.cubemap_step._set_export_method(normalize_sfm_route(route_id))
        self.show_tool(_PAGE_CUBEMAP)

    def current_tool(self) -> str:
        return self._page

    def _active_step(self) -> BaseStepWidget:
        if self._page == _PAGE_CUBEMAP:
            return self.cubemap_step
        if self._page == _PAGE_REALITYSCAN:
            return self.realityscan_tool
        if self._page == _PAGE_SCALE:
            return self.scale_tool
        return self

    def primary_action_text(self) -> str:
        if self._page == _PAGE_MENU:
            return i18n.t("DATASET_SELECT_TOOL")
        return self._active_step().primary_action_text()

    def primary_action_tooltip(self) -> str:
        if self._page == _PAGE_MENU:
            return i18n.tip("DATASET_SELECT_TOOL")
        return self._active_step().primary_action_tooltip()

    def primary_action_enabled(self) -> bool:
        if self._page == _PAGE_MENU:
            return False
        return self._active_step().primary_action_enabled()

    def build_commands(self) -> list[tuple[str, list[str]]]:
        if self._page == _PAGE_MENU:
            return []
        return self._active_step().build_commands()

    def process_log_dir(self) -> Path | None:
        if self._page == _PAGE_MENU:
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
