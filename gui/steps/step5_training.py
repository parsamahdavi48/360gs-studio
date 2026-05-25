"""Step 5: 学習アプリの起動."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from gui import i18n
from gui.common.runner_types import StepCommandQueue
from gui.steps.base_step import BaseStepWidget
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_LICHTFELD,
    TRAINING_BACKEND_POSTSHOT,
    training_backend_specs,
)
from gui.steps.workflow_cards import WorkflowCardGrid, WorkflowCardSpec

_PAGE_MENU = "menu"
_PAGE_DETAIL = "detail"


class TrainingStep(BaseStepWidget):
    """Training UI and command entrypoint backed by the Step 4 dataset state."""

    def __init__(self, base_dir: Path, dataset_step: CubemapStep, parent=None) -> None:
        super().__init__(base_dir, parent)
        self.dataset_step = dataset_step
        self._page = _PAGE_MENU
        self._selected_backend = TRAINING_BACKEND_LICHTFELD

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stack = QStackedWidget()
        self._page_indices: dict[str, int] = {}
        self._page_indices[_PAGE_MENU] = self.stack.addWidget(self._build_menu_page())
        self._page_indices[_PAGE_DETAIL] = self.stack.addWidget(self._build_detail_page())
        layout.addWidget(self.stack)

    def _build_menu_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        description = QLabel(i18n.t("TRAINING_STEP_DESC"))
        description.setObjectName("workflowNote")
        description.setWordWrap(True)
        layout.addWidget(description)

        specs = tuple(
            WorkflowCardSpec(
                spec.backend_id,
                i18n.t(spec.label_key),
                i18n.t(f"TRAINING_TOOL_{spec.backend_id.upper()}_CARD_BODY"),
                i18n.t(f"TRAINING_TOOL_{spec.backend_id.upper()}_CARD_FOOTER"),
                i18n.tip(spec.tooltip_key),
            )
            for spec in training_backend_specs(category="primary", visible_only=True)
        )
        self.training_card_grid = WorkflowCardGrid(specs)
        self.training_card_grid.buttons[TRAINING_BACKEND_LICHTFELD].clicked.connect(
            lambda _checked=False: self.show_backend(TRAINING_BACKEND_LICHTFELD)
        )
        self.training_card_grid.buttons[TRAINING_BACKEND_POSTSHOT].clicked.connect(
            lambda _checked=False: self.show_backend(TRAINING_BACKEND_POSTSHOT)
        )
        layout.addWidget(self.training_card_grid)
        layout.addStretch()
        return page

    def _build_detail_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.training_detail_note = QLabel("")
        self.training_detail_note.setObjectName("workflowNote")
        self.training_detail_note.setWordWrap(True)
        layout.addWidget(self.training_detail_note)

        training_widget = self.dataset_step.apply_training_wide_layout()
        self.dataset_step.training_backend_row.setVisible(False)
        training_layout = training_widget.layout()
        if training_layout is not None:
            training_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(training_widget, stretch=1)
        return page

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if self.dataset_step.scene_dir != path:
            self.dataset_step.set_scene_dir(path)

    def on_activated(self) -> None:
        self.dataset_step.prepare_training_step()

    def show_backend(self, backend: str) -> None:
        self._selected_backend = backend
        self.dataset_step._set_training_backend(backend)
        self.training_detail_note.setText(i18n.t(f"TRAINING_TOOL_{backend.upper()}_DESC"))
        self._page = _PAGE_DETAIL
        self.stack.setCurrentIndex(self._page_indices[_PAGE_DETAIL])
        self.primary_action_state_changed.emit()

    def header_title(self) -> str:
        if self._page == _PAGE_DETAIL:
            return self.dataset_step._training_backend_display_name(self._selected_backend)
        return ""

    def header_back_enabled(self) -> bool:
        return self._page == _PAGE_DETAIL

    def header_back_tooltip(self) -> str:
        return i18n.tip("TRAINING_BACK_TO_APPS")

    def header_back(self) -> None:
        self._page = _PAGE_MENU
        self.stack.setCurrentIndex(self._page_indices[_PAGE_MENU])
        self.primary_action_state_changed.emit()

    def primary_action_text(self) -> str:
        return i18n.t("LAUNCH") if self._page == _PAGE_DETAIL else i18n.t("TRAINING_SELECT_APP")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("LAUNCH_TRAINING") if self._page == _PAGE_DETAIL else i18n.tip("TRAINING_SELECT_APP")

    def primary_action_enabled(self) -> bool:
        if self._page != _PAGE_DETAIL:
            return False
        return self.dataset_step.training_primary_action_enabled()

    def build_commands(self) -> StepCommandQueue:
        if self._page != _PAGE_DETAIL:
            return []
        return self.dataset_step.build_training_launch_commands()

    def process_log_dir(self) -> Path | None:
        return self.dataset_step.training_process_log_dir()

    def phase_display_name(self, phase: str) -> str:
        return self.dataset_step.phase_display_name(phase)

    def on_line(self, line: str) -> tuple[int, int] | None:
        return self.dataset_step.on_line(line)

    def on_phase_started(self, phase: str) -> tuple[int, int] | None:
        return self.dataset_step.on_phase_started(phase)

    def on_phase_log_started(self, phase: str, path: str) -> None:
        self.dataset_step.on_phase_log_started(phase, path)

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        self.dataset_step.on_phase_finished(phase, exit_code, canceled)

    def on_queue_finished(self, success: bool) -> None:
        self.dataset_step.on_training_queue_finished(success)
