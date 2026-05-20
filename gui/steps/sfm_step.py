"""SfM route selection step."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui import i18n
from gui.steps.base_step import BaseStepWidget
from gui.steps.sfm_route_specs import SFM_ROUTE_COLMAP, SFM_ROUTE_METASHAPE, SFM_ROUTE_SPHERESFM
from gui.steps.workflow_cards import WorkflowCardGrid, WorkflowCardSpec


class SfmStep(BaseStepWidget):
    """Explains the available camera-alignment routes and forwards selection."""

    route_requested = Signal(str)

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        title = QLabel(i18n.t("SFM_MENU_TITLE"))
        title.setObjectName("paneTitle")
        layout.addWidget(title)

        description = QLabel(i18n.t("SFM_MENU_DESC"))
        description.setObjectName("workflowNote")
        description.setWordWrap(True)
        layout.addWidget(description)

        specs = (
            WorkflowCardSpec(
                SFM_ROUTE_METASHAPE,
                i18n.t("SFM_ROUTE_EXTERNAL_TITLE"),
                i18n.t("SFM_ROUTE_EXTERNAL_BODY"),
                i18n.t("SFM_ROUTE_EXTERNAL_FOOTER"),
                i18n.tip("SFM_ROUTE_EXTERNAL"),
            ),
            WorkflowCardSpec(
                SFM_ROUTE_COLMAP,
                i18n.t("SFM_ROUTE_COLMAP_TITLE"),
                i18n.t("SFM_ROUTE_COLMAP_BODY"),
                i18n.t("SFM_ROUTE_COLMAP_FOOTER"),
                i18n.tip("SFM_ROUTE_COLMAP"),
            ),
            WorkflowCardSpec(
                SFM_ROUTE_SPHERESFM,
                i18n.t("SFM_ROUTE_SPHERESFM_TITLE"),
                i18n.t("SFM_ROUTE_SPHERESFM_BODY"),
                i18n.t("SFM_ROUTE_SPHERESFM_FOOTER"),
                i18n.tip("SFM_ROUTE_SPHERESFM"),
            ),
        )
        self.card_grid = WorkflowCardGrid(specs)
        for route_id, button in self.card_grid.buttons.items():
            button.clicked.connect(lambda _checked=False, r=route_id: self.route_requested.emit(r))
        layout.addWidget(self.card_grid)
        layout.addStretch()

    def primary_action_text(self) -> str:
        return i18n.t("SFM_SELECT_ROUTE")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("SFM_SELECT_ROUTE")

    def primary_action_enabled(self) -> bool:
        return False

    def build_commands(self) -> list[tuple[str, list[str]]]:
        return []
