"""Reusable card widgets for workflow route menus."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QGridLayout, QPushButton, QSizePolicy, QWidget


@dataclass(frozen=True, slots=True)
class WorkflowCardSpec:
    card_id: str
    title: str
    body: str
    footer: str = ""
    tooltip: str = ""


class WorkflowCardButton(QPushButton):
    def __init__(self, spec: WorkflowCardSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.card_id = spec.card_id
        self.setObjectName("workflowCard")
        footer = f"\n\n{spec.footer}" if spec.footer else ""
        self.setText(f"{spec.title}\n{spec.body}{footer}")
        self.setToolTip(spec.tooltip or spec.body)
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class WorkflowCardGrid(QWidget):
    def __init__(self, specs: tuple[WorkflowCardSpec, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowCardGrid")
        self.buttons: dict[str, WorkflowCardButton] = {}
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        for index, spec in enumerate(specs):
            button = WorkflowCardButton(spec)
            self.buttons[spec.card_id] = button
            layout.addWidget(button, index // 2, index % 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
