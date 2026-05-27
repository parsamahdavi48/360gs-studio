"""Reusable card widgets for workflow route menus."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_CARD_HEIGHT = 184


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
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(spec.title)
        self.setAccessibleDescription(spec.body)
        self.setToolTip(spec.tooltip or spec.body)
        self.setMinimumHeight(_CARD_HEIGHT)
        self.setMaximumHeight(_CARD_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.title_label = QLabel(spec.title, self)
        self.title_label.setObjectName("workflowCardTitle")
        self.title_label.setWordWrap(True)
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.title_label)

        self.body_label = QLabel(spec.body, self)
        self.body_label.setObjectName("workflowCardBody")
        self.body_label.setWordWrap(True)
        self.body_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.body_label)
        layout.addStretch(1)

        self.footer_label = QLabel(spec.footer, self)
        self.footer_label.setObjectName("workflowCardFooter")
        self.footer_label.setWordWrap(True)
        self.footer_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.footer_label)
        self.footer_label.setVisible(bool(spec.footer))


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
            button = WorkflowCardButton(spec, self)
            self.buttons[spec.card_id] = button
            layout.addWidget(button, index // 2, index % 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
