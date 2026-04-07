"""折りたたみセクション"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """展開/折りたたみ可能なセクションウィジェット"""

    def __init__(self, title: str, expanded: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 4)
        root.setSpacing(4)

        self.toggle_button = QToolButton()
        self.toggle_button.setText(f"  {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_button.setSizePolicy(
            self.toggle_button.sizePolicy().horizontalPolicy(),
            self.toggle_button.sizePolicy().verticalPolicy(),
        )
        self.toggle_button.setMinimumHeight(30)
        self.toggle_button.toggled.connect(self.set_expanded)
        root.addWidget(self.toggle_button)

        self.content_widget = QWidget()
        self.content_widget.setVisible(expanded)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(12, 4, 0, 4)
        self.content_layout.setSpacing(6)
        root.addWidget(self.content_widget)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content_widget.setVisible(expanded)
