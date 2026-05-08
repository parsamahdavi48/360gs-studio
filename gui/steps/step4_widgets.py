"""Small Step 4 widgets shared by the main panel and mixins."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QStackedWidget, QWidget


class CurrentPageStack(QStackedWidget):
    """Stacked widget whose size hint follows only the visible page."""

    def sizeHint(self) -> QSize:
        widget = self.currentWidget()
        return widget.sizeHint() if widget is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        widget = self.currentWidget()
        return widget.minimumSizeHint() if widget is not None else super().minimumSizeHint()


class ElidedPathLabel(QLabel):
    """Keep long paths on one line while preserving the full path internally."""

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = text or "-"
        self._apply_elide()

    def full_text(self) -> str:
        return self._full_text

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        return QSize(0, base.height())

    def minimumSizeHint(self) -> QSize:
        base = super().minimumSizeHint()
        return QSize(0, base.height())

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        width = max(0, self.contentsRect().width())
        text = self._full_text
        if width > 0:
            text = QFontMetrics(self.font()).elidedText(text, Qt.ElideMiddle, width)
        QLabel.setText(self, text)
