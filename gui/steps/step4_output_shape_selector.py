"""Step 4 output-shape selector widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QWidget


class OutputShapeSelector(QWidget):
    """Two-choice selector with the small API surface CubemapStep already uses."""

    currentIndexChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("radioOptionRow")
        self._items: list[tuple[str, object]] = []
        self._buttons: list[QRadioButton] = []
        self._current_index = -1
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)
        self._layout.addStretch()

    def addItem(self, text: str, data: object) -> None:
        index = len(self._items)
        button = QRadioButton(text)
        button.setObjectName("optionRadio")
        button.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        button.setToolTip(self.toolTip())
        button.clicked.connect(lambda _checked=False, idx=index: self.setCurrentIndex(idx))
        self._group.addButton(button)
        self._items.append((text, data))
        self._buttons.append(button)
        self._layout.insertWidget(index, button)
        if self._current_index < 0:
            self.setCurrentIndex(index)

    def count(self) -> int:
        return len(self._items)

    def currentData(self) -> object | None:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def currentIndex(self) -> int:
        return self._current_index

    def findData(self, data: object) -> int:
        for index, (_text, item_data) in enumerate(self._items):
            if item_data == data:
                return index
        return -1

    def itemText(self, index: int) -> str:
        if 0 <= index < len(self._items):
            return self._items[index][0]
        return ""

    def isItemEnabled(self, index: int) -> bool:  # noqa: N802 - Qt-style API
        if 0 <= index < len(self._buttons):
            return self._buttons[index].isEnabled()
        return False

    def itemToolTip(self, index: int) -> str:  # noqa: N802 - Qt-style API
        if 0 <= index < len(self._buttons):
            return self._buttons[index].toolTip()
        return ""

    def setItemEnabled(self, index: int, enabled: bool) -> None:  # noqa: N802 - Qt-style API
        if 0 <= index < len(self._buttons):
            self._buttons[index].setEnabled(enabled)

    def setItemToolTip(self, index: int, tooltip: str) -> None:  # noqa: N802 - Qt-style API
        if 0 <= index < len(self._buttons):
            self._buttons[index].setToolTip(tooltip)

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items):
            return
        changed = index != self._current_index
        self._current_index = index
        button = self._buttons[index]
        if not button.isChecked():
            button.setChecked(True)
        if changed:
            self.currentIndexChanged.emit(index)

    def setToolTip(self, tooltip: str) -> None:  # noqa: N802 - Qt API
        super().setToolTip(tooltip)
        for button in self._buttons:
            button.setToolTip(tooltip)
