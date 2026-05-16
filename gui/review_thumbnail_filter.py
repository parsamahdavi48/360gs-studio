"""Thumbnail filter menu button used by the Step 2 review widget."""
from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu, QToolButton, QWidget

from gui.common.icons import filter_icon


class ReviewThumbnailFilterButton(QToolButton):
    filter_changed = Signal(str)

    def __init__(self, filter_keys: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._actions: dict[str, QAction] = {}
        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)

        menu = QMenu(self)
        for key in filter_keys:
            action = QAction("", menu)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, filter_key=key: self.filter_changed.emit(filter_key))
            self._action_group.addAction(action)
            menu.addAction(action)
            self._actions[str(key)] = action

        self.setObjectName("iconToolButton")
        self.setProperty("hideMenuIndicator", "true")
        self.setCheckable(True)
        self.setIcon(filter_icon())
        self.setIconSize(QSize(20, 20))
        self.setFixedSize(36, 32)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMenu(menu)

    def set_filter_options(self, options: list[dict[str, str]], active_key: str, tooltip: str) -> None:
        option_by_key = {str(option["key"]): option for option in options}
        for key, action in self._actions.items():
            option = option_by_key.get(key, {"label": key})
            action.setText(str(option["label"]))
            action.setChecked(key == active_key)
        self.setToolTip(tooltip)
