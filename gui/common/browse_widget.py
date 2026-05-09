"""パス入力 + 参照ボタン"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractButton,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.common.icons import folder_icon
from gui.i18n import BROWSE


class BrowseWidget(QWidget):
    """QLineEdit + 参照ボタン。ディレクトリ or ファイル選択。"""

    path_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        mode: str = "dir",
        filter_str: str = "",
        placeholder: str = "",
        button_position: str = "side",
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._filter = filter_str
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._button_position = button_position

        layout = QVBoxLayout(self) if button_position == "below" else QHBoxLayout(self)
        self._layout = layout
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        self.line_edit = QLineEdit()
        self.line_edit.setMinimumWidth(0)
        self.line_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        if placeholder:
            self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.textChanged.connect(self.path_changed.emit)
        if button_position == "below":
            self._layout.addWidget(self.line_edit)
        else:
            self._layout.addWidget(self.line_edit, stretch=1)

        self.browse_button = QToolButton()
        self.browse_button.setObjectName("iconToolButton")
        self.browse_button.setIcon(folder_icon())
        self.browse_button.setToolTip(BROWSE)
        self.browse_button.setAccessibleName(BROWSE)
        self.browse_button.setFixedSize(32, 32)
        self.browse_button.clicked.connect(self.browse)
        if button_position == "below":
            self._layout.addWidget(self.browse_button, alignment=Qt.AlignLeft)
        else:
            self._layout.addWidget(self.browse_button)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def set_text(self, path: str) -> None:
        self.line_edit.setText(path)

    def browse(self) -> None:
        self._browse()

    def add_icon_button(
        self,
        icon: QIcon,
        tooltip: str,
        callback,
        *,
        accessible_name: str = "",
    ) -> QAbstractButton:
        button = QToolButton(self)
        button.setObjectName("iconToolButton")
        button.setIcon(icon)
        button.setToolTip(tooltip)
        button.setFixedSize(32, 32)
        if accessible_name:
            button.setAccessibleName(accessible_name)
        if callback is not None:
            button.clicked.connect(callback)
        if self._button_position == "below":
            self._layout.addWidget(button, alignment=Qt.AlignLeft)
        else:
            self._layout.addWidget(button)
        return button

    def _browse(self) -> None:
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "", self.text())
        elif self._mode == "files":
            paths, _ = QFileDialog.getOpenFileNames(self, "", self._dialog_start_path(), self._filter)
            path = "; ".join(paths)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "", self._dialog_start_path(), self._filter)
        if path:
            self.set_text(path)

    def _dialog_start_path(self) -> str:
        text = self.text()
        if ";" in text:
            return text.split(";", 1)[0].strip()
        return text
