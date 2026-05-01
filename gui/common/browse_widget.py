"""パス入力 + 参照ボタン"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

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
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._filter = filter_str

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.line_edit = QLineEdit()
        if placeholder:
            self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.textChanged.connect(self.path_changed.emit)
        layout.addWidget(self.line_edit)

        btn = QPushButton(BROWSE)
        btn.setFixedWidth(104)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def set_text(self, path: str) -> None:
        self.line_edit.setText(path)

    def _browse(self) -> None:
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "", self.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "", self.text(), self._filter)
        if path:
            self.set_text(path)
