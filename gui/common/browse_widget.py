"""パス入力 + 参照ボタン"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
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
        button_position: str = "side",
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._filter = filter_str
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._button_position = button_position

        layout = QVBoxLayout(self) if button_position == "below" else QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.line_edit = QLineEdit()
        self.line_edit.setMinimumWidth(0)
        self.line_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        if placeholder:
            self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.textChanged.connect(self.path_changed.emit)
        if button_position == "below":
            layout.addWidget(self.line_edit)
        else:
            layout.addWidget(self.line_edit, stretch=1)

        btn = QPushButton(BROWSE)
        btn.setFixedWidth(104)
        btn.clicked.connect(self._browse)
        if button_position == "below":
            layout.addWidget(btn, alignment=Qt.AlignLeft)
        else:
            layout.addWidget(btn)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def set_text(self, path: str) -> None:
        self.line_edit.setText(path)

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
