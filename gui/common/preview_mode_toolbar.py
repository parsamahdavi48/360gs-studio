"""Shared single-image / thumbnail preview mode controls."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

from gui import i18n
from gui.common.icons import single_preview_icon, thumbnail_preview_icon

PREVIEW_MODE_SINGLE = "single"
PREVIEW_MODE_THUMBNAILS = "thumbnails"


class PreviewModeToolbar(QWidget):
    """Compact icon toolbar for switching between single and thumbnail previews."""

    mode_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        single_text_key: str = "MASK_PREVIEW_MODE_SINGLE",
        thumbnail_text_key: str = "MASK_PREVIEW_MODE_THUMBNAILS",
        single_tip_key: str | None = None,
        thumbnail_tip_key: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = PREVIEW_MODE_SINGLE
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        single_tip_key = single_tip_key or single_text_key
        thumbnail_tip_key = thumbnail_tip_key or thumbnail_text_key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.single_preview_btn = QToolButton()
        self.single_preview_btn.setObjectName("iconToolButton")
        self.single_preview_btn.setCheckable(True)
        self.single_preview_btn.setChecked(True)
        self.single_preview_btn.setIcon(single_preview_icon())
        self.single_preview_btn.setToolTip(i18n.tip(single_tip_key))
        self.single_preview_btn.setAccessibleName(i18n.t(single_text_key))
        self.single_preview_btn.setFixedSize(28, 28)
        self.single_preview_btn.clicked.connect(
            lambda _checked=False: self.set_mode(PREVIEW_MODE_SINGLE, emit=True)
        )
        self.button_group.addButton(self.single_preview_btn)
        layout.addWidget(self.single_preview_btn)

        self.thumbnail_preview_btn = QToolButton()
        self.thumbnail_preview_btn.setObjectName("iconToolButton")
        self.thumbnail_preview_btn.setCheckable(True)
        self.thumbnail_preview_btn.setIcon(thumbnail_preview_icon())
        self.thumbnail_preview_btn.setToolTip(i18n.tip(thumbnail_tip_key))
        self.thumbnail_preview_btn.setAccessibleName(i18n.t(thumbnail_text_key))
        self.thumbnail_preview_btn.setFixedSize(28, 28)
        self.thumbnail_preview_btn.clicked.connect(
            lambda _checked=False: self.set_mode(PREVIEW_MODE_THUMBNAILS, emit=True)
        )
        self.button_group.addButton(self.thumbnail_preview_btn)
        layout.addWidget(self.thumbnail_preview_btn)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str, *, emit: bool = False) -> None:
        if mode not in {PREVIEW_MODE_SINGLE, PREVIEW_MODE_THUMBNAILS}:
            return
        changed = mode != self._mode
        self._mode = mode
        self.single_preview_btn.setChecked(mode == PREVIEW_MODE_SINGLE)
        self.thumbnail_preview_btn.setChecked(mode == PREVIEW_MODE_THUMBNAILS)
        if changed and emit:
            self.mode_changed.emit(mode)
