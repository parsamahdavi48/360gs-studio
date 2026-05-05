"""Shared single-image / thumbnail preview mode controls."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

from gui import i18n
from gui.common.icons import perspective_preview_icon, single_preview_icon, thumbnail_preview_icon

PREVIEW_MODE_PERSPECTIVE = "perspective"
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
        perspective_text_key: str = "PREVIEW_PROJECTION_TOGGLE",
        single_tip_key: str | None = None,
        thumbnail_tip_key: str | None = None,
        perspective_tip_key: str | None = None,
        include_perspective: bool = False,
    ) -> None:
        super().__init__(parent)
        self._mode = PREVIEW_MODE_SINGLE
        self._allowed_modes = {PREVIEW_MODE_SINGLE, PREVIEW_MODE_THUMBNAILS}
        if include_perspective:
            self._allowed_modes.add(PREVIEW_MODE_PERSPECTIVE)
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        single_tip_key = single_tip_key or single_text_key
        thumbnail_tip_key = thumbnail_tip_key or thumbnail_text_key
        perspective_tip_key = perspective_tip_key or perspective_text_key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.perspective_preview_btn: QToolButton | None = None
        if include_perspective:
            self.perspective_preview_btn = QToolButton()
            self.perspective_preview_btn.setObjectName("iconToolButton")
            self.perspective_preview_btn.setCheckable(True)
            self.perspective_preview_btn.setIcon(perspective_preview_icon())
            self.perspective_preview_btn.setToolTip(i18n.tip(perspective_tip_key))
            self.perspective_preview_btn.setAccessibleName(i18n.t(perspective_text_key))
            self.perspective_preview_btn.setFixedSize(28, 28)
            self.perspective_preview_btn.clicked.connect(
                lambda _checked=False: self.set_mode(PREVIEW_MODE_PERSPECTIVE, emit=True)
            )
            self.button_group.addButton(self.perspective_preview_btn)
            layout.addWidget(self.perspective_preview_btn)

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
        if mode not in self._allowed_modes:
            return
        changed = mode != self._mode
        self._mode = mode
        if self.perspective_preview_btn is not None:
            self.perspective_preview_btn.setChecked(mode == PREVIEW_MODE_PERSPECTIVE)
        self.single_preview_btn.setChecked(mode == PREVIEW_MODE_SINGLE)
        self.thumbnail_preview_btn.setChecked(mode == PREVIEW_MODE_THUMBNAILS)
        if changed and emit:
            self.mode_changed.emit(mode)
