"""Delegate for painting mask overlays over cached base thumbnails."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QStyleOptionViewItem

from gui.common.thumbnail_delegate import ThumbnailSelectionDelegate
from gui.common.thumbnail_list_model import THUMBNAIL_PAYLOAD_ROLE


class MaskThumbnailDelegate(ThumbnailSelectionDelegate):
    """Paint cached mask overlays without changing thumbnail model cache keys."""

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._overlay_visible = True

    def set_overlay_visible(self, visible: bool) -> bool:
        visible = bool(visible)
        if visible == self._overlay_visible:
            return False
        self._overlay_visible = visible
        return True

    def overlay_visible(self) -> bool:
        return self._overlay_visible

    def paint_thumbnail_overlay(
        self,
        painter: QPainter,
        _option: QStyleOptionViewItem,
        index,  # noqa: ANN001
        image_rect,
    ) -> None:
        if not self._overlay_visible:
            return

        payload = index.data(THUMBNAIL_PAYLOAD_ROLE)
        overlay = getattr(payload, "overlay", None)
        base_image = getattr(payload, "image", None)
        if not isinstance(overlay, QImage) or overlay.isNull():
            return
        if not isinstance(base_image, QImage) or base_image.isNull():
            return

        if image_rect.isEmpty():
            return
        painter.drawImage(image_rect, _red_overlay_from_alpha(overlay))


def _red_overlay_from_alpha(alpha_mask: QImage) -> QImage:
    overlay = QImage(alpha_mask.size(), QImage.Format_ARGB32_Premultiplied)
    overlay.fill(Qt.transparent)
    painter = QPainter(overlay)
    painter.fillRect(overlay.rect(), QColor(255, 0, 0))
    painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    painter.drawImage(overlay.rect(), alpha_mask)
    painter.end()
    return overlay
