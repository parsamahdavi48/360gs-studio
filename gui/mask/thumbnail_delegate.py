"""Delegate for painting mask overlays over cached base thumbnails."""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from gui.common.thumbnail_list_model import THUMBNAIL_PAYLOAD_ROLE


class MaskThumbnailDelegate(QStyledItemDelegate):
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

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # noqa: ANN001
        super().paint(painter, option, index)
        if not self._overlay_visible:
            return

        payload = index.data(THUMBNAIL_PAYLOAD_ROLE)
        overlay = getattr(payload, "overlay", None)
        base_image = getattr(payload, "image", None)
        if not isinstance(overlay, QImage) or overlay.isNull():
            return
        if not isinstance(base_image, QImage) or base_image.isNull():
            return

        target = _decoration_image_rect(option, index, base_image.size())
        if target.isEmpty():
            return
        painter.drawImage(target, _red_overlay_from_alpha(overlay))


def _decoration_image_rect(
    option: QStyleOptionViewItem,
    index,  # noqa: ANN001
    image_size: QSize,
) -> QRect:
    opt = QStyleOptionViewItem(option)
    delegate = option.widget.itemDelegate(index) if option.widget is not None else None
    if isinstance(delegate, QStyledItemDelegate):
        delegate.initStyleOption(opt, index)

    widget = opt.widget
    style = widget.style() if widget is not None else QApplication.style()
    decoration_rect = style.subElementRect(QStyle.SE_ItemViewItemDecoration, opt, widget)
    if decoration_rect.isEmpty() or image_size.isEmpty():
        return QRect()

    scale = min(
        decoration_rect.width() / max(1, image_size.width()),
        decoration_rect.height() / max(1, image_size.height()),
    )
    width = max(1, int(round(image_size.width() * scale)))
    height = max(1, int(round(image_size.height() * scale)))
    x = decoration_rect.x()
    if opt.decorationAlignment & Qt.AlignHCenter:
        x += (decoration_rect.width() - width) // 2
    elif opt.decorationAlignment & Qt.AlignRight:
        x += decoration_rect.width() - width
    y = decoration_rect.y()
    if opt.decorationAlignment & Qt.AlignVCenter:
        y += (decoration_rect.height() - height) // 2
    elif opt.decorationAlignment & Qt.AlignBottom:
        y += decoration_rect.height() - height
    return QRect(x, y, width, height)


def _red_overlay_from_alpha(alpha_mask: QImage) -> QImage:
    overlay = QImage(alpha_mask.size(), QImage.Format_ARGB32_Premultiplied)
    overlay.fill(Qt.transparent)
    painter = QPainter(overlay)
    painter.fillRect(overlay.rect(), QColor(255, 0, 0))
    painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    painter.drawImage(overlay.rect(), alpha_mask)
    painter.end()
    return overlay
