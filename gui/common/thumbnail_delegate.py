"""Shared thumbnail delegate with explicit selection styling."""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from gui import theme

_ITEM_MARGIN = 5
_LABEL_HEIGHT = 22
_LABEL_GAP = 4
_SELECTION_BORDER_WIDTH = 3
_CURRENT_BORDER_WIDTH = 1


class ThumbnailSelectionDelegate(QStyledItemDelegate):
    """Paint thumbnails without the default translucent selection overlay."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # noqa: ANN001
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        selected = bool(opt.state & QStyle.State_Selected)
        current = _is_current_index(opt, index)
        item_rect = opt.rect.adjusted(3, 3, -3, -3)
        label_rect = _label_rect(item_rect)
        image_area = _image_area_rect(item_rect, label_rect)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        icon = _icon_from_option(opt, index)
        image_rect = _image_rect(image_area, icon, opt.decorationSize)
        if icon is not None and not image_rect.isEmpty():
            icon.paint(painter, image_rect, Qt.AlignCenter)
            self.paint_thumbnail_overlay(painter, opt, index, image_rect)
            self._paint_selection_frame(painter, image_rect, selected=selected, current=current)
        self._paint_label(painter, opt.text, label_rect, selected=selected)
        painter.restore()

    def paint_thumbnail_overlay(
        self,
        _painter: QPainter,
        _option: QStyleOptionViewItem,
        _index,  # noqa: ANN001
        _image_rect: QRect,
    ) -> None:
        """Hook for step-specific overlays painted on top of the thumbnail image."""

    def _paint_selection_frame(
        self,
        painter: QPainter,
        image_rect: QRect,
        *,
        selected: bool,
        current: bool,
    ) -> None:
        if selected:
            frame = image_rect.adjusted(-3, -3, 3, 3)
            painter.setPen(QPen(QColor(theme.ACCENT), _SELECTION_BORDER_WIDTH))
            painter.drawRect(frame.adjusted(1, 1, -1, -1))
        if current:
            frame = image_rect.adjusted(-6, -6, 6, 6)
            painter.setPen(QPen(QColor(theme.ACCENT_HOVER), _CURRENT_BORDER_WIDTH))
            painter.drawRect(frame.adjusted(1, 1, -1, -1))

    def _paint_label(self, painter: QPainter, text: str, label_rect: QRect, *, selected: bool) -> None:
        if label_rect.isEmpty():
            return
        metrics = QFontMetrics(painter.font())
        text_rect = label_rect.adjusted(4, 0, -4, 0)
        display_text = metrics.elidedText(text, Qt.ElideRight, max(1, text_rect.width()))
        if selected:
            painter.fillRect(label_rect.adjusted(0, 1, 0, -1), QColor(theme.ACCENT))
            painter.setPen(QColor(theme.BG_INPUT))
        else:
            painter.setPen(QColor(theme.TEXT))
        painter.drawText(text_rect, Qt.AlignCenter, display_text)


def _is_current_index(option: QStyleOptionViewItem, index) -> bool:  # noqa: ANN001
    widget = option.widget
    if widget is None:
        return False
    selection_model_getter = getattr(widget, "selectionModel", None)
    if not callable(selection_model_getter):
        return False
    selection_model = selection_model_getter()
    if selection_model is None:
        return False
    return selection_model.currentIndex() == index


def _label_rect(item_rect: QRect) -> QRect:
    if item_rect.isEmpty():
        return QRect()
    return QRect(
        item_rect.left() + _ITEM_MARGIN,
        item_rect.bottom() - _LABEL_HEIGHT + 1,
        max(1, item_rect.width() - (_ITEM_MARGIN * 2)),
        _LABEL_HEIGHT,
    )


def _image_area_rect(item_rect: QRect, label_rect: QRect) -> QRect:
    if item_rect.isEmpty() or label_rect.isEmpty():
        return QRect()
    top = item_rect.top() + _ITEM_MARGIN
    bottom = label_rect.top() - _LABEL_GAP
    return QRect(
        item_rect.left() + _ITEM_MARGIN,
        top,
        max(1, item_rect.width() - (_ITEM_MARGIN * 2)),
        max(1, bottom - top),
    )


def _icon_from_option(option: QStyleOptionViewItem, index) -> QIcon | None:  # noqa: ANN001
    icon = option.icon
    if isinstance(icon, QIcon) and not icon.isNull():
        return icon
    icon = index.data(Qt.DecorationRole)
    if isinstance(icon, QIcon) and not icon.isNull():
        return icon
    return None


def _image_rect(image_area: QRect, icon: QIcon | None, requested_size: QSize) -> QRect:
    if image_area.isEmpty() or icon is None:
        return QRect()
    size = QSize(requested_size)
    if size.isEmpty():
        size = icon.actualSize(image_area.size())
    if size.isEmpty():
        return QRect()

    scale = min(
        image_area.width() / max(1, size.width()),
        image_area.height() / max(1, size.height()),
        1.0,
    )
    width = max(1, int(round(size.width() * scale)))
    height = max(1, int(round(size.height() * scale)))
    x = image_area.left() + (image_area.width() - width) // 2
    y = image_area.top() + (image_area.height() - height) // 2
    return QRect(x, y, width, height)
