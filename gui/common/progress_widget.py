"""プログレスバー + ステータスラベル"""
from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from gui import i18n, theme


class ReadableProgressBar(QProgressBar):
    """テーマ色を保ったまま数値だけ背景に合わせて描き分ける進捗バー。"""

    _TRACK_RADIUS = 4
    _CHUNK_RADIUS = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextVisible(False)
        font = self.font()
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self.minimum() == 0 and self.maximum() == 0:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setFont(self.font())

        outer_rect = self.rect().adjusted(0, 0, -1, -1)
        content_rect = outer_rect.adjusted(1, 1, -1, -1)
        track_path = QPainterPath()
        track_path.addRoundedRect(QRectF(outer_rect), self._TRACK_RADIUS, self._TRACK_RADIUS)

        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.setBrush(QColor(theme.BG_INPUT))
        painter.drawPath(track_path)

        fraction = self._progress_fraction()
        chunk_rect = self._chunk_rect(content_rect, fraction)
        if not chunk_rect.isEmpty():
            gradient = QLinearGradient(content_rect.topLeft(), content_rect.topRight())
            gradient.setColorAt(0, QColor(theme.ACCENT))
            gradient.setColorAt(1, QColor(theme.ACCENT_HOVER))

            chunk_path = QPainterPath()
            chunk_path.addRoundedRect(QRectF(chunk_rect), self._CHUNK_RADIUS, self._CHUNK_RADIUS)
            painter.save()
            painter.setClipPath(track_path)
            painter.fillPath(chunk_path, gradient)
            painter.restore()

        text = self.text()
        if text:
            self._draw_progress_text(painter, outer_rect, chunk_rect, text)

    def _progress_fraction(self) -> float:
        span = self.maximum() - self.minimum()
        if span <= 0:
            return 0.0
        value = min(max(self.value(), self.minimum()), self.maximum())
        return (value - self.minimum()) / span

    def _chunk_rect(self, content_rect: QRect, fraction: float) -> QRect:
        width = int(round(content_rect.width() * max(0.0, min(1.0, fraction))))
        if width <= 0:
            return QRect()
        if self.invertedAppearance():
            return QRect(
                content_rect.right() - width + 1,
                content_rect.top(),
                width,
                content_rect.height(),
            )
        return QRect(content_rect.left(), content_rect.top(), width, content_rect.height())

    def _draw_progress_text(
        self,
        painter: QPainter,
        text_rect: QRect,
        chunk_rect: QRect,
        text: str,
    ) -> None:
        painter.save()
        painter.setPen(QColor(theme.TEXT))
        painter.drawText(text_rect, Qt.AlignCenter, text)
        painter.restore()

        if chunk_rect.isEmpty():
            return

        painter.save()
        painter.setClipRect(chunk_rect)
        painter.setPen(QColor(theme.BG_INPUT))
        painter.drawText(text_rect, Qt.AlignCenter, text)
        painter.restore()


class ProgressWidget(QWidget):
    """プログレスバーとステータスラベルのセット"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.bar = ReadableProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        layout.addWidget(self.bar, stretch=1)

        self.status_label = QLabel(i18n.STATUS_IDLE)
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setMinimumWidth(200)
        layout.addWidget(self.status_label)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(min(done, total))
        else:
            self.bar.setRange(0, 0)

    def reset(self) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.status_label.setText(i18n.STATUS_IDLE)
