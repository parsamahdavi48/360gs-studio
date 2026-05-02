"""Small SVG icons used by the PySide GUI."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


_RESET_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M5.6 16.7A8.4 8.4 0 1 0 7.4 6.2" stroke="#e5e7eb" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M4.1 7.5 8.1 4.2 8.7 9.4Z" fill="#e5e7eb"/>
</svg>
""".strip()


@lru_cache(maxsize=16)
def svg_icon(svg: str, size: int = 18) -> QIcon:
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


def reset_icon(size: int = 18) -> QIcon:
    return svg_icon(_RESET_SVG, size)
