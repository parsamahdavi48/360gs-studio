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

_SELECT_ALL_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <rect x="4" y="4" width="6" height="6" rx="1.2" fill="#2dd4bf"/>
  <rect x="14" y="4" width="6" height="6" rx="1.2" fill="#2dd4bf"/>
  <rect x="4" y="14" width="6" height="6" rx="1.2" fill="#2dd4bf"/>
  <rect x="14" y="14" width="6" height="6" rx="1.2" fill="#2dd4bf"/>
  <path d="m5.8 7 1.1 1.2L8.5 6" stroke="#101316" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="m15.8 7 1.1 1.2L18.5 6" stroke="#101316" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="m5.8 17 1.1 1.2 1.6-2.2" stroke="#101316" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="m15.8 17 1.1 1.2 1.6-2.2" stroke="#101316" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
""".strip()

_DESELECT_ALL_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <rect x="4" y="4" width="6" height="6" rx="1.2" stroke="#9ca3af" stroke-width="1.6"/>
  <rect x="14" y="4" width="6" height="6" rx="1.2" stroke="#9ca3af" stroke-width="1.6"/>
  <rect x="4" y="14" width="6" height="6" rx="1.2" stroke="#9ca3af" stroke-width="1.6"/>
  <rect x="14" y="14" width="6" height="6" rx="1.2" stroke="#9ca3af" stroke-width="1.6"/>
  <path d="m5.8 5.8 2.4 2.4m0-2.4L5.8 8.2" stroke="#e5e7eb" stroke-width="1.4"
        stroke-linecap="round"/>
  <path d="m15.8 5.8 2.4 2.4m0-2.4-2.4 2.4" stroke="#e5e7eb" stroke-width="1.4"
        stroke-linecap="round"/>
  <path d="m5.8 15.8 2.4 2.4m0-2.4-2.4 2.4" stroke="#e5e7eb" stroke-width="1.4"
        stroke-linecap="round"/>
  <path d="m15.8 15.8 2.4 2.4m0-2.4-2.4 2.4" stroke="#e5e7eb" stroke-width="1.4"
        stroke-linecap="round"/>
</svg>
""".strip()

_SINGLE_PREVIEW_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <rect x="4" y="5" width="16" height="14" rx="1.8" stroke="#e5e7eb" stroke-width="1.8"/>
  <path d="M7 16.2 10.2 12.8 12.5 15.1 14.2 13.2 17 16.2" stroke="#2dd4bf"
        stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="9" cy="9" r="1.3" fill="#2dd4bf"/>
</svg>
""".strip()

_THUMBNAIL_PREVIEW_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <rect x="4" y="5" width="6" height="5" rx="1.1" stroke="#e5e7eb" stroke-width="1.6"/>
  <rect x="14" y="5" width="6" height="5" rx="1.1" stroke="#e5e7eb" stroke-width="1.6"/>
  <rect x="4" y="14" width="6" height="5" rx="1.1" stroke="#e5e7eb" stroke-width="1.6"/>
  <rect x="14" y="14" width="6" height="5" rx="1.1" stroke="#e5e7eb" stroke-width="1.6"/>
  <path d="M6.2 8.1 7.3 7 8.6 8.1M16.2 17.1 17.3 16 18.6 17.1" stroke="#2dd4bf"
        stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
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


def select_all_icon(size: int = 18) -> QIcon:
    return svg_icon(_SELECT_ALL_SVG, size)


def deselect_all_icon(size: int = 18) -> QIcon:
    return svg_icon(_DESELECT_ALL_SVG, size)


def single_preview_icon(size: int = 18) -> QIcon:
    return svg_icon(_SINGLE_PREVIEW_SVG, size)


def thumbnail_preview_icon(size: int = 18) -> QIcon:
    return svg_icon(_THUMBNAIL_PREVIEW_SVG, size)
