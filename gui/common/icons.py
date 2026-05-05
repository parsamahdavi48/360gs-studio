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

_PLUS_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M12 5v14M5 12h14" stroke="#e5e7eb" stroke-width="2.2"
        stroke-linecap="round"/>
</svg>
""".strip()

_MINUS_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M5 12h14" stroke="#e5e7eb" stroke-width="2.2" stroke-linecap="round"/>
</svg>
""".strip()

_DELETE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M6 7h12M10 7V5h4v2M8 7l.6 12h6.8L16 7" stroke="#e5e7eb"
        stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
""".strip()

_FOLDER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M3.8 7.4h6l1.8 2.2h8.6v7.9a2.1 2.1 0 0 1-2.1 2.1H5.9a2.1 2.1 0 0 1-2.1-2.1V7.4Z"
        stroke="#e5e7eb" stroke-width="1.8" stroke-linejoin="round"/>
  <path d="M3.8 9.6h16.4" stroke="#2dd4bf" stroke-width="1.8" stroke-linecap="round"/>
</svg>
""".strip()

_FILE_PICKER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M7 3.8h6.2L17 7.6v12.6H7V3.8Z" stroke="#e5e7eb" stroke-width="1.8"
        stroke-linejoin="round"/>
  <path d="M13.1 4.1v3.8h3.8" stroke="#e5e7eb" stroke-width="1.8" stroke-linejoin="round"/>
  <path d="M9.4 15.6h5.2M12 13v5.2" stroke="#2dd4bf" stroke-width="1.8"
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

_MASK_OVERLAY_ON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M3.8 12s3-5.2 8.2-5.2 8.2 5.2 8.2 5.2-3 5.2-8.2 5.2S3.8 12 3.8 12Z"
        stroke="#e5e7eb" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="12" cy="12" r="2.7" fill="#ef4444"/>
  <circle cx="12" cy="12" r="1.1" fill="#101316"/>
</svg>
""".strip()

_MASK_OVERLAY_OFF_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="M4.2 4.2 19.8 19.8" stroke="#e5e7eb" stroke-width="1.9"
        stroke-linecap="round"/>
  <path d="M9.1 6.9A8.6 8.6 0 0 1 12 6.8c5.2 0 8.2 5.2 8.2 5.2a13.2 13.2 0 0 1-2.4 2.9"
        stroke="#9ca3af" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M14.2 16.9a8.5 8.5 0 0 1-2.2.3C6.8 17.2 3.8 12 3.8 12a13.4 13.4 0 0 1 2.5-3"
        stroke="#9ca3af" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M10 10a2.7 2.7 0 0 0 3.8 3.8" stroke="#ef4444" stroke-width="1.6"
        stroke-linecap="round"/>
</svg>
""".strip()

_PREVIEW_TEMP_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <rect x="4" y="5" width="16" height="14" rx="1.8" stroke="#e5e7eb" stroke-width="1.7"/>
  <path d="M7 16.3 10 13.2 12.4 15.2 14.3 12.8 17.2 16.3" stroke="#2dd4bf"
        stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M15.8 6.7v3.8M13.9 8.6h3.8" stroke="#2dd4bf" stroke-width="1.7"
        stroke-linecap="round"/>
  <path d="M6.2 7.2h5.1" stroke="#ef4444" stroke-width="1.8" stroke-linecap="round"/>
</svg>
""".strip()

_PREVIEW_SAVED_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <rect x="4" y="5" width="16" height="14" rx="1.8" stroke="#9ca3af" stroke-width="1.7"/>
  <path d="M7 16.3 10 13.2 12.4 15.2 14.3 12.8 17.2 16.3" stroke="#9ca3af"
        stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M8 8h8M8 11h5.5" stroke="#ef4444" stroke-width="1.7" stroke-linecap="round"/>
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


def plus_icon(size: int = 18) -> QIcon:
    return svg_icon(_PLUS_SVG, size)


def minus_icon(size: int = 18) -> QIcon:
    return svg_icon(_MINUS_SVG, size)


def delete_icon(size: int = 18) -> QIcon:
    return svg_icon(_DELETE_SVG, size)


def folder_icon(size: int = 18) -> QIcon:
    return svg_icon(_FOLDER_SVG, size)


def file_picker_icon(size: int = 18) -> QIcon:
    return svg_icon(_FILE_PICKER_SVG, size)


def single_preview_icon(size: int = 18) -> QIcon:
    return svg_icon(_SINGLE_PREVIEW_SVG, size)


def thumbnail_preview_icon(size: int = 18) -> QIcon:
    return svg_icon(_THUMBNAIL_PREVIEW_SVG, size)


def mask_overlay_on_icon(size: int = 18) -> QIcon:
    return svg_icon(_MASK_OVERLAY_ON_SVG, size)


def mask_overlay_off_icon(size: int = 18) -> QIcon:
    return svg_icon(_MASK_OVERLAY_OFF_SVG, size)


def preview_temp_icon(size: int = 18) -> QIcon:
    return svg_icon(_PREVIEW_TEMP_SVG, size)


def preview_saved_icon(size: int = 18) -> QIcon:
    return svg_icon(_PREVIEW_SAVED_SVG, size)
