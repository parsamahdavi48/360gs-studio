"""Small helpers for external documentation/repository links."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


def make_external_link(text: str, url: str, tooltip: str, object_name: str) -> QLabel:
    link = QLabel(f'<a href="{url}">{text}</a>')
    link.setObjectName(object_name)
    link.setOpenExternalLinks(True)
    link.setTextInteractionFlags(Qt.TextBrowserInteraction)
    link.setToolTip(tooltip)
    return link
