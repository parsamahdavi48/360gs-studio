from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.app import app_icon


def _app():
    return QApplication.instance() or QApplication([])


def test_app_icon_assets_exist() -> None:
    assets_dir = Path(__file__).resolve().parent.parent / "gui" / "assets"

    assert (assets_dir / "app_icon.svg").is_file()
    assert (assets_dir / "app_icon.ico").is_file()


def test_app_icon_loads_as_qicon() -> None:
    _app()

    icon = app_icon()

    assert not icon.isNull()
    assert icon.availableSizes()
