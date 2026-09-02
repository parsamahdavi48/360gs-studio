from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow

from gs360studio.i18n import LanguagePack, Translator, bundled_english_pack
from gs360studio.theme import bundled_midnight_theme
from gui.app import MainWindow
from gui.steps.perspective_export import PerspectiveExportStep


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_language_pack_fallback_and_placeholder_validation() -> None:
    fallback = bundled_english_pack()
    translated = LanguagePack("x", "Test", {"app.title": "Test", "jobs.interrupted": "Recovered {wrong} jobs"})
    issues = translated.validate_against(fallback)
    assert any("placeholder mismatch" in issue for issue in issues)
    assert Translator(fallback).text("jobs.interrupted", count=3) == "Recovered 3 interrupted job(s)"


def test_midnight_theme_contract() -> None:
    theme = bundled_midnight_theme()
    assert theme.theme_id == "midnight"
    assert theme.colors["primary"] != theme.colors["secondary"]
    assert theme.colors["background"].startswith("#")


def test_main_window_is_dockable_and_contains_perspective_workspace(tmp_path: Path) -> None:
    _app()
    window = MainWindow(str(tmp_path))
    try:
        assert isinstance(window, QMainWindow)
        assert isinstance(window.jobs_dock, QDockWidget)
        assert isinstance(window.steps[-1], PerspectiveExportStep)
        assert window.stack.count() == 7
        window._set_current_step(6)
        assert window.stack.currentWidget() is window.perspective_step
    finally:
        window.shutdown()
