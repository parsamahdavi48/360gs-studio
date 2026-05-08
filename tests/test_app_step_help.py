from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QToolButton

from gui import i18n
from gui.app import MainWindow, step_help_url


def _app():
    return QApplication.instance() or QApplication([])


def test_step_help_url_follows_language() -> None:
    assert (
        step_help_url(0, lang="ja")
        == "https://github.com/stechdrive/stechdrive-3dgs-utils/blob/main/doc/extract_frames_gui.ja.md"
    )
    assert (
        step_help_url(1, lang="en")
        == "https://github.com/stechdrive/stechdrive-3dgs-utils/blob/main/doc/review_frames_gui.md"
    )
    assert (
        step_help_url(2, lang="ja_JP")
        == "https://github.com/stechdrive/stechdrive-3dgs-utils/blob/main/doc/mask_tools_gui.ja.md"
    )
    assert (
        step_help_url(3, lang="en_US")
        == "https://github.com/stechdrive/stechdrive-3dgs-utils/blob/main/doc/cubemap_tools_gui.md"
    )


def test_main_window_step_help_button_opens_current_step_doc(tmp_path: Path, monkeypatch) -> None:
    _app()
    opened: list[str] = []
    monkeypatch.setattr("gui.app.QDesktopServices.openUrl", lambda url: opened.append(url.toString()) or True)
    window = MainWindow(str(tmp_path))

    assert isinstance(window.step_help_btn, QToolButton)
    assert window.step_help_btn.objectName() == "iconToolButton"
    assert window.step_help_btn.text() == ""
    assert not window.step_help_btn.icon().isNull()
    assert window.step_help_btn.toolTip() == i18n.tip("STEP_HELP_BUTTON")

    window._set_current_step(2)
    window.step_help_btn.click()

    assert opened == [step_help_url(2)]
    window.close()


def test_step_header_shows_scene_relative_work_root(tmp_path: Path) -> None:
    _app()
    window = MainWindow(str(tmp_path))

    expected = [
        f"{i18n.t('STEP_HEADER_OUTPUT')}: images/",
        f"{i18n.t('STEP_HEADER_TARGET')}: images/",
        f"{i18n.t('STEP_HEADER_OUTPUT')}: masks/",
        f"{i18n.t('STEP_HEADER_OUTPUT_ROOT')}: output/",
    ]
    for index, text in enumerate(expected):
        window._set_current_step(index)
        assert window.step_subheader.text() == text
        assert str(tmp_path) in window.step_subheader.toolTip()

    window.close()
