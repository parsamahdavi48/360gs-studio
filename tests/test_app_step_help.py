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
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/extract_frames_gui.ja.md"
    )
    assert (
        step_help_url(1, lang="en")
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/review_frames_gui.md"
    )
    assert (
        step_help_url(2, lang="ja_JP")
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/mask_tools_gui.ja.md"
    )
    assert (
        step_help_url(3, lang="en_US")
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/cubemap_tools_gui.md"
    )
    assert (
        step_help_url(4, lang="ja")
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/cubemap_tools_gui.ja.md"
    )
    assert (
        step_help_url(5, lang="ja")
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/training_gui.ja.md"
    )
    assert (
        step_help_url(5, lang="en")
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/training_gui.md"
    )
    assert step_help_url(6, lang="en") == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/perspective_export.md"
    assert (
        step_help_url(6, lang="fa")
        == "https://github.com/parsamahdavi48/360gs-studio/blob/main/doc/perspective_export.fa.md"
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


def test_step_header_uses_context_title_without_duplicate_scene_path(tmp_path: Path) -> None:
    _app()
    window = MainWindow(str(tmp_path))

    for index in range(len(window.steps)):
        window._set_current_step(index)
        assert window.step_subheader.text() == ""
        assert window.step_subheader.toolTip() == ""
        assert window.step_subheader.isHidden()

    window._set_current_step(3)
    assert window.step_header.text() == i18n.t("SFM_MENU_TITLE")
    assert window.step_back_btn.isHidden()

    window.sfm_step.card_grid.buttons["colmap"].click()
    assert window.step_header.text() == i18n.t("SFM_COLMAP_DETAIL_TITLE")
    assert not window.step_back_btn.isHidden()
    assert window.step_back_btn.text() == ""
    assert not window.step_back_btn.icon().isNull()
    assert window.step_back_btn.toolTip() == i18n.tip("SFM_BACK_TO_ROUTES")

    window.step_back_btn.click()
    assert window.sfm_step.current_route() == ""
    assert window.step_header.text() == i18n.t("SFM_MENU_TITLE")
    assert window.step_back_btn.isHidden()

    window._set_current_step(4)
    assert window.step_header.text() == i18n.t("DATASET_MENU_TITLE")
    assert window.step_back_btn.isHidden()

    window.dataset_step.card_grid.buttons["realityscan_lfs"].click()
    assert window.step_header.text() == i18n.t("DATASET_TOOL_RS_LFS_TITLE")
    assert not window.step_back_btn.isHidden()
    assert window.step_back_btn.text() == ""
    assert not window.step_back_btn.icon().isNull()
    assert window.step_back_btn.toolTip() == i18n.t("DATASET_BACK_TO_MENU")

    window.step_back_btn.click()
    assert window.dataset_step.current_tool() == "menu"
    assert window.step_header.text() == i18n.t("DATASET_MENU_TITLE")
    assert window.step_back_btn.isHidden()

    window.close()
