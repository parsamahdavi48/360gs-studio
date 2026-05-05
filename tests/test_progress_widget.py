import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui import i18n
from gui.common.progress_widget import ProgressWidget


def _app():
    return QApplication.instance() or QApplication([])


def test_progress_widget_shows_elapsed_time_on_bar() -> None:
    _app()
    widget = ProgressWidget()

    widget.start_phase()

    assert widget.bar.minimum() == 0
    assert widget.bar.maximum() == 0
    assert i18n.t("ELAPSED_TIME") in widget.bar._overlay_text

    widget.set_progress(2, 10)

    assert widget.bar.minimum() == 0
    assert widget.bar.maximum() == 10
    assert "2/10" in widget.bar._overlay_text
    assert i18n.t("ELAPSED_TIME") in widget.bar._overlay_text

    widget.finish_phase()

    assert i18n.t("ELAPSED_TIME") in widget.bar._overlay_text

    widget.reset()

    assert widget.bar._overlay_text == ""
    assert widget.status_label.text() == i18n.STATUS_IDLE


def test_progress_widget_finish_phase_stops_indeterminate_animation() -> None:
    _app()
    widget = ProgressWidget()

    widget.start_phase()
    widget.finish_phase()

    assert widget.bar.minimum() == 0
    assert widget.bar.maximum() == 100
    assert widget.bar.value() == 100
    assert "100/100" in widget.bar._overlay_text
    assert i18n.t("ELAPSED_TIME") in widget.bar._overlay_text


def test_progress_widget_finish_phase_can_stop_indeterminate_without_completion() -> None:
    _app()
    widget = ProgressWidget()

    widget.start_phase()
    widget.finish_phase(complete=False)

    assert widget.bar.minimum() == 0
    assert widget.bar.maximum() == 100
    assert widget.bar.value() == 0
    assert "0/100" in widget.bar._overlay_text
    assert i18n.t("ELAPSED_TIME") in widget.bar._overlay_text
