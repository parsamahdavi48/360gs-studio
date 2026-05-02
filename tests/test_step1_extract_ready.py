import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui import i18n
from gui.app import MainWindow
from gui.steps.step1_extract import ExtractStep


def _app():
    return QApplication.instance() or QApplication([])


def _video_info() -> dict:
    return {
        "width": 7680,
        "height": 3840,
        "fps": 29.97,
        "duration_sec": 10.0,
        "total_frames": 300,
    }


def _make_ready(step: ExtractStep, video: Path, scene: Path) -> None:
    step.video_browse.line_edit.blockSignals(True)
    step.video_browse.set_text(str(video))
    step.video_browse.line_edit.blockSignals(False)
    step.set_scene_dir(str(scene))
    step.video_info = _video_info()
    step._update_ready_status()


def test_extract_run_disabled_until_video_is_selected() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert not step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_NO_VIDEO")
    assert step.primary_action_tooltip() == i18n.t("EXTRACT_READY_NO_VIDEO")


def test_extract_run_enabled_when_required_inputs_are_ready(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())

    _make_ready(step, video, tmp_path)

    assert step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_OK")
    assert step.primary_action_tooltip() == i18n.tip("RUN")


def test_extract_run_disabled_for_invalid_analysis_width(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)

    step.analysis_width_edit.setText("wide")

    assert not step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_BAD_ANALYSIS_WIDTH")


def test_main_window_run_button_follows_extract_readiness(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    window = MainWindow(str(tmp_path))

    assert window.stack.currentIndex() == 0
    assert not window.run_btn.isEnabled()

    _make_ready(window.step1, video, tmp_path)

    assert window.run_btn.isEnabled()
    window.close()
