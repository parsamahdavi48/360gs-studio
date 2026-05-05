import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.app import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def _process_events_until(app: QApplication, predicate, timeout_sec: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def test_main_window_locks_workflow_controls_while_process_runs(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(str(tmp_path))

    try:
        assert window.scene_browse.isEnabled()
        assert window.clear_scene_btn.isEnabled()
        assert window.stack.isEnabled()
        assert all(btn.isEnabled() for btn in window.step_buttons)

        window.runner.start_queue([("sleep", [sys.executable, "-c", "import time; time.sleep(0.2)"])])
        app.processEvents()

        assert window.runner.is_running()
        assert window.cancel_btn.isEnabled()
        assert not window.scene_browse.isEnabled()
        assert not window.clear_scene_btn.isEnabled()
        assert not window.stack.isEnabled()
        assert all(not btn.isEnabled() for btn in window.step_buttons)

        assert _process_events_until(app, lambda: not window.runner.is_running())

        assert window.scene_browse.isEnabled()
        assert window.clear_scene_btn.isEnabled()
        assert window.stack.isEnabled()
        assert all(btn.isEnabled() for btn in window.step_buttons)
    finally:
        window.shutdown()
        if window.runner.is_running():
            window.runner.cancel()
            _process_events_until(app, lambda: not window.runner.is_running())
