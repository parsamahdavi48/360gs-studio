import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.common.process_runner import ProcessRunner


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


def test_process_runner_is_running_when_phase_started() -> None:
    app = _app()
    runner = ProcessRunner()
    running_during_phase_start: list[bool] = []
    finished: list[bool] = []

    runner.phase_started.connect(lambda _phase: running_during_phase_start.append(runner.is_running()))
    runner.queue_finished.connect(finished.append)

    runner.start_queue([("sleep", [sys.executable, "-c", "import time; time.sleep(0.2)"])])

    try:
        assert running_during_phase_start == [True]
        assert runner.is_running()
        assert _process_events_until(app, lambda: bool(finished))
        assert finished == [True]
        assert not runner.is_running()
    finally:
        if runner.is_running():
            runner.cancel()
            _process_events_until(app, lambda: not runner.is_running())
