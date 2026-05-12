import os
import sys
import time
from pathlib import Path

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


def test_process_runner_exposes_phase_queue_position() -> None:
    app = _app()
    runner = ProcessRunner()
    positions: list[tuple[str, int, int]] = []
    finished: list[bool] = []

    runner.phase_started.connect(
        lambda phase: positions.append((phase, runner.phase_index, runner.queue_total))
    )
    runner.queue_finished.connect(finished.append)

    runner.start_queue(
        [
            ("first", [sys.executable, "-c", "print('one')"]),
            ("second", [sys.executable, "-c", "print('two')"]),
        ]
    )

    try:
        assert _process_events_until(app, lambda: bool(finished))
        assert finished == [True]
        assert positions == [("first", 1, 2), ("second", 2, 2)]
    finally:
        if runner.is_running():
            runner.cancel()
            _process_events_until(app, lambda: not runner.is_running())


def test_process_runner_writes_phase_logs(tmp_path: Path) -> None:
    app = _app()
    runner = ProcessRunner()
    finished: list[bool] = []
    log_paths: list[str] = []

    runner.phase_log_started.connect(lambda _phase, path: log_paths.append(path))
    runner.queue_finished.connect(finished.append)

    runner.start_queue(
        [("sphere phase/1", [sys.executable, "-c", "print('alpha'); print('beta')"])],
        log_dir=tmp_path,
    )

    try:
        assert _process_events_until(app, lambda: bool(finished))
        assert finished == [True]
        assert len(log_paths) == 1
        log_path = Path(log_paths[0])
        assert log_path.parent == tmp_path
        assert log_path.name.endswith("_sphere_phase_1.log")
        text = log_path.read_text(encoding="utf-8")
        assert "$ " in text
        assert "alpha" in text
        assert "beta" in text
        assert "exit_code=0" in text
    finally:
        if runner.is_running():
            runner.cancel()
            _process_events_until(app, lambda: not runner.is_running())
