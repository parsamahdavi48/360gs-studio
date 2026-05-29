import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.app_job import AppJob
from core.apriltag_scale_job_spec import apriltag_scale_estimate_job
from core.cancellation import AppJobCancelled
from gui.common.process_runner import ProcessRunner, _external_command_for_app_job


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


def test_process_runner_runs_internal_app_job(monkeypatch, tmp_path: Path) -> None:
    app = _app()
    runner = ProcessRunner()
    lines: list[str] = []
    finished: list[bool] = []

    def fake_run_app_job(job: AppJob, *, cancel_event=None) -> None:
        assert cancel_event is not None
        print(f"internal {job.job_type}:{job.kind}")

    monkeypatch.setattr("gui.common.process_runner.run_app_job", fake_run_app_job)
    runner.line_received.connect(lines.append)
    runner.queue_finished.connect(finished.append)

    job = AppJob("frame", {"kind": "unit_test"}, tmp_path / "job.json")
    runner.start_queue([("internal", job)], log_dir=tmp_path)

    try:
        assert _process_events_until(app, lambda: bool(finished))
        assert finished == [True]
        assert any("app-job frame unit_test" in line for line in lines)
        assert any("internal frame:unit_test" in line for line in lines)
    finally:
        if runner.is_running():
            runner.cancel()
            _process_events_until(app, lambda: not runner.is_running())


def test_process_runner_runs_process_after_internal_app_job(monkeypatch, tmp_path: Path) -> None:
    app = _app()
    runner = ProcessRunner()
    lines: list[str] = []
    finished: list[bool] = []

    def fake_run_app_job(job: AppJob, *, cancel_event=None) -> None:
        assert job.kind == "unit_test"
        assert cancel_event is not None
        print("internal done")

    monkeypatch.setattr("gui.common.process_runner.run_app_job", fake_run_app_job)
    runner.line_received.connect(lines.append)
    runner.queue_finished.connect(finished.append)

    job = AppJob("frame", {"kind": "unit_test"}, tmp_path / "job.json")
    runner.start_queue(
        [
            ("internal", job),
            ("external", [sys.executable, "-c", "print('external done')"]),
        ],
        log_dir=tmp_path,
    )

    try:
        assert _process_events_until(app, lambda: bool(finished))
        assert finished == [True]
        assert any("internal done" in line for line in lines)
        assert any("external done" in line for line in lines)
    finally:
        if runner.is_running():
            runner.cancel()
            _process_events_until(app, lambda: not runner.is_running())


def test_process_runner_offloads_workflow_app_job(monkeypatch, tmp_path: Path) -> None:
    app = _app()
    runner = ProcessRunner()
    lines: list[str] = []
    finished: list[bool] = []

    def fail_internal_job(_job: AppJob, *, cancel_event=None) -> None:
        raise AssertionError("workflow jobs should run out of process")

    monkeypatch.setattr("gui.common.process_runner.run_app_job", fail_internal_job)
    monkeypatch.setattr(
        "gui.common.process_runner._external_command_for_app_job",
        lambda _job: [sys.executable, "-c", "print('external workflow ok')"],
    )
    runner.line_received.connect(lines.append)
    runner.queue_finished.connect(finished.append)

    job = AppJob("workflow", {"kind": "metashape_preprocess"}, tmp_path / "job.json")
    runner.start_queue([("metashape", job)], log_dir=tmp_path)

    try:
        assert _process_events_until(app, lambda: bool(finished))
        assert finished == [True]
        assert any("app-job workflow metashape_preprocess" in line for line in lines)
        assert any("external workflow ok" in line for line in lines)
    finally:
        if runner.is_running():
            runner.cancel()
            _process_events_until(app, lambda: not runner.is_running())


def test_external_command_for_app_job_uses_workflow_job_script(tmp_path: Path) -> None:
    job_path = tmp_path / "metashape.json"
    job = AppJob("workflow", {"kind": "metashape_preprocess"}, job_path)

    cmd = _external_command_for_app_job(job)

    assert cmd is not None
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts\\run_workflow_job.py") or cmd[1].endswith("scripts/run_workflow_job.py")
    assert cmd[2:] == ["--job", str(job_path)]


def test_external_command_for_app_job_keeps_non_workflow_jobs_internal(tmp_path: Path) -> None:
    job = AppJob("frame", {"kind": "unit_test"}, tmp_path / "job.json")

    assert _external_command_for_app_job(job) is None


def test_external_command_for_app_job_keeps_apriltag_jobs_internal(tmp_path: Path) -> None:
    payload = apriltag_scale_estimate_job(
        dataset=tmp_path / "transforms.json",
        report_json=tmp_path / "report.json",
        tag_size_m=0.16,
        family="tag36h11",
    )
    job = AppJob("apriltag", payload, tmp_path / "apriltag.json")

    assert _external_command_for_app_job(job) is None


def test_process_runner_cancels_internal_app_job(monkeypatch, tmp_path: Path) -> None:
    app = _app()
    runner = ProcessRunner()
    lines: list[str] = []
    finished: list[bool] = []
    cancel_events: list[object] = []

    def fake_run_app_job(job: AppJob, *, cancel_event=None) -> None:
        assert job.kind == "unit_test"
        assert cancel_event is not None
        cancel_events.append(cancel_event)
        while not cancel_event.is_set():
            time.sleep(0.01)
        raise AppJobCancelled()

    monkeypatch.setattr("gui.common.process_runner.run_app_job", fake_run_app_job)
    runner.line_received.connect(lines.append)
    runner.queue_finished.connect(finished.append)

    job = AppJob("frame", {"kind": "unit_test"}, tmp_path / "job.json")
    runner.start_queue([("internal", job)], log_dir=tmp_path)

    try:
        assert _process_events_until(app, lambda: bool(cancel_events))
        runner.cancel()
        assert _process_events_until(app, lambda: bool(finished))
        assert finished == [False]
        assert not runner.is_running()
        assert any("キャンセル中" in line for line in lines)
        assert any("Operation canceled" in line for line in lines)
    finally:
        if runner.is_running():
            runner.cancel()
            _process_events_until(app, lambda: not runner.is_running())
