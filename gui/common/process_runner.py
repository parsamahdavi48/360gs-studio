"""QProcess ライフサイクル管理 + ステップキュー"""
from __future__ import annotations

import os
import subprocess

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


class ProcessRunner(QObject):
    """複数ステップのCLIコマンドを順番に実行する共通ランナー。

    シグナル:
        line_received(str)  -- stdout の1行
        phase_started(str)  -- フェーズ名
        phase_finished(str, int, bool)  -- フェーズ名, exit_code, was_canceled
        queue_finished(bool)  -- 全フェーズが成功したか
    """

    line_received = Signal(str)
    phase_started = Signal(str)
    phase_finished = Signal(str, int, bool)
    queue_finished = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._current_phase = ""
        self._buffer = ""
        self._pending: list[tuple[str, list[str]]] = []
        self._cancel_requested = False
        self._all_ok = True
        self._running = False

    # -- public API --

    def is_running(self) -> bool:
        return self._running

    @property
    def current_phase(self) -> str:
        return self._current_phase

    def start_single(self, cmd: list[str], phase: str = "run") -> None:
        self.start_queue([(phase, cmd)])

    def start_queue(self, steps: list[tuple[str, list[str]]]) -> None:
        if self.is_running():
            return
        self._cancel_requested = False
        self._all_ok = True
        self._pending = list(steps)
        self._running = bool(self._pending)
        self._run_next()

    def cancel(self) -> None:
        if not self.is_running():
            return
        self._cancel_requested = True
        self._pending.clear()
        if self._proc is not None:
            self._terminate_gracefully(self._proc, self._current_phase)

    # -- internal --

    def _run_next(self) -> None:
        if not self._pending:
            self._running = False
            self.queue_finished.emit(self._all_ok)
            return

        phase, cmd = self._pending.pop(0)
        self._current_phase = phase
        self._buffer = ""
        self.line_received.emit("$ " + " ".join(cmd))
        self.phase_started.emit(phase)

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.errorOccurred.connect(self._on_error)
        proc.finished.connect(self._on_finished)
        self._proc = proc
        proc.start()

    def _on_output(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        data = data.replace("\r", "\n")
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.line_received.emit(line)

    def _on_error(self, _error: QProcess.ProcessError) -> None:
        if self._proc is None:
            return
        self.line_received.emit(f"[{self._current_phase}] プロセスエラーが発生しました")

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._buffer:
            tail = self._buffer.replace("\r", "\n").strip()
            for line in tail.splitlines():
                self.line_received.emit(line)
            self._buffer = ""

        phase = self._current_phase
        was_canceled = self._cancel_requested
        self._cancel_requested = False
        self._proc = None

        if was_canceled:
            self._running = False
            self._all_ok = False
            self._pending.clear()
            self.phase_finished.emit(phase, exit_code, True)
            self.queue_finished.emit(False)
        elif exit_code == 0:
            self.phase_finished.emit(phase, 0, False)
            self._run_next()
        else:
            self._running = False
            self._all_ok = False
            self._pending.clear()
            self.phase_finished.emit(phase, exit_code, False)
            self.queue_finished.emit(False)

    def _terminate_gracefully(self, proc: QProcess, phase: str, timeout_ms: int = 3000) -> None:
        if proc.state() == QProcess.NotRunning:
            return
        self.line_received.emit(f"[{phase}] キャンセル中...")
        proc.terminate()
        QTimer.singleShot(timeout_ms, lambda p=proc, ph=phase: self._force_kill(p, ph))

    def _force_kill(self, proc: QProcess, phase: str) -> None:
        if proc.state() == QProcess.NotRunning:
            return
        self.line_received.emit(f"[{phase}] タイムアウト; プロセスを強制終了")
        self._kill_process_tree(proc)
        proc.kill()

    def _kill_process_tree(self, proc: QProcess) -> None:
        if os.name != "nt":
            return
        pid = int(proc.processId())
        if pid <= 0:
            return
        kwargs: dict[str, object] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": 5,
        }
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], **kwargs)
        except (OSError, subprocess.TimeoutExpired):
            pass
