"""QProcess ライフサイクル管理 + ステップキュー"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


class ProcessRunner(QObject):
    """複数ステップのCLIコマンドを順番に実行する共通ランナー。

    シグナル:
        line_received(str)  -- stdout の1行
        phase_started(str)  -- フェーズ名
        phase_log_started(str, str)  -- フェーズ名, ログファイルパス
        phase_finished(str, int, bool)  -- フェーズ名, exit_code, was_canceled
        queue_finished(bool)  -- 全フェーズが成功したか
    """

    line_received = Signal(str)
    phase_started = Signal(str)
    phase_log_started = Signal(str, str)
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
        self._log_dir: Path | None = None
        self._log_file: TextIO | None = None
        self._current_log_path: Path | None = None
        self._run_log_stamp = ""
        self._phase_index = 0
        self._queue_total = 0

    # -- public API --

    def is_running(self) -> bool:
        return self._running

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @property
    def phase_index(self) -> int:
        return self._phase_index

    @property
    def queue_total(self) -> int:
        return self._queue_total

    def start_single(self, cmd: list[str], phase: str = "run", log_dir: str | Path | None = None) -> None:
        self.start_queue([(phase, cmd)], log_dir=log_dir)

    def start_queue(self, steps: list[tuple[str, list[str]]], log_dir: str | Path | None = None) -> None:
        if self.is_running():
            return
        self._cancel_requested = False
        self._all_ok = True
        self._pending = list(steps)
        self._running = bool(self._pending)
        self._phase_index = 0
        self._queue_total = len(self._pending)
        self._run_log_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_dir = Path(log_dir) if log_dir is not None else None
        if self._log_dir is not None:
            try:
                self._log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.line_received.emit(f"[log] Could not create process log directory: {exc}")
                self._log_dir = None
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
        self._phase_index += 1
        self._open_phase_log(phase)
        self._emit_line("$ " + " ".join(cmd))
        if self._current_log_path is not None:
            self._emit_line(f"[{phase}] log: {self._current_log_path}")
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
                self._emit_line(line)

    def _on_error(self, _error: QProcess.ProcessError) -> None:
        if self._proc is None:
            return
        self._emit_line(f"[{self._current_phase}] プロセスエラーが発生しました")

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._buffer:
            tail = self._buffer.replace("\r", "\n").strip()
            for line in tail.splitlines():
                self._emit_line(line)
            self._buffer = ""

        phase = self._current_phase
        was_canceled = self._cancel_requested
        self._cancel_requested = False
        self._proc = None
        self._emit_line(f"[{phase}] exit_code={exit_code} canceled={int(was_canceled)}")
        self._close_phase_log()

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
        self._emit_line(f"[{phase}] キャンセル中...")
        proc.terminate()
        QTimer.singleShot(timeout_ms, lambda p=proc, ph=phase: self._force_kill(p, ph))

    def _force_kill(self, proc: QProcess, phase: str) -> None:
        if proc.state() == QProcess.NotRunning:
            return
        self._emit_line(f"[{phase}] タイムアウト; プロセスを強制終了")
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

    def _open_phase_log(self, phase: str) -> None:
        self._close_phase_log()
        self._current_log_path = None
        if self._log_dir is None:
            return
        safe_phase = re.sub(r"[^A-Za-z0-9_.-]+", "_", phase).strip("._") or "phase"
        path = self._log_dir / f"{self._run_log_stamp}_{self._phase_index:02d}_{safe_phase}.log"
        try:
            self._log_file = path.open("w", encoding="utf-8", newline="\n")
        except OSError as exc:
            self.line_received.emit(f"[log] Could not open process log file: {exc}")
            self._log_file = None
            return
        self._current_log_path = path
        self.phase_log_started.emit(phase, str(path))

    def _close_phase_log(self) -> None:
        if self._log_file is None:
            return
        self._log_file.close()
        self._log_file = None

    def _emit_line(self, line: str) -> None:
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()
        self.line_received.emit(line)
