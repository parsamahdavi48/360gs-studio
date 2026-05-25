from __future__ import annotations

import subprocess
from typing import Protocol


class CancellationToken(Protocol):
    def is_set(self) -> bool: ...


class AppJobCancelled(RuntimeError):
    """Raised when an in-app job cooperatively stops after a cancel request."""

    def __init__(self, message: str = "Operation canceled") -> None:
        super().__init__(message)


def is_cancelled(cancel_event: CancellationToken | None) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def raise_if_cancelled(cancel_event: CancellationToken | None) -> None:
    if is_cancelled(cancel_event):
        raise AppJobCancelled()


def terminate_process(proc: subprocess.Popen, *, timeout: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.kill()
    except OSError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return
