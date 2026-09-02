"""Stable adapter boundary for FFmpeg, SfM tools, and trainers."""

from __future__ import annotations

import abc
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProbeResult:
    available: bool
    executable: str = ""
    version: str = ""
    capabilities: frozenset[str] = field(default_factory=frozenset)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "executable": self.executable,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
            "message": self.message,
        }


class ToolAdapter(abc.ABC):
    tool_id: str
    display_name: str

    @abc.abstractmethod
    def probe(self) -> ProbeResult:
        raise NotImplementedError

    def capabilities(self) -> frozenset[str]:
        return self.probe().capabilities

    def preflight(self, request: dict[str, Any]) -> list[str]:
        del request
        result = self.probe()
        return [] if result.available else [result.message or f"{self.display_name} is unavailable"]

    @abc.abstractmethod
    def build_command(self, request: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def parse_progress(self, line: str) -> tuple[int, int] | None:
        del line
        return None

    def cancel(self, process: subprocess.Popen[Any]) -> None:
        process.terminate()


class ExecutableAdapter(ToolAdapter):
    executable_names: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)

    def __init__(self, executable: str | Path | None = None) -> None:
        self.executable = str(executable) if executable else ""

    def resolve_executable(self) -> str:
        if self.executable and Path(self.executable).is_file():
            return str(Path(self.executable).resolve())
        for name in self.executable_names:
            found = shutil.which(name)
            if found:
                return found
        return ""

    def probe(self) -> ProbeResult:
        executable = self.resolve_executable()
        if not executable:
            return ProbeResult(False, message=f"{self.display_name} executable was not found")
        try:
            completed = subprocess.run(
                [executable, *self.version_args],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            return ProbeResult(False, executable=executable, message=str(exc))
        version = (completed.stdout or completed.stderr).splitlines()[0].strip() if (completed.stdout or completed.stderr) else ""
        return ProbeResult(completed.returncode == 0, executable=executable, version=version)
