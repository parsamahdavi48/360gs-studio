"""Adapters for supported local command-line tools."""

from __future__ import annotations

import re
from typing import Any

from gs360studio.adapters.base import ExecutableAdapter, ProbeResult


class FFmpegAdapter(ExecutableAdapter):
    tool_id = "ffmpeg"
    display_name = "FFmpeg"
    executable_names = ("ffmpeg.exe", "ffmpeg")
    version_args = ("-version",)

    def probe(self) -> ProbeResult:
        base = super().probe()
        if not base.available:
            return base
        capabilities: set[str] = {"decode", "v360"}
        executable = base.executable
        import subprocess

        completed = subprocess.run(
            [executable, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        text = f"{completed.stdout}\n{completed.stderr}".lower()
        if "hevc_nvenc" in text:
            capabilities.add("hevc_nvenc")
        if "h264_nvenc" in text:
            capabilities.add("h264_nvenc")
        if "libx265" in text:
            capabilities.add("libx265")
        return ProbeResult(True, executable=executable, version=base.version, capabilities=frozenset(capabilities))

    def build_command(self, request: dict[str, Any]) -> list[str]:
        executable = self.resolve_executable()
        if not executable:
            raise FileNotFoundError("FFmpeg was not found")
        args = request.get("args")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError("FFmpeg request must contain a string args list")
        return [executable, *args]

    def parse_progress(self, line: str) -> tuple[int, int] | None:
        match = re.search(r"frame=\s*(\d+)", line)
        if not match:
            return None
        return int(match.group(1)), 0


class ColmapAdapter(ExecutableAdapter):
    tool_id = "colmap"
    display_name = "COLMAP"
    executable_names = ("colmap.exe", "colmap")

    def build_command(self, request: dict[str, Any]) -> list[str]:
        executable = self.resolve_executable()
        if not executable:
            raise FileNotFoundError("COLMAP was not found")
        command = str(request.get("command") or "").strip()
        args = request.get("args", [])
        if not command or not isinstance(args, list):
            raise ValueError("COLMAP request requires command and args")
        return [executable, command, *[str(item) for item in args]]


class TrainerAdapter(ExecutableAdapter):
    def build_command(self, request: dict[str, Any]) -> list[str]:
        executable = self.resolve_executable()
        if not executable:
            raise FileNotFoundError(f"{self.display_name} was not found")
        dataset = str(request.get("dataset") or "").strip()
        if not dataset:
            raise ValueError("trainer request requires dataset")
        return [executable, dataset, *[str(item) for item in request.get("args", [])]]


class LichtFeldAdapter(TrainerAdapter):
    tool_id = "lichtfeld"
    display_name = "LichtFeld Studio"
    executable_names = ("LichtFeld-Studio.exe", "LichtFeldStudio.exe")


class PostshotAdapter(TrainerAdapter):
    tool_id = "postshot"
    display_name = "Postshot"
    executable_names = ("postshot-cli.exe",)


class BrushAdapter(TrainerAdapter):
    tool_id = "brush"
    display_name = "Brush"
    executable_names = ("brush.exe",)
