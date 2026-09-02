from __future__ import annotations

from pathlib import Path

from gs360studio.adapters.tools import FFmpegAdapter


def test_ffmpeg_probe_uses_native_version_flag(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"")
    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = "ffmpeg version test\n hevc_nvenc libx265"
        stderr = ""

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Completed()

    monkeypatch.setattr("gs360studio.adapters.base.subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.run", fake_run)
    result = FFmpegAdapter(executable).probe()

    assert result.available
    assert calls[0][1:] == ["-version"]
    assert "hevc_nvenc" in result.capabilities
