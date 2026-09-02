from __future__ import annotations

import os
from pathlib import Path

from scripts.run_pyinstaller_clean import isolate_dll_search_path


def test_isolate_dll_search_path_removes_unrelated_entries(monkeypatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join((r"C:\tools\poppler", r"D:\other")))
    monkeypatch.setenv("SystemRoot", r"C:\Windows")

    value = isolate_dll_search_path()

    assert "poppler" not in value.lower()
    assert str(Path(r"C:\Windows") / "System32") in value
    assert os.environ["PATH"] == value
