"""Run PyInstaller without collecting DLLs from unrelated PATH entries."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def isolate_dll_search_path() -> str:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    paths = (
        Path(sys.executable).resolve().parent,
        system_root / "System32",
        system_root,
        system_root / "System32" / "Wbem",
        system_root / "System32" / "WindowsPowerShell" / "v1.0",
    )
    value = os.pathsep.join(str(path) for path in paths)
    os.environ["PATH"] = value
    return value


def main() -> int:
    isolate_dll_search_path()
    from PyInstaller.__main__ import run

    run(sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
