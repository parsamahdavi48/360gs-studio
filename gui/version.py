"""Application version helpers."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT_PATH = _PROJECT_ROOT / "pyproject.toml"
_FALLBACK_VERSION = "0.0.0"
_PACKAGED_VERSION = "0.1.0"
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def read_app_version(pyproject_path: Path = _PYPROJECT_PATH) -> str:
    """Read the app version from pyproject.toml."""
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        version = data["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return _FALLBACK_VERSION
    if not isinstance(version, str):
        return _FALLBACK_VERSION
    version = version.strip()
    return version if _VERSION_RE.fullmatch(version) else _FALLBACK_VERSION


APP_VERSION = read_app_version()
if APP_VERSION == _FALLBACK_VERSION:
    # Frozen one-directory builds do not ship the repository-level pyproject.
    APP_VERSION = _PACKAGED_VERSION


def app_version_label() -> str:
    return f"v{APP_VERSION}"
