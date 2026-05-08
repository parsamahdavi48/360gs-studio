"""Step 4 project settings persistence helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scene_layout import step4_export_settings_path
from scene_project import load_json, write_json

STEP4_SETTINGS_VERSION = 2


def _settings_version(payload: dict[str, Any]) -> int:
    try:
        return int(payload.get("settings_version", 0))
    except (TypeError, ValueError):
        return 0


def load_step4_export_settings(scene_dir: Path) -> dict[str, Any]:
    """Load current Step 4 project settings."""
    data = load_json(step4_export_settings_path(scene_dir), {})
    if not data or _settings_version(data) < STEP4_SETTINGS_VERSION:
        return {}
    return data


def write_step4_export_settings(scene_dir: Path, payload: dict[str, Any]) -> Path:
    """Write Step 4 settings to the current project metadata path."""
    path = step4_export_settings_path(scene_dir)
    write_json(path, payload)
    return path
