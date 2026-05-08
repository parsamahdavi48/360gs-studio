"""Step 4 project settings persistence helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scene_layout import (
    legacy_step4_export_settings_path,
    step4_export_settings_path,
)
from scene_project import load_json, write_json

STEP4_SETTINGS_VERSION = 2


def step4_export_settings_candidates(scene_dir: Path) -> tuple[Path, ...]:
    """Return current and legacy Step 4 settings paths, in read preference order."""
    return (
        step4_export_settings_path(scene_dir),
        legacy_step4_export_settings_path(scene_dir),
    )


def load_step4_export_settings(scene_dir: Path) -> dict[str, Any]:
    """Load Step 4 settings from the current path, falling back to the legacy path."""
    for path in step4_export_settings_candidates(scene_dir):
        data = load_json(path, {})
        if data:
            return data
    return {}


def write_step4_export_settings(scene_dir: Path, payload: dict[str, Any]) -> Path:
    """Write Step 4 settings to the current project metadata path."""
    path = step4_export_settings_path(scene_dir)
    write_json(path, payload)
    return path
