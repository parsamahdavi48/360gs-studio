"""Per-user GUI settings stored outside project and scene folders."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

APP_SETTINGS_DIRNAME = "stechdrive-3dgs-utils"
SETTINGS_FILENAME = "user_settings.json"
SETTINGS_PATH_ENV = "STECHDRIVE_USER_SETTINGS_PATH"


def user_settings_path() -> Path:
    override = os.environ.get(SETTINGS_PATH_ENV, "").strip()
    if override:
        return Path(override)

    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        return Path(appdata) / APP_SETTINGS_DIRNAME / SETTINGS_FILENAME

    return Path.home() / ".config" / APP_SETTINGS_DIRNAME / SETTINGS_FILENAME


def load_user_settings() -> dict[str, Any]:
    path = user_settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_user_settings(data: dict[str, Any]) -> None:
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_user_settings_section(section: str) -> dict[str, Any]:
    data = load_user_settings()
    value = data.get(section)
    return value if isinstance(value, dict) else {}


def update_user_settings_section(section: str, values: dict[str, Any]) -> None:
    data = load_user_settings()
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        section_data = {}
    section_data.update(values)
    data[section] = section_data
    save_user_settings(data)
