"""Resolve the persisted UI language before translated widget modules load."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QSettings


def bootstrap_language_preference() -> str:
    for index, value in enumerate(sys.argv):
        if value == "--language" and index + 1 < len(sys.argv):
            os.environ["STUDIO_LANG"] = sys.argv[index + 1]
            return sys.argv[index + 1]
        if value.startswith("--language="):
            selected = value.split("=", 1)[1]
            os.environ["STUDIO_LANG"] = selected
            return selected
    override = os.environ.get("STUDIO_LANG", "").strip()
    if override:
        return override
    saved = str(QSettings("360GS Studio", "360GS Studio").value("language/id", "") or "").strip()
    if saved:
        os.environ["STUDIO_LANG"] = saved
    return saved


ACTIVE_LANGUAGE = bootstrap_language_preference()
