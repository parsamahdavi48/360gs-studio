"""Validated external theme packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_COLORS = frozenset({"background", "surface", "surface_bright", "primary", "secondary", "text", "text_dim", "border", "success", "warning", "error"})


@dataclass(frozen=True, slots=True)
class ThemePack:
    theme_id: str
    display_name: str
    colors: dict[str, str]
    metrics: dict[str, float]
    schema_version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> ThemePack:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported theme schema: {path}")
        colors = payload.get("colors")
        metrics = payload.get("metrics", {})
        if not isinstance(colors, dict) or _REQUIRED_COLORS - colors.keys():
            missing = sorted(_REQUIRED_COLORS - (colors.keys() if isinstance(colors, dict) else set()))
            raise ValueError(f"theme is missing required colors: {missing}")
        if not all(isinstance(value, str) and value.startswith("#") for value in colors.values()):
            raise ValueError("theme colors must use hexadecimal strings")
        if not isinstance(metrics, dict) or not all(isinstance(value, (int, float)) for value in metrics.values()):
            raise ValueError("theme metrics must be numeric")
        return cls(str(payload["theme_id"]), str(payload["display_name"]), dict(colors), {key: float(value) for key, value in metrics.items()})


def bundled_midnight_theme() -> ThemePack:
    return ThemePack.load(Path(__file__).resolve().parent / "resources" / "themes" / "midnight.json")
