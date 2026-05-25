from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_VIEW_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class CubemapViewSpec:
    name: str
    yaw: float
    pitch: float

    def as_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "yaw": self.yaw, "pitch": self.pitch}


@dataclass(frozen=True, slots=True)
class RemapSpec:
    input_size: tuple[int, int]
    output_size: int
    fov_deg: float
    yaw_deg: float
    pitch_deg: float


def normalize_views(raw_views: Any) -> tuple[CubemapViewSpec, ...]:
    if not isinstance(raw_views, list):
        raise ValueError("views-json must be a list or an object with 'views' list")

    views: list[CubemapViewSpec] = []
    used_names: set[str] = set()
    for idx, item in enumerate(raw_views):
        if not isinstance(item, dict):
            raise ValueError(f"views[{idx}] must be an object")
        if not bool(item.get("enabled", True)):
            continue

        name = _parse_view_name(item, idx)
        if name in used_names:
            raise ValueError(f"views has duplicated name: {name}")
        try:
            yaw = _finite_float(item["yaw"])
            pitch = _finite_float(item["pitch"])
        except KeyError as e:
            raise ValueError(f"views[{idx}] missing field: {e}") from e
        except Exception as e:
            raise ValueError(f"views[{idx}] yaw/pitch parse error: {e}") from e

        views.append(CubemapViewSpec(name=name, yaw=yaw, pitch=pitch))
        used_names.add(name)

    if not views:
        raise ValueError("views-json has no enabled views")
    return tuple(views)


def load_views_json(path: str | Path) -> tuple[CubemapViewSpec, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_views = data.get("views") if isinstance(data, dict) else data
    return normalize_views(raw_views)


def make_default_cube6_views(
    yaw: float,
    stitch: float,
    *,
    no_top: bool = False,
    no_bottom: bool = False,
) -> tuple[CubemapViewSpec, ...]:
    views = (
        CubemapViewSpec("px", 90.0 - float(yaw) - float(stitch), 0.0),
        CubemapViewSpec("nx", -90.0 - float(yaw) - float(stitch), 0.0),
        CubemapViewSpec("py", 0.0 - float(yaw), -90.0),
        CubemapViewSpec("ny", 0.0 - float(yaw), 90.0),
        CubemapViewSpec("pz", 0.0 - float(yaw) + float(stitch), 0.0),
        CubemapViewSpec("nz", 180.0 - float(yaw) + float(stitch), 0.0),
    )
    return tuple(
        view
        for view in views
        if (not no_top or view.name != "py") and (not no_bottom or view.name != "ny")
    )


def views_to_dicts(views: tuple[CubemapViewSpec, ...] | list[CubemapViewSpec]) -> list[dict[str, float | str]]:
    return [view.as_dict() for view in views]


def build_remap_spec(
    *,
    input_size: tuple[int, int],
    output_size: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
) -> RemapSpec:
    try:
        width, height = int(input_size[0]), int(input_size[1])
        output = int(output_size)
        fov = _finite_float(fov_deg)
        yaw = _finite_float(yaw_deg)
        pitch = _finite_float(pitch_deg)
    except Exception as e:
        raise ValueError("remap spec values must be numeric") from e
    if width <= 0 or height <= 0:
        raise ValueError("input_size must contain positive width and height")
    if output <= 0:
        raise ValueError("output_size must be positive")
    if fov <= 0.0 or fov >= 180.0:
        raise ValueError("fov_deg must be in (0, 180)")
    return RemapSpec(
        input_size=(width, height),
        output_size=output,
        fov_deg=fov,
        yaw_deg=yaw,
        pitch_deg=pitch,
    )


def _parse_view_name(item: dict[str, Any], idx: int) -> str:
    name = str(item.get("name", "")).strip()
    if not name:
        raise ValueError(f"views[{idx}].name is required")
    if not SAFE_VIEW_NAME_RE.match(name):
        raise ValueError(f"views[{idx}].name '{name}' is invalid; use letters/numbers/_/- only")
    return name


def _finite_float(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("value must be finite")
    return parsed
