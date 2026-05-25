from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def require_mapping(payload: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload must be a JSON object")
    return payload


def require_schema_version(payload: Mapping[str, Any], *, expected: int, label: str) -> None:
    version = payload.get("schema_version")
    if not isinstance(version, int) or version != expected:
        raise ValueError(f"Unsupported {label} job schema version: {version!r}")


def require_kind(payload: Mapping[str, Any], *, allowed: set[str], label: str) -> str:
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{label} job kind is required")
    if kind not in allowed:
        raise ValueError(f"Unsupported {label} job kind: {kind}")
    return kind


def require_str(payload: Mapping[str, Any], key: str, *, label: str, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{label} job field '{key}' must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} job field '{key}' is required")
    return value


def require_bool(payload: Mapping[str, Any], key: str, *, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label} job field '{key}' must be a boolean")
    return value


def require_finite_float(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} job field '{key}' must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} job field '{key}' must be finite")
    if min_value is not None:
        if number < min_value or (number == min_value and not min_inclusive):
            op = ">=" if min_inclusive else ">"
            raise ValueError(f"{label} job field '{key}' must be {op} {min_value:g}")
    if max_value is not None:
        if number > max_value or (number == max_value and not max_inclusive):
            op = "<=" if max_inclusive else "<"
            raise ValueError(f"{label} job field '{key}' must be {op} {max_value:g}")
    return number


def require_int_range(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} job field '{key}' must be an integer")
    if min_value is not None and value < min_value:
        raise ValueError(f"{label} job field '{key}' must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{label} job field '{key}' must be <= {max_value}")
    return value


def require_views(
    payload: Mapping[str, Any],
    *,
    label: str,
    require_orientation: bool = True,
    require_enabled_view: bool = True,
) -> list[Mapping[str, Any]]:
    value = payload.get("views")
    if not isinstance(value, list):
        raise ValueError(f"{label} job field 'views' must be a list")
    enabled_count = 0
    views: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} job field 'views[{index}]' must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} job field 'views[{index}].name' is required")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{label} job field 'views[{index}].enabled' must be a boolean")
        if enabled:
            enabled_count += 1
        if require_orientation:
            _require_view_number(item, "yaw", index=index, label=label)
            _require_view_number(item, "pitch", index=index, label=label)
        views.append(item)
    if require_enabled_view and enabled_count == 0:
        raise ValueError(f"{label} job requires at least one enabled view")
    return views


def _require_view_number(item: Mapping[str, Any], key: str, *, index: int, label: str) -> float:
    value = item.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} job field 'views[{index}].{key}' must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} job field 'views[{index}].{key}' must be finite")
    return number
