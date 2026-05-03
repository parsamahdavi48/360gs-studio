from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

Pathish = str | PathLike[str]

DEFAULT_MAX_SAFE_PATH_CHARS = 240


@dataclass(frozen=True)
class PathSafetyIssue:
    code: str
    value: str = ""
    length: int = 0
    limit: int = 0


def normalized_path_text(path: Pathish) -> str:
    try:
        return str(Path(path).resolve(strict=False))
    except Exception:
        return str(path)


def check_path_safety(
    path: Pathish,
    *,
    max_chars: int = DEFAULT_MAX_SAFE_PATH_CHARS,
) -> list[PathSafetyIssue]:
    text = normalized_path_text(path)
    issues: list[PathSafetyIssue] = []

    if any(ord(ch) > 127 for ch in text):
        issues.append(PathSafetyIssue("non_ascii"))

    if len(text) >= max_chars:
        issues.append(PathSafetyIssue("too_long", length=len(text), limit=max_chars))

    bad_controls = sorted({ch for ch in text if ord(ch) < 32})
    if bad_controls:
        issues.append(PathSafetyIssue("control_chars", value=" ".join(f"U+{ord(ch):04X}" for ch in bad_controls)))

    if '"' in text:
        issues.append(PathSafetyIssue("quote", value='"'))

    return issues
