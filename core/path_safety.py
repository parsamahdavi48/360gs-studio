from __future__ import annotations

import shutil
from collections.abc import Sequence
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


def resolve_for_safety(path: Pathish) -> Path:
    return Path(path).resolve(strict=False)


def is_path_inside(path: Pathish, root: Pathish, *, allow_equal: bool = True) -> bool:
    try:
        resolved = resolve_for_safety(path)
        resolved_root = resolve_for_safety(root)
    except Exception:
        return False
    if allow_equal and resolved == resolved_root:
        return True
    try:
        resolved.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def _is_dangerous_root(path: Path) -> bool:
    resolved = resolve_for_safety(path)
    if resolved.parent == resolved:
        return True
    anchor = Path(resolved.anchor) if resolved.anchor else None
    return anchor is not None and resolved == anchor


def ensure_within_allowed_roots(
    path: Pathish,
    allowed_roots: Sequence[Pathish],
    *,
    allow_root: bool = False,
) -> Path:
    resolved = resolve_for_safety(path)
    if _is_dangerous_root(resolved):
        raise RuntimeError(f"Refusing to operate on filesystem root: {resolved}")
    for root in allowed_roots:
        resolved_root = resolve_for_safety(root)
        if resolved == resolved_root:
            if allow_root:
                return resolved
            raise RuntimeError(f"Refusing to operate on allowed root itself: {resolved}")
        if is_path_inside(resolved, resolved_root, allow_equal=False):
            return resolved
    roots = ", ".join(str(resolve_for_safety(root)) for root in allowed_roots)
    raise RuntimeError(f"Path is outside allowed roots: {resolved} (allowed: {roots})")


def safe_clear_path(
    path: Pathish,
    *,
    allowed_roots: Sequence[Pathish],
    allow_root: bool = False,
) -> None:
    target = ensure_within_allowed_roots(path, allowed_roots, allow_root=allow_root)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
