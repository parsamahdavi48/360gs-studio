"""Output reset helpers for workflow steps."""

from __future__ import annotations

from pathlib import Path

from path_safety import safe_clear_path


def path_has_contents(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.exists()


def dedupe_nested_paths(paths: list[Path]) -> list[Path]:
    kept: list[Path] = []
    for path in sorted(paths, key=lambda p: len(p.parts)):
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        nested = False
        for parent in kept:
            try:
                parent_resolved = parent.resolve()
            except OSError:
                parent_resolved = parent.absolute()
            if resolved == parent_resolved:
                nested = True
                break
            try:
                resolved.relative_to(parent_resolved)
                nested = True
                break
            except ValueError:
                pass
        if not nested:
            kept.append(path)
    return kept


def clear_path(path: Path, *, allowed_roots: list[Path]) -> None:
    safe_clear_path(path, allowed_roots=allowed_roots)


def clear_output_dir(output: Path) -> None:
    for child in output.iterdir():
        safe_clear_path(child, allowed_roots=[output])
