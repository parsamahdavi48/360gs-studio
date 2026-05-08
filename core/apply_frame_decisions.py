#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from core.path_safety import is_path_inside, safe_clear_path
from core.scene_layout import frame_backups_dir, scene_images_dir, selected_frames_keep_path, selected_frames_path


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clean_output_dir(output_dir: Path) -> int:
    removed = 0
    if not output_dir.exists():
        return removed

    for p in output_dir.iterdir():
        if p.is_file() and is_image_file(p):
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")
    return rows


def write_rows(csv_path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_unique_backup_path(csv_path: Path, backup_dir: Path | None = None) -> Path:
    stem = csv_path.stem
    suffix = csv_path.suffix
    root = backup_dir or csv_path.parent
    candidate = root / f"{stem}.before_finalize{suffix}"
    if not candidate.exists():
        return candidate

    for i in range(1, 1000):
        candidate = root / f"{stem}.before_finalize.{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Failed to allocate backup path near: {csv_path}")


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def normalize_decision(row: dict) -> str:
    return row.get("decision", "keep").strip().lower()


def pending_drop_image_paths(
    scene_dir: Path,
    csv_name: str = "selected_frames.csv",
    images_dir: Path | None = None,
) -> list[Path]:
    """Return drop-marked image files that still exist under images_dir."""
    csv_path = selected_frames_path(scene_dir, csv_name)
    if not csv_path.exists():
        return []

    rows = load_rows(csv_path)
    root = images_dir if images_dir is not None else scene_images_dir(scene_dir)
    if not root.exists():
        return []

    pending: list[Path] = []
    for row in rows:
        if normalize_decision(row) != "drop":
            continue
        rel_path = row.get("output_file", "").strip()
        if not rel_path:
            continue
        src = scene_dir / rel_path
        if src.exists() and src.is_file() and is_under(src, root):
            pending.append(src)
    return pending


def untracked_image_paths(
    scene_dir: Path,
    csv_name: str = "selected_frames.csv",
    images_dir: Path | None = None,
) -> list[Path]:
    """Return image files in images_dir that are not referenced by the selected CSV."""
    csv_path = selected_frames_path(scene_dir, csv_name)
    root = images_dir if images_dir is not None else scene_images_dir(scene_dir)
    if not csv_path.exists() or not root.exists():
        return []

    rows = load_rows(csv_path)
    tracked = {
        str((scene_dir / row.get("output_file", "").strip()).resolve())
        for row in rows
        if row.get("output_file", "").strip()
    }
    pending: list[Path] = []
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or not is_image_file(path):
            continue
        if str(path.resolve()) not in tracked:
            pending.append(path)
    return pending


def copy_keep_rows(
    scene_dir: Path,
    rows: Sequence[dict],
    output_dir: Path,
    clean_output: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        removed = clean_output_dir(output_dir)
        if removed > 0:
            print(f"Removed {removed} existing image files from {output_dir}")

    copied = 0
    skipped = 0
    missing: list[str] = []

    for row in rows:
        decision = normalize_decision(row)
        if decision == "drop":
            skipped += 1
            continue

        rel_path = row.get("output_file", "").strip()
        if not rel_path:
            skipped += 1
            continue

        src = scene_dir / rel_path
        if not src.exists():
            missing.append(str(src))
            continue

        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied += 1

    print(f"Kept frames: {copied}")
    print(f"Dropped frames: {skipped}")
    print(f"Output dir: {output_dir}")

    if missing:
        print("Missing source files:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


def _looks_like_backup_path(path: Path) -> bool:
    name = path.name.casefold()
    parent_name = path.parent.name.casefold()
    return "backup" in name or parent_name in {"backup", "backups"}


def _validate_backup_target(images_dir: Path, backup_dir: Path) -> None:
    try:
        source = images_dir.resolve(strict=False)
        target = backup_dir.resolve(strict=False)
    except OSError as exc:
        raise RuntimeError(f"Failed to resolve backup paths: {exc}") from exc
    if target == source:
        raise RuntimeError(f"Backup directory must be separate from images/: {backup_dir}")
    if is_path_inside(target, source, allow_equal=False):
        raise RuntimeError(f"Backup directory must not be inside images/: {backup_dir}")
    if is_path_inside(source, target, allow_equal=False):
        raise RuntimeError(f"Backup directory must not contain images/: {backup_dir}")
    if backup_dir.exists() and not _looks_like_backup_path(backup_dir):
        raise RuntimeError(f"Refusing to replace a directory that does not look like a backup path: {backup_dir}")


def backup_images_dir(images_dir: Path, backup_dir: Path) -> int:
    """images/ の現状を backup_dir にフルコピー（既存 backup は事前に削除）。

    Returns:
        コピーしたファイル数。
    """
    if not images_dir.is_dir():
        return 0
    _validate_backup_target(images_dir, backup_dir)
    if backup_dir.exists():
        safe_clear_path(backup_dir, allowed_roots=[backup_dir.parent])
    shutil.copytree(images_dir, backup_dir)
    return sum(1 for p in backup_dir.rglob("*") if p.is_file())


def finalize_in_place(
    scene_dir: Path,
    csv_name: str,
    backup_dir: Path | None = None,
) -> None:
    csv_path = selected_frames_path(scene_dir, csv_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    fieldnames = list(rows[0].keys())
    images_dir = scene_images_dir(scene_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    if backup_dir is not None:
        backup_count = backup_images_dir(images_dir, backup_dir)
        print(f"Backed up {backup_count} files to {backup_dir}")

    keep_entries: list[tuple[dict, Path]] = []
    dropped_paths: list[Path] = []
    missing: list[str] = []
    dropped_rows = 0

    for row in rows:
        rel_path = row.get("output_file", "").strip()
        if not rel_path:
            if normalize_decision(row) == "drop":
                dropped_rows += 1
            else:
                missing.append("<empty output_file>")
            continue

        src = scene_dir / rel_path
        if normalize_decision(row) == "drop":
            dropped_rows += 1
            if src.exists() and is_under(src, images_dir) and src.is_file():
                dropped_paths.append(src)
            continue

        if not src.exists():
            missing.append(str(src))
            continue

        if not is_under(src, images_dir):
            raise RuntimeError(f"Keep frame is outside images/: {src}")
        keep_entries.append((row, src))

    if not keep_entries:
        raise RuntimeError("No keep frames to finalize")
    if missing:
        preview = ", ".join(missing[:3])
        raise RuntimeError(f"Missing keep frame files ({len(missing)}). Example: {preview}")

    keep_source_set = {str(src.resolve()) for _row, src in keep_entries}
    if len(keep_source_set) != len(keep_entries):
        raise RuntimeError("Duplicate keep source detected")

    removed_drop_files = 0
    for p in dropped_paths:
        if str(p.resolve()) in keep_source_set:
            continue
        if p.exists():
            p.unlink()
            removed_drop_files += 1

    updated_rows: list[dict] = []
    for seq, (row, src) in enumerate(keep_entries, start=1):
        new_row = dict(row)
        new_row["seq"] = str(seq)
        new_row["decision"] = "keep"
        new_row["output_file"] = src.relative_to(scene_dir).as_posix()
        updated_rows.append(new_row)

    csv_backup_dir = frame_backups_dir(scene_dir)
    csv_backup_dir.mkdir(parents=True, exist_ok=True)
    backup_csv = ensure_unique_backup_path(csv_path, csv_backup_dir)
    shutil.copy2(csv_path, backup_csv)
    write_rows(csv_path, fieldnames, updated_rows)

    keep_csv = selected_frames_keep_path(scene_dir)
    write_rows(keep_csv, fieldnames, updated_rows)

    print("Finalize mode: inplace images")
    print(f"Kept frames: {len(updated_rows)}")
    print(f"Dropped rows: {dropped_rows}")
    print(f"Removed dropped image files: {removed_drop_files}")
    print(f"Images dir: {images_dir}")
    print("Kept filenames: preserved")
    print(f"CSV backup: {backup_csv}")
    print(f"Updated CSV: {csv_path}")
    print(f"Keep CSV: {keep_csv}")


def apply_decisions(
    scene_dir: Path,
    csv_name: str,
    output_name: str,
    clean_output: bool,
    finalize_inplace: bool,
    backup_dir: Path | None = None,
) -> None:
    csv_path = selected_frames_path(scene_dir, csv_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if finalize_inplace:
        finalize_in_place(scene_dir, csv_name, backup_dir=backup_dir)
        return

    rows = load_rows(csv_path)
    output_dir = scene_dir / output_name
    copy_keep_rows(scene_dir, rows, output_dir, clean_output)

    fieldnames = list(rows[0].keys())
    keep_rows = [row for row in rows if normalize_decision(row) != "drop"]
    keep_csv = output_dir / "selected_frames_keep.csv"
    write_rows(keep_csv, fieldnames, keep_rows)
    print(f"Keep CSV: {keep_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply keep/drop decisions: either finalize in images/ or copy keeps to another folder."
    )
    parser.add_argument(
        "scene_dir",
        nargs="?",
        default=".",
        help="Scene directory containing _stechdrive/frames/selected_frames.csv and images/",
    )
    parser.add_argument(
        "--csv",
        default="selected_frames.csv",
        help="CSV filename under scene_dir/_stechdrive/frames, or an absolute path (default=selected_frames.csv)",
    )
    parser.add_argument(
        "--output",
        default="metashape_images",
        help="Output directory name under scene_dir (default=metashape_images)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing image files in output directory before exporting",
    )
    parser.add_argument(
        "--finalize-in-place",
        action="store_true",
        help="Drop=remove inside scene_dir/images, preserve kept filenames, then update _stechdrive/frames/selected_frames.csv",
    )
    parser.add_argument(
        "--backup-dir",
        default="",
        help=(
            "If set together with --finalize-in-place, snapshot images/ to this directory before "
            "modification (full copy; existing target is replaced). Provide a path relative to scene_dir "
            "or an absolute path. Default empty = no backup."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).resolve()

    backup_dir: Path | None = None
    if args.backup_dir:
        bp = Path(args.backup_dir)
        backup_dir = bp if bp.is_absolute() else (scene_dir / bp)

    try:
        apply_decisions(
            scene_dir=scene_dir,
            csv_name=args.csv,
            output_name=args.output,
            clean_output=args.clean_output,
            finalize_inplace=args.finalize_in_place,
            backup_dir=backup_dir,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
