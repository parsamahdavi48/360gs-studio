#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


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


def load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")
    return rows


def write_rows(csv_path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_unique_backup_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    suffix = csv_path.suffix
    candidate = csv_path.with_name(f"{stem}.before_finalize{suffix}")
    if not candidate.exists():
        return candidate

    for i in range(1, 1000):
        candidate = csv_path.with_name(f"{stem}.before_finalize.{i}{suffix}")
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


def sanitize_filename_prefix(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        return ""
    return text


def infer_prefix_from_output_file(rel_path: str) -> str:
    name = Path(rel_path).stem
    m = re.match(r"(.+)_\d+$", name)
    if m:
        return sanitize_filename_prefix(m.group(1))
    return sanitize_filename_prefix(name)


def detect_default_prefix(scene_dir: Path, keep_entries: Sequence[Tuple[dict, Path]]) -> str:
    report_path = scene_dir / "extract_report.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            params = data.get("params", {})
            report_prefix = sanitize_filename_prefix(str(params.get("filename_prefix", "")))
            if report_prefix:
                return report_prefix
            input_video = data.get("input_video", "")
            if input_video:
                stem_prefix = sanitize_filename_prefix(Path(str(input_video)).stem)
                if stem_prefix:
                    return stem_prefix
        except Exception:
            pass

    for row, _src in keep_entries:
        rel_path = row.get("output_file", "").strip()
        if rel_path:
            inferred = infer_prefix_from_output_file(rel_path)
            if inferred:
                return inferred
    return "frame"


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
    missing: List[str] = []

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


def stage_rename_keep_files(
    keep_entries: Sequence[Tuple[dict, Path]],
    images_dir: Path,
) -> List[Tuple[dict, Path, Path, str]]:
    staged: List[Tuple[dict, Path, Path, str]] = []
    seen_sources = set()

    for i, (row, src) in enumerate(keep_entries, start=1):
        key = str(src.resolve())
        if key in seen_sources:
            raise RuntimeError(f"Duplicate keep source detected: {src}")
        seen_sources.add(key)

        ext = src.suffix.lower()
        if not ext:
            ext = ".jpg"
        tmp_path = images_dir / f".tmp_finalize_{i:06d}{ext}"
        if tmp_path.exists():
            tmp_path.unlink()
        src.rename(tmp_path)
        staged.append((row, src, tmp_path, ext))

    return staged


def rollback_staged(
    staged: Sequence[Tuple[dict, Path, Path, str]],
    finalized: Sequence[Tuple[Path, Path]],
) -> None:
    for dst, src in reversed(list(finalized)):
        if dst.exists() and not src.exists():
            dst.rename(src)

    for _row, src, tmp_path, _ext in staged:
        if tmp_path.exists() and not src.exists():
            tmp_path.rename(src)


def backup_images_dir(images_dir: Path, backup_dir: Path) -> int:
    """images/ の現状を backup_dir にフルコピー（既存 backup は事前に削除）。

    Returns:
        コピーしたファイル数。
    """
    if not images_dir.is_dir():
        return 0
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(images_dir, backup_dir)
    return sum(1 for p in backup_dir.rglob("*") if p.is_file())


def finalize_in_place(
    scene_dir: Path,
    csv_name: str,
    filename_prefix: str,
    backup_dir: Optional[Path] = None,
) -> None:
    csv_path = scene_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    fieldnames = list(rows[0].keys())
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if backup_dir is not None:
        backup_count = backup_images_dir(images_dir, backup_dir)
        print(f"Backed up {backup_count} files to {backup_dir}")

    keep_entries: List[Tuple[dict, Path]] = []
    dropped_paths: List[Path] = []
    missing: List[str] = []
    dropped_rows = 0

    for row in rows:
        rel_path = row.get("output_file", "").strip()
        if not rel_path:
            if normalize_decision(row) == "drop":
                dropped_rows += 1
            else:
                missing.append("<empty output_file>")
            continue

        src = (scene_dir / rel_path)
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

    resolved_prefix = sanitize_filename_prefix(filename_prefix)
    if not resolved_prefix:
        resolved_prefix = detect_default_prefix(scene_dir, keep_entries)

    keep_source_set = {str(src.resolve()) for _row, src in keep_entries}
    removed_drop_files = 0
    for p in dropped_paths:
        if str(p.resolve()) in keep_source_set:
            continue
        if p.exists():
            p.unlink()
            removed_drop_files += 1

    staged: List[Tuple[dict, Path, Path, str]] = []
    finalized: List[Tuple[Path, Path]] = []
    updated_rows: List[dict] = []

    try:
        staged = stage_rename_keep_files(keep_entries, images_dir)
        digits = max(6, len(str(len(staged))))
        for seq, (row, src, tmp_path, ext) in enumerate(staged, start=1):
            final_name = f"{resolved_prefix}_{seq:0{digits}d}{ext}"
            dst = images_dir / final_name
            if dst.exists():
                dst.unlink()
            tmp_path.rename(dst)
            finalized.append((dst, src))

            new_row = dict(row)
            new_row["seq"] = str(seq)
            new_row["decision"] = "keep"
            new_row["output_file"] = f"images/{final_name}"
            updated_rows.append(new_row)
    except Exception:
        rollback_staged(staged, finalized)
        raise

    backup_csv = ensure_unique_backup_path(csv_path)
    shutil.copy2(csv_path, backup_csv)
    write_rows(csv_path, fieldnames, updated_rows)

    keep_csv = scene_dir / "selected_frames_keep.csv"
    write_rows(keep_csv, fieldnames, updated_rows)

    print("Finalize mode: inplace images")
    print(f"Kept frames: {len(updated_rows)}")
    print(f"Dropped rows: {dropped_rows}")
    print(f"Removed dropped image files: {removed_drop_files}")
    print(f"Images dir: {images_dir}")
    print(f"Filename prefix: {resolved_prefix}")
    print(f"CSV backup: {backup_csv}")
    print(f"Updated CSV: {csv_path}")
    print(f"Keep CSV: {keep_csv}")


def apply_decisions(
    scene_dir: Path,
    csv_name: str,
    output_name: str,
    clean_output: bool,
    finalize_inplace: bool,
    filename_prefix: str,
    backup_dir: Optional[Path] = None,
) -> None:
    csv_path = scene_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if finalize_inplace:
        finalize_in_place(scene_dir, csv_name, filename_prefix, backup_dir=backup_dir)
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
        help="Scene directory containing selected_frames.csv and images/",
    )
    parser.add_argument(
        "--csv",
        default="selected_frames.csv",
        help="CSV filename under scene_dir (default=selected_frames.csv)",
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
        "--filename-prefix",
        default="",
        help="Filename prefix used by --finalize-in-place. Default is extracted from report/video.",
    )
    parser.add_argument(
        "--finalize-in-place",
        action="store_true",
        help="Drop=remove and Keep=renumber inside scene_dir/images, then update selected_frames.csv",
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

    backup_dir: Optional[Path] = None
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
            filename_prefix=args.filename_prefix,
            backup_dir=backup_dir,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
