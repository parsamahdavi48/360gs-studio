#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import List


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


def apply_decisions(scene_dir: Path, csv_name: str, output_name: str, clean_output: bool) -> None:
    csv_path = scene_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = scene_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if clean_output:
        removed = clean_output_dir(output_dir)
        if removed > 0:
            print(f"Removed {removed} existing image files from {output_dir}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    copied = 0
    skipped = 0
    missing: List[str] = []

    for row in rows:
        decision = row.get("decision", "keep").strip().lower()
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

    keep_csv = output_dir / "selected_frames_keep.csv"
    with keep_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            if row.get("decision", "keep").strip().lower() != "drop":
                writer.writerow(row)

    print(f"Keep CSV: {keep_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply keep/drop decisions from selected_frames.csv and export keep frames for Metashape."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).resolve()

    try:
        apply_decisions(scene_dir, args.csv, args.output, args.clean_output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
