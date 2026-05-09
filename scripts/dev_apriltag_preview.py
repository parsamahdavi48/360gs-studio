#!/usr/bin/env python3
"""Create a contact sheet and projection CSV for a synthetic AprilTag run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.apriltag_detection import available_families
from devtools.apriltag.preview import create_synthetic_preview


def _parse_tag_ids(values: list[str]) -> set[int] | None:
    ids: set[int] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            ids.add(int(part))
    return ids or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create AprilTag synthetic run preview artifacts.")
    parser.add_argument("synthetic_dir", type=Path, help="Synthetic run directory containing transforms.json")
    parser.add_argument("--family", default="tag36h11", choices=available_families(), help="AprilTag family")
    parser.add_argument("--tag-size-m", type=float, required=True, help="Physical detected tag square side length")
    parser.add_argument("--tag-id", action="append", default=[], help="Allowed tag ID. May be repeated or comma-separated")
    parser.add_argument("--out", type=Path, default=None, help="Preview contact sheet output path")
    parser.add_argument("--csv", type=Path, default=None, help="Projection CSV output path")
    parser.add_argument("--max-images", type=int, default=12, help="Maximum detected images in the contact sheet")
    args = parser.parse_args()
    if args.tag_size_m <= 0.0:
        parser.error("--tag-size-m must be positive")
    try:
        args.tag_ids = _parse_tag_ids(args.tag_id)
    except ValueError as e:
        parser.error(f"--tag-id must contain integers: {e}")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = create_synthetic_preview(
            args.synthetic_dir,
            family=args.family,
            tag_size_m=args.tag_size_m,
            tag_ids=args.tag_ids,
            preview_path=args.out,
            csv_path=args.csv,
            max_images=max(1, args.max_images),
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    print(f"Preview: {result.preview_path}")
    print(f"Projection CSV: {result.csv_path}")
    print(f"Detected preview images: {result.detected_images}")
    print(f"Projected injection rows: {result.projection_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

