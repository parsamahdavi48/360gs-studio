"""CLI adapter for applying Step 2 keep/drop frame decisions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from core.apply_frame_decisions import apply_decisions


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--renumber-kept-images",
        action="store_true",
        help=(
            "With --finalize-in-place, rename kept files in images/ to frame_000001.ext order "
            "and update frame metadata paths. Refuses to run after masks/output/Step 4 metadata exist."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
            renumber_kept_images=args.renumber_kept_images,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
