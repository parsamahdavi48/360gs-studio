"""CLI adapter for RealityScan CSV/PLY to NeRF-style transforms conversion."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from core.realityscan_to_transforms import IMAGE_PATH_MODES, TARGET_PROFILE_CHOICES, TARGET_PROFILE_LICHTFELD, convert


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert RealityScan CSV + PLY exports to NeRF-style transforms.json + pointcloud.ply.",
    )
    parser.add_argument("csv_path", help="RealityScan registration CSV exported for Postshot")
    parser.add_argument("output_dir", help="Output dataset directory")
    parser.add_argument("--images-dir", help="Directory containing images referenced by the CSV")
    parser.add_argument("--ply", help="RealityScan PLY to convert to pointcloud.ply")
    parser.add_argument(
        "--source-transforms",
        help="Existing transforms.json used only for per-image metadata/intrinsics; defaults to CSV folder transforms.json",
    )
    parser.add_argument("--masks-dir", help="Optional masks directory; matching masks are written as frame mask_path")
    parser.add_argument(
        "--image-path-mode",
        choices=IMAGE_PATH_MODES,
        default="images-prefix",
        help="How frame file_path values are written (default: images-prefix)",
    )
    parser.add_argument(
        "--target-profile",
        choices=TARGET_PROFILE_CHOICES,
        default=TARGET_PROFILE_LICHTFELD,
        help="Output coordinate profile (default: lichtfeld)",
    )
    parser.add_argument("--json-name", default="transforms.json", help="Output JSON filename")
    parser.add_argument("--pointcloud-name", default="pointcloud.ply", help="Output PLY filename")
    parser.add_argument("--no-mask-paths", action="store_true", help="Do not write frame mask_path entries")
    parser.add_argument("--skip-missing-images", action="store_true", help="Skip CSV rows whose images are missing")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = convert(
            Path(args.csv_path),
            Path(args.output_dir),
            images_dir=Path(args.images_dir) if args.images_dir else None,
            ply_path=Path(args.ply) if args.ply else None,
            source_transforms=Path(args.source_transforms) if args.source_transforms else None,
            masks_dir=Path(args.masks_dir) if args.masks_dir else None,
            image_path_mode=args.image_path_mode,
            target_profile=args.target_profile,
            json_name=args.json_name,
            pointcloud_name=args.pointcloud_name,
            write_mask_paths=not args.no_mask_paths,
            skip_missing_images=args.skip_missing_images,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved transforms.json: {result['transforms']}")
    if result["pointcloud"]:
        print(f"Saved pointcloud.ply: {result['pointcloud']}")
    print(f"Frames: {result['num_frames']} / CSV rows: {result['num_csv_rows']}")
    print(f"OpenCV/distorted frames: {result['num_opencv_frames']}")
    print(f"Mask paths: {result['num_mask_paths']}")
    print(f"Target profile: {result['target_profile']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
