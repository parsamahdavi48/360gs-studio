"""CLI adapter for converting transforms.json datasets to COLMAP text."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from core.transforms_to_colmap import convert


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert nerfstudio-style transforms.json to COLMAP text format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m core.transforms_to_colmap ./output\n"
            "  python -m core.transforms_to_colmap ./output ./output/colmap --ply ./output/pointcloud.ply\n"
        ),
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing transforms.json (output from cubemap_transforms_json.py)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Output directory for COLMAP text files (default=<input_dir>/colmap)",
    )
    parser.add_argument(
        "--json",
        default="transforms.json",
        help="Input transforms.json filename (default=transforms.json)",
    )
    parser.add_argument(
        "--ply",
        help="Optional PLY file for points3D.txt output",
    )
    parser.add_argument(
        "--image-prefix",
        "--image_prefix",
        dest="image_prefix",
        default="images/",
        help=(
            "Prefix in transforms.json file_path entries to strip when writing image names "
            "(default='images/'). Set to '' to keep paths as-is."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input_dir '{input_dir}' not found", file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "colmap"
    ply_path = Path(args.ply) if args.ply else None

    try:
        result = convert(
            input_dir=input_dir,
            json_name=args.json,
            output_dir=output_dir,
            ply_path=ply_path,
            image_prefix=args.image_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote cameras.txt, images.txt, points3D.txt to {result['output_dir']}")
    print(f"  Camera model: {result['camera_model']}")
    print(f"  Images: {result['num_images']}")
    print(f"  3D points: {result['num_points']}")


if __name__ == "__main__":
    main()
