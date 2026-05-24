"""CLI adapter for SphereSfM sparse model to transforms.json conversion."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from core.spheresfm_to_transforms import convert


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a SphereSfM sparse model to equirectangular transforms.json.",
    )
    parser.add_argument("model_dir", help="SphereSfM sparse model directory, or sparse root containing 0/")
    parser.add_argument("output_dir", help="Output directory for transforms.json and pointcloud.ply")
    parser.add_argument("--images-dir", required=True, help="Source equirectangular images directory")
    parser.add_argument(
        "--image-path-mode",
        choices=("relative", "images-prefix", "relative-to-output", "absolute"),
        default="relative",
        help="How frame file_path values are written (default: relative to --images-dir)",
    )
    parser.add_argument(
        "--keep-colmap-camera-convention",
        action="store_true",
        help="Do not flip camera Y/Z axes from COLMAP/OpenCV to OpenGL-style transforms.",
    )
    parser.add_argument("--no-pointcloud", action="store_true", help="Do not write pointcloud.ply")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        raise SystemExit(1)

    try:
        result = convert(
            Path(args.model_dir),
            Path(args.output_dir),
            images_dir,
            image_path_mode=args.image_path_mode,
            opengl_camera=not args.keep_colmap_camera_convention,
            write_pointcloud=not args.no_pointcloud,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Using sparse model: {result['model_dir']}")
    print(f"Saved transforms.json: {result['transforms']}")
    if result["pointcloud"]:
        print(f"Saved pointcloud.ply: {result['pointcloud']}")
    print(f"Images: {result['num_images']}")
    print(f"Points: {result['num_points']}")


if __name__ == "__main__":
    main()
