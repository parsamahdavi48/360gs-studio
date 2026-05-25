"""CLI adapter for SphereSfM GPU SIFT preflight."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import core.spheresfm_gpu_preflight as preflight


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether SphereSfM GPU SIFT can run on one image.")
    parser.add_argument("--colmap", required=True, help="SphereSfM colmap executable")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--camera-params", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {args.images_dir}")
    images = preflight.iter_images(args.images_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found: {args.images_dir}")

    preflight.validate_spheresfm_colmap(args.colmap)

    preflight_images = preflight.reset_preflight_workspace(args.work_dir)
    source = images[0]
    target = preflight_images / f"preflight_000001{source.suffix.lower()}"
    shutil.copy2(source, target)
    print(f"SphereSfM GPU preflight image: {source}", flush=True)

    database = args.work_dir / "database.db"
    preflight.run_colmap_command(
        [
            args.colmap,
            "database_creator",
            "--database_path",
            str(database),
        ],
        "SphereSfM preflight database_creator",
    )
    preflight.run_colmap_command(
        preflight.build_feature_command(args.colmap, database, preflight_images, args.camera_params),
        "SphereSfM preflight feature_extractor",
    )
    print("SphereSfM GPU preflight passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
