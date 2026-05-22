"""Prepare a COLMAP SfM project from mixed ERP and normal scene images."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.colmap_mixed_project import prepare_colmap_mixed_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", help="Scene directory containing images/ and optional masks/")
    parser.add_argument("output_dir", help="Scene output directory; the project is written to output/colmap_rig/")
    parser.add_argument("--views-json", "--views_json", dest="views_json", required=True)
    parser.add_argument("--output-scale", "--output_scale", dest="output_scale", type=float, default=0.5)
    parser.add_argument("--output-format", "--output_format", dest="output_format", default="auto")
    parser.add_argument("--output-bit-depth", "--output_bit_depth", dest="output_bit_depth", default="8")
    parser.add_argument("--jpg-quality", "--jpg_quality", dest="jpg_quality", type=int, default=95)
    parser.add_argument("--skip-images", "--skip_images", dest="skip_images", action="store_true")
    parser.add_argument("--skip-masks", "--skip_masks", dest="skip_masks", action="store_true")
    parser.add_argument("--invert-masks", "--invert_masks", dest="invert_masks", action="store_true")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--remap-cache-limit", "--remap_cache_limit", dest="remap_cache_limit", default="auto")
    parser.add_argument("--rig-name", "--rig_name", dest="rig_name", default="rig1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = prepare_colmap_mixed_project(
            args.scene_dir,
            args.output_dir,
            views_json=args.views_json,
            output_scale=args.output_scale,
            output_format=args.output_format,
            output_bit_depth=args.output_bit_depth,
            jpg_quality=args.jpg_quality,
            write_images=not args.skip_images,
            write_masks=not args.skip_masks,
            invert_masks=args.invert_masks,
            workers=args.workers,
            remap_cache_limit=args.remap_cache_limit,
            rig_name=args.rig_name,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for warning in result.warnings:
        print(f"Warning: {warning}", flush=True)
    print(f"Saved COLMAP mixed project manifest: {result.manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
