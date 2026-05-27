"""CLI entry point for COLMAP mixed-project preparation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from core.colmap_mixed_project import prepare_colmap_mixed_project
from core.sfm_job_runner import run_sfm_job_payload
from core.sfm_job_spec import JOB_KIND_COLMAP_MIXED_PROJECT, load_sfm_job


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a COLMAP SfM project from mixed ERP and normal scene images."
    )
    parser.add_argument("scene_dir", nargs="?", help="Scene directory containing images/ and optional masks/")
    parser.add_argument("output_dir", nargs="?", help="Scene output directory; the project is written to output/colmap_rig/")
    parser.add_argument("--job", default="", help="Versioned SfM job JSON")
    parser.add_argument("--views-json", "--views_json", dest="views_json", default="")
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.job:
        try:
            job = load_sfm_job(args.job, expected_kind=JOB_KIND_COLMAP_MIXED_PROJECT)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return _run_job(job)

    if not args.scene_dir or not args.output_dir or not args.views_json:
        print("Error: scene_dir, output_dir, and --views-json are required unless --job is used", file=sys.stderr)
        return 1
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
    print(f"Saved COLMAP mixed project metadata: {result.manifest_path}", flush=True)
    return 0


def _run_job(job: dict[str, Any]) -> int:
    try:
        run_sfm_job_payload(job)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
