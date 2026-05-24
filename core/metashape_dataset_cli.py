"""CLI helpers for Metashape-derived dataset exports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.dataset_job_runner import run_dataset_job_payload
from core.dataset_job_spec import (
    JOB_KIND_METASHAPE_COLMAP,
    JOB_KIND_METASHAPE_NERF,
    load_dataset_job,
    metashape_colmap_job,
    metashape_nerf_job,
)


def main_colmap(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser(
        "Export a mixed Metashape XML/PLY result as a COLMAP text dataset.",
        final_orientation_choices=("none", "lichtfeld", "realityscan"),
    )
    args = parser.parse_args(argv)
    try:
        payload = _payload_from_args(args, kind=JOB_KIND_METASHAPE_COLMAP)
        run_dataset_job_payload(payload)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main_nerf(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser("Export a mixed Metashape XML/PLY result as a NeRF JSON/PLY dataset.")
    args = parser.parse_args(argv)
    try:
        payload = _payload_from_args(args, kind=JOB_KIND_METASHAPE_NERF)
        run_dataset_job_payload(payload)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser(
    description: str,
    *,
    final_orientation_choices: tuple[str, ...] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--job", default="", help="Versioned dataset job JSON")
    parser.add_argument("--scene", default="", help="Scene folder")
    parser.add_argument("--images", default="", help="Source images directory")
    parser.add_argument("--masks", default="", help="Source masks directory")
    parser.add_argument("--xml", default="", help="Metashape camera XML")
    parser.add_argument("--ply", default="", help="Metashape point-cloud PLY")
    parser.add_argument("--output", default="", help="Output dataset root")
    parser.add_argument("--views-json", default="", help="View configuration JSON")
    parser.add_argument("--scale", type=float, default=1.0, help="ERP output view scale relative to source height")
    parser.add_argument("--output-format", default="jpg", help="Output image format for generated views")
    parser.add_argument("--output-bit-depth", default="8", choices=("8", "source"), help="Generated image bit depth")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPG/WebP output quality")
    parser.add_argument("--undistort-alpha", type=float, default=1.0, help="OpenCV undistort alpha for frame cameras")
    parser.add_argument("--axis-transform", default="none", choices=("none", "postshot", "brush"), help="World axis profile")
    parser.add_argument(
        "--final-orientation",
        default="none",
        choices=final_orientation_choices,
        help="Optional final orientation profile",
    )
    return parser


def _payload_from_args(args: argparse.Namespace, *, kind: str) -> dict[str, Any]:
    if args.job:
        return load_dataset_job(args.job, expected_kind=kind)
    if not all((args.scene, args.images, args.xml, args.output, args.views_json)):
        raise ValueError("--scene, --images, --xml, --output, and --views-json are required unless --job is used")
    views = _load_views(Path(args.views_json))
    common = {
        "scene_dir": Path(args.scene),
        "images_dir": Path(args.images),
        "masks_dir": Path(args.masks) if args.masks else None,
        "xml_path": Path(args.xml),
        "ply_path": Path(args.ply) if args.ply else None,
        "output_dir": Path(args.output),
        "views": views,
        "output_scale": args.scale,
        "output_format": args.output_format,
        "output_bit_depth": args.output_bit_depth,
        "jpg_quality": args.jpg_quality,
        "undistort_alpha": args.undistort_alpha,
        "axis_transform": args.axis_transform,
        "final_orientation": args.final_orientation,
    }
    if kind == JOB_KIND_METASHAPE_COLMAP:
        return metashape_colmap_job(**common)
    if kind == JOB_KIND_METASHAPE_NERF:
        return metashape_nerf_job(**common)
    raise ValueError(f"Unsupported Metashape dataset kind: {kind}")


def _load_views(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("views") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("views-json must contain a views list")
    return [dict(item) for item in raw if isinstance(item, dict) and bool(item.get("enabled", True))]
