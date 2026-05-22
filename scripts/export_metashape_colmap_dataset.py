from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.metashape_colmap_dataset import export_metashape_colmap_dataset


def _load_views(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("views") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("views-json must contain a views list")
    return [dict(item) for item in raw if isinstance(item, dict) and bool(item.get("enabled", True))]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a mixed Metashape XML/PLY result as a COLMAP text dataset.")
    parser.add_argument("--scene", required=True, help="Scene folder")
    parser.add_argument("--images", required=True, help="Source images directory")
    parser.add_argument("--masks", default="", help="Source masks directory")
    parser.add_argument("--xml", required=True, help="Metashape camera XML")
    parser.add_argument("--ply", default="", help="Metashape point-cloud PLY")
    parser.add_argument("--output", required=True, help="Output COLMAP dataset root")
    parser.add_argument("--views-json", required=True, help="View configuration JSON")
    parser.add_argument("--scale", type=float, default=1.0, help="ERP output view scale relative to source height")
    parser.add_argument("--output-format", default="jpg", help="Output image format for generated views")
    parser.add_argument("--undistort-alpha", type=float, default=1.0, help="OpenCV undistort alpha for frame cameras")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = export_metashape_colmap_dataset(
            scene_dir=Path(args.scene),
            images_dir=Path(args.images),
            masks_dir=Path(args.masks) if args.masks else None,
            xml_path=Path(args.xml),
            ply_path=Path(args.ply) if args.ply else None,
            output_dir=Path(args.output),
            views=_load_views(Path(args.views_json)),
            output_scale=args.scale,
            output_format=args.output_format,
            undistort_alpha=args.undistort_alpha,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved mixed Metashape COLMAP dataset: {result.output_dir}")
    print(f"Images: {result.image_count}")
    print(f"Cameras: {result.camera_count}")
    print(f"Actions: {json.dumps(result.action_counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
