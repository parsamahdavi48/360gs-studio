from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dataset_job_spec import JOB_KIND_METASHAPE_NERF, load_dataset_job
from core.metashape_nerf_dataset import export_metashape_nerf_dataset


def _load_views(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("views") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("views-json must contain a views list")
    return [dict(item) for item in raw if isinstance(item, dict) and bool(item.get("enabled", True))]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a mixed Metashape XML/PLY result as a NeRF JSON/PLY dataset.")
    parser.add_argument("--job", default="", help="Versioned dataset job JSON")
    parser.add_argument("--scene", default="", help="Scene folder")
    parser.add_argument("--images", default="", help="Source images directory")
    parser.add_argument("--masks", default="", help="Source masks directory")
    parser.add_argument("--xml", default="", help="Metashape camera XML")
    parser.add_argument("--ply", default="", help="Metashape point-cloud PLY")
    parser.add_argument("--output", default="", help="Output NeRF dataset root")
    parser.add_argument("--views-json", default="", help="View configuration JSON")
    parser.add_argument("--scale", type=float, default=1.0, help="ERP output view scale relative to source height")
    parser.add_argument("--output-format", default="jpg", help="Output image format for generated views")
    parser.add_argument("--output-bit-depth", default="8", choices=("8", "source"), help="Generated image bit depth")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPG/WebP quality")
    parser.add_argument("--undistort-alpha", type=float, default=1.0, help="OpenCV undistort alpha for frame cameras")
    parser.add_argument("--axis-transform", default="none", choices=("none", "postshot", "brush"), help="Output axis profile")
    parser.add_argument("--final-orientation", default="none", help="Optional final orientation profile")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.job:
            job = load_dataset_job(args.job, expected_kind=JOB_KIND_METASHAPE_NERF)
            views = [dict(item) for item in job.get("views", []) if isinstance(item, dict) and bool(item.get("enabled", True))]
            result = export_metashape_nerf_dataset(
                scene_dir=Path(str(job["scene_dir"])),
                images_dir=Path(str(job["images_dir"])),
                masks_dir=Path(str(job["masks_dir"])) if str(job.get("masks_dir") or "") else None,
                xml_path=Path(str(job["xml_path"])),
                ply_path=Path(str(job["ply_path"])) if str(job.get("ply_path") or "") else None,
                output_dir=Path(str(job["output_dir"])),
                views=views,
                output_scale=float(job.get("output_scale", 1.0)),
                output_format=str(job.get("output_format") or "jpg"),
                output_bit_depth=str(job.get("output_bit_depth") or "8"),
                jpg_quality=int(job.get("jpg_quality", 95)),
                undistort_alpha=float(job.get("undistort_alpha", 1.0)),
                axis_transform=str(job.get("axis_transform") or "none"),
                final_orientation=str(job.get("final_orientation") or "none"),
            )
        else:
            if not all((args.scene, args.images, args.xml, args.output, args.views_json)):
                raise ValueError("--scene, --images, --xml, --output, and --views-json are required unless --job is used")
            result = export_metashape_nerf_dataset(
                scene_dir=Path(args.scene),
                images_dir=Path(args.images),
                masks_dir=Path(args.masks) if args.masks else None,
                xml_path=Path(args.xml),
                ply_path=Path(args.ply) if args.ply else None,
                output_dir=Path(args.output),
                views=_load_views(Path(args.views_json)),
                output_scale=args.scale,
                output_format=args.output_format,
                output_bit_depth=args.output_bit_depth,
                jpg_quality=args.jpg_quality,
                undistort_alpha=args.undistort_alpha,
                axis_transform=args.axis_transform,
                final_orientation=args.final_orientation,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved mixed Metashape NeRF dataset: {result.output_dir}")
    print(f"transforms.json: {result.transforms_json}")
    if result.pointcloud:
        print(f"pointcloud.ply: {result.pointcloud}")
    print(f"Frames: {result.frame_count}")
    print(f"Actions: {json.dumps(result.action_counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
