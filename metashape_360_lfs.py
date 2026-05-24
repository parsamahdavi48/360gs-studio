from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from core.metashape_preprocess import export_metashape_equirectangular_dataset


def convert_metashape_to_lichtfeld(
    images_dir: str | Path,
    xml_path: str | Path,
    output_dir: str | Path | None = None,
    ply_path: str | Path | None = None,
    fix_upside_down: bool = True,
    scale: float = 1.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compatibility API for the historical metashape_360_lfs command."""
    xml = Path(xml_path)
    result = export_metashape_equirectangular_dataset(
        images_dir=images_dir,
        xml_path=xml,
        output_dir=Path(output_dir) if output_dir is not None else xml.parent,
        ply_path=ply_path,
        fix_upside_down=fix_upside_down,
        scale=scale,
        verbose=verbose,
    )
    return result.as_legacy_summary()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Metashape spherical XML/PLY to transforms.json.")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--xml", type=Path, required=True, help="Path to Metashape camera XML")
    parser.add_argument("--ply", type=Path, default=None, help="Optional Metashape point-cloud PLY")
    parser.add_argument("--output", type=Path, default=None, help="Output directory; defaults to the XML folder")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for camera positions and points")
    parser.add_argument("--no-fix-rotation", action="store_true", help="Disable the Metashape orientation correction")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.images.is_dir():
            raise FileNotFoundError(f"Images directory not found: {args.images}")
        if not args.xml.is_file():
            raise FileNotFoundError(f"Metashape XML not found: {args.xml}")
        if args.ply is not None and not args.ply.is_file():
            raise FileNotFoundError(f"Metashape PLY not found: {args.ply}")
        result = convert_metashape_to_lichtfeld(
            images_dir=args.images,
            xml_path=args.xml,
            output_dir=args.output,
            ply_path=args.ply,
            fix_upside_down=not args.no_fix_rotation,
            scale=args.scale,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1

    if not args.quiet:
        print("Conversion complete!", flush=True)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
