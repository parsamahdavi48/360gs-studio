"""CLI adapter for cubemap transform/image export."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from core.colmap_rig_export import DEFAULT_RIG_NAME, prepare_views_for_colmap, write_rig_config_json
from core.cubemap_transforms_json import (
    FINAL_ORIENTATION_NONE,
    _parse_positive_int_or_auto,
    collect_image_files,
    convert_images,
    convert_images_colmap_rig,
    frame_yaw_offset,
    infer_image_only_sizes,
    load_custom_views,
    make_default_views,
    transform_json,
    write_colmap_rig_metadata,
    write_image_only_metadata,
)
from core.orientation_correction import FINAL_ORIENTATION_CHOICES
from core.realityscan_xmp import (
    REALITYSCAN_CALIBRATION_PRIORS,
    REALITYSCAN_COORDINATE_MODES,
    REALITYSCAN_POSE_PRIORS,
    write_realityscan_mask_layers,
    write_realityscan_xmp_sidecars,
)

EXAMPLE_TEXT = """Example:
  python -m core.cubemap_transforms_json .
  python -m core.cubemap_transforms_json . ./output --yaw 45 --stitch 2.5
  python -m core.cubemap_transforms_json . ./output --views-json views_config.json
  python -m core.cubemap_transforms_json . ./output --image-only --views-json views_config.json
  python -m core.cubemap_transforms_json . ./output --image-only --colmap-rig --views-json views_config.json
  python -m core.cubemap_transforms_json . ./output/realityscan --views-json views_config.json --realityscan-xmp
"""


class CubemapArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help()
        sys.stderr.write(f"\n{message}\n")
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = CubemapArgumentParser(
        description="Convert transforms.json from equirectangular to cubemap views.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_TEXT,
    )
    parser.add_argument("input_dir", help="Input directory containing transforms.json and images")
    parser.add_argument("output_dir", nargs="?", help="Output directory (default=<input_dir>/output)")
    parser.add_argument("--json", help="transforms.json filename override (default='transforms.json')")
    parser.add_argument(
        "--image-dir",
        "--image_dir",
        dest="image_dir",
        help="Input equirectangular image directory for transforms.json conversion (default=<input_dir>)",
    )
    parser.add_argument("--mask_dir", help="Input mask images directory (default=<input_dir>/masks)")
    parser.add_argument("--mask_from_alpha", action="store_true", help="Extract masks from alpha channel")
    parser.add_argument("--invert_masks", action="store_true", help="Invert output masks (black/white)")
    parser.add_argument("--yaw", type=float, default=45.0, help="Yaw offset for default 6 views")
    parser.add_argument("--stitch", type=float, default=0.0, help="Stitch avoid angle for default 6 views")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of view for each output view")
    parser.add_argument(
        "--output_scale",
        "--output-scale",
        dest="output_scale",
        type=float,
        default=0.5,
        help="Output face size ratio to input image height (0.5=half, 1.0=full)",
    )
    parser.add_argument("--views-json", dest="views_json", help="Custom views JSON path")
    parser.add_argument("--no_bottom", action="store_true", help="Exclude bottom face in default mode")
    parser.add_argument("--no_top", action="store_true", help="Exclude top face in default mode")
    parser.add_argument("--no_image", action="store_true", help="Convert transforms.json only")
    parser.add_argument(
        "--skip-images",
        "--skip_images",
        dest="skip_images",
        action="store_true",
        help="Do not write converted view images",
    )
    parser.add_argument(
        "--skip-masks",
        "--skip_masks",
        dest="skip_masks",
        action="store_true",
        help="Do not write converted view masks",
    )
    parser.add_argument(
        "--image-only",
        "--image_only",
        dest="image_only",
        action="store_true",
        help="Convert equirectangular images/masks without reading or writing transforms.json",
    )
    parser.add_argument(
        "--colmap-rig",
        "--colmap_rig",
        dest="colmap_rig",
        action="store_true",
        help=(
            "Export image-only outputs as a COLMAP rig dataset under output/colmap_rig. "
            "This implies image-only mode and writes rig_config.json."
        ),
    )
    parser.add_argument(
        "--colmap-rig-name",
        "--colmap_rig_name",
        dest="colmap_rig_name",
        default=DEFAULT_RIG_NAME,
        help=f"COLMAP rig name for --colmap-rig (default: {DEFAULT_RIG_NAME})",
    )
    parser.add_argument("--no_transform", action="store_true", help="Disable axis transform (for LichtFeld Studio)")
    parser.add_argument("--duplicate", action="store_true", help="Allow duplicated image files")
    parser.add_argument("--brush", action="store_true", help="Transform axes for Brush")
    parser.add_argument(
        "--final-orientation",
        "--final_orientation",
        dest="final_orientation",
        default=FINAL_ORIENTATION_NONE,
        choices=FINAL_ORIENTATION_CHOICES,
        help=(
            "Apply a final dataset orientation correction to output camera poses and pointcloud.ply. "
            "'lichtfeld' matches LichtFeld Studio Cube6/3DGUT coordinates; default is 'none'."
        ),
    )
    parser.add_argument(
        "--output-format",
        "--output_format",
        dest="output_format",
        default="auto",
        choices=["auto", "jpg", "png", "tiff", "tif", "webp"],
        help="Output image format. 'auto' (default) preserves the input format.",
    )
    parser.add_argument(
        "--output-bit-depth",
        "--output_bit_depth",
        dest="output_bit_depth",
        default="8",
        choices=["8", "source"],
        help=(
            "Output image bit depth. '8' (default) down-converts images for broad "
            "3DGS tool compatibility; 'source' preserves PNG/TIFF source bit depth."
        ),
    )
    parser.add_argument(
        "--jpg-quality",
        "--jpg_quality",
        dest="jpg_quality",
        type=int,
        default=95,
        help="JPEG/WebP quality (1-100, default 95).",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Image conversion worker processes: 'auto' or a positive integer (default=auto).",
    )
    parser.add_argument(
        "--remap-cache-limit",
        "--remap_cache_limit",
        dest="remap_cache_limit",
        default="auto",
        help="Per-worker yaw remap table cache limit: 'auto' or a positive integer (default=auto, memory-aware).",
    )
    parser.add_argument(
        "--yaw-offset-per-frame",
        "--yaw_offset_per_frame",
        dest="yaw_offset_per_frame",
        type=float,
        default=30.0,
        help=(
            "Per-frame cubemap yaw rotation step (degrees, default 30.0). "
            "Each unique input image gets yaw offset = frame_index * step (mod 360). "
            "Diversifies sampling angles to reduce 3DGS face-boundary artifacts. "
            "Set to 0 to disable (matches legacy behavior)."
        ),
    )
    parser.add_argument(
        "--realityscan-xmp",
        "--realityscan_xmp",
        dest="realityscan_xmp",
        action="store_true",
        help="Write RealityScan XMP sidecars next to the exported cubemap images.",
    )
    parser.add_argument(
        "--realityscan-pose-prior",
        "--realityscan_pose_prior",
        dest="realityscan_pose_prior",
        choices=list(REALITYSCAN_POSE_PRIORS),
        default="exact",
        help="RealityScan xcr:PosePrior value for generated XMP sidecars (default=exact).",
    )
    parser.add_argument(
        "--realityscan-calibration-prior",
        "--realityscan_calibration_prior",
        dest="realityscan_calibration_prior",
        choices=list(REALITYSCAN_CALIBRATION_PRIORS),
        default="exact",
        help="RealityScan xcr:CalibrationPrior value for generated XMP sidecars (default=exact).",
    )
    parser.add_argument(
        "--realityscan-coordinates",
        "--realityscan_coordinates",
        dest="realityscan_coordinates",
        choices=list(REALITYSCAN_COORDINATE_MODES),
        default="auto",
        help=(
            "RealityScan xcr:Coordinates mode for generated XMP sidecars. "
            "auto writes relative coordinates for exact pose priors and absolute otherwise."
        ),
    )
    parser.add_argument(
        "--realityscan-rig-name",
        "--realityscan_rig_name",
        dest="realityscan_rig_name",
        default="stechdrive-cubemap",
        help="Stable rig name used when --realityscan-include-rig writes RealityScan XMP Rig GUIDs.",
    )
    parser.add_argument(
        "--realityscan-include-rig",
        "--realityscan_include_rig",
        dest="realityscan_include_rig",
        action="store_true",
        help=(
            "Also write Rig/RigInstance/RigPoseIndex XMP metadata. "
            "This is experimental because RealityScan 2.1.1 can skip sparse tie-point export for these image rigs."
        ),
    )
    parser.add_argument(
        "--no-realityscan-mask-layers",
        "--no_realityscan_mask_layers",
        dest="realityscan_mask_layers",
        action="store_false",
        help="Do not copy converted masks to RealityScan image-layer names (*.jpg.mask.png).",
    )
    parser.set_defaults(realityscan_mask_layers=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.colmap_rig:
        args.image_only = True
    if args.image_only and args.final_orientation != FINAL_ORIENTATION_NONE:
        _exit_error("--final-orientation requires transforms.json conversion, not --image-only")
    if args.image_only and args.realityscan_xmp:
        _exit_error("--realityscan-xmp requires transforms.json conversion, not --image-only")

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else f"{input_dir}/output"
    input_json = args.json if args.json else "transforms.json"

    image_dir = args.image_dir if args.image_dir else input_dir
    mask_dir = args.mask_dir if args.mask_dir else f"{input_dir}/masks"
    output_image_dir = f"{output_dir}/images"
    output_mask_dir = f"{output_dir}/masks"

    if args.mask_dir and not os.path.isdir(mask_dir):
        _exit_error(f"mask_dir '{mask_dir}' not found")
    if args.image_dir and not os.path.isdir(image_dir):
        _exit_error(f"image_dir '{image_dir}' not found")

    if args.fov <= 0 or args.fov >= 180:
        _exit_error("fov must be in (0, 180)")
    if args.output_scale <= 0 or args.output_scale > 1.0:
        _exit_error("output_scale must be in (0, 1.0]")
    try:
        _parse_positive_int_or_auto(args.workers, "--workers")
        _parse_positive_int_or_auto(args.remap_cache_limit, "--remap-cache-limit")
    except ValueError as exc:
        _exit_error(str(exc))

    if args.views_json:
        try:
            views = load_custom_views(args.views_json)
        except Exception as exc:
            _exit_error(f"failed to parse views-json: {exc}")
    else:
        views = make_default_views(args.yaw, args.stitch, args.no_top, args.no_bottom)

    if not views:
        _exit_error("no views to export")

    for view in views:
        print(f"{view['name']}: yaw={view['yaw']},pitch={view['pitch']}")

    if args.image_only:
        image_dir = os.path.join(input_dir, "images")
        if not os.path.isdir(image_dir):
            _exit_error(f"images directory not found: {image_dir}")
        image_files = collect_image_files(image_dir)
        if not image_files:
            _exit_error(f"no images found in {image_dir}")
        input_size, output_size = infer_image_only_sizes(image_dir, image_files, args.output_scale)
        if args.colmap_rig:
            if args.yaw_offset_per_frame != 0.0:
                print("COLMAP rig export fixes per-frame yaw rotation to 0 degrees.")
            frame_yaw_offsets = [0.0 for _ in image_files]
            prepared_views = prepare_views_for_colmap([{**view, "fov": float(args.fov)} for view in views])
            rig_path = write_rig_config_json(
                output_dir,
                prepared_views,
                (output_size, output_size),
                rig_name=args.colmap_rig_name,
            )
            write_colmap_rig_metadata(
                output_dir=output_dir,
                image_dir=image_dir,
                mask_dir=mask_dir,
                image_files=image_files,
                prepared_views=prepared_views,
                fov=args.fov,
                output_scale=args.output_scale,
                input_size=input_size,
                output_size=output_size,
                rig_name=args.colmap_rig_name,
                export_images=not args.no_image and not args.skip_images,
                export_masks=not args.no_image and not args.skip_masks,
            )
            print(f"COLMAP rig export: {len(image_files)} source images")
            print(f"Saved rig_config.json: {rig_path}")
            views = prepared_views
        else:
            frame_yaw_offsets = [frame_yaw_offset(i, args.yaw_offset_per_frame) for i in range(len(image_files))]
            write_image_only_metadata(
                output_dir=output_dir,
                image_dir=image_dir,
                mask_dir=mask_dir,
                image_files=image_files,
                views=views,
                fov=args.fov,
                output_scale=args.output_scale,
                input_size=input_size,
                output_size=output_size,
                yaw_offset_per_frame=args.yaw_offset_per_frame,
                export_images=not args.no_image and not args.skip_images,
                export_masks=not args.no_image and not args.skip_masks,
            )
            print(f"Image-only export: {len(image_files)} source images")
    else:
        image_files, frame_yaw_offsets, input_size, output_size = transform_json(
            input_dir=input_dir,
            input_json=input_json,
            image_dir=image_dir,
            output_dir=output_dir,
            views=views,
            fov=args.fov,
            output_scale=args.output_scale,
            no_transform=args.no_transform,
            allow_duplicate=args.duplicate,
            brush_mode=args.brush,
            yaw_offset_per_frame=args.yaw_offset_per_frame,
            final_orientation=args.final_orientation,
            output_format=args.output_format,
        )
    if not image_files:
        raise SystemExit(1)

    realityscan_manifest = None
    if args.realityscan_xmp:
        try:
            realityscan_manifest = write_realityscan_xmp_sidecars(
                Path(output_dir),
                pose_prior=args.realityscan_pose_prior,
                calibration_prior=args.realityscan_calibration_prior,
                coordinates=args.realityscan_coordinates,
                rig_name=args.realityscan_rig_name,
                include_rig=args.realityscan_include_rig,
            )
        except Exception as exc:
            _exit_error(f"failed to write RealityScan XMP sidecars: {exc}")
        print(f"RealityScan XMP sidecars: {realityscan_manifest['xmp_count']}")

    if args.yaw_offset_per_frame != 0.0 and not args.colmap_rig:
        unique_offsets = sorted({round(y, 3) for y in frame_yaw_offsets})
        print(f"Per-frame yaw rotation: step={args.yaw_offset_per_frame:g}deg, unique offsets={len(unique_offsets)}")

    export_images = not args.no_image and not args.skip_images
    export_masks = not args.no_image and not args.skip_masks

    if export_images or export_masks:
        if args.colmap_rig:
            convert_images_colmap_rig(
                image_files=image_files,
                input_size=input_size,
                output_size=output_size,
                views=views,
                fov=args.fov,
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=output_dir,
                rig_name=args.colmap_rig_name,
                mask_from_alpha=args.mask_from_alpha,
                invert_masks=args.invert_masks,
                output_format=args.output_format,
                output_bit_depth=args.output_bit_depth,
                jpg_quality=args.jpg_quality,
                export_images=export_images,
                export_masks=export_masks,
                workers=args.workers,
                remap_cache_limit=args.remap_cache_limit,
            )
            return
        convert_images(
            image_files=image_files,
            input_size=input_size,
            output_size=output_size,
            views=views,
            fov=args.fov,
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_image_dir=output_image_dir,
            output_mask_dir=output_mask_dir,
            mask_from_alpha=args.mask_from_alpha,
            invert_masks=args.invert_masks,
            output_format=args.output_format,
            output_bit_depth=args.output_bit_depth,
            jpg_quality=args.jpg_quality,
            frame_yaw_offsets=frame_yaw_offsets,
            export_images=export_images,
            export_masks=export_masks,
            workers=args.workers,
            remap_cache_limit=args.remap_cache_limit,
        )
    if args.realityscan_xmp and args.realityscan_mask_layers and export_masks:
        try:
            realityscan_manifest = write_realityscan_mask_layers(
                Path(output_dir),
                manifest=realityscan_manifest,
            )
        except Exception as exc:
            _exit_error(f"failed to write RealityScan mask layers: {exc}")
        print(f"RealityScan mask layers: {realityscan_manifest['mask_layer_count']}")


def _exit_error(message: str) -> None:
    print(f"Error: {message}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
