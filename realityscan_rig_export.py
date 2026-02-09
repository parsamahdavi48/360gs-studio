#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cubemap_transforms_json import (
    build_remap,
    load_custom_views,
    make_default_views,
    mask_candidates,
    rot4,
    rotation_matrix,
)


EXAMPLE_TEXT = """Example:
  python realityscan_rig_export.py . ./realityscan_rig --views-json ./views_config.json
  python realityscan_rig_export.py . ./realityscan_rig --views-json ./views_config.json --mask_dir ./masks
"""

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
POSE_PRIOR_CHOICES = ("initial", "exact", "locked")
CALIB_PRIOR_CHOICES = ("initial", "fixed", "exact", "locked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export perspective crops + XMP pairs for RealityScan rig workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_TEXT,
    )
    parser.add_argument("input_dir", help="Input scene directory containing transforms.json and images")
    parser.add_argument("output_dir", help="RealityScan output root directory")
    parser.add_argument("--json", default="transforms.json", help="Input transforms JSON name")
    parser.add_argument("--views-json", dest="views_json", help="Custom views JSON path")
    parser.add_argument("--yaw", type=float, default=45.0, help="Yaw offset for default 6 views")
    parser.add_argument("--stitch", type=float, default=0.0, help="Stitch avoid angle for default 6 views")
    parser.add_argument("--no_top", action="store_true", help="Exclude top face in default mode")
    parser.add_argument("--no_bottom", action="store_true", help="Exclude bottom face in default mode")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of view for output views")
    parser.add_argument("--mask_dir", help="Input mask directory (default=<input_dir>/masks)")
    parser.add_argument("--mask_from_alpha", action="store_true", help="Extract masks from alpha channel")
    parser.add_argument("--invert_masks", action="store_true", help="Invert exported masks")
    parser.add_argument("--duplicate", action="store_true", help="Allow duplicate frame file_path")
    parser.add_argument("--no_transform", action="store_true", help="Disable axis transform on input poses")
    parser.add_argument(
        "--pose_prior",
        choices=POSE_PRIOR_CHOICES,
        default="exact",
        help="XMP xcr:PosePrior",
    )
    parser.add_argument(
        "--calibration_prior",
        choices=CALIB_PRIOR_CHOICES,
        default="fixed",
        help="XMP xcr:CalibrationPrior",
    )
    parser.add_argument("--focal35mm", type=float, help="Override XMP xcr:FocalLength35mm (default=derived from FOV)")
    parser.add_argument("--rig_id", help="Optional rig GUID (with or without braces)")
    parser.add_argument(
        "--coordinates",
        choices=["absolute", "relative"],
        default="absolute",
        help="XMP xcr:Coordinates",
    )
    return parser.parse_args()


def _load_views(args: argparse.Namespace) -> list[dict]:
    if args.views_json:
        return load_custom_views(args.views_json)
    return make_default_views(args.yaw, args.stitch, args.no_top, args.no_bottom)


def _load_frames(input_dir: Path, json_name: str, allow_duplicate: bool) -> list[dict]:
    json_path = input_dir / json_name
    if not json_path.is_file():
        raise FileNotFoundError(f"Input transforms JSON not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("camera_model") != "EQUIRECTANGULAR":
        raise ValueError("camera_model must be EQUIRECTANGULAR")

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames is empty in transforms JSON")

    valid: list[dict] = []
    seen: set[str] = set()
    for idx, frame in enumerate(frames):
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue

        key = file_path.lower()
        if not allow_duplicate and key in seen:
            continue

        src = input_dir / file_path
        if not src.is_file():
            continue

        try:
            transform = np.array(frame["transform_matrix"], dtype=np.float64)
        except Exception:
            print(f"Skipped frame with invalid transform_matrix: {file_path}")
            continue
        if transform.shape != (4, 4):
            print(f"Skipped frame with non 4x4 transform_matrix: {file_path}")
            continue

        seen.add(key)
        valid.append(
            {
                "index": idx,
                "file_path": file_path,
                "transform": transform,
            }
        )

    if not valid:
        raise ValueError("No readable input frames found from transforms JSON")
    return valid


def _determine_io_shape(input_dir: Path, frames: list[dict]) -> tuple[tuple[int, int], int]:
    for frame in frames:
        src = input_dir / str(frame["file_path"])
        if src.is_file():
            with Image.open(src) as img:
                w, h = img.size
            out_size = max(1, h // 2)
            return (w, h), out_size
    raise ValueError("Failed to read input image size")


def _source_key(file_path: str) -> str:
    p = Path(file_path)
    parts = list(p.parts)
    if parts and parts[0].lower() == "images" and len(parts) > 1:
        p = Path(*parts[1:])
    rel = str(p).replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    ext = Path(rel).suffix
    stem = rel[: -len(ext)] if ext else rel
    token = stem.replace("/", "__")
    if not ext:
        ext = ".jpg"
    if not SAFE_NAME_RE.match(token):
        token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
    if not token:
        token = "frame"
    return f"{token}{ext}"


def _unique_output_names(frames: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for frame in frames:
        file_path = str(frame["file_path"])
        base = _source_key(file_path)
        stem, ext = os.path.splitext(base)

        candidate = base
        idx = 1
        while candidate.lower() in used:
            candidate = f"{stem}__{idx:03d}{ext}"
            idx += 1

        used.add(candidate.lower())
        mapping[file_path] = candidate

    return mapping


def _build_remap_tables(
    input_size: tuple[int, int],
    output_size: int,
    fov: float,
    views: list[dict],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view in views:
        tables[view["name"]] = build_remap(
            input_size=input_size,
            fov_deg=fov,
            yaw_deg=float(view["yaw"]),
            pitch_deg=float(view["pitch"]),
            output_size=output_size,
        )
    return tables


def _prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "inputs": output_root / "inputs",
    }
    dirs["root"].mkdir(parents=True, exist_ok=True)
    dirs["inputs"].mkdir(parents=True, exist_ok=True)
    return dirs


def _load_equi_image(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as img:
        if img.mode == "L":
            return np.array(img), "L"
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            return np.array(img.convert("RGBA")), "RGBA"
        return np.array(img.convert("RGB")), "RGB"


def _remap_rgb_or_luma(equi: np.ndarray, mode: str, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    converted = cv2.remap(
        equi,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return converted


def _remap_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    converted = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_WRAP,
    )
    _, converted = cv2.threshold(converted, 127, 255, cv2.THRESH_BINARY)
    return converted


def _save_image(path: Path, arr: np.ndarray, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)


def _maybe_load_mask(mask_dir: Path, frame_file: str) -> np.ndarray | None:
    if not mask_dir.is_dir():
        return None
    for c in mask_candidates(str(mask_dir), frame_file):
        p = Path(c)
        if p.is_file():
            with Image.open(p) as img:
                return np.array(img.convert("L"))
    return None


def _axis_transform(no_transform: bool) -> np.ndarray:
    if no_transform:
        return np.eye(4, dtype=np.float64)
    return rot4(np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64))


def _camera_transform_for_view(world_transform: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    r = rotation_matrix(yaw, pitch, True)
    return world_transform @ rot4(r.T)


def _focal35_from_fov(fov: float) -> float:
    return 18.0 / math.tan(math.radians(fov) / 2.0)


def _normalize_guid(guid_text: str) -> tuple[str, uuid.UUID]:
    raw = guid_text.strip()
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]
    val = uuid.UUID(raw)
    return "{" + str(val) + "}", val


def _default_rig_id(views: list[dict]) -> tuple[str, uuid.UUID]:
    token = json.dumps(
        [{"name": str(v["name"]), "yaw": float(v["yaw"]), "pitch": float(v["pitch"])} for v in views],
        sort_keys=True,
    )
    val = uuid.uuid5(uuid.NAMESPACE_URL, f"realityscan-rig:{token}")
    return "{" + str(val) + "}", val


def _fmt_values(values: np.ndarray | list[float]) -> str:
    arr = np.array(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{float(v):.12g}" for v in arr)


def _write_xmp(
    path: Path,
    pose: np.ndarray,
    pose_prior: str,
    calibration_prior: str,
    focal35mm: float,
    coordinates: str,
    rig_id: str,
    rig_instance_id: str,
    rig_pose_index: int,
) -> None:
    r = pose[:3, :3]
    t = pose[:3, 3]
    rotation_text = _fmt_values(r)
    position_text = _fmt_values(t)

    lines = [
        '<?xpacket begin="?" id="W5M0MpCehiHzreSzNTczkc9d"?>',
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"',
        '      xcr:Version="3"',
        f'      xcr:PosePrior="{pose_prior}"',
        f'      xcr:Coordinates="{coordinates}"',
        '      xcr:DistortionModel="division"',
        f'      xcr:FocalLength35mm="{focal35mm:.12g}"',
        f'      xcr:CalibrationPrior="{calibration_prior}"',
        '      xcr:CalibrationGroup="-1"',
        '      xcr:DistortionGroup="-1"',
        f'      xcr:Rig="{rig_id}"',
        f'      xcr:RigInstance="{rig_instance_id}"',
        f'      xcr:RigPoseIndex="{rig_pose_index}"',
        '      xcr:InTexturing="1"',
        '      xcr:InMeshing="1"',
        '      xcr:PrincipalPointU="0"',
        '      xcr:PrincipalPointV="0"',
        '      xcr:Skew="0">',
        f'      <xcr:Position>{position_text}</xcr:Position>',
        f'      <xcr:Rotation>{rotation_text}</xcr:Rotation>',
        '      <xcr:DistortionCoeficients>0 0 0 0 0 0</xcr:DistortionCoeficients>',
        '    </rdf:Description>',
        '  </rdf:RDF>',
        '</x:xmpmeta>',
        '<?xpacket end="w"?>',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_views_config(path: Path, views: list[dict], fov: float) -> None:
    payload = {
        "fov": float(fov),
        "views": [
            {
                "name": str(v["name"]),
                "yaw": float(v["yaw"]),
                "pitch": float(v["pitch"]),
                "enabled": True,
            }
            for v in views
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file_path",
                "view_name",
                "image_name",
                "xmp_name",
                "mask_name",
                "rig_pose_index",
                "rig_instance",
                "yaw_deg",
                "pitch_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_project_config(
    path: Path,
    output_root: Path,
    rig_id: str,
    pose_prior: str,
    calibration_prior: str,
    fov: float,
    focal35mm: float,
) -> None:
    payload = {
        "version": 1,
        "format": "realityscan_rig_export",
        "output_root": str(output_root.resolve()),
        "inputs_dir": str((output_root / "inputs").resolve()),
        "manifest_path": str((output_root / "manifest.csv").resolve()),
        "views_config_path": str((output_root / "views_config.json").resolve()),
        "rig_id": rig_id,
        "pose_prior": pose_prior,
        "calibration_prior": calibration_prior,
        "fov_deg": float(fov),
        "focal_length_35mm": float(focal35mm),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_realityscan_rig(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    if not (0.0 < args.fov < 180.0):
        raise ValueError("fov must be in (0, 180)")

    views = _load_views(args)
    if not views:
        raise ValueError("No enabled views to export")

    mask_dir = Path(args.mask_dir).resolve() if args.mask_dir else (input_dir / "masks")
    frames = _load_frames(input_dir, args.json, args.duplicate)
    input_size, output_size = _determine_io_shape(input_dir, frames)
    remap_tables = _build_remap_tables(input_size, output_size, args.fov, views)
    output_name_map = _unique_output_names(frames)
    axis_transform = _axis_transform(args.no_transform)

    focal35mm = float(args.focal35mm) if args.focal35mm is not None else _focal35_from_fov(float(args.fov))
    if not math.isfinite(focal35mm) or focal35mm <= 0.0:
        raise ValueError("focal35mm must be a positive finite value")

    if args.rig_id:
        rig_id_text, rig_uuid = _normalize_guid(args.rig_id)
    else:
        rig_id_text, rig_uuid = _default_rig_id(views)

    dirs = _prepare_output_dirs(output_root)

    manifest_rows: list[dict[str, str]] = []
    total = len(frames)
    print(f"Converting {total} images...")
    for idx, frame in enumerate(frames, start=1):
        frame_file = str(frame["file_path"])
        print(f"Processing: {frame_file} ({idx}/{total})")
        src_path = input_dir / frame_file
        source_name = output_name_map[frame_file]
        source_stem, source_ext = os.path.splitext(source_name)
        if not source_ext:
            source_ext = ".jpg"

        rig_instance = "{" + str(uuid.uuid5(rig_uuid, frame_file.lower())) + "}"

        equi_img, mode = _load_equi_image(src_path)
        src_mask = _maybe_load_mask(mask_dir, frame_file)
        world_transform = axis_transform @ np.array(frame["transform"], dtype=np.float64)

        for rig_pose_index, view in enumerate(views):
            view_name = str(view["name"])
            map_x, map_y = remap_tables[view_name]

            converted = _remap_rgb_or_luma(equi_img, mode, map_x, map_y)
            image_name = f"{source_stem}__{view_name}{source_ext}"
            out_img_path = dirs["inputs"] / image_name

            alpha_mask: np.ndarray | None = None
            if mode == "RGBA":
                converted_rgb = np.array(Image.fromarray(converted, mode="RGBA").convert("RGB"))
                _save_image(out_img_path, converted_rgb, "RGB")
                if args.mask_from_alpha:
                    alpha = converted[..., -1]
                    _, alpha_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            else:
                _save_image(out_img_path, converted, "L" if mode == "L" else mode)

            remapped_src_mask: np.ndarray | None = None
            if src_mask is not None:
                remapped_src_mask = _remap_mask(src_mask, map_x, map_y)

            merged_mask: np.ndarray | None = None
            if alpha_mask is not None and remapped_src_mask is not None:
                merged_mask = cv2.bitwise_and(alpha_mask, remapped_src_mask)
            elif alpha_mask is not None:
                merged_mask = alpha_mask
            elif remapped_src_mask is not None:
                merged_mask = remapped_src_mask

            mask_name = ""
            if merged_mask is not None:
                if args.invert_masks:
                    merged_mask = cv2.bitwise_not(merged_mask)
                mask_name = f"{image_name}.mask.png"
                _save_image(dirs["inputs"] / mask_name, merged_mask, "L")

            pose = _camera_transform_for_view(world_transform, float(view["yaw"]), float(view["pitch"]))
            xmp_name = f"{image_name}.xmp"
            _write_xmp(
                dirs["inputs"] / xmp_name,
                pose=pose,
                pose_prior=args.pose_prior,
                calibration_prior=args.calibration_prior,
                focal35mm=focal35mm,
                coordinates=args.coordinates,
                rig_id=rig_id_text,
                rig_instance_id=rig_instance,
                rig_pose_index=rig_pose_index,
            )

            manifest_rows.append(
                {
                    "source_file_path": frame_file,
                    "view_name": view_name,
                    "image_name": image_name,
                    "xmp_name": xmp_name,
                    "mask_name": mask_name,
                    "rig_pose_index": str(rig_pose_index),
                    "rig_instance": rig_instance,
                    "yaw_deg": f"{float(view['yaw']):.9g}",
                    "pitch_deg": f"{float(view['pitch']):.9g}",
                }
            )

    _write_views_config(output_root / "views_config.json", views, float(args.fov))
    _write_manifest(output_root / "manifest.csv", manifest_rows)
    _write_project_config(
        output_root / "realityscan_project.json",
        output_root,
        rig_id=rig_id_text,
        pose_prior=args.pose_prior,
        calibration_prior=args.calibration_prior,
        fov=float(args.fov),
        focal35mm=focal35mm,
    )

    print(f"Saved RealityScan rig package in: {output_root}")
    print(f"Import this folder in RealityScan: {dirs['inputs']}")


def main() -> None:
    args = parse_args()
    try:
        export_realityscan_rig(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
