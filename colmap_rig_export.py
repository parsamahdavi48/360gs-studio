#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cubemap_transforms_json import build_remap, load_custom_views, make_default_views, mask_candidates


EXAMPLE_TEXT = """Example:
  python colmap_rig_export.py . ./colmap_rig --views-json ./views_config.json
  python colmap_rig_export.py . ./colmap_rig --views-json ./views_config.json --mask_dir ./masks
"""

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cubemap-selected views as a COLMAP rig dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_TEXT,
    )
    parser.add_argument("input_dir", help="Input scene directory containing transforms.json and images")
    parser.add_argument("output_dir", help="COLMAP output root directory")
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
    return parser.parse_args()


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )

    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )

    r = rx @ ry
    r[np.abs(r) < 1e-10] = 0.0
    return r


def _quat_from_rotmat(r: np.ndarray) -> list[float]:
    m = r.astype(float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    q /= n
    if q[0] < 0.0:
        q = -q
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _load_views(args: argparse.Namespace) -> list[dict]:
    if args.views_json:
        return load_custom_views(args.views_json)
    return make_default_views(args.yaw, args.stitch, args.no_top, args.no_bottom)


def _load_frame_files(input_dir: Path, json_name: str, allow_duplicate: bool) -> list[str]:
    json_path = input_dir / json_name
    if not json_path.is_file():
        raise FileNotFoundError(f"Input transforms JSON not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("camera_model") != "EQUIRECTANGULAR":
        raise ValueError("camera_model must be EQUIRECTANGULAR")

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames is empty in transforms JSON")

    image_files: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue

        key = file_path.lower()
        if not allow_duplicate and key in seen:
            continue

        src = input_dir / file_path
        if not src.is_file():
            continue

        seen.add(key)
        image_files.append(file_path)

    if not image_files:
        raise ValueError("No readable input images found from transforms frames")
    return image_files


def _determine_io_shape(input_dir: Path, image_files: list[str]) -> tuple[tuple[int, int], int]:
    for rel in image_files:
        src = input_dir / rel
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


def _unique_output_names(image_files: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for file_path in image_files:
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
    if mode == "L":
        _, converted = cv2.threshold(converted, 127, 255, cv2.THRESH_BINARY)
    return converted


def _remap_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, invert: bool) -> np.ndarray:
    converted = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_WRAP,
    )
    _, converted = cv2.threshold(converted, 127, 255, cv2.THRESH_BINARY)
    if invert:
        converted = cv2.bitwise_not(converted)
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


def _prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "dataset": output_root / "dataset",
        "images": output_root / "dataset" / "images",
        "masks": output_root / "dataset" / "masks",
        "workspace": output_root / "workspace",
        "sparse": output_root / "workspace" / "sparse",
        "logs": output_root / "logs",
    }
    for key in ["dataset", "images", "masks", "workspace", "sparse", "logs"]:
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def _build_remap_tables(input_size: tuple[int, int], output_size: int, fov: float, views: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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


def _write_manifest(
    path: Path,
    image_files: list[str],
    output_name_map: dict[str, str],
    views: list[dict],
) -> None:
    rows: list[dict[str, str]] = []
    for frame_file in image_files:
        out_name = output_name_map[frame_file]
        for view in views:
            view_name = str(view["name"])
            rows.append(
                {
                    "source_file_path": frame_file,
                    "view_name": view_name,
                    "image_relpath": f"images/{view_name}/{out_name}",
                    "mask_relpath": f"masks/{view_name}/{out_name}",
                    "yaw_deg": f"{float(view['yaw']):.9g}",
                    "pitch_deg": f"{float(view['pitch']):.9g}",
                }
            )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_file_path", "view_name", "image_relpath", "mask_relpath", "yaw_deg", "pitch_deg"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_rig_template(path: Path, views: list[dict]) -> None:
    if not views:
        raise ValueError("views is empty")

    entries: list[dict] = []
    for view in views:
        view_name = str(view["name"])
        r = _rotation_matrix(float(view["yaw"]), float(view["pitch"]))
        q = _quat_from_rotmat(r)
        entries.append(
            {
                "view_name": view_name,
                "image_prefix": f"{view_name}/",
                "cam_from_rig_rotation": q,
                "cam_from_rig_translation": [0.0, 0.0, 0.0],
            }
        )

    payload = {
        "ref_view_name": str(views[0]["name"]),
        "cameras": entries,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_project_config(path: Path, output_root: Path) -> None:
    payload = {
        "version": 1,
        "dataset_dir": str((output_root / "dataset").resolve()),
        "images_dir": str((output_root / "dataset" / "images").resolve()),
        "masks_dir": str((output_root / "dataset" / "masks").resolve()),
        "workspace_dir": str((output_root / "workspace").resolve()),
        "database_path": str((output_root / "workspace" / "database.db").resolve()),
        "sparse_dir": str((output_root / "workspace" / "sparse").resolve()),
        "rig_template_path": str((output_root / "dataset" / "rig_template.json").resolve()),
        "rig_config_path": str((output_root / "dataset" / "rig_config.json").resolve()),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_colmap_rig(args: argparse.Namespace) -> None:
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

    image_files = _load_frame_files(input_dir, args.json, args.duplicate)
    input_size, output_size = _determine_io_shape(input_dir, image_files)
    remap_tables = _build_remap_tables(input_size, output_size, args.fov, views)
    output_name_map = _unique_output_names(image_files)

    dirs = _prepare_output_dirs(output_root)

    total = len(image_files)
    print(f"Converting {total} images...")
    for idx, frame_file in enumerate(image_files, start=1):
        print(f"Processing: {frame_file} ({idx}/{total})")
        src_path = input_dir / frame_file
        out_name = output_name_map[frame_file]

        equi_img, mode = _load_equi_image(src_path)
        src_mask = _maybe_load_mask(mask_dir, frame_file)

        for view in views:
            view_name = str(view["name"])
            map_x, map_y = remap_tables[view_name]

            converted = _remap_rgb_or_luma(equi_img, mode, map_x, map_y)
            out_img_path = dirs["images"] / view_name / out_name

            if mode == "RGBA" and args.mask_from_alpha:
                converted_rgb = np.array(Image.fromarray(converted, mode="RGBA").convert("RGB"))
                _save_image(out_img_path, converted_rgb, "RGB")

                alpha = converted[..., -1]
                _, alpha_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
                if args.invert_masks:
                    alpha_mask = cv2.bitwise_not(alpha_mask)
                mask_path = dirs["masks"] / view_name / out_name
                _save_image(mask_path, alpha_mask, "L")
            elif mode == "RGBA":
                converted_rgb = np.array(Image.fromarray(converted, mode="RGBA").convert("RGB"))
                _save_image(out_img_path, converted_rgb, "RGB")
            else:
                _save_image(out_img_path, converted, "L" if mode == "L" else mode)

            if src_mask is not None:
                out_mask = _remap_mask(src_mask, map_x, map_y, args.invert_masks)
                mask_path = dirs["masks"] / view_name / out_name
                _save_image(mask_path, out_mask, "L")

    views_config_payload = {
        "fov": float(args.fov),
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
    (dirs["dataset"] / "views_config.json").write_text(json.dumps(views_config_payload, indent=2), encoding="utf-8")

    _write_manifest(dirs["dataset"] / "manifest.csv", image_files, output_name_map, views)
    _write_rig_template(dirs["dataset"] / "rig_template.json", views)
    _write_project_config(output_root / "colmap_project.json", output_root)

    print(f"Saved COLMAP rig dataset in: {output_root}")


def main() -> None:
    args = parse_args()
    try:
        export_colmap_rig(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
