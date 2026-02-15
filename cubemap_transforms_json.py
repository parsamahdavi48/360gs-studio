import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXAMPLE_TEXT = """Example:
  python cubemap_transforms_json.py .
  python cubemap_transforms_json.py . ./cubic --yaw 45 --stitch 2.5
  python cubemap_transforms_json.py . ./cubic --views-json views_config.json
"""

SAFE_VIEW_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_WORKER_REMAP_TABLES: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
_WORKER_VIEWS: list[dict] | None = None
_WORKER_IMAGE_DIR = ""
_WORKER_MASK_DIR = ""
_WORKER_OUTPUT_IMAGE_DIR = ""
_WORKER_OUTPUT_MASK_DIR = ""
_WORKER_MASK_FROM_ALPHA = False
_WORKER_INVERT_MASKS = False


class MyParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help()
        sys.stderr.write(f"\n{message}\n")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = MyParser(
        description="Convert transforms.json from equirectangular to cubemap views.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_TEXT,
    )
    parser.add_argument("input_dir", help="Input directory containing transforms.json and images")
    parser.add_argument("output_dir", nargs="?", help="Output directory (default=<input_dir>/cubic)")
    parser.add_argument("--json", help="transforms.json filename override (default='transforms.json')")
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
    parser.add_argument("--no_transform", action="store_true", help="Disable axis transform (for LichtFeld Studio)")
    parser.add_argument("--duplicate", action="store_true", help="Allow duplicated image files")
    return parser.parse_args()


def rot4(r3: np.ndarray) -> np.ndarray:
    r4 = np.eye(4)
    r4[:3, :3] = r3
    return r4


def rotation_matrix(yaw_deg: float, pitch_deg: float, forward: bool) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ]
    )

    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )

    r = rx @ ry if forward else ry @ rx
    r[np.abs(r) < 1e-10] = 0.0
    return r


def build_remap(
    input_size: tuple[int, int],
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(output_size), np.arange(output_size))

    cx = xs - output_size / 2
    cy = ys - output_size / 2

    focal = 0.5 * output_size / np.tan(np.deg2rad(fov_deg) / 2)

    rays = np.stack([cx, -cy, np.full_like(cx, focal)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    r = rotation_matrix(yaw_deg, pitch_deg, False)
    rays = rays @ r.T

    dx, dy, dz = rays[..., 0], rays[..., 1], rays[..., 2]

    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))

    map_x = (lon / np.pi + 1.0) * 0.5 * input_size[0]
    map_y = (0.5 - lat / np.pi) * input_size[1]

    return map_x.astype(np.float32), map_y.astype(np.float32)


def make_default_views(yaw: float, stitch: float, no_top: bool, no_bottom: bool) -> list[dict]:
    views = [
        {"name": "px", "yaw": 90.0 - yaw - stitch, "pitch": 0.0},
        {"name": "nx", "yaw": -90.0 - yaw - stitch, "pitch": 0.0},
        {"name": "py", "yaw": 0.0 - yaw, "pitch": -90.0},
        {"name": "ny", "yaw": 0.0 - yaw, "pitch": 90.0},
        {"name": "pz", "yaw": 0.0 - yaw + stitch, "pitch": 0.0},
        {"name": "nz", "yaw": 180.0 - yaw + stitch, "pitch": 0.0},
    ]
    if no_top:
        views = [v for v in views if v["name"] != "py"]
    if no_bottom:
        views = [v for v in views if v["name"] != "ny"]
    return views


def load_custom_views(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_views = data.get("views")
    else:
        raw_views = data

    if not isinstance(raw_views, list):
        raise ValueError("views-json must be a list or an object with 'views' list")

    views: list[dict] = []
    used_names: set[str] = set()

    for idx, item in enumerate(raw_views):
        if not isinstance(item, dict):
            raise ValueError(f"views[{idx}] must be an object")

        if not bool(item.get("enabled", True)):
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"views[{idx}].name is required")
        if not SAFE_VIEW_NAME_RE.match(name):
            raise ValueError(
                f"views[{idx}].name '{name}' is invalid; use letters/numbers/_/- only"
            )
        if name in used_names:
            raise ValueError(f"views has duplicated name: {name}")

        try:
            yaw = float(item["yaw"])
            pitch = float(item["pitch"])
        except KeyError as e:
            raise ValueError(f"views[{idx}] missing field: {e}") from e
        except Exception as e:
            raise ValueError(f"views[{idx}] yaw/pitch parse error: {e}") from e

        views.append({"name": name, "yaw": yaw, "pitch": pitch})
        used_names.add(name)

    if not views:
        raise ValueError("views-json has no enabled views")
    return views


def split_filename_for_output(input_file: str) -> tuple[str, str, str]:
    basename, ext = os.path.splitext(os.path.basename(input_file))
    ext2 = ""
    lower = basename.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
        basename, ext2 = os.path.splitext(basename)
    return basename, ext2, ext


def remap_image(
    input_file: str,
    output_dir: str,
    remap_tables: dict[str, tuple[np.ndarray, np.ndarray]],
    views: list[dict],
    mask_from_alpha: bool,
    output_mask_dir: str,
    invert_masks: bool,
) -> None:
    basename, ext2, ext = split_filename_for_output(input_file)

    print(f"Processing: {input_file}")
    with Image.open(input_file) as img:
        if img.mode == "L":
            equi = np.array(img)
            mode_for_pil = "L"
        elif img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            equi = np.array(img.convert("RGBA"))
            mode_for_pil = "RGBA"
        else:
            equi = np.array(img.convert("RGB"))
            mode_for_pil = "RGB"

    for view in views:
        view_name = view["name"]
        map_x, map_y = remap_tables[view_name]

        converted = cv2.remap(
            equi,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )

        if mode_for_pil == "L":
            _, converted = cv2.threshold(converted, 127, 255, cv2.THRESH_BINARY)
            if invert_masks:
                converted = cv2.bitwise_not(converted)

        out_path = os.path.join(output_dir, f"{basename}_{view_name}{ext2}{ext}")
        if mask_from_alpha and mode_for_pil == "RGBA":
            converted_rgb = np.array(Image.fromarray(converted, mode="RGBA").convert("RGB"))
            Image.fromarray(converted_rgb, mode="RGB").save(out_path)

            alpha_channel = converted[..., -1]
            _, mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
            if invert_masks:
                mask = cv2.bitwise_not(mask)
            mask_out_path = os.path.join(output_mask_dir, f"{basename}_{view_name}{ext2}.png")
            Image.fromarray(mask, mode="L").save(mask_out_path)
        else:
            Image.fromarray(converted, mode=mode_for_pil).save(out_path)


def rotation_angle_diff(r1: np.ndarray, r2: np.ndarray) -> float:
    r = r1.T @ r2
    cos_theta = (np.trace(r) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def make_output_file_path(file_path: str, view_name: str) -> str:
    root, ext = os.path.splitext(file_path)
    if ext:
        return f"{root}_{view_name}{ext}"
    return f"{file_path}_{view_name}"


def transform_json(
    input_dir: str,
    input_json: str,
    image_dir: str,
    output_dir: str,
    views: list[dict],
    fov: float,
    output_scale: float,
    no_transform: bool,
    allow_duplicate: bool,
) -> tuple[list[str], tuple[int, int], int]:
    json_path = os.path.join(input_dir, input_json)
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return [], (0, 0), 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("camera_model") != "EQUIRECTANGULAR":
        print("Error: camera_model is not EQUIRECTANGULAR")
        return [], (0, 0), 0

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        print("Error: frames in transforms.json is empty")
        return [], (0, 0), 0

    input_size = (7840, 3920)
    output_size = max(1, int(round(input_size[1] * output_scale)))

    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        probe = os.path.join(image_dir, file_path)
        if os.path.exists(probe):
            with Image.open(probe) as first_img:
                input_size = first_img.size
            output_size = max(1, int(round(input_size[1] * output_scale)))
            break

    if no_transform:
        axis_transform = np.eye(4)
    else:
        axis_transform = rot4(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))

    new_frames: list[dict] = []
    image_files: list[str] = []
    image_map: dict[str, np.ndarray] = {}

    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            print("Skipped frame without file_path")
            continue

        try:
            t = np.array(frame["transform_matrix"], dtype=float)
        except Exception:
            print(f"Skipped frame with invalid transform_matrix: {file_path}")
            continue

        if t.shape != (4, 4):
            print(f"Skipped frame with non 4x4 transform_matrix: {file_path}")
            continue

        if not allow_duplicate and file_path in image_map:
            r_diff = rotation_angle_diff(image_map[file_path][:3, :3], t[:3, :3])
            t_diff = image_map[file_path][:3, 3] - t[:3, 3]
            print(
                "Skipped duplicated image: "
                f"{file_path} (diff={np.rad2deg(r_diff):.3f} deg, {np.linalg.norm(t_diff):.4f} dist.)"
            )
            continue

        t_world = axis_transform @ t

        image_map[file_path] = t
        image_files.append(file_path)

        for view in views:
            view_name = view["name"]
            yaw = view["yaw"]
            pitch = view["pitch"]

            new_frame: dict = {"file_path": make_output_file_path(file_path, view_name)}

            r = rotation_matrix(yaw, pitch, True)
            t_face = t_world @ rot4(r.T)
            new_frame["transform_matrix"] = t_face.tolist()

            new_frames.append(new_frame)

    out = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": output_size,
        "h": output_size,
        "fl_x": output_size / 2 / np.tan(np.deg2rad(fov) / 2),
        "fl_y": output_size / 2 / np.tan(np.deg2rad(fov) / 2),
        "cx": output_size / 2,
        "cy": output_size / 2,
        "frames": new_frames,
    }
    if data.get("ply_file_path"):
        out["ply_file_path"] = data["ply_file_path"]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved transforms.json in {output_dir}")

    return image_files, input_size, output_size


def mask_candidates(mask_dir: str, frame_file: str) -> list[str]:
    frame_path = Path(frame_file)
    candidates: list[Path] = []

    variants: list[Path] = [frame_path, Path(frame_path.name)]
    if frame_path.parts and frame_path.parts[0].lower() == "images" and len(frame_path.parts) > 1:
        variants.append(Path(*frame_path.parts[1:]))

    for rel in variants:
        candidates.append(Path(mask_dir) / rel)
        candidates.append(Path(mask_dir) / f"{rel.name}.png")
        candidates.append(Path(mask_dir) / f"{rel.stem}.png")

    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(str(c))
    return uniq


def worker_init(
    input_size: tuple[int, int],
    fov: float,
    output_size: int,
    views: list[dict],
    image_dir: str,
    mask_dir: str,
    output_image_dir: str,
    output_mask_dir: str,
    mask_from_alpha: bool,
    invert_masks: bool,
) -> None:
    global _WORKER_REMAP_TABLES
    global _WORKER_VIEWS
    global _WORKER_IMAGE_DIR
    global _WORKER_MASK_DIR
    global _WORKER_OUTPUT_IMAGE_DIR
    global _WORKER_OUTPUT_MASK_DIR
    global _WORKER_MASK_FROM_ALPHA
    global _WORKER_INVERT_MASKS

    _WORKER_REMAP_TABLES = {}
    for view in views:
        _WORKER_REMAP_TABLES[view["name"]] = build_remap(
            input_size,
            fov,
            float(view["yaw"]),
            float(view["pitch"]),
            output_size,
        )

    _WORKER_VIEWS = views
    _WORKER_IMAGE_DIR = image_dir
    _WORKER_MASK_DIR = mask_dir
    _WORKER_OUTPUT_IMAGE_DIR = output_image_dir
    _WORKER_OUTPUT_MASK_DIR = output_mask_dir
    _WORKER_MASK_FROM_ALPHA = mask_from_alpha
    _WORKER_INVERT_MASKS = invert_masks


def proc_convert_images(frame_file: str) -> None:
    if _WORKER_REMAP_TABLES is None or _WORKER_VIEWS is None:
        raise RuntimeError("worker remap tables are not initialized")

    image = os.path.join(_WORKER_IMAGE_DIR, frame_file)
    if os.path.exists(image):
        remap_image(
            image,
            _WORKER_OUTPUT_IMAGE_DIR,
            _WORKER_REMAP_TABLES,
            _WORKER_VIEWS,
            _WORKER_MASK_FROM_ALPHA,
            _WORKER_OUTPUT_MASK_DIR,
            _WORKER_INVERT_MASKS,
        )

    if _WORKER_MASK_FROM_ALPHA or not _WORKER_MASK_DIR or not os.path.isdir(_WORKER_MASK_DIR):
        return

    for mask in mask_candidates(_WORKER_MASK_DIR, frame_file):
        if os.path.exists(mask):
            remap_image(
                mask,
                _WORKER_OUTPUT_MASK_DIR,
                _WORKER_REMAP_TABLES,
                _WORKER_VIEWS,
                False,
                _WORKER_OUTPUT_MASK_DIR,
                _WORKER_INVERT_MASKS,
            )
            break


def convert_images(
    image_files: list[str],
    input_size: tuple[int, int],
    output_size: int,
    views: list[dict],
    fov: float,
    image_dir: str,
    mask_dir: str,
    output_image_dir: str,
    output_mask_dir: str,
    mask_from_alpha: bool,
    invert_masks: bool,
) -> None:
    print(f"Converting {len(image_files)} images...")

    max_workers = min(16, os.cpu_count() or 1)

    os.makedirs(output_image_dir, exist_ok=True)
    if mask_from_alpha or os.path.isdir(mask_dir):
        os.makedirs(output_mask_dir, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init,
        initargs=(
            input_size,
            fov,
            output_size,
            views,
            image_dir,
            mask_dir,
            output_image_dir,
            output_mask_dir,
            mask_from_alpha,
            invert_masks,
        ),
    ) as executor:
        futures = [executor.submit(proc_convert_images, frame_file) for frame_file in image_files]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Worker failed:", e)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else f"{input_dir}/cubic"
    input_json = args.json if args.json else "transforms.json"

    image_dir = input_dir
    mask_dir = args.mask_dir if args.mask_dir else f"{input_dir}/masks"
    output_image_dir = f"{output_dir}/images"
    output_mask_dir = f"{output_dir}/masks"

    if args.mask_dir and not os.path.isdir(mask_dir):
        print(f"Error: mask_dir '{mask_dir}' not found")
        sys.exit(1)

    if args.fov <= 0 or args.fov >= 180:
        print("Error: fov must be in (0, 180)")
        sys.exit(1)
    if args.output_scale <= 0 or args.output_scale > 1.0:
        print("Error: output_scale must be in (0, 1.0]")
        sys.exit(1)

    if args.views_json:
        try:
            views = load_custom_views(args.views_json)
        except Exception as e:
            print(f"Error: failed to parse views-json: {e}")
            sys.exit(1)
    else:
        views = make_default_views(args.yaw, args.stitch, args.no_top, args.no_bottom)

    if not views:
        print("Error: no views to export")
        sys.exit(1)

    for view in views:
        print(f"{view['name']}: yaw={view['yaw']},pitch={view['pitch']}")

    image_files, input_size, output_size = transform_json(
        input_dir=input_dir,
        input_json=input_json,
        image_dir=image_dir,
        output_dir=output_dir,
        views=views,
        fov=args.fov,
        output_scale=args.output_scale,
        no_transform=args.no_transform,
        allow_duplicate=args.duplicate,
    )
    if not image_files:
        sys.exit(1)

    if not args.no_image:
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
        )


if __name__ == "__main__":
    main()
