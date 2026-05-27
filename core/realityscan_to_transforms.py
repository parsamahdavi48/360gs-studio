"""Convert RealityScan registration CSV exports to NeRF-style transforms.json."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from core import realityscan_layout as _rs_layout
from core.dataset_writer_nerf import write_nerf_json_ply_dataset

try:
    from PIL import Image
except Exception:  # pragma: no cover - import error is reported when image sizes are needed
    Image = None  # type: ignore[assignment]


TARGET_PROFILE_REALITYSCAN = "realityscan"
TARGET_PROFILE_LICHTFELD = "lichtfeld"
TARGET_PROFILE_CHOICES = (TARGET_PROFILE_REALITYSCAN, TARGET_PROFILE_LICHTFELD)
REALITYSCAN_IMAGE_DIR_NAMES = _rs_layout.REALITYSCAN_IMAGE_DIR_NAMES
REALITYSCAN_MASK_DIR_NAMES = _rs_layout.REALITYSCAN_MASK_DIR_NAMES
related_realityscan_asset_roots = _rs_layout.related_realityscan_asset_roots
strip_leading_realityscan_asset_dir = _rs_layout.strip_leading_realityscan_asset_dir
realityscan_image_asset_relative_path = _rs_layout.realityscan_image_asset_relative_path

# The repository's RealityScan preset writes XMP poses in the coordinate frame
# RealityScan exports back to CSV.  This maps that frame to the same LichtFeld
# final orientation used by cubemap_transforms_json.py.
REALITYSCAN_TO_LICHTFELD_MATRIX = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# LichtFeld's transforms.json loader applies an additional Y/Z flip to the PLY
# after loading it.  The camera JSON and referenced PLY therefore need different
# file-space transforms to land in the same LichtFeld internal world frame.
REALITYSCAN_PLY_TO_LICHTFELD_FILE_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

IMAGE_PATH_MODES = ("images-prefix", "relative-to-output", "absolute", "relative")
DISTORTION_EPSILON = 1e-12
PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

SOURCE_FRAME_KEYS = (
    "source_file_path",
    "source_image_index",
    "view_name",
    "view_index",
    "yaw_offset_deg",
)
CAMERA_KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy")


@dataclass(frozen=True)
class RealityScanCameraRow:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    f_35mm: float
    px_norm: float
    py_norm: float
    k1: float
    k2: float
    k3: float
    k4: float
    t1: float
    t2: float


def target_profile_matrix(profile: str) -> np.ndarray:
    normalized = profile.strip().lower()
    if normalized == TARGET_PROFILE_REALITYSCAN:
        return np.eye(4, dtype=np.float64)
    if normalized == TARGET_PROFILE_LICHTFELD:
        return REALITYSCAN_TO_LICHTFELD_MATRIX.copy()
    raise ValueError(f"Unsupported target profile: {profile}")


def pointcloud_target_profile_matrix(profile: str) -> np.ndarray:
    normalized = profile.strip().lower()
    if normalized == TARGET_PROFILE_REALITYSCAN:
        return np.eye(4, dtype=np.float64)
    if normalized == TARGET_PROFILE_LICHTFELD:
        return REALITYSCAN_PLY_TO_LICHTFELD_FILE_MATRIX.copy()
    raise ValueError(f"Unsupported target profile: {profile}")


def parse_float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def read_realityscan_csv(path: Path) -> list[RealityScanCameraRow]:
    rows: list[RealityScanCameraRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            name = str(raw.get("#name") or raw.get("name") or "").strip()
            if not name:
                continue
            rows.append(
                RealityScanCameraRow(
                    name=name,
                    x=parse_float(raw.get("x")),
                    y=parse_float(raw.get("y")),
                    z=parse_float(raw.get("alt")),
                    yaw=parse_float(raw.get("yaw")),
                    pitch=parse_float(raw.get("pitch")),
                    roll=parse_float(raw.get("roll")),
                    f_35mm=parse_float(raw.get("f_35mm")),
                    px_norm=parse_float(raw.get("px_norm")),
                    py_norm=parse_float(raw.get("py_norm")),
                    k1=parse_float(raw.get("k1")),
                    k2=parse_float(raw.get("k2")),
                    k3=parse_float(raw.get("k3")),
                    k4=parse_float(raw.get("k4")),
                    t1=parse_float(raw.get("t1")),
                    t2=parse_float(raw.get("t2")),
                )
            )
    if not rows:
        raise ValueError(f"No camera rows found in RealityScan CSV: {path}")
    return rows


def realityscan_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return the camera-to-world rotation exported by RealityScan CSV rows."""
    yaw_rad = np.deg2rad(-yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)

    sy, cy = np.sin(yaw_rad), np.cos(yaw_rad)
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)
    sr, cr = np.sin(roll_rad), np.cos(roll_rad)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float64)
    return rz @ rx @ ry


def row_to_transform(row: RealityScanCameraRow) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = realityscan_rotation_matrix(row.yaw, row.pitch, row.roll)
    transform[:3, 3] = [row.x, row.y, row.z]
    return transform


def output_file_path(path: Path, root_dir: Path, output_dir: Path, mode: str) -> str:
    if mode == "absolute":
        return path.resolve().as_posix()
    try:
        rel_to_root = path.resolve().relative_to(root_dir.resolve())
    except ValueError:
        rel_to_root = Path(path.name)
    if mode == "relative":
        return rel_to_root.as_posix()
    if mode == "images-prefix":
        asset_rel = realityscan_image_asset_relative_path(path, root_dir)
        if asset_rel is not None:
            return asset_rel.as_posix()
        return (Path("images") / rel_to_root).as_posix()
    if mode == "relative-to-output":
        return os.path.relpath(path.resolve(), output_dir.resolve()).replace(os.sep, "/")
    raise ValueError(f"Unsupported image path mode: {mode}")


def resolve_image_path(images_dir: Path, name: str) -> Path:
    return _rs_layout.resolve_realityscan_image_path(images_dir, name)


def image_size(path: Path) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError("Pillow is required to infer image sizes")
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def principal_point_from_normalized(width: int, height: int, px_norm: float, py_norm: float) -> tuple[float, float]:
    scale = float(min(width, height))
    return ((float(width) - 1.0) / 2.0 + px_norm * scale, (float(height) - 1.0) / 2.0 + py_norm * scale)


def camera_from_csv_row(row: RealityScanCameraRow, width: int, height: int) -> dict[str, Any]:
    scale = float(min(width, height))
    focal = float(row.f_35mm) * scale / 36.0 if row.f_35mm > 0.0 else scale * 0.5
    cx, cy = principal_point_from_normalized(width, height, row.px_norm, row.py_norm)
    camera: dict[str, Any] = {
        "w": int(width),
        "h": int(height),
        "fl_x": focal,
        "fl_y": focal,
        "cx": cx,
        "cy": cy,
    }
    if row_has_distortion(row):
        camera.update(
            {
                "camera_model": "OPENCV",
                "k1": row.k1,
                "k2": row.k2,
                "k3": row.k3,
                "k4": row.k4,
                "p1": row.t1,
                "p2": row.t2,
            }
        )
    else:
        camera["camera_model"] = "PINHOLE"
    return camera


def row_has_distortion(row: RealityScanCameraRow) -> bool:
    return max(abs(row.k1), abs(row.k2), abs(row.k3), abs(row.k4), abs(row.t1), abs(row.t2)) > DISTORTION_EPSILON


def load_source_transforms(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None or not path.is_file():
        return {}, None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"transforms.json root must be an object: {path}")
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"transforms.json frames must be a list: {path}")
    by_key: dict[str, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        raw_path = str(frame.get("file_path") or "").replace("\\", "/")
        if not raw_path:
            continue
        by_key[raw_path.lower()] = frame
        by_key[Path(raw_path).name.lower()] = frame
    return by_key, data


def source_frame_for(row: RealityScanCameraRow, source_frames: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    normalized = row.name.replace("\\", "/").lower()
    return source_frames.get(normalized) or source_frames.get(Path(normalized).name)


def frame_camera_from_source(source_frame: dict[str, Any], source_data: dict[str, Any] | None) -> dict[str, Any]:
    camera: dict[str, Any] = {}
    root = source_data or {}
    for key in CAMERA_KEYS:
        value = source_frame.get(key, root.get(key))
        if value is not None:
            camera[key] = value
    model = source_frame.get("camera_model", root.get("camera_model"))
    if model:
        camera["camera_model"] = str(model)
    return camera


def camera_signature(frame: dict[str, Any]) -> tuple[Any, ...]:
    keys = ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2")
    values: list[Any] = []
    for key in keys:
        value = frame.get(key)
        if isinstance(value, float):
            value = round(value, 9)
        values.append(value)
    return tuple(values)


def top_level_camera_payload(frames: list[dict[str, Any]]) -> tuple[dict[str, Any], int, int]:
    counts: dict[tuple[Any, ...], int] = {}
    first_index: dict[tuple[Any, ...], int] = {}
    for index, frame in enumerate(frames):
        signature = camera_signature(frame)
        counts[signature] = counts.get(signature, 0) + 1
        first_index.setdefault(signature, index)

    best = max(counts, key=lambda signature: (counts[signature], -first_index[signature]))
    source = frames[first_index[best]]
    keys = ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2")
    return {key: source[key] for key in keys if key in source}, counts[best], len(counts)


def find_mask_path(image_path: Path, masks_dir: Path | None) -> Path | None:
    if masks_dir is not None:
        roots = tuple(root for root in related_realityscan_asset_roots(masks_dir, REALITYSCAN_MASK_DIR_NAMES) if root.is_dir())
        for rel in _rs_layout.mask_lookup_candidates(realityscan_image_asset_relative_path(image_path, masks_dir) or image_path.name):
            for root in roots:
                candidate = root / rel
                if candidate.is_file():
                    return candidate
    layer = Path(f"{image_path}.mask.png")
    return layer if layer.is_file() else None


def convert(
    csv_path: Path,
    output_dir: Path,
    *,
    images_dir: Path | None = None,
    ply_path: Path | None = None,
    source_transforms: Path | None = None,
    masks_dir: Path | None = None,
    image_path_mode: str = "images-prefix",
    target_profile: str = TARGET_PROFILE_LICHTFELD,
    json_name: str = "transforms.json",
    pointcloud_name: str = "pointcloud.ply",
    write_mask_paths: bool = True,
    skip_missing_images: bool = False,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir) if images_dir is not None else csv_path.parent / "images"
    masks_dir = Path(masks_dir) if masks_dir is not None else csv_path.parent / "masks"
    if not masks_dir.is_dir():
        image_layer_masks = tuple(
            root for root in related_realityscan_asset_roots(images_dir, REALITYSCAN_MASK_DIR_NAMES) if root.is_dir()
        )
        masks_dir = images_dir if image_layer_masks else None

    if image_path_mode not in IMAGE_PATH_MODES:
        raise ValueError(f"Unsupported image path mode: {image_path_mode}")
    if target_profile not in TARGET_PROFILE_CHOICES:
        raise ValueError(f"Unsupported target profile: {target_profile}")

    rows = read_realityscan_csv(csv_path)
    if source_transforms is None:
        candidate = csv_path.parent / "transforms.json"
        source_transforms = candidate if candidate.is_file() else None
    source_frames, source_data = load_source_transforms(source_transforms)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_matrix = target_profile_matrix(target_profile)

    frames: list[dict[str, Any]] = []
    missing_images: list[str] = []
    opencv_frames = 0
    mask_paths = 0
    for row in rows:
        image_path = resolve_image_path(images_dir, row.name)
        source_frame = source_frame_for(row, source_frames)
        if not image_path.is_file():
            if skip_missing_images:
                missing_images.append(row.name)
                continue
            raise FileNotFoundError(f"Image referenced by RealityScan CSV was not found: {image_path}")

        frame_camera = frame_camera_from_source(source_frame, source_data) if source_frame is not None else {}
        if not all(key in frame_camera for key in CAMERA_KEYS):
            width, height = image_size(image_path)
            frame_camera.update(camera_from_csv_row(row, width, height))
        else:
            frame_camera.setdefault("camera_model", "OPENCV" if row_has_distortion(row) else "PINHOLE")
            if row_has_distortion(row):
                frame_camera.update(
                    {
                        "k1": row.k1,
                        "k2": row.k2,
                        "k3": row.k3,
                        "k4": row.k4,
                        "p1": row.t1,
                        "p2": row.t2,
                        "camera_model": "OPENCV",
                    }
                )

        transform = profile_matrix @ row_to_transform(row)
        frame: dict[str, Any] = {
            "file_path": output_file_path(image_path, images_dir, output_dir, image_path_mode),
            "transform_matrix": transform.tolist(),
            **frame_camera,
        }
        if source_frame is not None:
            for key in SOURCE_FRAME_KEYS:
                if key in source_frame:
                    frame[key] = source_frame[key]
        if write_mask_paths:
            mask_path = find_mask_path(image_path, masks_dir)
            if mask_path is not None:
                frame["mask_path"] = output_file_path(mask_path, masks_dir or mask_path.parent, output_dir, "relative-to-output")
                mask_paths += 1
        if str(frame.get("camera_model") or "").upper() == "OPENCV":
            opencv_frames += 1
        frames.append(frame)

    if not frames:
        raise ValueError("No frames were converted")

    top_camera, top_camera_count, camera_group_count = top_level_camera_payload(frames)
    top_camera_model = str(top_camera.pop("camera_model", "PINHOLE") or "PINHOLE")
    data: dict[str, Any] = {
        "camera_model": top_camera_model,
        **top_camera,
        "frames": frames,
        "source": {
            "type": "realityscan_csv",
            "csv_path": str(csv_path),
            "images_dir": str(images_dir),
            "source_transforms": str(source_transforms) if source_transforms is not None else "",
            "target_profile": target_profile,
            "target_profile_matrix": profile_matrix.tolist(),
            "pointcloud_profile_matrix": pointcloud_target_profile_matrix(target_profile).tolist(),
            "per_frame_intrinsics": True,
            "per_frame_camera_model": True,
            "top_level_camera_group_count": top_camera_count,
            "camera_group_count": camera_group_count,
        },
    }

    pointcloud_output = ""
    if ply_path is not None:
        ply_path = Path(ply_path)
        pointcloud_dest = output_dir / pointcloud_name
        write_transformed_ply(ply_path, pointcloud_dest, pointcloud_target_profile_matrix(target_profile))
        data["ply_file_path"] = pointcloud_dest.name
        pointcloud_output = str(pointcloud_dest)

    write_result = write_nerf_json_ply_dataset(
        output_dir,
        data,
        transforms_name=json_name,
        pointcloud_name=pointcloud_name,
        manifest={
            "source_kind": "realityscan_csv_ply",
            "target_profile": target_profile,
            "num_opencv_frames": opencv_frames,
            "num_mask_paths": mask_paths,
        },
    )
    return {
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "transforms": str(write_result.transforms_json),
        "pointcloud": pointcloud_output,
        "num_csv_rows": len(rows),
        "num_frames": len(frames),
        "num_missing_images": len(missing_images),
        "num_opencv_frames": opencv_frames,
        "num_mask_paths": mask_paths,
        "target_profile": target_profile,
        "metadata": write_result.metadata,
    }


def write_transformed_ply(source: Path, dest: Path, matrix: np.ndarray) -> None:
    source = Path(source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == dest.resolve():
        if np.allclose(matrix, np.eye(4), atol=0.0):
            return
        raise ValueError("Refusing to overwrite the input PLY while applying a coordinate transform")
    if np.allclose(matrix, np.eye(4), atol=0.0):
        shutil.copy2(source, dest)
        return
    with source.open("rb") as f:
        header_bytes, header_lines = read_ply_header_bytes(f)
        fmt, vertex_count, vertex_properties, vertex_is_first = parse_ply_header(header_lines)
        if fmt == "ascii":
            write_transformed_ascii_ply(f, dest, header_bytes, vertex_count, vertex_properties, matrix)
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            write_transformed_binary_ply(
                f,
                dest,
                header_bytes,
                vertex_count,
                vertex_properties,
                matrix,
                endian=endian,
                vertex_is_first=vertex_is_first,
            )
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")


def read_ply_header_bytes(f: BinaryIO) -> tuple[bytes, list[str]]:
    header = bytearray()
    lines: list[str] = []
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("PLY header ended unexpectedly")
        header.extend(raw)
        line = raw.decode("ascii", errors="replace").strip()
        lines.append(line)
        if line == "end_header":
            break
    if not lines or lines[0] != "ply":
        raise ValueError("Not a PLY file")
    return bytes(header), lines


def parse_ply_header(lines: list[str]) -> tuple[str, int, list[tuple[str, str]], bool]:
    fmt = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    elements: list[str] = []
    for line in lines:
        if not line or line.startswith("comment "):
            continue
        parts = line.split()
        if parts[:1] == ["format"] and len(parts) >= 2:
            fmt = parts[1]
        elif parts[:1] == ["element"] and len(parts) >= 3:
            current_element = parts[1]
            elements.append(current_element)
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[:1] == ["property"] and current_element == "vertex":
            if len(parts) >= 3 and parts[1] == "list":
                raise ValueError("PLY vertex list properties are not supported")
            if len(parts) >= 3:
                vertex_properties.append((parts[2], parts[1]))
    if not fmt:
        raise ValueError("PLY format is missing")
    if vertex_count < 0 or not vertex_properties:
        raise ValueError("PLY vertex element is missing")
    names = [name.lower() for name, _type in vertex_properties]
    if not all(name in names for name in ("x", "y", "z")):
        raise ValueError("PLY vertex element must contain x/y/z properties")
    return fmt, vertex_count, vertex_properties, bool(elements and elements[0] == "vertex")


def vertex_dtype(properties: list[tuple[str, str]], endian: str) -> np.dtype:
    fields = []
    for name, type_name in properties:
        code = PLY_TYPES.get(type_name.lower())
        if code is None:
            raise ValueError(f"Unsupported PLY vertex property type: {type_name}")
        fields.append((name, np.dtype(code).newbyteorder(endian)))
    return np.dtype(fields)


def write_transformed_binary_ply(
    f: BinaryIO,
    dest: Path,
    header_bytes: bytes,
    vertex_count: int,
    vertex_properties: list[tuple[str, str]],
    matrix: np.ndarray,
    *,
    endian: str,
    vertex_is_first: bool,
) -> None:
    if not vertex_is_first:
        raise ValueError("Binary PLY transform currently requires the vertex element to be first")
    dtype = vertex_dtype(vertex_properties, endian)
    vertex_bytes = f.read(dtype.itemsize * vertex_count)
    if len(vertex_bytes) != dtype.itemsize * vertex_count:
        raise ValueError("PLY vertex data ended unexpectedly")
    trailing = f.read()
    data = np.frombuffer(vertex_bytes, dtype=dtype, count=vertex_count).copy()
    transformed = transform_points(np.column_stack([data["x"], data["y"], data["z"]]), matrix)
    data["x"] = transformed[:, 0]
    data["y"] = transformed[:, 1]
    data["z"] = transformed[:, 2]
    with dest.open("wb") as out:
        out.write(header_bytes)
        out.write(data.tobytes())
        out.write(trailing)


def write_transformed_ascii_ply(
    f: BinaryIO,
    dest: Path,
    header_bytes: bytes,
    vertex_count: int,
    vertex_properties: list[tuple[str, str]],
    matrix: np.ndarray,
) -> None:
    rest = f.read().decode("ascii", errors="strict")
    lines = rest.splitlines(keepends=True)
    if len(lines) < vertex_count:
        raise ValueError("PLY vertex data ended unexpectedly")
    names = [name.lower() for name, _type in vertex_properties]
    x_idx, y_idx, z_idx = names.index("x"), names.index("y"), names.index("z")
    output_lines: list[str] = []
    for index in range(vertex_count):
        line = lines[index]
        newline = "\n" if line.endswith("\n") else ""
        values = line.split()
        if len(values) < len(vertex_properties):
            raise ValueError(f"PLY vertex row {index} is truncated")
        point = np.array([[float(values[x_idx]), float(values[y_idx]), float(values[z_idx])]], dtype=np.float64)
        transformed = transform_points(point, matrix)[0]
        values[x_idx] = f"{transformed[0]:.9g}"
        values[y_idx] = f"{transformed[1]:.9g}"
        values[z_idx] = f"{transformed[2]:.9g}"
        output_lines.append(" ".join(values) + newline)
    output_lines.extend(lines[vertex_count:])
    with dest.open("wb") as out:
        out.write(header_bytes)
        out.write("".join(output_lines).encode("ascii"))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("point transform must be a 4x4 matrix")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ transform[:3, :3].T + transform[:3, 3]


def parse_args(argv: list[str] | None = None):
    from core.realityscan_to_transforms_cli import parse_args as _parse_args

    return _parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from core.realityscan_to_transforms_cli import main as _main

    return _main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
