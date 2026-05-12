"""Shared final-orientation correction helpers for 3DGS exports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

FINAL_ORIENTATION_NONE = "none"
FINAL_ORIENTATION_LICHTFELD = "lichtfeld"
FINAL_ORIENTATION_CHOICES = (FINAL_ORIENTATION_NONE, FINAL_ORIENTATION_LICHTFELD)

FINAL_ORIENTATION_STAGE_NONE = "none"
FINAL_ORIENTATION_STAGE_CUBEMAP_CLI = "cubemap_cli"
FINAL_ORIENTATION_STAGE_DIRECT_FINALIZE = "direct_finalize"

LICHTFELD_FINAL_ORIENTATION_MATRIX = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def normalize_final_orientation(value: object) -> str:
    raw = str(value or FINAL_ORIENTATION_NONE).strip().lower().replace("_", "-")
    if raw in {"", "none", "off", "false", "no"}:
        return FINAL_ORIENTATION_NONE
    if raw in {"lichtfeld", "lichtfeld-studio", "lfs"}:
        return FINAL_ORIENTATION_LICHTFELD
    raise ValueError(f"Unsupported final orientation: {value}")


def final_orientation_matrix(value: object) -> np.ndarray:
    orientation = normalize_final_orientation(value)
    if orientation == FINAL_ORIENTATION_NONE:
        return np.eye(4, dtype=np.float64)
    if orientation == FINAL_ORIENTATION_LICHTFELD:
        return LICHTFELD_FINAL_ORIENTATION_MATRIX.copy()
    raise ValueError(f"Unsupported final orientation: {value}")


def final_orientation_is_applied(data: object, orientation: object) -> bool:
    orientation_id = normalize_final_orientation(orientation)
    if orientation_id == FINAL_ORIENTATION_NONE or not isinstance(data, dict):
        return False
    postprocess = data.get("postprocess")
    if not isinstance(postprocess, dict):
        return False
    recorded = str(postprocess.get("final_orientation") or "").strip().lower().replace("_", "-")
    if recorded == orientation_id:
        return True
    return orientation_id == FINAL_ORIENTATION_LICHTFELD and bool(
        postprocess.get("lichtfeld_final_orientation_correction")
    )


def mark_final_orientation(data: dict[str, Any], orientation: object, stage: str) -> None:
    orientation_id = normalize_final_orientation(orientation)
    postprocess = data.setdefault("postprocess", {})
    if not isinstance(postprocess, dict):
        postprocess = {}
        data["postprocess"] = postprocess

    postprocess["final_orientation"] = orientation_id
    postprocess["final_orientation_stage"] = stage
    if orientation_id == FINAL_ORIENTATION_NONE:
        postprocess["final_orientation_matrix"] = None
        postprocess["lichtfeld_final_orientation_correction"] = False
        postprocess["lichtfeld_final_orientation_stage"] = FINAL_ORIENTATION_STAGE_NONE
        postprocess["lichtfeld_final_orientation_matrix"] = None
        return

    matrix = final_orientation_matrix(orientation_id).tolist()
    postprocess["final_orientation_matrix"] = matrix
    if orientation_id == FINAL_ORIENTATION_LICHTFELD:
        postprocess["lichtfeld_final_orientation_correction"] = True
        postprocess["lichtfeld_final_orientation_stage"] = stage
        postprocess["lichtfeld_final_orientation_matrix"] = matrix


def apply_final_orientation_to_transforms_data(
    data: dict[str, Any],
    orientation: object,
    *,
    stage: str,
) -> bool:
    orientation_id = normalize_final_orientation(orientation)
    if orientation_id == FINAL_ORIENTATION_NONE:
        mark_final_orientation(data, orientation_id, FINAL_ORIENTATION_STAGE_NONE)
        return False

    already_applied = final_orientation_is_applied(data, orientation_id)
    matrix = final_orientation_matrix(orientation_id)
    if not already_applied:
        frames = data.get("frames", [])
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict) or "transform_matrix" not in frame:
                    continue
                transform = np.array(frame["transform_matrix"], dtype=np.float64)
                if transform.shape != (4, 4):
                    continue
                frame["transform_matrix"] = (matrix @ transform).tolist()

    mark_final_orientation(data, orientation_id, stage)
    return not already_applied


def transform_transforms_json(path: Path, orientation: object, *, stage: str) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"transforms.json root must be an object: {path}")
    changed = apply_final_orientation_to_transforms_data(data, orientation, stage=stage)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changed


def transform_ply_points(path: Path, matrix: np.ndarray) -> None:
    if transform_ply_with_open3d(path, matrix):
        return
    transform_ascii_ply(path, matrix)


def transform_ply_with_open3d(path: Path, matrix: np.ndarray) -> bool:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return False
    try:
        pc = o3d.io.read_point_cloud(str(path))
        if pc.is_empty():
            return False
        pc.transform(matrix)
        return bool(o3d.io.write_point_cloud(str(path), pc))
    except Exception:
        return False


def transform_ascii_ply(path: Path, matrix: np.ndarray) -> None:
    text = path.read_text(encoding="ascii", errors="strict")
    lines = text.splitlines(keepends=True)
    try:
        end_idx = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
    except StopIteration as e:
        raise ValueError(f"PLY header is missing end_header: {path}") from e

    header = lines[: end_idx + 1]
    if not any(line.strip().startswith("format ascii") for line in header):
        raise ValueError(f"Binary PLY correction requires open3d, but open3d could not transform: {path}")

    vertex_count = 0
    vertex_props: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and parts[0] == "property" and len(parts) >= 3:
            vertex_props.append(parts[-1])

    try:
        x_idx = vertex_props.index("x")
        y_idx = vertex_props.index("y")
        z_idx = vertex_props.index("z")
    except ValueError as e:
        raise ValueError(f"PLY vertex element must contain x/y/z properties: {path}") from e

    data_start = end_idx + 1
    if len(lines) < data_start + vertex_count:
        raise ValueError(f"PLY vertex data is truncated: {path}")

    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    for i in range(vertex_count):
        line_idx = data_start + i
        line = lines[line_idx]
        newline = "\n" if line.endswith("\n") else ""
        tokens = line.split()
        if len(tokens) < len(vertex_props):
            raise ValueError(f"PLY vertex row is truncated at row {i}: {path}")
        point = np.array(
            [float(tokens[x_idx]), float(tokens[y_idx]), float(tokens[z_idx])],
            dtype=np.float64,
        )
        corrected = rot @ point + trans
        tokens[x_idx] = f"{corrected[0]:.9g}"
        tokens[y_idx] = f"{corrected[1]:.9g}"
        tokens[z_idx] = f"{corrected[2]:.9g}"
        lines[line_idx] = " ".join(tokens) + newline

    path.write_text("".join(lines), encoding="ascii")


def resolve_pointcloud_path(base_dir: Path, ply_file_path: object) -> Path | None:
    if not isinstance(ply_file_path, str) or not ply_file_path.strip():
        return None
    path = Path(ply_file_path)
    if not path.is_absolute():
        path = base_dir / path
    return path if path.is_file() else None


def write_final_orientation_pointcloud(
    source: Path,
    dest: Path,
    orientation: object,
    *,
    already_applied: bool = False,
) -> None:
    orientation_id = normalize_final_orientation(orientation)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    if orientation_id == FINAL_ORIENTATION_NONE or already_applied:
        return
    transform_ply_points(dest, final_orientation_matrix(orientation_id))


def apply_final_orientation_to_dataset(
    output: Path,
    orientation: object = FINAL_ORIENTATION_LICHTFELD,
    *,
    stage: str = FINAL_ORIENTATION_STAGE_DIRECT_FINALIZE,
) -> None:
    transforms = output / "transforms.json"
    already_applied = False
    if transforms.is_file():
        data = json.loads(transforms.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"transforms.json root must be an object: {transforms}")
        already_applied = final_orientation_is_applied(data, orientation)
        apply_final_orientation_to_transforms_data(data, orientation, stage=stage)
        transforms.write_text(json.dumps(data, indent=2), encoding="utf-8")

    pointcloud = output / "pointcloud.ply"
    if pointcloud.is_file() and not already_applied and normalize_final_orientation(orientation) != FINAL_ORIENTATION_NONE:
        transform_ply_points(pointcloud, final_orientation_matrix(orientation))
