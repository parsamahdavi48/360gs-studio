#!/usr/bin/env python3
"""nerfstudio 風 transforms.json → COLMAP テキスト形式 (cameras.txt / images.txt / points3D.txt)。

cubemap_transforms_json.py が出力した transforms.json と pointcloud.ply を読み込み、
PostShot / Brush / 公式 gaussian-splatting / nerfstudio など COLMAP テキストを食う
ツールに渡せる形式に変換する。

注意: 入力 transforms.json は既に profile (Postshot/Brush/LichtFeld) に応じた軸変換が
適用済みである前提。本スクリプトは追加の座標変換は行わない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    from core.transforms_to_colmap_cli import parse_args as _parse_args

    return _parse_args()


def quaternion_from_matrix(r: np.ndarray) -> np.ndarray:
    """3x3 回転行列 → クォータニオン (qw, qx, qy, qz) (Hamilton 規約、qw >= 0)。"""
    trace = float(np.trace(r))
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r[2, 1] - r[1, 2]) * s
        qy = (r[0, 2] - r[2, 0]) * s
        qz = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("Invalid near-zero quaternion")
    q /= n
    if q[0] < 0.0:
        q = -q
    return q


def c2w_to_w2c(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 c2w → (R_w2c, t_w2c)。"""
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    r_c2w = transform[:3, :3]
    t_c2w = transform[:3, 3]
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ t_c2w
    return r_w2c, t_w2c


def strip_prefix(path: str, prefix: str) -> str:
    """transforms.json 内の file_path から先頭 prefix を取り除く。"""
    if not prefix:
        return path
    p = path.replace("\\", "/")
    normalized = prefix.replace("\\", "/").strip("/")
    if not normalized:
        return p
    directory_prefix = f"{normalized}/"
    if p.startswith(directory_prefix):
        return p[len(directory_prefix):]
    return p


def _camera_model_params(
    w: int,
    h: int,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
) -> tuple[str, tuple[float, ...]]:
    use_simple = abs(fl_x - fl_y) < 1e-6
    model = "SIMPLE_PINHOLE" if use_simple else "PINHOLE"
    if use_simple:
        return model, (float(fl_x), float(cx), float(cy))
    return model, (float(fl_x), float(fl_y), float(cx), float(cy))


def write_cameras_txt(out_path: Path, w: int, h: int, fl_x: float, fl_y: float,
                      cx: float, cy: float) -> None:
    """SIMPLE_PINHOLE / PINHOLE モデルで cameras.txt を出力（カメラ ID = 1 で全画像共有）。"""
    model, values = _camera_model_params(w, h, fl_x, fl_y, cx, cy)
    params = " ".join(str(value) for value in values)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 {model} {w} {h} {params}\n")


def write_camera_records_txt(
    out_path: Path,
    cameras: list[tuple[int, str, int, int, tuple[float, ...]]],
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id, model, width, height, params in cameras:
            params_text = " ".join(str(value) for value in params)
            f.write(f"{camera_id} {model} {width} {height} {params_text}\n")


def write_images_txt(
    out_path: Path,
    frames: list[dict],
    image_prefix: str,
    camera_ids: list[int] | None = None,
) -> int:
    """images.txt 出力。POINTS2D は空。

    Returns:
        書き出された画像数。
    """
    if camera_ids is None:
        camera_ids = [1 for _frame in frames]
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")
        for idx, frame in enumerate(frames, start=1):
            file_path = frame.get("file_path", "")
            if not isinstance(file_path, str) or not file_path:
                continue
            try:
                t = np.array(frame["transform_matrix"], dtype=np.float64)
            except Exception:
                continue

            r_w2c, t_w2c = c2w_to_w2c(t)
            try:
                qw, qx, qy, qz = quaternion_from_matrix(r_w2c)
            except ValueError:
                continue

            name = strip_prefix(file_path, image_prefix)
            camera_id = int(camera_ids[idx - 1]) if idx - 1 < len(camera_ids) else 1
            f.write(
                f"{idx} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                f"{t_w2c[0]:.10f} {t_w2c[1]:.10f} {t_w2c[2]:.10f} {camera_id} {name}\n"
            )
            f.write("\n")  # POINTS2D 空行
            written += 1
    return written


def _frame_intrinsics(data: dict, frame: dict) -> tuple[int, int, float, float, float, float]:
    width = int(frame.get("w") or data.get("w") or 0)
    height = int(frame.get("h") or data.get("h") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid w/h in transforms.json: w={width}, h={height}")
    fl_x = float(frame.get("fl_x") or data.get("fl_x") or 0.0)
    fl_y = float(frame.get("fl_y") or data.get("fl_y") or fl_x)
    cx = float(frame.get("cx") if frame.get("cx") is not None else data.get("cx", (width - 1) / 2.0))
    cy = float(frame.get("cy") if frame.get("cy") is not None else data.get("cy", (height - 1) / 2.0))
    if fl_x <= 0.0 or fl_y <= 0.0:
        raise ValueError("Invalid fl_x/fl_y in transforms.json")
    return width, height, fl_x, fl_y, cx, cy


def _build_camera_records(data: dict, frames: list[dict]) -> tuple[list[tuple[int, str, int, int, tuple[float, ...]]], list[int]]:
    camera_ids_by_key: dict[tuple[object, ...], int] = {}
    cameras: list[tuple[int, str, int, int, tuple[float, ...]]] = []
    frame_camera_ids: list[int] = []
    for frame in frames:
        width, height, fl_x, fl_y, cx, cy = _frame_intrinsics(data, frame)
        model, params = _camera_model_params(width, height, fl_x, fl_y, cx, cy)
        key = (
            model,
            int(width),
            int(height),
            tuple(round(float(value), 9) for value in params),
        )
        camera_id = camera_ids_by_key.get(key)
        if camera_id is None:
            camera_id = len(cameras) + 1
            camera_ids_by_key[key] = camera_id
            cameras.append((camera_id, model, width, height, params))
        frame_camera_ids.append(camera_id)
    return cameras, frame_camera_ids


def read_ply_points(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """PLY 読み込み: open3d → plyfile → 内蔵 ASCII パーサ の順でフォールバック。

    Returns:
        (points (N,3), colors (N,3) uint8 or None)
    """
    try:
        import open3d as o3d  # type: ignore
        pc = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pc.points)
        if pc.has_colors():
            colors = (np.asarray(pc.colors) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            colors = None
        return points, colors
    except ImportError:
        pass

    try:
        from plyfile import PlyData  # type: ignore
        plydata = PlyData.read(str(ply_path))
        v = plydata["vertex"].data
        points = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
        if all(c in v.dtype.names for c in ("red", "green", "blue")):
            colors = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.uint8)
        else:
            colors = None
        return points, colors
    except ImportError:
        pass

    # 最低限の ASCII PLY パーサ（依存ゼロでもエラーにならないように）
    return _read_ply_ascii_fallback(ply_path)


def _read_ply_ascii_fallback(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with ply_path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
            if not line:
                raise OSError(f"Unexpected EOF reading PLY header: {ply_path}")

        format_line = next((ln for ln in header_lines if ln.startswith("format")), "")
        if "ascii" not in format_line:
            raise OSError(
                "Binary PLY without open3d/plyfile is not supported by the fallback reader. "
                "Install open3d or plyfile: pip install open3d  (or: pip install plyfile)"
            )

        vertex_count = 0
        properties: list[str] = []
        in_vertex = False
        for ln in header_lines:
            if ln.startswith("element vertex"):
                vertex_count = int(ln.split()[-1])
                in_vertex = True
            elif ln.startswith("element"):
                in_vertex = False
            elif ln.startswith("property") and in_vertex:
                properties.append(ln.split()[-1])

        x_idx = properties.index("x") if "x" in properties else None
        y_idx = properties.index("y") if "y" in properties else None
        z_idx = properties.index("z") if "z" in properties else None
        if x_idx is None or y_idx is None or z_idx is None:
            raise OSError(f"PLY missing x/y/z vertex properties: {ply_path}")
        r_idx = properties.index("red") if "red" in properties else None
        g_idx = properties.index("green") if "green" in properties else None
        b_idx = properties.index("blue") if "blue" in properties else None

        points = np.zeros((vertex_count, 3), dtype=np.float64)
        has_color = r_idx is not None and g_idx is not None and b_idx is not None
        colors = np.zeros((vertex_count, 3), dtype=np.uint8) if has_color else None

        for i in range(vertex_count):
            tokens = f.readline().decode("ascii", errors="replace").split()
            if len(tokens) < len(properties):
                raise OSError(f"Truncated vertex row {i} in {ply_path}")
            points[i, 0] = float(tokens[x_idx])
            points[i, 1] = float(tokens[y_idx])
            points[i, 2] = float(tokens[z_idx])
            if colors is not None:
                colors[i, 0] = int(tokens[r_idx]) & 0xFF
                colors[i, 1] = int(tokens[g_idx]) & 0xFF
                colors[i, 2] = int(tokens[b_idx]) & 0xFF

    return points, colors


def write_points3d_txt(out_path: Path, points: np.ndarray, colors: np.ndarray | None) -> int:
    """points3D.txt をストリーム書き出し。track 情報は空。"""
    n = len(points)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {n}\n")
        chunk = 10000
        buf: list[str] = []
        for i in range(n):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            else:
                r = g = b = 128
            buf.append(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0\n")
            if len(buf) >= chunk:
                f.write("".join(buf))
                buf.clear()
        if buf:
            f.write("".join(buf))
    return n


def write_empty_points3d_txt(out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")


def convert(
    input_dir: Path,
    json_name: str,
    output_dir: Path,
    ply_path: Path | None,
    image_prefix: str,
) -> dict:
    json_path = input_dir / json_name
    if not json_path.is_file():
        raise FileNotFoundError(f"transforms.json not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    cm = data.get("camera_model", "")
    if cm not in {"SIMPLE_PINHOLE", "PINHOLE"}:
        raise ValueError(
            f"Unsupported camera_model '{cm}' (only SIMPLE_PINHOLE/PINHOLE allowed; "
            "run cubemap_transforms_json.py first to convert from EQUIRECTANGULAR)"
        )

    frames = data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames in transforms.json is empty")
    frames = [frame for frame in frames if isinstance(frame, dict)]
    if not frames:
        raise ValueError("frames in transforms.json is empty")
    cameras, frame_camera_ids = _build_camera_records(data, frames)

    output_dir.mkdir(parents=True, exist_ok=True)

    cameras_path = output_dir / "cameras.txt"
    images_path = output_dir / "images.txt"
    points_path = output_dir / "points3D.txt"

    write_camera_records_txt(cameras_path, cameras)
    num_images = write_images_txt(images_path, frames, image_prefix, frame_camera_ids)

    num_points = 0
    if ply_path is not None and ply_path.is_file():
        try:
            points, colors = read_ply_points(ply_path)
        except Exception as e:
            print(f"Warning: failed to read PLY ({e}); writing empty points3D.txt", file=sys.stderr)
            write_empty_points3d_txt(points_path)
        else:
            num_points = write_points3d_txt(points_path, points, colors)
            # PLY をそのまま COLMAP 慣例の points3D.ply としてもコピー出力
            try:
                import shutil
                shutil.copy2(ply_path, output_dir / "points3D.ply")
            except Exception:
                pass
    else:
        if ply_path is not None:
            print(f"Warning: PLY not found: {ply_path}; writing empty points3D.txt", file=sys.stderr)
        write_empty_points3d_txt(points_path)

    first_camera_model = cameras[0][1] if cameras else "PINHOLE"
    return {
        "num_images": num_images,
        "num_points": num_points,
        "camera_model": first_camera_model if len(cameras) == 1 else "mixed",
        "num_cameras": len(cameras),
        "output_dir": str(output_dir),
    }


def main() -> None:
    from core.transforms_to_colmap_cli import main as _main

    _main()


if __name__ == "__main__":
    main()
