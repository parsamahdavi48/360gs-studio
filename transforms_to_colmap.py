#!/usr/bin/env python3
"""nerfstudio 風 transforms.json → COLMAP テキスト形式 (cameras.txt / images.txt / points3D.txt)。

cubemap_transforms_json.py が出力した transforms.json と pointcloud.ply を読み込み、
PostShot / Brush / 公式 gaussian-splatting / nerfstudio など COLMAP テキストを食う
ツールに渡せる形式に変換する。

注意: 入力 transforms.json は既に profile (Postshot/Brush/LichtFeld) に応じた軸変換が
適用済みである前提。本スクリプトは追加の座標変換は行わない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert nerfstudio-style transforms.json to COLMAP text format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python transforms_to_colmap.py ./output\n"
            "  python transforms_to_colmap.py ./output ./output/colmap --ply ./output/pointcloud.ply\n"
        ),
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing transforms.json (output from cubemap_transforms_json.py)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Output directory for COLMAP text files (default=<input_dir>/colmap)",
    )
    parser.add_argument(
        "--json",
        default="transforms.json",
        help="Input transforms.json filename (default=transforms.json)",
    )
    parser.add_argument(
        "--ply",
        help="Optional PLY file for points3D.txt and points3D.ply output",
    )
    parser.add_argument(
        "--image-prefix",
        "--image_prefix",
        dest="image_prefix",
        default="images/",
        help=(
            "Prefix in transforms.json file_path entries to strip when writing image names "
            "(default='images/'). Set to '' to keep paths as-is."
        ),
    )
    return parser.parse_args()


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
    if p.startswith(prefix):
        return p[len(prefix):]
    return p


def write_cameras_txt(out_path: Path, w: int, h: int, fl_x: float, fl_y: float,
                      cx: float, cy: float) -> None:
    """SIMPLE_PINHOLE / PINHOLE モデルで cameras.txt を出力（カメラ ID = 1 で全画像共有）。"""
    use_simple = abs(fl_x - fl_y) < 1e-6
    model = "SIMPLE_PINHOLE" if use_simple else "PINHOLE"
    if use_simple:
        params = f"{fl_x} {cx} {cy}"
    else:
        params = f"{fl_x} {fl_y} {cx} {cy}"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 {model} {w} {h} {params}\n")


def write_images_txt(out_path: Path, frames: list[dict], image_prefix: str) -> int:
    """images.txt 出力。各画像は CAMERA_ID=1 を共有。POINTS2D は空。

    Returns:
        書き出された画像数。
    """
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
            f.write(
                f"{idx} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                f"{t_w2c[0]:.10f} {t_w2c[1]:.10f} {t_w2c[2]:.10f} 1 {name}\n"
            )
            f.write("\n")  # POINTS2D 空行
            written += 1
    return written


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

    w = int(data.get("w", 0))
    h = int(data.get("h", 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid w/h in transforms.json: w={w}, h={h}")

    fl_x = float(data["fl_x"])
    fl_y = float(data.get("fl_y", fl_x))
    cx = float(data.get("cx", (w - 1) / 2.0))
    cy = float(data.get("cy", (h - 1) / 2.0))

    frames = data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames in transforms.json is empty")

    output_dir.mkdir(parents=True, exist_ok=True)

    cameras_path = output_dir / "cameras.txt"
    images_path = output_dir / "images.txt"
    points_path = output_dir / "points3D.txt"

    write_cameras_txt(cameras_path, w, h, fl_x, fl_y, cx, cy)
    num_images = write_images_txt(images_path, frames, image_prefix)

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

    return {
        "num_images": num_images,
        "num_points": num_points,
        "camera_model": "SIMPLE_PINHOLE" if abs(fl_x - fl_y) < 1e-6 else "PINHOLE",
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input_dir '{input_dir}' not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "colmap"
    ply_path = Path(args.ply) if args.ply else None

    try:
        result = convert(
            input_dir=input_dir,
            json_name=args.json,
            output_dir=output_dir,
            ply_path=ply_path,
            image_prefix=args.image_prefix,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote cameras.txt, images.txt, points3D.txt to {result['output_dir']}")
    print(f"  Camera model: {result['camera_model']}")
    print(f"  Images: {result['num_images']}")
    print(f"  3D points: {result['num_points']}")


if __name__ == "__main__":
    main()
