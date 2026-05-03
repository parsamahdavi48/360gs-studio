"""Item 4: transforms_to_colmap.py のテスト。

cameras.txt / images.txt / points3D.txt のフォーマット整合性、
クォータニオン変換の精度、PLY 読み込みフォールバックを検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transforms_to_colmap import (
    c2w_to_w2c,
    convert,
    quaternion_from_matrix,
    read_ply_points,
    strip_prefix,
    write_cameras_txt,
    write_images_txt,
    write_points3d_txt,
)


# =============================================================================
# quaternion_from_matrix
# =============================================================================


def test_quaternion_identity():
    """単位行列 → (1, 0, 0, 0)。"""
    q = quaternion_from_matrix(np.eye(3))
    assert q[0] == pytest.approx(1.0)
    assert np.allclose(q[1:], [0.0, 0.0, 0.0], atol=1e-10)


def test_quaternion_180_x():
    """X 軸 180° 回転 = diag(1, -1, -1) → (0, 1, 0, 0)。"""
    r = np.diag([1.0, -1.0, -1.0])
    q = quaternion_from_matrix(r)
    # qw=0, qx=1 期待
    assert q[0] == pytest.approx(0.0, abs=1e-10)
    assert abs(q[1]) == pytest.approx(1.0, abs=1e-10)


def test_quaternion_180_y():
    """Y 軸 180° 回転 = diag(-1, 1, -1)。"""
    r = np.diag([-1.0, 1.0, -1.0])
    q = quaternion_from_matrix(r)
    assert q[0] == pytest.approx(0.0, abs=1e-10)
    assert abs(q[2]) == pytest.approx(1.0, abs=1e-10)


def test_quaternion_180_z():
    """Z 軸 180° 回転 = diag(-1, -1, 1)。"""
    r = np.diag([-1.0, -1.0, 1.0])
    q = quaternion_from_matrix(r)
    assert q[0] == pytest.approx(0.0, abs=1e-10)
    assert abs(q[3]) == pytest.approx(1.0, abs=1e-10)


def test_quaternion_normalized():
    """任意回転でも単位クォータニオン (norm=1)。"""
    rng = np.random.default_rng(42)
    for _ in range(20):
        # ランダムな直交行列を生成 (QR 分解)
        a = rng.standard_normal((3, 3))
        q_orth, _ = np.linalg.qr(a)
        if np.linalg.det(q_orth) < 0:
            q_orth[:, 0] *= -1
        quat = quaternion_from_matrix(q_orth)
        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-10)
        assert quat[0] >= 0  # qw >= 0 半球


def test_quaternion_rejects_non_finite_matrix():
    with pytest.raises(ValueError, match="quaternion"):
        quaternion_from_matrix(np.full((3, 3), np.nan))


def test_quaternion_round_trip():
    """quaternion → matrix → quaternion で元に戻る。"""
    rng = np.random.default_rng(123)
    for _ in range(10):
        # 既知の (yaw, pitch, roll) から構築
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi, 3)
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        r_yaw = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
        r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        r = r_yaw @ r_pitch @ r_roll

        q = quaternion_from_matrix(r)

        # quaternion → matrix
        qw, qx, qy, qz = q
        r_back = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        assert np.allclose(r, r_back, atol=1e-10)


# =============================================================================
# c2w_to_w2c
# =============================================================================


def test_c2w_to_w2c_identity():
    """c2w = identity → R_w2c = identity, t_w2c = 0。"""
    r, t = c2w_to_w2c(np.eye(4))
    assert np.allclose(r, np.eye(3))
    assert np.allclose(t, [0, 0, 0])


def test_c2w_to_w2c_translation_only():
    """t_c2w=(1,2,3) → t_w2c = -R^T @ (1,2,3) = -(1,2,3) (R=I なので)。"""
    c2w = np.eye(4)
    c2w[:3, 3] = [1, 2, 3]
    r, t = c2w_to_w2c(c2w)
    assert np.allclose(t, [-1, -2, -3])


def test_c2w_to_w2c_invariants():
    """w2c が w2c の逆変換として正しい: R_w2c @ R_c2w = I, t_w2c + R_w2c @ t_c2w = 0。"""
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = rng.standard_normal((3, 3))
        r_c2w, _ = np.linalg.qr(a)
        if np.linalg.det(r_c2w) < 0:
            r_c2w[:, 0] *= -1
        c2w = np.eye(4)
        c2w[:3, :3] = r_c2w
        c2w[:3, 3] = rng.standard_normal(3)

        r_w2c, t_w2c = c2w_to_w2c(c2w)
        assert np.allclose(r_w2c @ r_c2w, np.eye(3), atol=1e-10)
        assert np.allclose(t_w2c + r_w2c @ c2w[:3, 3], 0.0, atol=1e-10)


def test_c2w_to_w2c_invalid_shape():
    with pytest.raises(ValueError):
        c2w_to_w2c(np.eye(3))


# =============================================================================
# strip_prefix
# =============================================================================


def test_strip_prefix_basic():
    assert strip_prefix("images/foo.jpg", "images/") == "foo.jpg"
    assert strip_prefix("images/sub/foo.jpg", "images/") == "sub/foo.jpg"


def test_strip_prefix_no_match():
    assert strip_prefix("foo.jpg", "images/") == "foo.jpg"


def test_strip_prefix_empty():
    assert strip_prefix("images/foo.jpg", "") == "images/foo.jpg"


def test_strip_prefix_windows_separator():
    """バックスラッシュ含む path も正しく処理。"""
    assert strip_prefix("images\\foo.jpg", "images/") == "foo.jpg"


# =============================================================================
# write_cameras_txt
# =============================================================================


def test_write_cameras_simple_pinhole(tmp_path: Path):
    """fx == fy なら SIMPLE_PINHOLE。"""
    p = tmp_path / "cameras.txt"
    write_cameras_txt(p, w=1920, h=1920, fl_x=960.0, fl_y=960.0, cx=959.5, cy=959.5)
    text = p.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "1"  # CAMERA_ID
    assert parts[1] == "SIMPLE_PINHOLE"
    assert parts[2] == "1920"
    assert parts[3] == "1920"
    assert float(parts[4]) == pytest.approx(960.0)
    assert float(parts[5]) == pytest.approx(959.5)
    assert float(parts[6]) == pytest.approx(959.5)


def test_write_cameras_pinhole(tmp_path: Path):
    """fx != fy なら PINHOLE。"""
    p = tmp_path / "cameras.txt"
    write_cameras_txt(p, w=1920, h=1080, fl_x=960.0, fl_y=540.0, cx=959.5, cy=539.5)
    text = p.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    parts = lines[0].split()
    assert parts[1] == "PINHOLE"
    assert len(parts) == 8  # ID, MODEL, W, H, fx, fy, cx, cy


# =============================================================================
# write_images_txt
# =============================================================================


def test_write_images_basic(tmp_path: Path):
    p = tmp_path / "images.txt"
    frames = [
        {"file_path": "images/a.jpg", "transform_matrix": np.eye(4).tolist()},
        {"file_path": "images/b.jpg", "transform_matrix": np.eye(4).tolist()},
    ]
    n = write_images_txt(p, frames, "images/")
    assert n == 2

    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    # 各画像は 2 行 (data + 空 POINTS2D)
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    # 2 frames × 2 lines (data + empty) = 4 lines but empty lines are excluded
    assert len(data_lines) == 2
    parts = data_lines[0].split()
    assert parts[0] == "1"  # IMAGE_ID
    assert parts[8] == "1"  # CAMERA_ID
    assert parts[9] == "a.jpg"  # NAME (prefix stripped)


def test_write_images_skips_invalid_frames(tmp_path: Path):
    p = tmp_path / "images.txt"
    frames = [
        {"file_path": "images/a.jpg", "transform_matrix": np.eye(4).tolist()},
        {"file_path": "", "transform_matrix": np.eye(4).tolist()},  # empty file_path
        {"file_path": "images/c.jpg"},  # missing transform
        {"file_path": "images/d.jpg", "transform_matrix": np.eye(4).tolist()},
    ]
    n = write_images_txt(p, frames, "images/")
    assert n == 2  # only a and d


# =============================================================================
# write_points3d_txt
# =============================================================================


def test_write_points3d_with_colors(tmp_path: Path):
    p = tmp_path / "points3D.txt"
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    n = write_points3d_txt(p, points, colors)
    assert n == 2
    text = p.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert len(lines) == 2
    parts = lines[0].split()
    assert parts[0] == "1"  # POINT3D_ID
    assert float(parts[1]) == pytest.approx(1.0)
    assert int(parts[4]) == 255  # R
    assert int(parts[5]) == 0  # G


def test_write_points3d_no_colors(tmp_path: Path):
    """colors=None なら gray (128) で埋める。"""
    p = tmp_path / "points3D.txt"
    points = np.array([[1.0, 2.0, 3.0]])
    n = write_points3d_txt(p, points, None)
    assert n == 1
    text = p.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    parts = lines[0].split()
    assert int(parts[4]) == 128
    assert int(parts[5]) == 128
    assert int(parts[6]) == 128


# =============================================================================
# read_ply_points (ASCII fallback)
# =============================================================================


def test_read_ply_ascii(tmp_path: Path):
    """ASCII PLY を内蔵 fallback リーダーで読み込めること。"""
    p = tmp_path / "test.ply"
    p.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
        "1.0 2.0 3.0 255 0 0\n"
        "4.0 5.0 6.0 0 255 0\n"
        "7.0 8.0 9.0 0 0 255\n",
        encoding="ascii",
    )
    points, colors = read_ply_points(p)
    assert points.shape == (3, 3)
    assert np.allclose(points[0], [1, 2, 3])
    assert np.allclose(points[2], [7, 8, 9])
    assert colors is not None
    assert colors[0, 0] == 255  # R
    assert colors[1, 1] == 255  # G


def test_read_ply_ascii_no_colors(tmp_path: Path):
    p = tmp_path / "test.ply"
    p.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "1.0 2.0 3.0\n",
        encoding="ascii",
    )
    points, colors = read_ply_points(p)
    assert points.shape == (1, 3)
    assert colors is None


# =============================================================================
# end-to-end convert
# =============================================================================


def _make_test_transforms_json(path: Path, num_frames: int = 3) -> None:
    frames = []
    for i in range(num_frames):
        t = np.eye(4)
        t[0, 3] = float(i) * 0.5
        frames.append(
            {
                "file_path": f"images/frame_{i:04d}_front.jpg",
                "transform_matrix": t.tolist(),
            }
        )
    data = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 1920,
        "h": 1920,
        "fl_x": 960.0,
        "fl_y": 960.0,
        "cx": 959.5,
        "cy": 959.5,
        "frames": frames,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_convert_end_to_end_no_ply(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _make_test_transforms_json(input_dir / "transforms.json", num_frames=4)

    output_dir = tmp_path / "out"
    result = convert(
        input_dir=input_dir,
        json_name="transforms.json",
        output_dir=output_dir,
        ply_path=None,
        image_prefix="images/",
    )
    assert result["num_images"] == 4
    assert result["num_points"] == 0
    assert result["camera_model"] == "SIMPLE_PINHOLE"

    assert (output_dir / "cameras.txt").is_file()
    assert (output_dir / "images.txt").is_file()
    assert (output_dir / "points3D.txt").is_file()


def test_convert_end_to_end_with_ply(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _make_test_transforms_json(input_dir / "transforms.json", num_frames=2)

    ply_path = input_dir / "pointcloud.ply"
    ply_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
        "0 0 0 100 100 100\n"
        "1 1 1 200 200 200\n",
        encoding="ascii",
    )

    output_dir = tmp_path / "out"
    result = convert(
        input_dir=input_dir,
        json_name="transforms.json",
        output_dir=output_dir,
        ply_path=ply_path,
        image_prefix="images/",
    )
    assert result["num_points"] == 2
    # PLY コピーも確認
    assert (output_dir / "points3D.ply").is_file()


def test_convert_rejects_equirect(tmp_path: Path):
    """EQUIRECTANGULAR モデルの transforms.json は弾く（cubemap_transforms_json.py を先に通す必要）。"""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    data = {"camera_model": "EQUIRECTANGULAR", "frames": []}
    (input_dir / "transforms.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported camera_model"):
        convert(input_dir, "transforms.json", tmp_path / "out", None, "images/")


def test_convert_missing_json(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        convert(tmp_path, "transforms.json", tmp_path / "out", None, "images/")
