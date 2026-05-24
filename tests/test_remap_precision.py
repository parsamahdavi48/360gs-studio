"""Item 1: リマップ精度のテスト。

build_remap がピクセル中心規約と arctan2 緯度で正しく動作するかを検証する。
"""
from __future__ import annotations

import numpy as np
import pytest

from core.cubemap_transforms_json import build_remap, rotation_matrix


def _equirect_to_xyz(map_x: float, map_y: float, w: int, h: int) -> tuple[float, float, float]:
    """build_remap の逆: equirect 座標 (map_x, map_y) を 3D 単位ベクトルに変換。"""
    lon = (map_x / w * 2.0 - 1.0) * np.pi
    lat = (0.5 - map_y / h) * np.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return float(x), float(y), float(z)


def test_principal_point_at_pixel_center():
    """主点が出力画像の幾何中心 (W-1)/2 に来ること。

    yaw=0, pitch=0 の正面ビューで、出力画像の中心ピクセルが equirect の (W/2, H/2) を指すべき。
    """
    input_size = (4096, 2048)
    output_size = 1920
    fov = 90.0

    map_x, map_y = build_remap(input_size, fov, yaw_deg=0.0, pitch_deg=0.0, output_size=output_size)

    # 出力画像中心の座標は ((output_size-1)/2, (output_size-1)/2) = (959.5, 959.5)
    # その近傍 (959, 959) と (960, 960) は対称に center 周りに分布
    # 整数ピクセル (960, 960) の equirect サンプリング位置を確認
    # 主点 (W-1)/2 = 959.5 なので、ピクセル 960 は中心から +0.5 の位置
    sample_at_960 = (map_x[960, 960], map_y[960, 960])
    sample_at_959 = (map_x[959, 959], map_y[959, 959])

    # equirect 中心 (W/2, H/2) からの偏位が対称的であるべき
    eq_center_x = input_size[0] / 2.0
    eq_center_y = input_size[1] / 2.0
    dx_960 = sample_at_960[0] - eq_center_x
    dx_959 = sample_at_959[0] - eq_center_x
    dy_960 = sample_at_960[1] - eq_center_y
    dy_959 = sample_at_959[1] - eq_center_y

    # 対称性: ピクセル (959, 959) と (960, 960) は中心から等距離反対側にあるべき。
    # map は float32 で保持されるため、float32 量子化精度 (~1e-4) を許容する。
    assert dx_960 == pytest.approx(-dx_959, abs=1e-3), (
        f"x asymmetry: dx_959={dx_959}, dx_960={dx_960}"
    )
    assert dy_960 == pytest.approx(-dy_959, abs=1e-3), (
        f"y asymmetry: dy_959={dy_959}, dy_960={dy_960}"
    )


def test_round_trip_known_directions():
    """既知方向の往復テスト: 正面 (yaw=0) ビューの中心ピクセルは equirect の (W/2, H/2) 付近を指す。"""
    input_size = (4096, 2048)
    output_size = 512
    fov = 90.0

    map_x, map_y = build_remap(input_size, fov, yaw_deg=0.0, pitch_deg=0.0, output_size=output_size)

    # 中心ピクセル付近の平均座標は equirect の幾何中心 (W/2, H/2) に一致するべき
    cx = (output_size - 1) / 2.0
    # 主点まわり 4 ピクセルの平均
    sx = (map_x[int(cx), int(cx)] + map_x[int(cx) + 1, int(cx)]
          + map_x[int(cx), int(cx) + 1] + map_x[int(cx) + 1, int(cx) + 1]) / 4.0
    sy = (map_y[int(cx), int(cx)] + map_y[int(cx) + 1, int(cx)]
          + map_y[int(cx), int(cx) + 1] + map_y[int(cx) + 1, int(cx) + 1]) / 4.0

    assert sx == pytest.approx(input_size[0] / 2.0, abs=0.5)
    assert sy == pytest.approx(input_size[1] / 2.0, abs=0.5)


def test_yaw_90_points_to_right():
    """yaw=90° は equirect の +X 軸方向 (経度 +π/2 = 右側) を中心にサンプリングする。"""
    input_size = (4096, 2048)
    output_size = 512
    fov = 90.0

    map_x, _ = build_remap(input_size, fov, yaw_deg=90.0, pitch_deg=0.0, output_size=output_size)

    cx = (output_size - 1) / 2.0
    # +X 方向 = lon = +π/2 → equirect map_x = (π/2/π + 1)/2 * W = 3/4 * W = 3072
    expected = 0.75 * input_size[0]
    assert map_x[int(cx), int(cx)] == pytest.approx(expected, abs=2.0)


def test_pole_stability_arctan2():
    """極近傍 pitch で計算しても緯度が正しい範囲に収まり、極端な値や反転が起きないこと。

    本リポの規約: rotation_matrix(yaw, pitch, forward=False) では +pitch がカメラを下方向へ回転。
    pitch=85° なら中心ピクセル付近の緯度はおよそ -80° ～ -90° の範囲になる。
    """
    input_size = (4096, 2048)
    output_size = 257  # 奇数: 中心ピクセル (128, 128) が光軸上に位置する
    fov = 90.0

    map_x, map_y = build_remap(input_size, fov, yaw_deg=0.0, pitch_deg=85.0, output_size=output_size)

    cx_idx = output_size // 2  # = 128
    # 中心ピクセルが光軸上なら、正確に lat = -85° を指す
    expected_lat = np.deg2rad(-85.0)
    expected_map_y = (0.5 - expected_lat / np.pi) * input_size[1]

    assert map_y[cx_idx, cx_idx] == pytest.approx(expected_map_y, abs=1.0)
    # NaN や Inf が混入しないこと
    assert np.isfinite(map_x).all()
    assert np.isfinite(map_y).all()


def test_pole_no_nan_at_extreme():
    """完全な極 (pitch=±90°) で計算しても NaN/Inf が発生しないこと。"""
    map_x, map_y = build_remap((2048, 1024), 90.0, 0.0, 90.0, 128)
    assert np.isfinite(map_x).all() and np.isfinite(map_y).all()
    map_x2, map_y2 = build_remap((2048, 1024), 90.0, 0.0, -90.0, 128)
    assert np.isfinite(map_x2).all() and np.isfinite(map_y2).all()


def test_focal_length_consistency():
    """fov=90° のとき、焦点距離は output_size/2 になる（正接公式）。"""
    output_size = 1920
    fov = 90.0
    expected_focal = output_size / 2.0 / np.tan(np.deg2rad(fov) / 2.0)
    # fov=90° では tan(45°)=1 なので focal = output_size/2 = 960
    assert expected_focal == pytest.approx(output_size / 2.0)


def test_dtype_is_float32():
    """出力 map は cv2.remap が要求する float32 であること。"""
    map_x, map_y = build_remap((1024, 512), 90.0, 0.0, 0.0, 256)
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32


def test_rotation_matrix_orthogonality():
    """rotation_matrix が直交行列を返すこと（数値精度範囲内）。"""
    for yaw, pitch in [(0, 0), (45, 0), (0, 45), (90, -30), (180, 60), (-45, -90)]:
        for forward in [True, False]:
            r = rotation_matrix(yaw, pitch, forward)
            should_be_identity = r @ r.T
            assert np.allclose(should_be_identity, np.eye(3), atol=1e-10), (
                f"R @ R.T not identity for yaw={yaw}, pitch={pitch}, forward={forward}"
            )
            assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-10)
