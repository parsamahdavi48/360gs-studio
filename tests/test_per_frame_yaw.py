"""Item 3: per-frame yaw rotation のテスト。

frame index に応じた yaw オフセット計算と、ワーカー側のテーブルキャッシュ動作を検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import core.cubemap_transforms_json as cube
import core.cubemap_worker_plan as worker_plan
from core.cubemap_remap import quantize_yaw_offset
from core.cubemap_transform_export import frame_yaw_offset, transform_json
from core.cubemap_transforms_json import (
    build_remap,
    get_remap_tables_for_offset,
    worker_init,
)
from core.cubemap_worker_plan import resolve_remap_cache_limit, resolve_worker_count

# =============================================================================
# frame_yaw_offset
# =============================================================================


def test_frame_yaw_offset_zero_step_returns_zero():
    """step=0 なら常に 0（旧動作の後方互換）。"""
    for i in range(10):
        assert frame_yaw_offset(i, 0.0) == 0.0


def test_frame_yaw_offset_basic():
    """step=30° で frame N の yaw = (N * 30) mod 360。"""
    assert frame_yaw_offset(0, 30.0) == 0.0
    assert frame_yaw_offset(1, 30.0) == 30.0
    assert frame_yaw_offset(2, 30.0) == 60.0
    assert frame_yaw_offset(11, 30.0) == 330.0
    assert frame_yaw_offset(12, 30.0) == 0.0  # 1 周期完了


def test_frame_yaw_offset_unique_count_30deg():
    """step=30° の場合、ユニーク値は 12 個（{0, 30, ..., 330}）。"""
    offsets = {frame_yaw_offset(i, 30.0) for i in range(100)}
    assert len(offsets) == 12
    assert offsets == {float(i * 30) for i in range(12)}


def test_frame_yaw_offset_unique_count_45deg():
    """step=45° の場合、ユニーク値は 8 個（{0, 45, ..., 315}）。"""
    offsets = {frame_yaw_offset(i, 45.0) for i in range(100)}
    assert len(offsets) == 8


def test_frame_yaw_offset_irrational_step():
    """step が 360 を割り切らない値（黄金比的）でも正しく動作。"""
    step = 90.0 * (np.sqrt(5) - 1) / 2.0  # ≈ 55.62°
    offsets_100 = {round(frame_yaw_offset(i, step), 6) for i in range(100)}
    # 非周期なので 100 個ほぼすべてユニークになる
    assert len(offsets_100) >= 95


# =============================================================================
# _quantize_yaw_offset
# =============================================================================


def test_quantize_yaw_offset_mod_360():
    assert quantize_yaw_offset(0.0) == 0.0
    assert quantize_yaw_offset(30.0) == 30.0
    assert quantize_yaw_offset(360.0) == 0.0
    assert quantize_yaw_offset(390.0) == 30.0
    assert quantize_yaw_offset(-30.0) == 330.0


def test_quantize_yaw_offset_decimal_precision():
    """3 桁丸めでキャッシュキーが安定すること。"""
    assert quantize_yaw_offset(30.1234567) == 30.123
    assert quantize_yaw_offset(30.1235) == quantize_yaw_offset(30.1235)


# =============================================================================
# get_remap_tables_for_offset (cache behavior)
# =============================================================================


def _setup_worker(tmp_path: Path):
    """ワーカー globals を初期化（テスト用）。"""
    views = [
        {"name": "front", "yaw": 0.0, "pitch": 0.0},
        {"name": "right", "yaw": 90.0, "pitch": 0.0},
    ]
    worker_init(
        input_size=(256, 128),
        fov=90.0,
        output_size=64,
        views=views,
        image_dir=str(tmp_path),
        mask_dir="",
        output_image_dir=str(tmp_path / "out"),
        output_mask_dir=str(tmp_path / "mask"),
        mask_from_alpha=False,
        invert_masks=False,
        output_format=None,
        output_bit_depth="8",
        jpg_quality=95,
    )


def _cache_key(offset: float, size: tuple[int, int] = (256, 128)) -> tuple[int, int, float]:
    return cube._remap_cache_key(size, offset)


def test_table_cache_offset_zero_prebuilt(tmp_path):
    """worker_init 後、offset=0 のテーブルが既にキャッシュされていること。"""
    _setup_worker(tmp_path)
    key = _cache_key(0.0)
    assert key in cube._WORKER_REMAP_CACHE
    assert "front" in cube._WORKER_REMAP_CACHE[key]
    assert "right" in cube._WORKER_REMAP_CACHE[key]


def test_table_cache_lazy_build(tmp_path):
    """offset=30° 等は lazy に生成され、再度同じ key を要求するとキャッシュヒット。"""
    _setup_worker(tmp_path)
    key = _cache_key(30.0)
    assert key not in cube._WORKER_REMAP_CACHE

    tables_30 = get_remap_tables_for_offset(30.0)
    assert key in cube._WORKER_REMAP_CACHE
    assert tables_30 is cube._WORKER_REMAP_CACHE[key]

    # 2 回目: 同じ dict object が返る
    tables_30_again = get_remap_tables_for_offset(30.0)
    assert tables_30_again is tables_30


def test_table_cache_size_after_one_period(tmp_path):
    """step=30° の 1 周期分（12 frames）回せば、キャッシュは 12 個（+ 0=既存）の lazily-built エントリ。"""
    _setup_worker(tmp_path)
    for i in range(12):
        offset = frame_yaw_offset(i, 30.0)
        get_remap_tables_for_offset(offset)
    # 0, 30, 60, ..., 330 の 12 個
    assert len(cube._WORKER_REMAP_CACHE) == 12


def test_table_cache_respects_lru_limit(tmp_path):
    _setup_worker(tmp_path)
    cube._WORKER_REMAP_CACHE_LIMIT = 2

    get_remap_tables_for_offset(30.0)
    get_remap_tables_for_offset(60.0)

    assert len(cube._WORKER_REMAP_CACHE) == 2
    assert _cache_key(0.0) not in cube._WORKER_REMAP_CACHE
    assert _cache_key(30.0) in cube._WORKER_REMAP_CACHE
    assert _cache_key(60.0) in cube._WORKER_REMAP_CACHE


def test_table_cache_offset_yields_different_tables(tmp_path):
    """offset≠0 のテーブルは offset=0 と異なる内容（実際に yaw 回転が反映される）。"""
    _setup_worker(tmp_path)
    t0 = cube._WORKER_REMAP_CACHE[_cache_key(0.0)]["front"]
    t30 = get_remap_tables_for_offset(30.0)["front"]
    # map_x が異なるはず
    assert not np.allclose(t0[0], t30[0])


def test_table_cache_offset_30_equals_view_at_30(tmp_path):
    """offset=30° + view at yaw=0° は、view at yaw=30° と同じテーブルになる。"""
    _setup_worker(tmp_path)
    cached_30 = get_remap_tables_for_offset(30.0)["front"]
    direct_30 = build_remap((256, 128), 90.0, 30.0, 0.0, 64)
    assert np.allclose(cached_30[0], direct_30[0])
    assert np.allclose(cached_30[1], direct_30[1])


def test_auto_remap_cache_limit_uses_memory_budget(monkeypatch):
    monkeypatch.setattr(worker_plan, "available_memory_bytes", lambda: 1_000_000)
    offsets = [float(i * 30) for i in range(12)]

    limit = resolve_remap_cache_limit(
        "auto",
        offsets,
        output_size=256,
        view_count=6,
        worker_count=4,
    )

    assert limit == 1


def test_explicit_worker_count_is_respected(monkeypatch):
    monkeypatch.setattr(worker_plan, "available_memory_bytes", lambda: 1)

    workers = resolve_worker_count(
        "3",
        input_size=(7840, 3920),
        output_size=1960,
        view_count=6,
        remap_cache_limit=12,
    )

    assert workers == 3


# =============================================================================
# transform_json end-to-end with yaw_offset_per_frame
# =============================================================================


def _write_input_transforms(tmp_path: Path, num_frames: int = 5) -> Path:
    """入力 transforms.json (EQUIRECTANGULAR) を生成。"""
    frames = []
    for i in range(num_frames):
        # 単純な camera-to-world (前進する camera)
        t = np.eye(4)
        t[0, 3] = float(i) * 0.1
        frames.append(
            {
                "file_path": f"images/frame_{i:04d}.png",
                "transform_matrix": t.tolist(),
            }
        )
    data = {"camera_model": "EQUIRECTANGULAR", "frames": frames}
    p = tmp_path / "transforms.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_transform_json_returns_yaw_offsets(tmp_path: Path):
    """transform_json が image_files と並列の frame_yaw_offsets を返す。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    images_dir = input_dir / "images"
    images_dir.mkdir()
    _write_input_transforms(input_dir, num_frames=5)
    # ダミー画像（input_size 推定用）
    cv2.imwrite(str(images_dir / "frame_0000.png"), np.zeros((128, 256, 3), dtype=np.uint8))

    output_dir = tmp_path / "out"
    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]

    image_files, yaw_offsets, _, _ = transform_json(
        input_dir=str(input_dir),
        input_json="transforms.json",
        image_dir=str(input_dir),
        output_dir=str(output_dir),
        views=views,
        fov=90.0,
        output_scale=0.5,
        no_transform=True,
        allow_duplicate=False,
        brush_mode=False,
        yaw_offset_per_frame=30.0,
    )

    assert len(image_files) == 5
    assert len(yaw_offsets) == 5
    assert yaw_offsets == [0.0, 30.0, 60.0, 90.0, 120.0]


def test_transform_json_zero_offset_legacy(tmp_path: Path):
    """yaw_offset_per_frame=0 で frame_yaw_offsets は全て 0（旧動作互換）。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    images_dir = input_dir / "images"
    images_dir.mkdir()
    _write_input_transforms(input_dir, num_frames=3)
    cv2.imwrite(str(images_dir / "frame_0000.png"), np.zeros((128, 256, 3), dtype=np.uint8))

    output_dir = tmp_path / "out"
    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]

    _, yaw_offsets, _, _ = transform_json(
        input_dir=str(input_dir),
        input_json="transforms.json",
        image_dir=str(input_dir),
        output_dir=str(output_dir),
        views=views,
        fov=90.0,
        output_scale=0.5,
        no_transform=True,
        allow_duplicate=False,
        brush_mode=False,
        yaw_offset_per_frame=0.0,
    )
    assert yaw_offsets == [0.0, 0.0, 0.0]


def test_transform_json_extrinsics_reflect_yaw_offset(tmp_path: Path):
    """yaw_offset_per_frame≠0 のとき、出力 transforms.json の transform_matrix は
    yaw 回転が反映された値になる（同じ入力フレームでも frame_index で異なる）。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    images_dir = input_dir / "images"
    images_dir.mkdir()
    _write_input_transforms(input_dir, num_frames=2)
    cv2.imwrite(str(images_dir / "frame_0000.png"), np.zeros((128, 256, 3), dtype=np.uint8))

    output_dir = tmp_path / "out"
    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]

    transform_json(
        input_dir=str(input_dir),
        input_json="transforms.json",
        image_dir=str(input_dir),
        output_dir=str(output_dir),
        views=views,
        fov=90.0,
        output_scale=0.5,
        no_transform=True,
        allow_duplicate=False,
        brush_mode=False,
        yaw_offset_per_frame=30.0,
    )

    out_data = json.loads((output_dir / "transforms.json").read_text(encoding="utf-8"))
    frames = out_data["frames"]
    assert len(frames) == 2  # 1 view × 2 frames

    # frame 0 と frame 1 で transform_matrix が異なる（yaw offset が効いている）
    t0 = np.array(frames[0]["transform_matrix"])
    t1 = np.array(frames[1]["transform_matrix"])
    # 入力 transform は frame 1 で +0.1 in x のみ違うので、yaw 回転が無ければ R 部分は同じはず
    # yaw 回転が反映されると R 部分が変わる
    assert not np.allclose(t0[:3, :3], t1[:3, :3])
