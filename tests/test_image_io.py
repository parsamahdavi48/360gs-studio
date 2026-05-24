"""Item 2: ビット深度 / フォーマット対応のテスト。

8/16-bit、RGB/RGBA/grayscale、jpg/png/tiff/webp のラウンドトリップを検証する。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.cubemap_image_conversion import (
    proc_convert_images,
    remap_image,
    worker_init,
)
from core.cubemap_image_io import (
    load_equirect,
    remap_with_channels,
    resolve_output_ext,
    save_image,
)
from core.cubemap_remap import build_remap

# =============================================================================
# resolve_output_ext
# =============================================================================


def test_resolve_output_ext_auto_preserves_input():
    assert resolve_output_ext(".png", "auto") == ".png"
    assert resolve_output_ext(".jpg", "auto") == ".jpg"
    assert resolve_output_ext(".jpeg", "auto") == ".jpg"
    assert resolve_output_ext(".tiff", "auto") == ".tiff"
    assert resolve_output_ext(".webp", "auto") == ".webp"


def test_resolve_output_ext_explicit():
    assert resolve_output_ext(".jpg", "png") == ".png"
    assert resolve_output_ext(".png", "tiff") == ".tiff"
    assert resolve_output_ext(".jpg", "webp") == ".webp"


def test_resolve_output_ext_unknown_input_falls_back_to_jpg():
    assert resolve_output_ext(".gif", "auto") == ".jpg"
    assert resolve_output_ext("", "auto") == ".jpg"


def test_resolve_output_ext_invalid_format_raises():
    with pytest.raises(ValueError):
        resolve_output_ext(".jpg", "exr")


# =============================================================================
# save_image / load_equirect roundtrip
# =============================================================================


def test_save_load_roundtrip_8bit_rgb_png(tmp_path: Path):
    arr = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
    p = tmp_path / "test.png"
    save_image(arr, str(p))
    loaded = load_equirect(str(p))
    assert loaded.shape == arr.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(arr, loaded)


def test_save_load_roundtrip_16bit_png(tmp_path: Path):
    arr = np.random.randint(0, 65536, (64, 128, 3), dtype=np.uint16)
    p = tmp_path / "test.png"
    save_image(arr, str(p))
    loaded = load_equirect(str(p))
    assert loaded.dtype == np.uint16
    assert loaded.shape == arr.shape
    assert np.array_equal(arr, loaded)


def test_save_load_roundtrip_16bit_tiff(tmp_path: Path):
    arr = np.random.randint(0, 65536, (32, 64, 3), dtype=np.uint16)
    p = tmp_path / "test.tiff"
    save_image(arr, str(p))
    loaded = load_equirect(str(p))
    assert loaded.dtype == np.uint16
    assert np.array_equal(arr, loaded)


def test_save_load_roundtrip_rgba_png(tmp_path: Path):
    arr = np.zeros((32, 64, 4), dtype=np.uint8)
    arr[..., 0] = 100  # B
    arr[..., 1] = 150  # G
    arr[..., 2] = 200  # R
    arr[..., 3] = 128  # A
    p = tmp_path / "test.png"
    save_image(arr, str(p))
    loaded = load_equirect(str(p))
    assert loaded.shape == (32, 64, 4)
    assert np.array_equal(arr, loaded)


def test_save_jpg_strips_alpha(tmp_path: Path):
    """JPG 出力は α 非対応 → α が落ちる挙動。"""
    arr = np.zeros((32, 64, 4), dtype=np.uint8)
    arr[..., :3] = 128
    arr[..., 3] = 0  # 完全透明
    p = tmp_path / "test.jpg"
    save_image(arr, str(p))
    loaded = load_equirect(str(p))
    # JPG なので 3 チャネルになっているはず
    assert loaded.ndim == 3
    assert loaded.shape[2] == 3


def test_save_jpg_downconverts_16bit(tmp_path: Path):
    """JPG は 16-bit 非対応 → 8-bit にダウンコンバート。"""
    arr = np.full((32, 64, 3), 32768, dtype=np.uint16)  # 中間値
    p = tmp_path / "test.jpg"
    save_image(arr, str(p))
    loaded = load_equirect(str(p))
    assert loaded.dtype == np.uint8
    # 16-bit 32768 → 8-bit 128 になる（256 で除算）
    assert abs(int(loaded[0, 0, 0]) - 128) <= 2  # JPG 圧縮による若干の誤差を許容


# =============================================================================
# remap_with_channels
# =============================================================================


def test_remap_preserves_dtype_uint8():
    arr = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
    map_x, map_y = build_remap((128, 64), 90.0, 0.0, 0.0, 32)
    out = remap_with_channels(arr, map_x, map_y)
    assert out.dtype == np.uint8


def test_remap_preserves_dtype_uint16():
    arr = np.random.randint(0, 65536, (64, 128, 3), dtype=np.uint16)
    map_x, map_y = build_remap((128, 64), 90.0, 0.0, 0.0, 32)
    out = remap_with_channels(arr, map_x, map_y)
    assert out.dtype == np.uint16


def test_remap_preserves_alpha_channel():
    arr = np.zeros((64, 128, 4), dtype=np.uint8)
    arr[..., :3] = 200
    arr[..., 3] = 100
    map_x, map_y = build_remap((128, 64), 90.0, 0.0, 0.0, 32)
    out = remap_with_channels(arr, map_x, map_y)
    assert out.shape[2] == 4
    # α チャネルが remap 後も存在し、想定値に近い
    assert out[..., 3].mean() == pytest.approx(100, abs=5)


def test_remap_grayscale():
    arr = np.full((64, 128), 200, dtype=np.uint8)
    map_x, map_y = build_remap((128, 64), 90.0, 0.0, 0.0, 32)
    out = remap_with_channels(arr, map_x, map_y)
    assert out.ndim == 2  # グレースケールは 2D のまま
    assert out.dtype == np.uint8


# =============================================================================
# end-to-end remap_image
# =============================================================================


def _make_test_equirect(path: Path, w: int = 256, h: int = 128, dtype=np.uint8, channels: int = 3):
    """テスト用 equirect 画像を生成（経度方向にグラデーション）。"""
    if channels == 1:
        arr = np.tile(np.linspace(0, 255 if dtype == np.uint8 else 65535, w, dtype=dtype), (h, 1))
    else:
        max_val = 255 if dtype == np.uint8 else 65535
        x_grad = np.linspace(0, max_val, w, dtype=dtype)
        arr = np.zeros((h, w, channels), dtype=dtype)
        for c in range(min(3, channels)):
            arr[..., c] = x_grad
        if channels == 4:
            arr[..., 3] = max_val // 2  # 中間 α
    cv2.imwrite(str(path), arr)


def test_end_to_end_8bit_rgb_png(tmp_path: Path):
    src = tmp_path / "input.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_test_equirect(src, dtype=np.uint8, channels=3)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(str(src), str(out_dir), tables, views, False, str(out_dir), False)

    out_files = list(out_dir.glob("*.png"))
    assert len(out_files) == 1
    loaded = cv2.imread(str(out_files[0]), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint8
    assert loaded.shape == (64, 64, 3)


def test_end_to_end_16bit_defaults_to_8bit_output(tmp_path: Path):
    src = tmp_path / "input.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_test_equirect(src, dtype=np.uint16, channels=3)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(
        str(src), str(out_dir), tables, views, False, str(out_dir), False, output_format="auto"
    )

    out_files = list(out_dir.glob("*.png"))
    assert len(out_files) == 1
    loaded = cv2.imread(str(out_files[0]), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint8, f"Expected uint8, got {loaded.dtype}"


def test_end_to_end_16bit_can_preserve_source_depth(tmp_path: Path):
    src = tmp_path / "input.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_test_equirect(src, dtype=np.uint16, channels=3)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(
        str(src),
        str(out_dir),
        tables,
        views,
        False,
        str(out_dir),
        False,
        output_format="auto",
        output_bit_depth="source",
    )

    out_files = list(out_dir.glob("*.png"))
    assert len(out_files) == 1
    loaded = cv2.imread(str(out_files[0]), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint16, f"Expected uint16, got {loaded.dtype}"


def test_end_to_end_rgba_preserved(tmp_path: Path):
    src = tmp_path / "input.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_test_equirect(src, dtype=np.uint8, channels=4)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(str(src), str(out_dir), tables, views, False, str(out_dir), False)

    out_files = list(out_dir.glob("*.png"))
    assert len(out_files) == 1
    loaded = cv2.imread(str(out_files[0]), cv2.IMREAD_UNCHANGED)
    assert loaded.shape[2] == 4, f"Expected 4 channels (RGBA), got shape {loaded.shape}"


def test_end_to_end_format_override(tmp_path: Path):
    """--output-format で入力 PNG → 出力 JPG を強制できる。"""
    src = tmp_path / "input.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_test_equirect(src, dtype=np.uint8, channels=3)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(
        str(src), str(out_dir), tables, views, False, str(out_dir), False, output_format="jpg"
    )

    jpg_files = list(out_dir.glob("*.jpg"))
    assert len(jpg_files) == 1
    png_files = list(out_dir.glob("*.png"))
    assert len(png_files) == 0


def test_end_to_end_mask_from_alpha(tmp_path: Path):
    """RGBA 入力 + --mask_from_alpha で RGB 画像と L マスクが分離して出力される。"""
    src = tmp_path / "input.png"
    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    _make_test_equirect(src, dtype=np.uint8, channels=4)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(
        str(src), str(img_dir), tables, views, mask_from_alpha=True, output_mask_dir=str(mask_dir),
        invert_masks=False,
    )

    img_files = list(img_dir.glob("*.png"))
    mask_files = list(mask_dir.glob("*.png"))
    assert len(img_files) == 1
    assert len(mask_files) == 1

    # RGB 画像は α が落ちている
    img = cv2.imread(str(img_files[0]), cv2.IMREAD_UNCHANGED)
    assert img.shape[2] == 3

    # マスクは L (1ch) で 2 値化されている
    mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
    assert mask.ndim == 2
    unique_vals = np.unique(mask)
    assert set(unique_vals.tolist()).issubset({0, 255})


def test_mask_from_16bit_alpha_writes_8bit_png(tmp_path: Path):
    src = tmp_path / "input.png"
    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    _make_test_equirect(src, dtype=np.uint16, channels=4)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(
        str(src),
        str(img_dir),
        tables,
        views,
        mask_from_alpha=True,
        output_mask_dir=str(mask_dir),
        invert_masks=False,
        output_bit_depth="source",
    )

    mask_files = list(mask_dir.glob("*.png"))
    assert len(mask_files) == 1
    mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
    assert mask.dtype == np.uint8
    assert mask.ndim == 2


def test_grayscale_mask_output_keeps_black_as_excluded_by_default(tmp_path: Path):
    src = tmp_path / "mask.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cv2.imwrite(str(src), np.zeros((128, 256), dtype=np.uint8))

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(str(src), str(out_dir), tables, views, False, str(out_dir), False)

    out_files = list(out_dir.glob("*.png"))
    assert len(out_files) == 1
    mask = cv2.imread(str(out_files[0]), cv2.IMREAD_UNCHANGED)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) == {0}


def test_grayscale_mask_output_can_invert_for_postshot_occluders(tmp_path: Path):
    src = tmp_path / "mask.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cv2.imwrite(str(src), np.zeros((128, 256), dtype=np.uint8))

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = build_remap((256, 128), 90.0, 0.0, 0.0, 64)
    tables = {"front": (map_x, map_y)}

    remap_image(str(src), str(out_dir), tables, views, False, str(out_dir), True)

    out_files = list(out_dir.glob("*.png"))
    assert len(out_files) == 1
    mask = cv2.imread(str(out_files[0]), cv2.IMREAD_UNCHANGED)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) == {255}


def _gradient_equirect(width: int, height: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    red = np.rint(xs * 255.0 / max(1, width - 1)).astype(np.uint8)
    green = np.rint(ys * 255.0 / max(1, height - 1)).astype(np.uint8)
    blue = np.rint((xs + ys) * 255.0 / max(1, width + height - 2)).astype(np.uint8)
    return np.dstack([blue, green, red])


def test_worker_remap_uses_each_source_image_size(tmp_path: Path):
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    out_dir.mkdir()
    mask_dir.mkdir()

    large = _gradient_equirect(12, 6)
    small = _gradient_equirect(8, 4)
    save_image(large, str(image_dir / "large.png"))
    save_image(small, str(image_dir / "small.png"))

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    worker_init(
        (12, 6),
        90.0,
        4,
        views,
        str(image_dir),
        str(mask_dir),
        str(out_dir),
        str(mask_dir),
        False,
        False,
        "png",
        "8",
        95,
        True,
        False,
        4,
    )

    proc_convert_images("large.png", 0.0)
    proc_convert_images("small.png", 0.0)

    actual = load_equirect(str(out_dir / "small_front.png"))
    map_x, map_y = build_remap((8, 4), 90.0, 0.0, 0.0, 4)
    expected = remap_with_channels(small, map_x, map_y)
    assert np.array_equal(actual, expected)

    wrong_map_x, wrong_map_y = build_remap((12, 6), 90.0, 0.0, 0.0, 4)
    wrong = remap_with_channels(small, wrong_map_x, wrong_map_y)
    assert not np.array_equal(actual, wrong)
