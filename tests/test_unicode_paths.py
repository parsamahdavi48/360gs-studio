from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import cubemap_transforms_json as cubemap
import overexposure_mask
import stitch_mask
import yolo_mask
from image_io import imread_unicode, imwrite_unicode
from path_safety import check_path_safety


def test_image_io_roundtrips_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "日本語 シーン" / "画像_日本語.png"
    path.parent.mkdir()
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[..., 1] = 180

    assert imwrite_unicode(path, image)
    loaded = imread_unicode(path, cv2.IMREAD_UNCHANGED)

    assert loaded is not None
    assert np.array_equal(loaded, image)


def test_cubemap_remap_image_accepts_unicode_paths(tmp_path: Path) -> None:
    src = tmp_path / "日本語 シーン" / "images" / "入力_日本語.png"
    out_dir = tmp_path / "日本語 シーン" / "output" / "images"
    src.parent.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[..., 2] = 255
    assert imwrite_unicode(src, image)

    views = [{"name": "front", "yaw": 0.0, "pitch": 0.0}]
    map_x, map_y = cubemap.build_remap((64, 32), 90.0, 0.0, 0.0, 16)
    written = cubemap.remap_image(
        str(src),
        str(out_dir),
        {"front": (map_x, map_y)},
        views,
        mask_from_alpha=False,
        output_mask_dir=str(out_dir),
        invert_masks=False,
    )

    out_path = out_dir / "入力_日本語_front.png"
    assert written == 1
    assert out_path.is_file()
    assert imread_unicode(out_path, cv2.IMREAD_UNCHANGED) is not None


def test_mask_tools_accept_unicode_paths(tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path / "日本語 シーン"
    images = scene / "images"
    masks = scene / "masks"
    output = scene / "output masks"
    images.mkdir(parents=True)
    masks.mkdir()
    output.mkdir()

    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[..., :] = 64
    image_path = images / "画像_日本語.png"
    mask_path = masks / "画像_日本語.png"
    assert imwrite_unicode(image_path, image)
    assert imwrite_unicode(mask_path, np.full((16, 32), 255, dtype=np.uint8))

    overexposure_mask._init_worker(254, 1)
    assert overexposure_mask._process_one((str(image_path), str(output / "白飛び_日本語.png"), None)) is None
    assert imread_unicode(output / "白飛び_日本語.png", cv2.IMREAD_GRAYSCALE) is not None

    stitch_mask.init_worker(np.full((16, 32), 255, dtype=np.uint8))
    assert stitch_mask.process_single_image((str(mask_path), str(output / "縫い目_日本語.png"))) is None
    assert imread_unicode(output / "縫い目_日本語.png", cv2.IMREAD_GRAYSCALE) is not None

    monkeypatch.setattr(yolo_mask, "add_yolo_mask", lambda img, mask, *args, **kwargs: (mask, 0))
    monkeypatch.setattr(yolo_mask, "should_run_bottom_redetection", lambda level, projection: False)
    result = yolo_mask.process_file(str(images), str(output), image_path.name, add_ext=False)
    assert result.output_path.is_file()
    assert imread_unicode(result.output_path, cv2.IMREAD_GRAYSCALE) is not None


def test_path_safety_allows_ascii_spaces_and_flags_non_ascii() -> None:
    assert check_path_safety(r"D:\work folder\scene01") == []

    issues = check_path_safety(r"D:\案件\scene01")
    assert [issue.code for issue in issues] == ["non_ascii"]


def test_path_safety_flags_only_clear_hazards() -> None:
    issues = check_path_safety("C:/" + "a" * 250 + '"')
    codes = {issue.code for issue in issues}
    assert "too_long" in codes
    assert "quote" in codes
