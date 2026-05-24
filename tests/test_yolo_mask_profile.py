from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import core.yolo_mask as yolo_mask


def test_profile_json_records_timing_without_changing_normal_masks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "images"
    output = tmp_path / "masks"
    profile_path = tmp_path / "profile.json"
    source.mkdir()
    image_path = source / "frame_0001.jpg"
    cv2.imwrite(str(image_path), np.full((8, 16, 3), 128, dtype=np.uint8))

    monkeypatch.setattr(yolo_mask, "load_models", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(yolo_mask, "add_yolo_mask", lambda img, mask, *args, **kwargs: (mask, 0))

    exit_code = yolo_mask.main(
        [
            str(source),
            str(output),
            "--projection",
            "normal",
            "--profile-json",
            str(profile_path),
        ]
    )

    assert exit_code == 0
    assert yolo_mask.PROFILE is None
    written = cv2.imread(str(output / "frame_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert np.all(written == 255)

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["schema_version"] == 1
    assert profile["settings"]["projection"] == "normal"
    assert profile["totals"]["images"] == 1
    assert profile["images"][0]["file"] == "frame_0001.jpg"
    assert profile["images"][0]["shape"] == {"height": 8, "width": 16}
    assert profile["images"][0]["timings_sec"]["image.total"] >= profile["images"][0]["timings_sec"]["image.read"]
