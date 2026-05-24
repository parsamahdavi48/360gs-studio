from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import core.yolo_mask as yolo_mask


def test_build_runtime_settings_normalizes_cli_values() -> None:
    args = yolo_mask.build_arg_parser().parse_args(
        [
            "images",
            "masks",
            "--quality",
            "high",
            "--projection",
            "normal",
            "--expand",
            "999",
            "--classes",
            "2,1,2",
            "--bottom-conf",
            "2",
            "--bottom-tta-rotations",
            "4",
            "--bottom-model",
            "x",
            "--bottom-filter",
            "--profile-json",
            "profile.json",
        ]
    )

    runtime = yolo_mask.build_runtime_settings(args)

    assert runtime.expand_was_clamped is True
    assert runtime.settings.class_ids == (1, 2)
    assert runtime.settings.expand == yolo_mask.EXPAND_MAX
    assert runtime.settings.bottom_conf == 1.0
    assert runtime.settings.bottom_tta_rotations == 4
    assert runtime.settings.bottom_model == "x"
    assert runtime.settings.bottom_filter is True
    assert runtime.settings.profile_json == "profile.json"


def test_apply_runtime_settings_updates_global_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    args = yolo_mask.build_arg_parser().parse_args(
        [
            "images",
            "masks",
            "--quality",
            "standard",
            "--projection",
            "normal",
            "--classes",
            "5",
        ]
    )
    runtime = yolo_mask.build_runtime_settings(args)
    monkeypatch.setattr(yolo_mask, "proc_count", 12)
    monkeypatch.setattr(yolo_mask, "PROFILE", object())

    yolo_mask.apply_runtime_settings(runtime.settings)

    assert yolo_mask.CLASS_IDS == [5]
    assert yolo_mask.LEVEL == runtime.settings.level
    assert yolo_mask.QUALITY == "standard"
    assert yolo_mask.PROJECTION == "normal"
    assert yolo_mask.EXPAND == runtime.settings.expand
    assert yolo_mask.BOTTOM_CONF == runtime.settings.bottom_conf
    assert yolo_mask.BOTTOM_TTA_ROTATIONS == runtime.settings.bottom_tta_rotations
    assert yolo_mask.BOTTOM_MODEL == runtime.settings.bottom_model
    assert yolo_mask.BOTTOM_FILTER == runtime.settings.bottom_filter
    assert yolo_mask.proc_count == 0
    assert yolo_mask.PROFILE is None


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
