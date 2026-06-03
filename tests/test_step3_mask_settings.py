from __future__ import annotations

import sys
from pathlib import Path

from gui.steps.mask_commands import SAM31_MERGE_REPLACE
from gui.steps.step3_mask_settings import (
    Step3MaskSettingsState,
    normalize_mask_merge_mode,
    normalize_sam31_merge_mode,
    split_sam_prompt_text,
)


def _settings_state() -> Step3MaskSettingsState:
    return Step3MaskSettingsState(
        projection="equirect",
        projection_mixed=True,
        tasks=("yolo", "stitch"),
        primary_backend="sam31",
        quality="high",
        yolo_level_index=1,
        yolo_level_label="High",
        yolo_expand="4",
        yolo_classes=(0, 2),
        yolo_extra_args=("--bottom-enhance",),
        semantic_labels=("person", "sky"),
        sam_prompts=("person", "tripod"),
        sam_subtract_prompts=("male icon",),
        mask_merge_mode="add",
        sky_backend="sam31",
        sky_inference_size="1008",
        sky_min_score="0.5",
        sky_min_area_ratio="0.01",
        sky_top_connected=True,
        stitch_enabled=True,
        stitch_boundary_width=5.0,
        stitch_workers="8",
        overexposure_enabled=True,
        overexposure_threshold="254",
        overexposure_dilate="1",
        custom_enabled=True,
        custom_mask="D:/masks/custom.png",
        images_dir="D:/scene/images",
        masks_dir="D:/scene/masks",
    )


def test_split_sam_prompt_text_accepts_common_separators() -> None:
    assert split_sam_prompt_text(" tripod, hand; selfie stick\ncamera ") == [
        "tripod",
        "hand",
        "selfie stick",
        "camera",
    ]


def test_normalize_sam31_merge_mode_rejects_unknown_values() -> None:
    assert normalize_mask_merge_mode("subtract") == "subtract"
    assert normalize_sam31_merge_mode("subtract") == "subtract"
    assert normalize_sam31_merge_mode("unexpected") == SAM31_MERGE_REPLACE
    assert normalize_sam31_merge_mode(None) == SAM31_MERGE_REPLACE


def test_step3_mask_settings_state_builds_command_context_with_projection_override(tmp_path: Path) -> None:
    state = _settings_state()

    context = state.command_context(
        python_executable=sys.executable,
        base_dir=tmp_path,
        projection="normal",
    )

    assert context.python_executable == sys.executable
    assert context.base_dir == tmp_path
    assert context.projection == "normal"
    assert context.quality == "high"
    assert context.yolo_classes == (0, 2)
    assert context.semantic_labels == ("person", "sky")
    assert context.sam_prompts == ("person", "tripod")
    assert context.mask_merge_mode == "add"
    assert context.sky_top_connected is True
    assert context.stitch_boundary_width == 5.0
    assert context.custom_mask == "D:/masks/custom.png"


def test_step3_mask_settings_state_snapshot_preserves_refresh_contract() -> None:
    snapshot = _settings_state().snapshot()

    assert snapshot["projection"] == "mixed"
    assert snapshot["mask_projection"] == "equirect"
    assert snapshot["tasks"] == ["yolo", "stitch"]
    assert snapshot["primary_backend"] == "sam31"
    assert snapshot["yolo"] == {
        "level_index": 1,
        "level_label": "High",
        "expand": "4",
        "classes": [0, 2],
        "extra_args": ["--bottom-enhance"],
    }
    assert snapshot["semantic"]["labels"] == ["person", "sky"]
    assert snapshot["mask_operation"]["merge_mode"] == "add"
    assert "merge_mode" not in snapshot["sam31"]
    assert snapshot["sky"]["top_connected"] is True
    assert snapshot["stitch"]["enabled"] is True
    assert snapshot["overexposure"]["threshold"] == "254"
    assert snapshot["custom_mask"]["path"] == "D:/masks/custom.png"
    assert snapshot["images_dir"] == "D:/scene/images"
