from __future__ import annotations

from pathlib import Path

from gui.steps.mask_commands import MASK_APPLY_ADD, MASK_APPLY_SUBTRACT
from gui.steps.step3_mask_plan import (
    MASK_TASK_CUSTOM,
    MASK_TASK_OVEREXPOSURE,
    MASK_TASK_STITCH,
    MASK_TASK_YOLO,
    PROJECTION_EQUIRECT,
    PROJECTION_NORMAL,
    build_mixed_mask_command_specs,
    build_uniform_mask_command_specs,
    needs_target_manifest,
)


def test_uniform_mask_plan_initializes_base_when_running_without_yolo(tmp_path: Path) -> None:
    manifest = tmp_path / "targets.jsonl"

    specs = build_uniform_mask_command_specs([MASK_TASK_STITCH, MASK_TASK_OVEREXPOSURE], target_manifest=manifest)

    assert [(spec.phase, spec.command, spec.image_list, spec.replace) for spec in specs] == [
        ("init_masks", "init_masks", manifest, False),
        ("stitch", "stitch", manifest, False),
        ("overexposure", "overexposure", manifest, False),
    ]


def test_uniform_mask_plan_replaces_when_custom_is_first() -> None:
    specs = build_uniform_mask_command_specs([MASK_TASK_CUSTOM], target_manifest=None)

    assert len(specs) == 1
    assert specs[0].phase == "custom"
    assert specs[0].replace is True


def test_uniform_mask_plan_uses_selected_merge_mode_for_primary_and_followups(tmp_path: Path) -> None:
    manifest = tmp_path / "targets.jsonl"

    specs = build_uniform_mask_command_specs(
        [MASK_TASK_YOLO, MASK_TASK_OVEREXPOSURE, MASK_TASK_CUSTOM],
        target_manifest=manifest,
        merge_mode=MASK_APPLY_SUBTRACT,
    )

    assert [(spec.phase, spec.replace, spec.merge_mode) for spec in specs] == [
        ("yolo", False, MASK_APPLY_SUBTRACT),
        ("overexposure", False, MASK_APPLY_SUBTRACT),
        ("custom", False, MASK_APPLY_SUBTRACT),
    ]


def test_mixed_mask_plan_splits_primary_by_projection(tmp_path: Path) -> None:
    manifests = {
        PROJECTION_EQUIRECT: tmp_path / "equirect.jsonl",
        PROJECTION_NORMAL: tmp_path / "normal.jsonl",
        "all": tmp_path / "all.jsonl",
    }

    specs = build_mixed_mask_command_specs(
        [MASK_TASK_YOLO, MASK_TASK_STITCH, MASK_TASK_OVEREXPOSURE, MASK_TASK_CUSTOM],
        manifests=manifests,
    )

    assert [(spec.phase, spec.projection, spec.image_list) for spec in specs] == [
        ("yolo_equirect", PROJECTION_EQUIRECT, manifests[PROJECTION_EQUIRECT]),
        ("yolo_normal", PROJECTION_NORMAL, manifests[PROJECTION_NORMAL]),
        ("stitch_equirect", None, manifests[PROJECTION_EQUIRECT]),
        ("overexposure", None, manifests["all"]),
        ("custom", None, manifests["all"]),
    ]
    assert specs[2].merge_mode is None
    assert specs[3].merge_mode == MASK_APPLY_ADD
    assert specs[4].merge_mode == MASK_APPLY_ADD


def test_target_manifest_is_needed_for_source_scope_or_partial_refresh() -> None:
    assert needs_target_manifest(source_is_all=True, target_count=3, all_image_count=3) is False
    assert needs_target_manifest(source_is_all=False, target_count=3, all_image_count=3) is True
    assert needs_target_manifest(source_is_all=True, target_count=2, all_image_count=3) is True
