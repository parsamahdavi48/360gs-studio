"""Batch command planning for Step 3 mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MASK_TASK_YOLO = "yolo"
MASK_TASK_STITCH = "stitch"
MASK_TASK_OVEREXPOSURE = "overexposure"
MASK_TASK_CUSTOM = "custom"

MASK_COMMAND_YOLO = "yolo"
MASK_COMMAND_INIT = "init_masks"
MASK_COMMAND_STITCH = "stitch"
MASK_COMMAND_OVEREXPOSURE = "overexposure"
MASK_COMMAND_CUSTOM = "custom"

PROJECTION_EQUIRECT = "equirect"
PROJECTION_NORMAL = "normal"
PROJECTION_ALL = "all"


@dataclass(frozen=True)
class MaskCommandSpec:
    phase: str
    command: str
    image_list: Path | None = None
    projection: str | None = None
    replace: bool = False


def needs_target_manifest(*, source_is_all: bool, target_count: int, all_image_count: int) -> bool:
    return not source_is_all or int(target_count) != int(all_image_count)


def build_uniform_mask_command_specs(
    requested_steps: list[str],
    *,
    target_manifest: Path | None,
) -> list[MaskCommandSpec]:
    specs: list[MaskCommandSpec] = []
    fresh_base_needed = True

    if MASK_TASK_YOLO in requested_steps:
        specs.append(MaskCommandSpec(phase="yolo", command=MASK_COMMAND_YOLO, image_list=target_manifest))
        fresh_base_needed = False

    if MASK_TASK_STITCH in requested_steps:
        if fresh_base_needed:
            specs.append(MaskCommandSpec(phase="init_masks", command=MASK_COMMAND_INIT, image_list=target_manifest))
            fresh_base_needed = False
        specs.append(MaskCommandSpec(phase="stitch", command=MASK_COMMAND_STITCH, image_list=target_manifest))

    if MASK_TASK_OVEREXPOSURE in requested_steps:
        specs.append(
            MaskCommandSpec(
                phase="overexposure",
                command=MASK_COMMAND_OVEREXPOSURE,
                image_list=target_manifest,
                replace=fresh_base_needed,
            )
        )
        fresh_base_needed = False

    if MASK_TASK_CUSTOM in requested_steps:
        specs.append(
            MaskCommandSpec(
                phase="custom",
                command=MASK_COMMAND_CUSTOM,
                image_list=target_manifest,
                replace=fresh_base_needed,
            )
        )

    return specs


def build_mixed_mask_command_specs(
    requested_steps: list[str],
    *,
    manifests: dict[str, Path],
) -> list[MaskCommandSpec]:
    specs: list[MaskCommandSpec] = []
    equirect_manifest = manifests.get(PROJECTION_EQUIRECT)
    normal_manifest = manifests.get(PROJECTION_NORMAL)
    all_manifest = manifests.get(PROJECTION_ALL)

    if MASK_TASK_YOLO in requested_steps:
        if equirect_manifest is not None:
            specs.append(
                MaskCommandSpec(
                    phase="yolo_equirect",
                    command=MASK_COMMAND_YOLO,
                    projection=PROJECTION_EQUIRECT,
                    image_list=equirect_manifest,
                )
            )
        if normal_manifest is not None:
            specs.append(
                MaskCommandSpec(
                    phase="yolo_normal",
                    command=MASK_COMMAND_YOLO,
                    projection=PROJECTION_NORMAL,
                    image_list=normal_manifest,
                )
            )

    if MASK_TASK_STITCH in requested_steps and equirect_manifest is not None:
        specs.append(MaskCommandSpec(phase="stitch_equirect", command=MASK_COMMAND_STITCH, image_list=equirect_manifest))

    if MASK_TASK_OVEREXPOSURE in requested_steps and all_manifest is not None:
        specs.append(MaskCommandSpec(phase="overexposure", command=MASK_COMMAND_OVEREXPOSURE, image_list=all_manifest))

    if MASK_TASK_CUSTOM in requested_steps and all_manifest is not None:
        specs.append(MaskCommandSpec(phase="custom", command=MASK_COMMAND_CUSTOM, image_list=all_manifest))

    return specs
