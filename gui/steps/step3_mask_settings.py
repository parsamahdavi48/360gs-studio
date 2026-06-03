"""Step 3 mask control-state helpers.

This module keeps formatting and validation-adjacent state conversion out of
the Qt widget. The widget still owns reading control values; this module owns
the stable contract passed to mask command builders and refresh-plan metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from gui.steps.mask_commands import (
    MASK_APPLY_MODES,
    MASK_APPLY_REPLACE,
    MaskCommandContext,
)


def split_sam_prompt_text(text: str) -> list[str]:
    """Split custom SAM prompt text from the UI into stable prompt tokens."""
    return [part.strip() for part in re.split(r"[,;\n]", text) if part.strip()]


def normalize_mask_merge_mode(mode: object) -> str:
    text = str(mode or MASK_APPLY_REPLACE)
    return text if text in MASK_APPLY_MODES else MASK_APPLY_REPLACE


normalize_sam31_merge_mode = normalize_mask_merge_mode


@dataclass(frozen=True)
class Step3MaskSettingsState:
    projection: str
    projection_mixed: bool
    tasks: tuple[str, ...]
    primary_backend: str
    quality: str
    yolo_level_index: int
    yolo_level_label: str
    yolo_expand: str
    yolo_classes: tuple[int, ...]
    yolo_extra_args: tuple[str, ...]
    semantic_labels: tuple[str, ...]
    sam_prompts: tuple[str, ...]
    sam_subtract_prompts: tuple[str, ...]
    mask_merge_mode: str
    sky_backend: str
    sky_inference_size: str
    sky_min_score: str
    sky_min_area_ratio: str
    sky_top_connected: bool
    stitch_enabled: bool
    stitch_boundary_width: float
    stitch_workers: str
    overexposure_enabled: bool
    overexposure_threshold: str
    overexposure_dilate: str
    custom_enabled: bool
    custom_mask: str
    images_dir: str
    masks_dir: str

    def command_context(
        self,
        *,
        python_executable: str,
        base_dir: Path,
        projection: str | None = None,
    ) -> MaskCommandContext:
        return MaskCommandContext(
            python_executable=python_executable,
            base_dir=base_dir,
            projection=projection or self.projection,
            quality=self.quality,
            yolo_expand=self.yolo_expand,
            sky_inference_size=self.sky_inference_size,
            sky_min_score=self.sky_min_score,
            sky_min_area_ratio=self.sky_min_area_ratio,
            sky_top_connected=self.sky_top_connected,
            stitch_boundary_width=self.stitch_boundary_width,
            stitch_workers=self.stitch_workers,
            overexposure_threshold=self.overexposure_threshold,
            overexposure_dilate=self.overexposure_dilate,
            custom_mask=self.custom_mask,
            yolo_classes=self.yolo_classes,
            yolo_extra_args=self.yolo_extra_args,
            semantic_labels=self.semantic_labels,
            sam_prompts=self.sam_prompts,
            sam_subtract_prompts=self.sam_subtract_prompts,
            mask_merge_mode=self.mask_merge_mode,
        )

    def snapshot(self) -> dict:
        return {
            "projection": "mixed" if self.projection_mixed else self.projection,
            "mask_projection": self.projection,
            "tasks": list(self.tasks),
            "primary_backend": self.primary_backend,
            "quality": self.quality,
            "yolo": {
                "level_index": self.yolo_level_index,
                "level_label": self.yolo_level_label,
                "expand": self.yolo_expand,
                "classes": list(self.yolo_classes),
                "extra_args": list(self.yolo_extra_args),
            },
            "semantic": {
                "labels": list(self.semantic_labels),
            },
            "mask_operation": {
                "merge_mode": self.mask_merge_mode,
            },
            "sam31": {
                "prompts": list(self.sam_prompts),
                "subtract_prompts": list(self.sam_subtract_prompts),
            },
            "sky": {
                "backend": self.sky_backend,
                "inference_size": self.sky_inference_size,
                "min_score": self.sky_min_score,
                "min_area_ratio": self.sky_min_area_ratio,
                "top_connected": self.sky_top_connected,
            },
            "stitch": {
                "enabled": self.stitch_enabled,
                "boundary_width": self.stitch_boundary_width,
                "workers": self.stitch_workers,
            },
            "overexposure": {
                "enabled": self.overexposure_enabled,
                "threshold": self.overexposure_threshold,
                "dilate": self.overexposure_dilate,
            },
            "custom_mask": {
                "enabled": self.custom_enabled,
                "path": self.custom_mask,
            },
            "images_dir": self.images_dir,
            "masks_dir": self.masks_dir,
        }
