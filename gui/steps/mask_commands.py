"""Command builders for Step 3 mask generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.mask_job_spec import (
    BACKEND_SAM31,
    BACKEND_YOLO26_SEM,
    custom_mask_job,
    init_masks_job,
    mask_job_to_command,
    overexposure_mask_job,
    sky_mask_job,
    stitch_mask_job,
    yolo_sam_mask_job,
)
from core.mask_merge import MASK_MERGE_ADD, MASK_MERGE_REPLACE, MASK_MERGE_SUBTRACT, SUPPORTED_MERGE_MODES

PERSON_BACKEND_YOLO_SAM = "yolo_sam"
PERSON_BACKEND_YOLO26_SEM = "yolo26_sem"
PERSON_BACKEND_SAM31 = "sam31"
MASK_APPLY_REPLACE = MASK_MERGE_REPLACE
MASK_APPLY_ADD = MASK_MERGE_ADD
MASK_APPLY_SUBTRACT = MASK_MERGE_SUBTRACT
MASK_APPLY_MODES = SUPPORTED_MERGE_MODES
SAM31_MERGE_REPLACE = MASK_APPLY_REPLACE
SAM31_MERGE_ADD = MASK_APPLY_ADD
SAM31_MERGE_SUBTRACT = MASK_APPLY_SUBTRACT
SAM31_MERGE_MODES = MASK_APPLY_MODES
PERSON_SAM31_INFERENCE_SIZE = "1008"
PERSON_SAM31_MIN_SCORE = "0.5"


@dataclass(frozen=True)
class MaskCommandContext:
    python_executable: str
    base_dir: Path
    projection: str
    quality: str
    yolo_expand: str
    sky_inference_size: str
    sky_min_score: str
    sky_min_area_ratio: str
    sky_top_connected: bool
    stitch_boundary_width: float
    stitch_workers: str
    overexposure_threshold: str
    overexposure_dilate: str
    custom_mask: str = ""
    yolo_classes: tuple[int, ...] = ()
    yolo_extra_args: tuple[str, ...] = ()
    semantic_labels: tuple[str, ...] = ()
    sam_prompts: tuple[str, ...] = ()
    sam_subtract_prompts: tuple[str, ...] = ()
    mask_merge_mode: str = MASK_APPLY_REPLACE


def _require_images_masks(images: str | Path, masks: str | Path) -> tuple[str, str]:
    images_text = str(images)
    masks_text = str(masks)
    if not images_text:
        raise ValueError("画像フォルダが指定されていません")
    if not masks_text:
        raise ValueError("マスクフォルダが指定されていません")
    return images_text, masks_text


def build_primary_mask_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    backend: str,
    image_list: str | Path | None = None,
) -> list[str]:
    if backend == PERSON_BACKEND_YOLO26_SEM:
        return build_yolo26_sem_cmd(context, images, masks, image_list=image_list)
    if backend == PERSON_BACKEND_SAM31:
        return build_sam31_prompt_cmd(
            context,
            images,
            masks,
            prompts=list(context.sam_prompts),
            subtract_prompts=list(context.sam_subtract_prompts),
            merge_mode=context.mask_merge_mode,
            image_list=image_list,
        )
    return build_yolo_sam_cmd(context, images, masks, image_list=image_list)


def build_yolo_sam_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    effective_merge_mode = _effective_merge_mode(context)
    return mask_job_to_command(
        context.python_executable,
        yolo_sam_mask_job(
            images=images_text,
            masks=masks_text,
            quality=context.quality,
            expand=int(context.yolo_expand),
            projection=context.projection,
            classes=context.yolo_classes or (0,),
            extra_args=context.yolo_extra_args,
            merge_mode=effective_merge_mode,
            image_list=image_list,
        ),
    )


def build_sam31_prompt_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    prompts: list[str],
    subtract_prompts: list[str] | None = None,
    merge_mode: str | None = None,
    replace: bool = False,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    effective_merge_mode = _effective_merge_mode(context, merge_mode=merge_mode, replace=replace)

    safe_batch = False
    try:
        if Path(images_text).is_dir():
            safe_batch = True
    except OSError:
        pass
    return mask_job_to_command(
        context.python_executable,
        sky_mask_job(
            images=images_text,
            masks=masks_text,
            backend=BACKEND_SAM31,
            projection=context.projection,
            quality=context.quality,
            inference_size=int(PERSON_SAM31_INFERENCE_SIZE),
            expand=int(context.yolo_expand),
            min_score=float(PERSON_SAM31_MIN_SCORE),
            min_area_ratio=float(context.sky_min_area_ratio),
            top_connected=context.sky_top_connected,
            sam_prompts=prompts,
            sam_subtract_prompts=subtract_prompts or (),
            merge_mode=effective_merge_mode,
            replace=effective_merge_mode == MASK_APPLY_REPLACE,
            safe_batch=safe_batch,
            image_list=image_list,
        ),
    )


def build_yolo26_sem_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    replace: bool = False,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    effective_merge_mode = _effective_merge_mode(context, replace=replace)
    return mask_job_to_command(
        context.python_executable,
        sky_mask_job(
            images=images_text,
            masks=masks_text,
            backend=BACKEND_YOLO26_SEM,
            projection=context.projection,
            quality=context.quality,
            inference_size=int(context.sky_inference_size),
            expand=int(context.yolo_expand),
            min_score=float(context.sky_min_score),
            min_area_ratio=float(context.sky_min_area_ratio),
            top_connected=context.sky_top_connected,
            labels=context.semantic_labels,
            merge_mode=effective_merge_mode,
            replace=effective_merge_mode == MASK_APPLY_REPLACE,
            image_list=image_list,
        ),
    )


def build_init_masks_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    return mask_job_to_command(
        context.python_executable,
        init_masks_job(images=images_text, masks=masks_text, image_list=image_list),
    )


def build_stitch_cmd(
    context: MaskCommandContext,
    masks: str | Path,
    *,
    image_list: str | Path | None = None,
) -> list[str]:
    masks_text = str(masks)
    if not masks_text:
        raise ValueError("マスクフォルダが指定されていません")
    return mask_job_to_command(
        context.python_executable,
        stitch_mask_job(
            masks=masks_text,
            boundary_width=context.stitch_boundary_width,
            workers=int(context.stitch_workers),
            image_list=image_list,
        ),
    )


def build_overexposure_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    replace: bool = False,
    merge_mode: str | None = None,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    effective_merge_mode = _effective_merge_mode(context, merge_mode=merge_mode, replace=replace)
    return mask_job_to_command(
        context.python_executable,
        overexposure_mask_job(
            images=images_text,
            masks=masks_text,
            threshold=int(context.overexposure_threshold),
            dilate=int(context.overexposure_dilate),
            workers=int(context.stitch_workers),
            merge_mode=effective_merge_mode,
            replace=effective_merge_mode == MASK_APPLY_REPLACE,
            image_list=image_list,
        ),
    )


def build_custom_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    replace: bool = False,
    merge_mode: str | None = None,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    if not context.custom_mask:
        raise ValueError("CUSTOM_MASK_REQUIRED")
    effective_merge_mode = _effective_merge_mode(context, merge_mode=merge_mode, replace=replace)
    return mask_job_to_command(
        context.python_executable,
        custom_mask_job(
            images=images_text,
            masks=masks_text,
            custom_mask=context.custom_mask,
            merge_mode=effective_merge_mode,
            replace=effective_merge_mode == MASK_APPLY_REPLACE,
            image_list=image_list,
        ),
    )


def _effective_merge_mode(
    context: MaskCommandContext,
    *,
    merge_mode: str | None = None,
    replace: bool = False,
) -> str:
    if replace:
        return MASK_APPLY_REPLACE
    candidate = merge_mode or context.mask_merge_mode
    return candidate if candidate in MASK_APPLY_MODES else MASK_APPLY_REPLACE
