"""Command builders for Step 3 mask generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PERSON_BACKEND_YOLO_SAM = "yolo_sam"
PERSON_BACKEND_MASK2FORMER = "mask2former"
PERSON_BACKEND_SAM31 = "sam31"
SAM31_MERGE_REPLACE = "replace"
SAM31_MERGE_ADD = "add"
SAM31_MERGE_SUBTRACT = "subtract"
SAM31_MERGE_MODES = (SAM31_MERGE_REPLACE, SAM31_MERGE_ADD, SAM31_MERGE_SUBTRACT)
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
    ade_labels: tuple[str, ...] = ()
    sam_prompts: tuple[str, ...] = ()
    sam_subtract_prompts: tuple[str, ...] = ()
    sam31_merge_mode: str = SAM31_MERGE_REPLACE


def _module_cmd(context: MaskCommandContext, module_name: str) -> list[str]:
    _ = context.base_dir
    return [context.python_executable, "-u", "-m", module_name]


def _require_images_masks(images: str | Path, masks: str | Path) -> tuple[str, str]:
    images_text = str(images)
    masks_text = str(masks)
    if not images_text:
        raise ValueError("画像フォルダが指定されていません")
    if not masks_text:
        raise ValueError("マスクフォルダが指定されていません")
    return images_text, masks_text


def _sky_postprocess_args(context: MaskCommandContext) -> list[str]:
    args = ["--min-area-ratio", context.sky_min_area_ratio]
    if context.sky_top_connected:
        args.append("--top-connected")
    return args


def build_primary_mask_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    backend: str,
    image_list: str | Path | None = None,
) -> list[str]:
    if backend == PERSON_BACKEND_MASK2FORMER:
        return build_mask2former_cmd(context, images, masks, replace=True, image_list=image_list)
    if backend == PERSON_BACKEND_SAM31:
        return build_sam31_prompt_cmd(
            context,
            images,
            masks,
            prompts=list(context.sam_prompts),
            subtract_prompts=list(context.sam_subtract_prompts),
            merge_mode=context.sam31_merge_mode,
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
    cmd = [
        *_module_cmd(context, "core.yolo_mask"),
        images_text,
        masks_text,
        "--quality",
        context.quality,
        "--expand",
        context.yolo_expand,
        "--projection",
        context.projection,
    ]
    if context.yolo_classes:
        cmd.extend(["--classes", ",".join(str(c) for c in context.yolo_classes)])
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    cmd.extend(context.yolo_extra_args)
    return cmd


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
    effective_merge_mode = SAM31_MERGE_REPLACE if replace else (merge_mode or context.sam31_merge_mode)
    if effective_merge_mode not in SAM31_MERGE_MODES:
        effective_merge_mode = SAM31_MERGE_REPLACE

    cmd = [
        *_module_cmd(context, "core.sky_mask"),
        images_text,
        masks_text,
        "--backend",
        "sam31",
        "--projection",
        context.projection,
        "--quality",
        context.quality,
        "--inference-size",
        PERSON_SAM31_INFERENCE_SIZE,
        "--expand",
        context.yolo_expand,
        "--min-score",
        PERSON_SAM31_MIN_SCORE,
        "--merge-mode",
        effective_merge_mode,
    ]
    cmd.extend(_sky_postprocess_args(context))
    for prompt in prompts:
        cmd.extend(["--sam-prompt", prompt])
    for prompt in subtract_prompts or []:
        cmd.extend(["--subtract-sam-prompt", prompt])
    if effective_merge_mode == SAM31_MERGE_REPLACE:
        cmd.append("--replace")
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    try:
        if Path(images_text).is_dir():
            cmd.append("--safe-batch")
    except OSError:
        pass
    return cmd


def build_mask2former_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    replace: bool = False,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    cmd = [
        *_module_cmd(context, "core.sky_mask"),
        images_text,
        masks_text,
        "--backend",
        "mask2former",
        "--projection",
        context.projection,
        "--quality",
        context.quality,
        "--inference-size",
        context.sky_inference_size,
        "--expand",
        context.yolo_expand,
        "--min-score",
        context.sky_min_score,
        "--labels",
        ",".join(context.ade_labels),
    ]
    cmd.extend(_sky_postprocess_args(context))
    if replace:
        cmd.append("--replace")
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    return cmd


def build_init_masks_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    cmd = [*_module_cmd(context, "core.init_masks"), images_text, masks_text]
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    return cmd


def build_stitch_cmd(
    context: MaskCommandContext,
    masks: str | Path,
    *,
    image_list: str | Path | None = None,
) -> list[str]:
    masks_text = str(masks)
    if not masks_text:
        raise ValueError("マスクフォルダが指定されていません")
    cmd = [
        *_module_cmd(context, "core.stitch_mask"),
        masks_text,
        masks_text,
        "--boundary-width",
        f"{context.stitch_boundary_width:g}",
        "--workers",
        context.stitch_workers,
    ]
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    return cmd


def build_overexposure_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    replace: bool = False,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    cmd = [
        *_module_cmd(context, "core.overexposure_mask"),
        images_text,
        masks_text,
        "--threshold",
        context.overexposure_threshold,
        "--dilate",
        context.overexposure_dilate,
        "--workers",
        context.stitch_workers,
    ]
    if replace:
        cmd.append("--replace")
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    return cmd


def build_custom_cmd(
    context: MaskCommandContext,
    images: str | Path,
    masks: str | Path,
    *,
    replace: bool = False,
    image_list: str | Path | None = None,
) -> list[str]:
    images_text, masks_text = _require_images_masks(images, masks)
    if not context.custom_mask:
        raise ValueError("CUSTOM_MASK_REQUIRED")
    cmd = [
        *_module_cmd(context, "core.custom_mask"),
        images_text,
        masks_text,
        context.custom_mask,
    ]
    if replace:
        cmd.append("--replace")
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    return cmd
