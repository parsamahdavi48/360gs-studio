import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from core.image_io import imread_unicode, imwrite_unicode
from core.mask_targets import collect_image_targets
from core.mask_view_recipes import (
    QUALITY_CHOICES,
    QUALITY_STANDARD,
    MaskViewRecipe,
    back_project_bottom_mask,
    extract_bottom_view,
    iter_tile_regions,
    normalize_quality,
    quarter_turn_angles,
    recipe_for,
    rotate_quarter_turn,
    transform_bbox_from_rotated_view,
)
from core.yolo_mask_utils import EXPAND_DEFAULT, EXPAND_MAX, EXPAND_MIN, clamp_expand_px

CLASS_IDS: list[int] = [0]
LEVEL = 1
QUALITY = QUALITY_STANDARD
PROJECTION = "equirect"
EXPAND = EXPAND_DEFAULT
YOLO_CONF_DEFAULT = 0.3
BOTTOM_CONF = YOLO_CONF_DEFAULT
BOTTOM_TTA_ROTATIONS = 1
BOTTOM_MODEL = "same"
BOTTOM_FILTER = False
PROFILE = None

yolo = None
yolo_models = {}
sam = None
proc_count = 0

YOLO_MODEL_PRIMARY = "primary"
YOLO_MODEL_BOTTOM = "bottom"


@dataclass(frozen=True)
class ProcessResult:
    output_path: Path
    group_key: str


@dataclass(frozen=True)
class YoloMaskRuntimeSettings:
    class_ids: tuple[int, ...]
    level: int
    quality: str
    projection: str
    expand: int
    bottom_conf: float
    bottom_tta_rotations: int
    bottom_model: str
    bottom_filter: bool
    recipe: MaskViewRecipe
    profile_json: str | None


@dataclass(frozen=True)
class YoloMaskRuntimeBuild:
    settings: YoloMaskRuntimeSettings
    expand_was_clamped: bool


class ProfileRecorder:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.started_at = datetime.now(UTC).isoformat()
        self.run_started = perf_counter()
        self.settings = {}
        self.images = []
        self.totals = {
            "timings_sec": {},
            "inference_calls": 0,
            "yolo_boxes": 0,
            "sam_masks": 0,
        }
        self._current = None

    def set_settings(self, settings: dict) -> None:
        self.settings = settings

    def begin_image(self, fname: str, input_path: str | Path) -> None:
        self._current = {
            "file": fname,
            "input_path": str(input_path),
            "output_path": None,
            "shape": None,
            "timings_sec": {},
            "inference_calls": [],
            "counts": {
                "yolo_boxes": 0,
                "sam_masks": 0,
                "detections": 0,
            },
        }

    def set_image_shape(self, shape: tuple[int, int]) -> None:
        if self._current is not None:
            self._current["shape"] = {"height": int(shape[0]), "width": int(shape[1])}

    def add_timing(self, key: str, elapsed: float) -> None:
        elapsed = float(elapsed)
        self.totals["timings_sec"][key] = self.totals["timings_sec"].get(key, 0.0) + elapsed
        if self._current is not None:
            timings = self._current["timings_sec"]
            timings[key] = timings.get(key, 0.0) + elapsed

    def record_inference(
        self,
        *,
        stage: str,
        model_key: str,
        conf: float,
        shape: tuple[int, int],
        box_count: int,
        sam_mask_count: int,
    ) -> None:
        self.totals["inference_calls"] += 1
        self.totals["yolo_boxes"] += int(box_count)
        self.totals["sam_masks"] += int(sam_mask_count)
        if self._current is None:
            return
        self._current["counts"]["yolo_boxes"] += int(box_count)
        self._current["counts"]["sam_masks"] += int(sam_mask_count)
        if sam_mask_count > 0:
            self._current["counts"]["detections"] += 1
        self._current["inference_calls"].append(
            {
                "stage": stage,
                "model_key": model_key,
                "conf": float(conf),
                "height": int(shape[0]),
                "width": int(shape[1]),
                "yolo_boxes": int(box_count),
                "sam_masks": int(sam_mask_count),
            }
        )

    def finish_image(self, output_path: str | Path) -> None:
        if self._current is None:
            return
        self._current["output_path"] = str(output_path)
        self.images.append(self._current)
        self._current = None

    def write(self) -> None:
        elapsed = perf_counter() - self.run_started
        data = {
            "schema_version": 1,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "settings": self.settings,
            "totals": {
                **self.totals,
                "elapsed_sec": elapsed,
                "images": len(self.images),
            },
            "images": self.images,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(slots=True)
class YoloMaskRuntimeContext:
    """Per-run YOLO/SAM settings and mutable worker state."""

    settings: YoloMaskRuntimeSettings
    profile: ProfileRecorder | None = None
    proc_count: int = 0

    @property
    def recipe(self) -> MaskViewRecipe:
        return self.settings.recipe


_runtime_context: YoloMaskRuntimeContext | None = None


def create_runtime_context(settings: YoloMaskRuntimeSettings) -> YoloMaskRuntimeContext:
    profile = ProfileRecorder(settings.profile_json) if settings.profile_json else None
    return YoloMaskRuntimeContext(settings=settings, profile=profile)


def _compat_runtime_settings() -> YoloMaskRuntimeSettings:
    recipe = recipe_for(QUALITY, PROJECTION)
    return YoloMaskRuntimeSettings(
        class_ids=tuple(int(item) for item in CLASS_IDS),
        level=int(LEVEL),
        quality=QUALITY,
        projection=PROJECTION,
        expand=int(EXPAND),
        bottom_conf=float(BOTTOM_CONF),
        bottom_tta_rotations=int(BOTTOM_TTA_ROTATIONS),
        bottom_model=str(BOTTOM_MODEL),
        bottom_filter=bool(BOTTOM_FILTER),
        recipe=recipe,
        profile_json=None,
    )


def active_runtime_context() -> YoloMaskRuntimeContext:
    if _runtime_context is None:
        context = create_runtime_context(_compat_runtime_settings())
        if PROFILE is not None:
            context.profile = PROFILE
        context.proc_count = int(proc_count)
        return context
    return _runtime_context


def clear_runtime_context() -> None:
    global _runtime_context, PROFILE
    _runtime_context = None
    PROFILE = None


def _profile_for(context: YoloMaskRuntimeContext | None = None) -> ProfileRecorder | None:
    if context is not None:
        return context.profile
    if _runtime_context is not None:
        return _runtime_context.profile
    return PROFILE


@contextmanager
def profile_timer(key: str, context: YoloMaskRuntimeContext | None = None):
    profile = _profile_for(context)
    if profile is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        profile.add_timing(key, perf_counter() - started)


def profile_record_inference(
    *,
    stage: str,
    model_key: str,
    conf: float,
    shape: tuple[int, int],
    box_count: int,
    sam_mask_count: int,
    context: YoloMaskRuntimeContext | None = None,
) -> None:
    profile = _profile_for(context)
    if profile is not None:
        profile.record_inference(
            stage=stage,
            model_key=model_key,
            conf=conf,
            shape=shape,
            box_count=box_count,
            sam_mask_count=sam_mask_count,
        )


def extract_yolo_bboxes(result) -> list[list[float]]:
    bboxes = []
    if result.boxes is None:
        return bboxes
    for box in result.boxes.xyxy:
        bboxes.append(box.tolist())
    return bboxes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make mask images for removing humans in the scene")
    parser.add_argument("images_dir", nargs="?", help="Input directory containing image files (default='./images')")
    parser.add_argument(
        "output_dir", nargs="?", help="Output directory for storing PNG mask images (default='./masks')"
    )
    parser.add_argument(
        "--add-ext",
        "--add_ext",
        action="store_true",
        dest="add_ext",
        help="Add a file extension forcibly (ex: hoge.jpg.png)",
    )
    parser.add_argument("--level", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--quality",
        choices=QUALITY_CHOICES,
        default=None,
        help="Input view recipe quality: standard, high, or best (default=high)",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=EXPAND_DEFAULT,
        help=f"Expand detected regions by N pixels (default={EXPAND_DEFAULT}, clamped {EXPAND_MIN}..{EXPAND_MAX})",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0",
        help="YOLO class ids (comma-separated, ex: '0,2,3'; default='0' person)",
    )
    parser.add_argument(
        "--projection",
        choices=("equirect", "normal"),
        default="equirect",
        help="Source image projection. equirect enables 360 panorama-specific bottom re-detection; normal disables it.",
    )
    parser.add_argument("--bottom-conf", type=float, default=None, help="Override bottom-view YOLO confidence")
    parser.add_argument(
        "--bottom-tta-rotations",
        type=int,
        choices=(1, 2, 4),
        default=None,
        help="Override bottom-view quarter-turn rotations and merge results",
    )
    parser.add_argument(
        "--bottom-model",
        choices=("same", "m", "l", "x"),
        default=None,
        help="Override bottom-view YOLO model. 'same' reuses the quality-selected model",
    )
    parser.add_argument(
        "--bottom-filter",
        action="store_true",
        help="Filter unreliable bottom-view mask components before merging into the final panorama mask",
    )
    parser.add_argument(
        "--profile-json",
        type=str,
        default=None,
        help="Write detailed timing and detection metrics to this JSON file (default=off)",
    )
    parser.add_argument("--image-list", default=None, help="JSON or JSONL list of images to process")
    return parser


def parse_classes(text):
    tokens = [t.strip() for t in str(text).split(",") if t.strip()]
    if len(tokens) == 0:
        raise ValueError("No class ids were provided")
    ids = []
    for tok in tokens:
        cls_id = int(tok)
        if cls_id < 0:
            raise ValueError(f"Class id must be >= 0: {cls_id}")
        ids.append(cls_id)
    return sorted(set(ids))


def build_runtime_settings(args: argparse.Namespace) -> YoloMaskRuntimeBuild:
    quality = normalize_quality(args.quality, legacy_level=args.level)
    projection = args.projection
    expand = clamp_expand_px(args.expand)
    recipe = recipe_for(quality, projection)
    bottom_conf = recipe.bottom_conf if args.bottom_conf is None else max(0.001, min(1.0, float(args.bottom_conf)))
    bottom_tta_rotations = (
        len(recipe.bottom_rotations) if args.bottom_tta_rotations is None else int(args.bottom_tta_rotations)
    )
    bottom_model = recipe.bottom_model if args.bottom_model is None else str(args.bottom_model)
    bottom_filter = bool(recipe.bottom_filter or args.bottom_filter)
    class_ids = tuple(parse_classes(args.classes))
    settings = YoloMaskRuntimeSettings(
        class_ids=class_ids,
        level=int(recipe.yolo_level),
        quality=quality,
        projection=projection,
        expand=expand,
        bottom_conf=bottom_conf,
        bottom_tta_rotations=bottom_tta_rotations,
        bottom_model=bottom_model,
        bottom_filter=bottom_filter,
        recipe=recipe,
        profile_json=args.profile_json,
    )
    return YoloMaskRuntimeBuild(settings=settings, expand_was_clamped=expand != args.expand)


def apply_runtime_settings(settings: YoloMaskRuntimeSettings) -> YoloMaskRuntimeContext:
    global \
        CLASS_IDS, \
        LEVEL, \
        QUALITY, \
        PROJECTION, \
        EXPAND, \
        BOTTOM_CONF, \
        BOTTOM_TTA_ROTATIONS, \
        BOTTOM_MODEL, \
        BOTTOM_FILTER, \
        PROFILE
    global proc_count, _runtime_context
    CLASS_IDS = list(settings.class_ids)
    LEVEL = int(settings.level)
    QUALITY = settings.quality
    PROJECTION = settings.projection
    EXPAND = int(settings.expand)
    BOTTOM_CONF = float(settings.bottom_conf)
    BOTTOM_TTA_ROTATIONS = int(settings.bottom_tta_rotations)
    BOTTOM_MODEL = settings.bottom_model
    BOTTOM_FILTER = bool(settings.bottom_filter)
    proc_count = 0
    _runtime_context = create_runtime_context(settings)
    PROFILE = _runtime_context.profile
    return _runtime_context


def current_recipe(context: YoloMaskRuntimeContext | None = None) -> MaskViewRecipe:
    runtime = context or active_runtime_context()
    return runtime.recipe


def should_run_bottom_redetection(
    level: int | str | None = None,
    projection: str | None = None,
    *,
    context: YoloMaskRuntimeContext | None = None,
) -> bool:
    if level is not None:
        try:
            if int(level) <= 0:
                return False
        except (TypeError, ValueError):
            pass
    if level is None:
        runtime = context or active_runtime_context()
        if projection is None or projection == runtime.settings.projection:
            recipe = runtime.recipe
        else:
            recipe = recipe_for(runtime.settings.quality, projection)
    else:
        runtime = context
        projection_value = projection or (runtime.settings.projection if runtime is not None else PROJECTION)
        recipe = recipe_for(None, projection_value, legacy_level=level)
    return recipe.bottom_view


def yolo_model_name(level: int) -> str:
    return "yolo26l.pt" if int(level) >= 2 else "yolo26m.pt"


def yolo_model_name_for_size(size: str) -> str:
    size = str(size).lower()
    if size not in {"m", "l", "x"}:
        raise ValueError(f"Unsupported YOLO model size: {size}")
    return f"yolo26{size}.pt"


def model_source(script_dir: str | Path, model_name: str) -> str:
    base_dir = Path(script_dir)
    local_path = base_dir / "models" / "ultralytics" / model_name
    if local_path.exists():
        return str(local_path)
    return model_name


def load_models(level: int, bottom_model: str = "same") -> None:
    global yolo, yolo_models, sam
    from ultralytics import SAM, YOLO

    script_dir = Path(__file__).resolve().parents[1]
    primary_name = yolo_model_name(level)
    yolo = YOLO(model_source(script_dir, primary_name))
    yolo_models = {YOLO_MODEL_PRIMARY: yolo}

    if bottom_model == "same":
        yolo_models[YOLO_MODEL_BOTTOM] = yolo
    else:
        yolo_models[YOLO_MODEL_BOTTOM] = YOLO(model_source(script_dir, yolo_model_name_for_size(bottom_model)))

    sam = SAM(model_source(script_dir, "sam2.1_l.pt"))


# =========================
# YOLO/SAM2によるマスク抽出
# =========================
def add_yolo_mask(
    img,
    mask,
    has_mask=0,
    *,
    model_key: str = YOLO_MODEL_PRIMARY,
    conf: float = YOLO_CONF_DEFAULT,
    profile_stage: str = "full",
    context: YoloMaskRuntimeContext | None = None,
):
    global yolo, sam
    detector = yolo_models.get(model_key) or yolo
    if detector is None or sam is None:
        raise RuntimeError("YOLO/SAM models are not loaded")

    # ---------- YOLO: 指定クラス検出 ----------
    runtime = context or active_runtime_context()
    bboxes = detect_yolo_bboxes(
        img,
        model_key=model_key,
        conf=conf,
        profile_stage=profile_stage,
        context=runtime,
    )
    return add_sam_mask(
        img,
        mask,
        bboxes,
        has_mask,
        model_key=model_key,
        conf=conf,
        profile_stage=profile_stage,
        context=runtime,
    )


def detect_yolo_bboxes(
    img,
    *,
    model_key: str = YOLO_MODEL_PRIMARY,
    conf: float = YOLO_CONF_DEFAULT,
    profile_stage: str = "full",
    context: YoloMaskRuntimeContext | None = None,
) -> list[list[float]]:
    return detect_yolo_bboxes_batch([img], model_key=model_key, conf=conf, profile_stage=profile_stage, context=context)[
        0
    ]


def detect_yolo_bboxes_batch(
    images: list[np.ndarray],
    *,
    model_key: str = YOLO_MODEL_PRIMARY,
    conf: float = YOLO_CONF_DEFAULT,
    profile_stage: str = "full",
    context: YoloMaskRuntimeContext | None = None,
) -> list[list[list[float]]]:
    global yolo
    if len(images) == 0:
        return []
    detector = yolo_models.get(model_key) or yolo
    if detector is None:
        raise RuntimeError("YOLO model is not loaded")

    runtime = context or active_runtime_context()
    source = images[0] if len(images) == 1 else images
    with profile_timer(f"{profile_stage}.yolo", context=runtime):
        results = detector(source, conf=conf, classes=list(runtime.settings.class_ids), verbose=False)
    return [extract_yolo_bboxes(result) for result in results]


def add_sam_mask(
    img,
    mask,
    bboxes: list[list[float]],
    has_mask=0,
    *,
    model_key: str = YOLO_MODEL_PRIMARY,
    conf: float = YOLO_CONF_DEFAULT,
    profile_stage: str = "full",
    context: YoloMaskRuntimeContext | None = None,
):
    global sam
    if sam is None:
        raise RuntimeError("SAM model is not loaded")

    if len(bboxes) == 0:
        profile_record_inference(
            stage=profile_stage,
            model_key=model_key,
            conf=conf,
            shape=img.shape[:2],
            box_count=0,
            sam_mask_count=0,
            context=context,
        )
        return mask, has_mask

    # ---------- SAM2: マスク生成 ----------
    runtime = context or active_runtime_context()
    with profile_timer(f"{profile_stage}.sam", context=runtime):
        sam_results = sam(
            img,
            bboxes=bboxes,
            verbose=False,
        )

    if not sam_results or sam_results[0].masks is None:
        profile_record_inference(
            stage=profile_stage,
            model_key=model_key,
            conf=conf,
            shape=img.shape[:2],
            box_count=len(bboxes),
            sam_mask_count=0,
            context=runtime,
        )
        return mask, has_mask

    sam_mask_count = len(sam_results[0].masks.data)
    profile_record_inference(
        stage=profile_stage,
        model_key=model_key,
        conf=conf,
        shape=img.shape[:2],
        box_count=len(bboxes),
        sam_mask_count=sam_mask_count,
        context=runtime,
    )
    with profile_timer(f"{profile_stage}.mask_merge", context=runtime):
        mask_data = sam_results[0].masks.data
        if sam_mask_count == 1:
            first_mask = next(iter(mask_data))
            combined_mask = first_mask.cpu().numpy().astype(np.uint8) * 255
        else:
            masks_np = mask_data.cpu().numpy().astype(np.uint8)
            combined_mask = np.max(masks_np, axis=0) * 255
        mask = np.maximum(mask, combined_mask)
    return mask, has_mask + 1


def bottom_rotation_angles(rotation_count: int) -> list[int]:
    return list(quarter_turn_angles(rotation_count))


def transform_bbox_from_rotated_bottom(
    bbox: list[float],
    angle: int,
    *,
    width: int,
    height: int,
) -> list[float] | None:
    return transform_bbox_from_rotated_view(bbox, angle, width=width, height=height)


def filter_bottom_mask_components(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return mask

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = mask.shape[:2]
    image_area = h * w
    min_area = max(16, int(image_area * 0.00005))
    max_area = int(image_area * 0.35)
    small_area = int(image_area * 0.004)
    edge_margin = max(2, int(min(h, w) * 0.02))
    cx0 = (w - 1) / 2.0
    cy0 = (h - 1) / 2.0
    half_diag = max(1.0, float(np.hypot(cx0, cy0)))

    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        bbox_area = max(1, bw * bh)
        fill_ratio = area / bbox_area
        aspect_ratio = max(bw / max(1, bh), bh / max(1, bw))
        if aspect_ratio >= 12.0 and min(bw, bh) <= max(3, int(min(h, w) * 0.03)):
            continue
        if aspect_ratio >= 6.0 and fill_ratio < 0.25:
            continue

        touches_edge = x <= edge_margin or y <= edge_margin or x + bw >= w - edge_margin or y + bh >= h - edge_margin
        if touches_edge and area < small_area:
            continue

        cx, cy = centroids[label]
        center_distance = float(np.hypot(cx - cx0, cy - cy0)) / half_diag
        if area < small_area and center_distance > 0.85:
            continue

        filtered[labels == label] = 255

    return filtered


# パノラマから下方向を抽出
def get_bottom_from_pano(pano_img, size=1024):
    return extract_bottom_view(pano_img, int(size))


# 下方向画像をパノラマに戻す
def back_to_pano_from_bottom(bottom_img, pano_width, pano_height):
    return back_project_bottom_mask(bottom_img, int(pano_width), int(pano_height))


def detect_bottom_mask(
    img,
    pano_width: int,
    pano_height: int,
    *,
    context: YoloMaskRuntimeContext | None = None,
) -> tuple[np.ndarray | None, int]:
    runtime = context or active_runtime_context()
    bsize = int(pano_width / 4)
    if bsize <= 0:
        return None, 0

    with profile_timer("bottom.extract", context=runtime):
        bottom = get_bottom_from_pano(img, size=bsize)
    merged_bottom_mask = np.zeros((bsize, bsize), dtype=np.uint8)
    total_has_bottom = 0

    bottom_bboxes = []
    for angle in bottom_rotation_angles(runtime.settings.bottom_tta_rotations):
        with profile_timer("bottom.rotate", context=runtime):
            rotated_bottom = rotate_quarter_turn(bottom, angle)
        rotated_bboxes = detect_yolo_bboxes(
            rotated_bottom,
            model_key=YOLO_MODEL_BOTTOM,
            conf=runtime.settings.bottom_conf,
            profile_stage="bottom",
            context=runtime,
        )
        for bbox in rotated_bboxes:
            original_bbox = transform_bbox_from_rotated_bottom(bbox, angle, width=bsize, height=bsize)
            if original_bbox is not None:
                bottom_bboxes.append(original_bbox)

    if len(bottom_bboxes) == 0:
        return None, 0

    merged_bottom_mask, total_has_bottom = add_sam_mask(
        bottom,
        merged_bottom_mask,
        bottom_bboxes,
        model_key=YOLO_MODEL_BOTTOM,
        conf=runtime.settings.bottom_conf,
        profile_stage="bottom",
        context=runtime,
    )
    if total_has_bottom == 0:
        return None, 0

    if runtime.settings.bottom_filter:
        with profile_timer("bottom.filter", context=runtime):
            merged_bottom_mask = filter_bottom_mask_components(merged_bottom_mask)
        if not np.any(merged_bottom_mask):
            return None, 0

    with profile_timer("bottom.back_project", context=runtime):
        pano_mask = back_to_pano_from_bottom(merged_bottom_mask, pano_width, pano_height)
    return pano_mask, total_has_bottom


# =========================
# メイン処理
# =========================
def process_image_path(
    image_path: str | Path,
    output_path: str | Path,
    display_name: str,
    *,
    context: YoloMaskRuntimeContext | None = None,
) -> ProcessResult:
    print(f"Processing: {display_name}", flush=True)
    runtime = context or active_runtime_context()
    settings = runtime.settings
    profile = runtime.profile

    # 画像読み込み
    img_path = Path(image_path)
    if profile is not None:
        profile.begin_image(display_name, img_path)
    image_started = perf_counter() if profile is not None else None
    with profile_timer("image.read", context=runtime):
        img = imread_unicode(img_path)
    if img is None:
        raise OSError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    if profile is not None:
        profile.set_image_shape((h, w))

    # マスク初期化
    mask = np.zeros((h, w), dtype=np.uint8)
    recipe = runtime.recipe

    # 全体で人物検出
    has_mask = 0
    if recipe.direct:
        with profile_timer("full.total", context=runtime):
            mask, has_mask = add_yolo_mask(img, mask, profile_stage="full", context=runtime)

    # 品質レシピに応じたタイル抽出
    tile_regions = iter_tile_regions(w, h, recipe.tile_spec)
    if tile_regions:
        with profile_timer("tile.total", context=runtime):
            for region in tile_regions:
                print(f"  Processing region {region.index}/{region.total} ...", flush=True)
                if runtime.proc_count == 0:
                    print(
                        f"  HQ extraction: region [{region.y1}:{region.y2}, {region.x1}:{region.x2}]",
                        flush=True,
                    )
                subimg = img[region.y1 : region.y2, region.x1 : region.x2]
                submask = np.zeros((region.y2 - region.y1, region.x2 - region.x1), dtype=np.uint8)
                submask, has_submask = add_yolo_mask(subimg, submask, profile_stage="tile", context=runtime)

                if has_submask > 0:
                    with profile_timer("tile.merge", context=runtime):
                        mask[region.y1 : region.y2, region.x1 : region.x2] = np.maximum(
                            mask[region.y1 : region.y2, region.x1 : region.x2],
                            submask,
                        )
                    has_mask += has_submask
        runtime.proc_count += 1

    # 下方向のみ展開画像で再検出（エクイレクタングラー360画像専用）
    if should_run_bottom_redetection(settings.level, settings.projection):
        with profile_timer("bottom.total", context=runtime):
            bottom_mask, has_bottom = detect_bottom_mask(img, w, h, context=runtime)
        if has_bottom > 0:
            mask = np.maximum(mask, bottom_mask)
            has_mask += has_bottom

    with profile_timer("mask.expand", context=runtime):
        if has_mask > 0 and settings.expand > 0:
            # 検出領域を指定pxぶん膨張
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=settings.expand)
        elif has_mask > 0 and settings.expand < 0:
            # 負値は検出領域を収縮
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=-settings.expand)

    # ここで反転（背景=白 / 人物=黒）
    with profile_timer("mask.invert", context=runtime):
        mask = 255 - mask

    # ---------- 保存 ----------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_timer("image.write", context=runtime):
        if not imwrite_unicode(out_path, mask):
            raise OSError(f"Failed to write mask: {out_path}")
    result = ProcessResult(
        output_path=Path(out_path),
        group_key=str(Path(out_path).parent.resolve()),
    )
    if profile is not None:
        if image_started is not None:
            profile.add_timing("image.total", perf_counter() - image_started)
        profile.finish_image(result.output_path)
    return result


def process_file(
    input_dir,
    output_dir,
    fname,
    add_ext=True,
    *,
    context: YoloMaskRuntimeContext | None = None,
) -> ProcessResult:
    img_path = Path(input_dir) / fname
    outname = fname + ".png" if add_ext else os.path.splitext(fname)[0] + ".png"
    return process_image_path(img_path, Path(output_dir) / outname, fname, context=context)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.images_dir if args.images_dir else "images"
    output_dir = args.output_dir if args.output_dir else "masks"
    add_ext = args.add_ext

    if not os.path.isdir(input_dir) and not os.path.isfile(input_dir):
        print("python -m core.yolo_mask {images_dir} {masks_dir}", flush=True)
        print(os.getcwd(), flush=True)
        clear_runtime_context()
        return 1

    try:
        runtime = build_runtime_settings(args)
    except Exception as e:
        print(f"Invalid --classes value: {e}", flush=True)
        clear_runtime_context()
        return 1
    settings = runtime.settings
    recipe = settings.recipe
    context = apply_runtime_settings(settings)
    if runtime.expand_was_clamped:
        print(f"Clamped --expand from {args.expand} to {settings.expand}", flush=True)

    print("YOLO classes:", ",".join(str(x) for x in settings.class_ids), flush=True)
    print(f"Projection: {settings.projection}", flush=True)
    print(f"Quality: {settings.quality}", flush=True)
    print(
        "Bottom detection:",
        f"conf={settings.bottom_conf:g}",
        f"rotations={settings.bottom_tta_rotations}",
        f"model={settings.bottom_model}",
        f"filter={settings.bottom_filter}",
        flush=True,
    )

    effective_bottom_model = settings.bottom_model if should_run_bottom_redetection(settings.level, settings.projection) else "same"
    if context.profile is not None:
        context.profile.set_settings(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "add_ext": bool(add_ext),
                "level": int(settings.level),
                "quality": settings.quality,
                "projection": settings.projection,
                "expand": int(settings.expand),
                "classes": list(settings.class_ids),
                "bottom_conf": float(settings.bottom_conf),
                "bottom_tta_rotations": int(settings.bottom_tta_rotations),
                "bottom_model": settings.bottom_model,
                "effective_bottom_model": effective_bottom_model,
                "bottom_filter": bool(settings.bottom_filter),
                "view_recipe": {
                    "direct": recipe.direct,
                    "tile_spec": recipe.tile_spec.__dict__ if recipe.tile_spec is not None else None,
                    "top_view": recipe.top_view,
                    "bottom_view": recipe.bottom_view,
                    "bottom_rotations": list(recipe.bottom_rotations),
                },
            }
        )
    with profile_timer("model.load", context=context):
        load_models(recipe.yolo_level, bottom_model=effective_bottom_model)

    # =========================
    # 連番画像を処理
    # =========================
    _images_root, targets = collect_image_targets(
        input_dir,
        output_dir,
        add_ext=add_ext,
        image_list=args.image_list,
    )
    total = len(targets)
    print(f"[progress] 0/{total}", flush=True)
    for done, target in enumerate(targets, start=1):
        display_name = target.rel_path or target.image_path.name
        process_image_path(target.image_path, target.mask_path, display_name, context=context)
        print(f"Processed: {display_name}", flush=True)
        print(f"[progress] {done}/{total}", flush=True)

    if context.profile is not None:
        profile = context.profile
        profile.write()
        print(f"[profile] wrote {profile.output_path}", flush=True)
    clear_runtime_context()

    return 0


if __name__ == "__main__":
    sys.exit(main())
