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

from image_io import imread_unicode, imwrite_unicode
from mask_view_recipes import (
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
from yolo_mask_utils import EXPAND_DEFAULT, EXPAND_MAX, EXPAND_MIN, clamp_expand_px

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


@contextmanager
def profile_timer(key: str):
    if PROFILE is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        PROFILE.add_timing(key, perf_counter() - started)


def profile_record_inference(
    *,
    stage: str,
    model_key: str,
    conf: float,
    shape: tuple[int, int],
    box_count: int,
    sam_mask_count: int,
) -> None:
    if PROFILE is not None:
        PROFILE.record_inference(
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
    parser.add_argument("output_dir", nargs="?", help="Output directory for storing PNG mask images (default='./masks')")
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


def current_recipe() -> MaskViewRecipe:
    return recipe_for(QUALITY, PROJECTION)


def should_run_bottom_redetection(level: int | str | None = None, projection: str | None = None) -> bool:
    if level is not None:
        try:
            if int(level) <= 0:
                return False
        except (TypeError, ValueError):
            pass
    recipe = (
        recipe_for(None, projection or PROJECTION, legacy_level=level)
        if level is not None
        else recipe_for(QUALITY, projection or PROJECTION)
    )
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

    script_dir = Path(__file__).resolve().parent
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
):
    global yolo, sam
    detector = yolo_models.get(model_key) or yolo
    if detector is None or sam is None:
        raise RuntimeError("YOLO/SAM models are not loaded")

    # ---------- YOLO: 指定クラス検出 ----------
    bboxes = detect_yolo_bboxes(img, model_key=model_key, conf=conf, profile_stage=profile_stage)
    return add_sam_mask(
        img,
        mask,
        bboxes,
        has_mask,
        model_key=model_key,
        conf=conf,
        profile_stage=profile_stage,
    )


def detect_yolo_bboxes(
    img,
    *,
    model_key: str = YOLO_MODEL_PRIMARY,
    conf: float = YOLO_CONF_DEFAULT,
    profile_stage: str = "full",
) -> list[list[float]]:
    return detect_yolo_bboxes_batch([img], model_key=model_key, conf=conf, profile_stage=profile_stage)[0]


def detect_yolo_bboxes_batch(
    images: list[np.ndarray],
    *,
    model_key: str = YOLO_MODEL_PRIMARY,
    conf: float = YOLO_CONF_DEFAULT,
    profile_stage: str = "full",
) -> list[list[list[float]]]:
    global yolo
    if len(images) == 0:
        return []
    detector = yolo_models.get(model_key) or yolo
    if detector is None:
        raise RuntimeError("YOLO model is not loaded")

    source = images[0] if len(images) == 1 else images
    with profile_timer(f"{profile_stage}.yolo"):
        results = detector(source, conf=conf, classes=CLASS_IDS, verbose=False)
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
        )
        return mask, has_mask

    # ---------- SAM2: マスク生成 ----------
    with profile_timer(f"{profile_stage}.sam"):
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
    )
    with profile_timer(f"{profile_stage}.mask_merge"):
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


def detect_bottom_mask(img, pano_width: int, pano_height: int) -> tuple[np.ndarray | None, int]:
    bsize = int(pano_width / 4)
    if bsize <= 0:
        return None, 0

    with profile_timer("bottom.extract"):
        bottom = get_bottom_from_pano(img, size=bsize)
    merged_bottom_mask = np.zeros((bsize, bsize), dtype=np.uint8)
    total_has_bottom = 0

    bottom_bboxes = []
    for angle in bottom_rotation_angles(BOTTOM_TTA_ROTATIONS):
        with profile_timer("bottom.rotate"):
            rotated_bottom = rotate_quarter_turn(bottom, angle)
        rotated_bboxes = detect_yolo_bboxes(
            rotated_bottom,
            model_key=YOLO_MODEL_BOTTOM,
            conf=BOTTOM_CONF,
            profile_stage="bottom",
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
        conf=BOTTOM_CONF,
        profile_stage="bottom",
    )
    if total_has_bottom == 0:
        return None, 0

    if BOTTOM_FILTER:
        with profile_timer("bottom.filter"):
            merged_bottom_mask = filter_bottom_mask_components(merged_bottom_mask)
        if not np.any(merged_bottom_mask):
            return None, 0

    with profile_timer("bottom.back_project"):
        pano_mask = back_to_pano_from_bottom(merged_bottom_mask, pano_width, pano_height)
    return pano_mask, total_has_bottom


# =========================
# メイン処理
# =========================
def process_file(input_dir, output_dir, fname, add_ext=True) -> ProcessResult:
    print(f"Processing: {fname}", flush=True)

    # 画像読み込み
    img_path = os.path.join(input_dir, fname)
    if PROFILE is not None:
        PROFILE.begin_image(fname, img_path)
    image_started = perf_counter() if PROFILE is not None else None
    with profile_timer("image.read"):
        img = imread_unicode(img_path)
    if img is None:
        raise OSError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    if PROFILE is not None:
        PROFILE.set_image_shape((h, w))

    # マスク初期化
    mask = np.zeros((h, w), dtype=np.uint8)
    recipe = current_recipe()

    # 全体で人物検出
    has_mask = 0
    if recipe.direct:
        with profile_timer("full.total"):
            mask, has_mask = add_yolo_mask(img, mask, profile_stage="full")

    # 品質レシピに応じたタイル抽出
    tile_regions = iter_tile_regions(w, h, recipe.tile_spec)
    if tile_regions:
        global proc_count
        with profile_timer("tile.total"):
            for region in tile_regions:
                print(f"  Processing region {region.index}/{region.total} ...", flush=True)
                if proc_count == 0:
                    print(
                        f"  HQ extraction: region [{region.y1}:{region.y2}, {region.x1}:{region.x2}]",
                        flush=True,
                    )
                subimg = img[region.y1:region.y2, region.x1:region.x2]
                submask = np.zeros((region.y2 - region.y1, region.x2 - region.x1), dtype=np.uint8)
                submask, has_submask = add_yolo_mask(subimg, submask, profile_stage="tile")

                if has_submask > 0:
                    with profile_timer("tile.merge"):
                        mask[region.y1:region.y2, region.x1:region.x2] = np.maximum(
                            mask[region.y1:region.y2, region.x1:region.x2],
                            submask,
                        )
                    has_mask += has_submask
        proc_count += 1

    # 下方向のみ展開画像で再検出（エクイレクタングラー360画像専用）
    if should_run_bottom_redetection(LEVEL, PROJECTION):
        with profile_timer("bottom.total"):
            bottom_mask, has_bottom = detect_bottom_mask(img, w, h)
        if has_bottom > 0:
            mask = np.maximum(mask, bottom_mask)
            has_mask += has_bottom

    with profile_timer("mask.expand"):
        if has_mask > 0 and EXPAND > 0:
            # 検出領域を指定pxぶん膨張
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=EXPAND)
        elif has_mask > 0 and EXPAND < 0:
            # 負値は検出領域を収縮
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=-EXPAND)

    # ここで反転（背景=白 / 人物=黒）
    with profile_timer("mask.invert"):
        mask = 255 - mask

    # ---------- 保存 ----------
    if add_ext:
        outname = fname + ".png"
    else:
        outname = os.path.splitext(fname)[0] + ".png"
    out_path = os.path.join(output_dir, outname)
    with profile_timer("image.write"):
        if not imwrite_unicode(out_path, mask):
            raise OSError(f"Failed to write mask: {out_path}")
    result = ProcessResult(
        output_path=Path(out_path),
        group_key=str(Path(output_dir).resolve()),
    )
    if PROFILE is not None:
        if image_started is not None:
            PROFILE.add_timing("image.total", perf_counter() - image_started)
        PROFILE.finish_image(result.output_path)
    return result


def main(argv: list[str] | None = None) -> int:
    global CLASS_IDS, LEVEL, QUALITY, PROJECTION, EXPAND, BOTTOM_CONF, BOTTOM_TTA_ROTATIONS, BOTTOM_MODEL, BOTTOM_FILTER, PROFILE
    global proc_count
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.images_dir if args.images_dir else "images"
    output_dir = args.output_dir if args.output_dir else "masks"
    add_ext = args.add_ext
    LEVEL = int(args.level) if args.level is not None else 2
    QUALITY = normalize_quality(args.quality, legacy_level=args.level)
    PROJECTION = args.projection
    EXPAND = clamp_expand_px(args.expand)
    recipe = current_recipe()
    BOTTOM_CONF = recipe.bottom_conf if args.bottom_conf is None else max(0.001, min(1.0, float(args.bottom_conf)))
    BOTTOM_TTA_ROTATIONS = len(recipe.bottom_rotations) if args.bottom_tta_rotations is None else args.bottom_tta_rotations
    BOTTOM_MODEL = recipe.bottom_model if args.bottom_model is None else args.bottom_model
    BOTTOM_FILTER = bool(recipe.bottom_filter or args.bottom_filter)
    LEVEL = int(recipe.yolo_level)
    proc_count = 0
    PROFILE = ProfileRecorder(args.profile_json) if args.profile_json else None
    if EXPAND != args.expand:
        print(f"Clamped --expand from {args.expand} to {EXPAND}", flush=True)

    if not os.path.isdir(input_dir) and not os.path.isfile(input_dir):
        print("python yolo_mask.py {images_dir} {masks_dir}", flush=True)
        print(os.getcwd(), flush=True)
        PROFILE = None
        return 1

    try:
        CLASS_IDS = parse_classes(args.classes)
    except Exception as e:
        print(f"Invalid --classes value: {e}", flush=True)
        PROFILE = None
        return 1

    print("YOLO classes:", ",".join(str(x) for x in CLASS_IDS), flush=True)
    print(f"Projection: {PROJECTION}", flush=True)
    print(f"Quality: {QUALITY}", flush=True)
    print(
        "Bottom detection:",
        f"conf={BOTTOM_CONF:g}",
        f"rotations={BOTTOM_TTA_ROTATIONS}",
        f"model={BOTTOM_MODEL}",
        f"filter={BOTTOM_FILTER}",
        flush=True,
    )

    effective_bottom_model = BOTTOM_MODEL if should_run_bottom_redetection() else "same"
    if PROFILE is not None:
        PROFILE.set_settings(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "add_ext": bool(add_ext),
                "level": int(LEVEL),
                "quality": QUALITY,
                "projection": PROJECTION,
                "expand": int(EXPAND),
                "classes": CLASS_IDS,
                "bottom_conf": float(BOTTOM_CONF),
                "bottom_tta_rotations": int(BOTTOM_TTA_ROTATIONS),
                "bottom_model": BOTTOM_MODEL,
                "effective_bottom_model": effective_bottom_model,
                "bottom_filter": bool(BOTTOM_FILTER),
                "view_recipe": {
                    "direct": recipe.direct,
                    "tile_spec": recipe.tile_spec.__dict__ if recipe.tile_spec is not None else None,
                    "top_view": recipe.top_view,
                    "bottom_view": recipe.bottom_view,
                    "bottom_rotations": list(recipe.bottom_rotations),
                },
            }
        )
    with profile_timer("model.load"):
        load_models(recipe.yolo_level, bottom_model=effective_bottom_model)

    # =========================
    # 連番画像を処理
    # =========================
    if os.path.isdir(input_dir):
        # サブディレクトリを含めて処理
        base = Path(input_dir)
        dirs = [p.relative_to(base) for p in [base, *base.rglob("*")] if p.is_dir()]
        tasks = []
        for subdir in dirs:
            dir = input_dir if subdir == "." else os.path.join(input_dir, subdir)
            task_output_dir = output_dir if subdir == "." else os.path.join(output_dir, subdir)
            os.makedirs(task_output_dir, exist_ok=True)

            image_files = sorted([
                f for f in os.listdir(dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
            ])
            tasks.extend((dir, task_output_dir, fname) for fname in image_files)

        total = len(tasks)
        print(f"[progress] 0/{total}", flush=True)
        for done, (dir, task_output_dir, fname) in enumerate(tasks, start=1):
            process_file(dir, task_output_dir, fname, add_ext)
            print(f"Processed: {fname}", flush=True)
            print(f"[progress] {done}/{total}", flush=True)
    else:
        # 単一ファイルの処理
        fname = os.path.basename(input_dir)
        source_dir = os.path.dirname(input_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("[progress] 0/1", flush=True)
        process_file(source_dir, output_dir, fname, add_ext)
        print(f"Processed: {fname}", flush=True)
        print("[progress] 1/1", flush=True)

    if PROFILE is not None:
        profile = PROFILE
        profile.write()
        print(f"[profile] wrote {profile.output_path}", flush=True)
        PROFILE = None

    return 0


if __name__ == "__main__":
    sys.exit(main())
