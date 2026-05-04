import argparse
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import sys
import tempfile
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode, imwrite_unicode
from yolo_mask_utils import EXPAND_DEFAULT, EXPAND_MAX, EXPAND_MIN, clamp_expand_px

CLASS_IDS: list[int] = [0]
LEVEL = 1
PROJECTION = "equirect"
EXPAND = EXPAND_DEFAULT
YOLO_CONF_DEFAULT = 0.3
BOTTOM_CONF = YOLO_CONF_DEFAULT
BOTTOM_TTA_ROTATIONS = 1
BOTTOM_MODEL = "same"
BOTTOM_FILTER = False
BOTTOM_TEMPORAL_WINDOW = 0
BOTTOM_TEMPORAL_MIN_VOTES = 1
PROFILE = None

yolo = None
yolo_models = {}
sam = None
px, py = None, None
ux, uy = None, None
is_bottom = None
_bottom_extract_cache = {}
_bottom_back_cache = {}
proc_count = 0

YOLO_MODEL_PRIMARY = "primary"
YOLO_MODEL_BOTTOM = "bottom"


@dataclass(frozen=True)
class ProcessResult:
    output_path: Path
    group_key: str
    bottom_mask_path: Path | None = None


class ProfileRecorder:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.started_at = datetime.now(timezone.utc).isoformat()
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
            "finished_at": datetime.now(timezone.utc).isoformat(),
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
    parser.add_argument("--level", type=int, default=1, help="Detection level [0:3] (default=1)")
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
    parser.add_argument(
        "--bottom-conf",
        type=float,
        default=YOLO_CONF_DEFAULT,
        help=f"YOLO confidence threshold for equirect bottom re-detection (default={YOLO_CONF_DEFAULT})",
    )
    parser.add_argument(
        "--bottom-tta-rotations",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Run bottom re-detection with 1, 2, or 4 quarter-turn rotations and merge results (default=1)",
    )
    parser.add_argument(
        "--bottom-model",
        choices=("same", "m", "l", "x"),
        default="same",
        help="YOLO model for equirect bottom re-detection. 'same' reuses the level-selected model (default=same)",
    )
    parser.add_argument(
        "--bottom-filter",
        action="store_true",
        help="Filter unreliable bottom-view mask components before merging into the final panorama mask",
    )
    parser.add_argument(
        "--bottom-temporal-window",
        type=int,
        default=0,
        help="Merge bottom detections from neighboring frames within N frames after per-frame YOLO/SAM (default=0/off)",
    )
    parser.add_argument(
        "--bottom-temporal-min-votes",
        type=int,
        default=1,
        help="Minimum neighboring bottom detections required per pixel for temporal fill (default=1)",
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


def should_run_bottom_redetection(level: int, projection: str) -> bool:
    return int(level) >= 1 and projection == "equirect"


def yolo_model_name(level: int) -> str:
    return "yolo26l.pt" if int(level) >= 2 else "yolo26m.pt"


def yolo_model_name_for_size(size: str) -> str:
    size = str(size).lower()
    if size not in {"m", "l", "x"}:
        raise ValueError(f"Unsupported YOLO model size: {size}")
    return f"yolo26{size}.pt"


def model_source(script_dir: str | Path, model_name: str) -> str:
    local_path = Path(script_dir) / model_name
    return str(local_path) if local_path.exists() else model_name


def load_models(level: int, bottom_model: str = "same") -> None:
    global yolo, yolo_models, sam
    from ultralytics import YOLO, SAM

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
    for m in sam_results[0].masks.data:
        m = m.cpu().numpy().astype(np.uint8) * 255
        mask = np.maximum(mask, m)
    return mask, has_mask + 1


def bottom_rotation_angles(rotation_count: int) -> list[int]:
    if rotation_count == 1:
        return [0]
    if rotation_count == 2:
        return [0, 180]
    if rotation_count == 4:
        return [0, 90, 180, 270]
    raise ValueError("bottom TTA rotations must be one of: 1, 2, 4")


def rotate_quarter_turn(img, angle: int):
    angle = angle % 360
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("angle must be a multiple of 90")


def transform_bbox_from_rotated_bottom(
    bbox: list[float],
    angle: int,
    *,
    width: int,
    height: int,
) -> list[float] | None:
    angle = angle % 360
    x1, y1, x2, y2 = [float(v) for v in bbox]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def to_original(x: float, y: float) -> tuple[float, float]:
        if angle == 0:
            return x, y
        if angle == 90:
            return y, height - x
        if angle == 180:
            return width - x, height - y
        if angle == 270:
            return width - y, x
        raise ValueError("angle must be a multiple of 90")

    transformed = [to_original(x, y) for x, y in corners]
    xs = [x for x, _y in transformed]
    ys = [y for _x, y in transformed]
    out_x1 = max(0.0, min(float(width), min(xs)))
    out_y1 = max(0.0, min(float(height), min(ys)))
    out_x2 = max(0.0, min(float(width), max(xs)))
    out_y2 = max(0.0, min(float(height), max(ys)))
    if out_x2 <= out_x1 or out_y2 <= out_y1:
        return None
    return [out_x1, out_y1, out_x2, out_y2]


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
    global px, py
    h, w = pano_img.shape[:2]

    key = (w, h, size)
    if key not in _bottom_extract_cache:
        # 下方向の座標系 (u, v) を作成
        u = np.linspace(-1, 1, size, dtype=np.float32)
        v = np.linspace(-1, 1, size, dtype=np.float32)
        U, V = np.meshgrid(u, v)

        # キューブマップの底面から3Dベクトルへの変換
        # 底面なので X=U, Y=V, Z=-1
        X = U
        Y = V
        Z = -np.ones_like(U)

        # 3Dベクトルからパノラマ座標 (経度, 緯度) へ
        lon = np.arctan2(Y, X)
        lat = np.arctan2(Z, np.sqrt(X**2 + Y**2))

        # ピクセル座標へ変換
        px = ((lon + np.pi) / (2 * np.pi) * (w - 1)).astype(np.float32)
        py = ((np.pi/2 - lat) / np.pi * (h - 1)).astype(np.float32)
        _bottom_extract_cache[key] = (px, py)
    else:
        px, py = _bottom_extract_cache[key]

    # 再サンプリング
    bottom_img = cv2.remap(pano_img, px, py, cv2.INTER_LINEAR)
    return bottom_img


# 下方向画像をパノラマに戻す
def back_to_pano_from_bottom(bottom_img, pano_width, pano_height):
    global ux, uy, is_bottom
    """
    キューブマップの底面画像をパノラマ形状に引き延ばして戻す
    (検出マスクなので底面以外の範囲は 0 で埋める)
    """
    bsize = bottom_img.shape[0]

    key = (pano_width, pano_height, bsize)
    if key not in _bottom_back_cache:
        # パノラマの全ピクセルの3Dベクトルを計算
        lon = np.linspace(-np.pi, np.pi, pano_width, dtype=np.float32)
        lat = np.linspace(np.pi / 2, -np.pi / 2, pano_height, dtype=np.float32)
        Lon, Lat = np.meshgrid(lon, lat)

        X = np.cos(Lat) * np.cos(Lon)
        Y = np.cos(Lat) * np.sin(Lon)
        Z = np.sin(Lat)

        # Zが負（下方向）かつ、底面(Z=-1)の面に投影したときに範囲内にあるピクセルを探す
        # 投影点 (u, v) = (X/|Z|, Y/|Z|)  ※Zは常に負
        is_bottom = (Z < 0) & (np.abs(Z) >= np.abs(X)) & (np.abs(Z) >= np.abs(Y))

        # 投影点 (u, v) = (X/|Z|, Y/|Z|) を計算
        # ※ Z=0 での除算を防ぐため、is_bottom の領域のみ計算
        U = np.zeros_like(Z)
        V = np.zeros_like(Z)
        U[is_bottom] = X[is_bottom] / np.abs(Z[is_bottom])
        V[is_bottom] = Y[is_bottom] / np.abs(Z[is_bottom])

        # キューブマップ座標 (-1~1) -> ピクセル座標
        ux = ((U + 1) / 2 * (bsize - 1)).astype(np.float32)
        uy = ((V + 1) / 2 * (bsize - 1)).astype(np.float32)
        _bottom_back_cache[key] = (ux, uy, is_bottom)
    else:
        ux, uy, is_bottom = _bottom_back_cache[key]

    # 背景を0で作成
    res_pano = np.zeros(
        (pano_height, pano_width, 3) if len(bottom_img.shape) == 3 else (pano_height, pano_width),
        dtype=np.uint8,
    )

    # マッピング実行
    mapped = cv2.remap(
        bottom_img,
        ux,
        uy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 底面判定されたピクセルのみ、マッピング結果を上書き
    res_pano[is_bottom] = mapped[is_bottom]

    return res_pano


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


def apply_temporal_bottom_propagation(results: list[ProcessResult], window: int, min_votes: int = 1) -> int:
    if window <= 0:
        return 0
    min_votes = max(1, int(min_votes))

    by_group: dict[str, list[ProcessResult]] = defaultdict(list)
    for result in results:
        by_group[result.group_key].append(result)

    updated = 0
    for group_results in by_group.values():
        source_indices = [
            idx
            for idx, result in enumerate(group_results)
            if result.bottom_mask_path is not None and result.bottom_mask_path.exists()
        ]
        if not source_indices:
            continue

        source_cache: dict[int, np.ndarray] = {}
        for idx, result in enumerate(group_results):
            frame_mask = imread_unicode(result.output_path, cv2.IMREAD_GRAYSCALE)
            if frame_mask is None:
                continue

            nearby = [src_idx for src_idx in source_indices if abs(src_idx - idx) <= window]
            if len(nearby) < min_votes:
                continue

            temporal_votes = np.zeros(frame_mask.shape, dtype=np.uint16)
            for src_idx in nearby:
                source_mask = source_cache.get(src_idx)
                if source_mask is None:
                    source_path = group_results[src_idx].bottom_mask_path
                    if source_path is None:
                        continue
                    source_mask = imread_unicode(source_path, cv2.IMREAD_GRAYSCALE)
                    if source_mask is None:
                        continue
                    source_cache[src_idx] = source_mask
                if source_mask.shape != frame_mask.shape:
                    continue
                temporal_votes += (source_mask > 0).astype(np.uint16)

            temporal_mask = np.where(temporal_votes >= min_votes, 255, 0).astype(np.uint8)
            if not np.any(temporal_mask):
                continue

            merged = cv2.bitwise_and(frame_mask, 255 - temporal_mask)
            if not np.array_equal(merged, frame_mask):
                if imwrite_unicode(result.output_path, merged):
                    updated += 1

    return updated


# =========================
# メイン処理
# =========================
def process_file(input_dir, output_dir, fname, add_ext=True, bottom_mask_path: str | Path | None = None) -> ProcessResult:
    print(f"Processing: {fname}", flush=True)

    # 画像読み込み
    img_path = os.path.join(input_dir, fname)
    if PROFILE is not None:
        PROFILE.begin_image(fname, img_path)
    image_started = perf_counter() if PROFILE is not None else None
    with profile_timer("image.read"):
        img = imread_unicode(img_path)
    if img is None:
        raise IOError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    if PROFILE is not None:
        PROFILE.set_image_shape((h, w))

    # マスク初期化
    mask = np.zeros((h, w), dtype=np.uint8)
    raw_bottom_mask = None

    # 全体で人物検出
    with profile_timer("full.total"):
        mask, has_mask = add_yolo_mask(img, mask, profile_stage="full")

    # 水平線付近の高品質抽出
    if LEVEL >= 2:
        # [ni, nj, top_y, bottom_h]
        level_defs = [
            [4, 1, 0.25, 0.75],
            [8, 2, 0.25, 0.75],
            [16, 6, 0.20, 0.80],
        ]
        level_idx = min(LEVEL - 2, len(level_defs) - 1)
        ni, nj = level_defs[level_idx][0], level_defs[level_idx][1]
        top_y = int(h * level_defs[level_idx][2])
        bottom_y = int(h * level_defs[level_idx][3])
        subw = int(w // ni)
        subh = int((bottom_y - top_y) / nj)
        pad = 20  # 重なり部分
        global proc_count
        with profile_timer("tile.total"):
            for i in range(ni):
                x1 = max(0, i * subw - pad)
                x2 = min(w, x1 + subw + pad)
                for j in range(nj):
                    y1 = max(0, top_y + j * subh - pad)
                    y2 = min(h, y1 + subh + pad)
                    # 一部を切り出して検出
                    print(f"  Processing region {i * nj + j}/{ni * nj} ...", flush=True)
                    if proc_count == 0:
                        print(f"  HQ extraction: region [{y1}:{y2}, {x1}:{x2}]", flush=True)
                    subimg = img[y1:y2, x1:x2]
                    submask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    submask, has_submask = add_yolo_mask(subimg, submask, profile_stage="tile")

                    # 元の画像に反映
                    if has_submask > 0:
                        with profile_timer("tile.merge"):
                            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], submask)
                        has_mask += has_submask
        proc_count += 1

    # 下方向のみ展開画像で再検出（エクイレクタングラー360画像専用）
    if should_run_bottom_redetection(LEVEL, PROJECTION):
        with profile_timer("bottom.total"):
            bottom_mask, has_bottom = detect_bottom_mask(img, w, h)
        if has_bottom > 0:
            mask = np.maximum(mask, bottom_mask)
            raw_bottom_mask = bottom_mask
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
            raise IOError(f"Failed to write mask: {out_path}")
    written_bottom_mask_path = None
    if raw_bottom_mask is not None and bottom_mask_path is not None:
        written_bottom_mask_path = Path(bottom_mask_path)
        written_bottom_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_timer("bottom.write"):
            if not imwrite_unicode(written_bottom_mask_path, raw_bottom_mask):
                raise IOError(f"Failed to write bottom mask: {written_bottom_mask_path}")

    result = ProcessResult(
        output_path=Path(out_path),
        group_key=str(Path(output_dir).resolve()),
        bottom_mask_path=written_bottom_mask_path,
    )
    if PROFILE is not None:
        if image_started is not None:
            PROFILE.add_timing("image.total", perf_counter() - image_started)
        PROFILE.finish_image(result.output_path)
    return result


def main(argv: list[str] | None = None) -> int:
    global CLASS_IDS, LEVEL, PROJECTION, EXPAND, BOTTOM_CONF, BOTTOM_TTA_ROTATIONS, BOTTOM_MODEL, BOTTOM_FILTER, PROFILE
    global BOTTOM_TEMPORAL_WINDOW, BOTTOM_TEMPORAL_MIN_VOTES, proc_count
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.images_dir if args.images_dir else "images"
    output_dir = args.output_dir if args.output_dir else "masks"
    add_ext = args.add_ext
    LEVEL = args.level
    PROJECTION = args.projection
    EXPAND = clamp_expand_px(args.expand)
    BOTTOM_CONF = max(0.001, min(1.0, float(args.bottom_conf)))
    BOTTOM_TTA_ROTATIONS = args.bottom_tta_rotations
    BOTTOM_MODEL = args.bottom_model
    BOTTOM_FILTER = bool(args.bottom_filter)
    BOTTOM_TEMPORAL_WINDOW = max(0, int(args.bottom_temporal_window))
    BOTTOM_TEMPORAL_MIN_VOTES = max(1, int(args.bottom_temporal_min_votes))
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
    print(
        "Bottom detection:",
        f"conf={BOTTOM_CONF:g}",
        f"rotations={BOTTOM_TTA_ROTATIONS}",
        f"model={BOTTOM_MODEL}",
        f"filter={BOTTOM_FILTER}",
        f"temporal_window={BOTTOM_TEMPORAL_WINDOW}",
        f"temporal_min_votes={BOTTOM_TEMPORAL_MIN_VOTES}",
        flush=True,
    )

    effective_bottom_model = BOTTOM_MODEL if should_run_bottom_redetection(LEVEL, PROJECTION) else "same"
    if PROFILE is not None:
        PROFILE.set_settings(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "add_ext": bool(add_ext),
                "level": int(LEVEL),
                "projection": PROJECTION,
                "expand": int(EXPAND),
                "classes": CLASS_IDS,
                "bottom_conf": float(BOTTOM_CONF),
                "bottom_tta_rotations": int(BOTTOM_TTA_ROTATIONS),
                "bottom_model": BOTTOM_MODEL,
                "effective_bottom_model": effective_bottom_model,
                "bottom_filter": bool(BOTTOM_FILTER),
                "bottom_temporal_window": int(BOTTOM_TEMPORAL_WINDOW),
                "bottom_temporal_min_votes": int(BOTTOM_TEMPORAL_MIN_VOTES),
            }
        )
    with profile_timer("model.load"):
        load_models(LEVEL, bottom_model=effective_bottom_model)

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
        process_results: list[ProcessResult] = []
        with tempfile.TemporaryDirectory(prefix="stechdrive_bottom_masks_") as bottom_tmp:
            bottom_tmp_dir = Path(bottom_tmp)
            print(f"[progress] 0/{total}", flush=True)
            for done, (dir, task_output_dir, fname) in enumerate(tasks, start=1):
                temp_bottom_path = bottom_tmp_dir / f"{done:06d}.png"
                result = process_file(dir, task_output_dir, fname, add_ext, bottom_mask_path=temp_bottom_path)
                process_results.append(result)
                print(f"Processed: {fname}", flush=True)
                print(f"[progress] {done}/{total}", flush=True)

            if BOTTOM_TEMPORAL_WINDOW > 0 and should_run_bottom_redetection(LEVEL, PROJECTION):
                with profile_timer("temporal.total"):
                    updated = apply_temporal_bottom_propagation(
                        process_results,
                        BOTTOM_TEMPORAL_WINDOW,
                        min_votes=BOTTOM_TEMPORAL_MIN_VOTES,
                    )
                print(f"[temporal] bottom masks updated: {updated}", flush=True)
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
