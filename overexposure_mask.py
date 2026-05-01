"""白飛び（過露出）領域をマスクに合成するツール。

元画像のRGB全チャンネルが8bit相当の閾値を超えるピクセルを検出し、
膨張処理でフリンジを含めた領域を既存マスクに黒として合成する。
マスクが存在しない場合は白地に黒で新規作成。
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def detect_overexposure(
    image: np.ndarray,
    threshold: int = 254,
    dilate_px: int = 1,
) -> np.ndarray:
    """BGR/BGRA/グレースケール画像から白飛び領域のバイナリマスクを返す。

    threshold は8bit相当値として扱う。uint16画像では同じ比率の16bit値へ
    スケールしてから比較する。

    Returns:
        白飛び領域=0 (黒), それ以外=255 (白) の uint8 マスク。
        既存マスクとのAND合成ですぐ使える形式。
    """
    threshold_value = _scale_threshold_for_dtype(threshold, image.dtype)
    if image.ndim == 2:
        blown = image > threshold_value
    else:
        color = image[..., :3]
        blown = np.all(color > threshold_value, axis=-1)

    mask_blown = blown.astype(np.uint8) * 255

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask_blown = cv2.dilate(mask_blown, kernel)

    # 反転: 白飛び=黒(0), 正常=白(255) → AND合成用
    return cv2.bitwise_not(mask_blown)


def _scale_threshold_for_dtype(threshold: int, dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        max_value = np.iinfo(dtype).max
        if max_value == 255:
            return float(threshold)
        return float(threshold) / 255.0 * float(max_value)
    return float(threshold)


def read_image_preserve_depth(path: str) -> np.ndarray | None:
    """ビット深度とαを保持して画像を読む。"""
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


# -- ワーカー用グローバル --
_worker_threshold: int = 254
_worker_dilate: int = 1


def _init_worker(threshold: int, dilate_px: int) -> None:
    global _worker_threshold, _worker_dilate
    _worker_threshold = threshold
    _worker_dilate = dilate_px


def _process_one(args: tuple[str, str, str | None]) -> str | None:
    """1枚処理。(image_path, mask_output_path, existing_mask_path or None)"""
    image_path, mask_out, existing_mask = args

    img = read_image_preserve_depth(image_path)
    if img is None:
        return f"Skipped (read error): {os.path.basename(image_path)}"

    overexp = detect_overexposure(img, _worker_threshold, _worker_dilate)

    if existing_mask is not None:
        mask = cv2.imread(existing_mask, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape != overexp.shape:
                mask = cv2.resize(mask, (overexp.shape[1], overexp.shape[0]))
            overexp = cv2.bitwise_and(mask, overexp)

    cv2.imwrite(mask_out, overexp)
    return None


def find_image_for_mask(mask_name: str, images_dir: Path) -> Path | None:
    """マスクファイル名に対応する元画像を探す。"""
    stem = Path(mask_name).stem
    # .png.png パターン対応
    if stem.lower().endswith((".jpg", ".jpeg", ".png")):
        stem = Path(stem).stem

    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def run(
    images_dir: str,
    masks_dir: str,
    threshold: int = 254,
    dilate_px: int = 1,
    workers: int | None = None,
) -> None:
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    masks_path.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = os.cpu_count() or 4

    # 画像一覧を収集
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted(
        [p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name.lower(),
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # タスク構築
    tasks: list[tuple[str, str, str | None]] = []
    for img_path in image_files:
        mask_name = f"{img_path.stem}.png"
        mask_out = str(masks_path / mask_name)

        # 既存マスクがあればAND合成
        existing = masks_path / mask_name
        existing_str = str(existing) if existing.is_file() else None

        tasks.append((str(img_path), mask_out, existing_str))

    print(f"Processing {len(tasks)} images (threshold={threshold}, dilate={dilate_px}px)")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(threshold, dilate_px),
    ) as executor:
        results = list(tqdm(executor.map(_process_one, tasks), total=len(tasks), unit="img"))

    errors = [r for r in results if r is not None]
    for e in errors:
        print(e)
    print(f"Done: {len(tasks) - len(errors)} succeeded, {len(errors)} skipped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect overexposed (blown-out) pixels and merge into mask images.",
    )
    parser.add_argument("images_dir", help="Source images directory")
    parser.add_argument("masks_dir", help="Mask output directory (existing masks are AND-merged)")
    parser.add_argument(
        "--threshold", type=int, default=254,
        help="8-bit-equivalent RGB threshold for overexposure detection (default=254, range 1-254)",
    )
    parser.add_argument(
        "--dilate", type=int, default=1,
        help="Dilation radius in pixels to cover fringe artifacts (default=1, 0=disable)",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    if args.threshold < 1 or args.threshold > 254:
        print("Error: threshold must be between 1 and 254")
        sys.exit(1)

    run(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        threshold=args.threshold,
        dilate_px=args.dilate,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
