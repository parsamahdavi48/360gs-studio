"""Detect sky regions with Mask2Former ADE20K and merge them into masks.

Mask convention: white means keep, black means exclude. Sky pixels are written
as black and AND-merged with any existing mask for the same source image.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from image_io import imread_unicode, imwrite_unicode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "mask2former-swin-large-ade-semantic"
DEFAULT_INFERENCE_SIZE = 768
DEFAULT_MODE = "hybrid"
DEFAULT_EXPAND = 0
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MIN_AREA_RATIO = 0.0005

_top_extract_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
_top_back_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class SkyMaskOptions:
    projection: str = "equirect"
    mode: str = DEFAULT_MODE
    inference_size: int = DEFAULT_INFERENCE_SIZE
    view_size: int | None = None
    min_score: float = DEFAULT_MIN_SCORE
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    expand_px: int = DEFAULT_EXPAND
    top_connected: bool = True
    add_ext: bool = False


@dataclass
class SkyMaskRunResult:
    total: int = 0
    applied: int = 0
    skipped: int = 0
    failed: int = 0
    messages: list[str] | None = None

    @property
    def ok(self) -> bool:
        return self.failed == 0 and (self.total == 0 or self.applied > 0)

    def add_message(self, message: str) -> None:
        if self.messages is None:
            self.messages = []
        self.messages.append(message)


class SkySegmenter:
    """Thin wrapper around a Mask2Former semantic segmentation model."""

    def __init__(self, model_source: str | Path, *, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as e:
            raise RuntimeError(
                "Sky masking requires transformers and safetensors. Run setup_windows.bat or update_venv.bat."
            ) from e

        self.torch = torch
        self.device = self._resolve_device(device)
        source_text = str(model_source)
        local_files_only = Path(source_text).exists()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The following named arguments are not valid")
            self.processor = AutoImageProcessor.from_pretrained(
                source_text,
                local_files_only=local_files_only,
                use_fast=False,
            )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            source_text,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.sky_label_id = self._find_label_id("sky")

    def _resolve_device(self, device: str) -> str:
        value = str(device).strip().lower()
        if value == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if value == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for sky masking, but CUDA is not available.")
        if value not in {"cpu", "cuda"}:
            raise ValueError("--device must be auto, cpu, or cuda")
        return value

    def _find_label_id(self, label_name: str) -> int:
        labels = getattr(self.model.config, "id2label", {})
        for idx, label in labels.items():
            if str(label).strip().lower() == label_name:
                return int(idx)
        raise RuntimeError(f"Mask2Former model does not expose a '{label_name}' label.")

    def detect_sky(self, bgr: np.ndarray, options: SkyMaskOptions) -> np.ndarray:
        rgb = bgr_to_rgb8(bgr)
        pil = Image.fromarray(rgb)
        inputs = self.processor(
            images=pil,
            return_tensors="pt",
            size={"height": options.inference_size, "width": options.inference_size},
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            class_queries = outputs.class_queries_logits
            mask_queries = outputs.masks_queries_logits
            target_hw = tuple(int(v) for v in inputs["pixel_values"].shape[-2:])
            mask_queries = self.torch.nn.functional.interpolate(
                mask_queries,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            class_probs = class_queries.softmax(dim=-1)[..., :-1]
            mask_probs = mask_queries.sigmoid()
            scores = self.torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)[0]
            labels = scores.argmax(dim=0)
            sky_scores = scores[self.sky_label_id]
            sky = labels == self.sky_label_id
            if options.min_score > 0.0:
                sky = sky & (sky_scores >= float(options.min_score))
            sky_small = sky.detach().to("cpu").numpy().astype(np.uint8)
        h, w = bgr.shape[:2]
        if sky_small.shape != (h, w):
            sky_small = cv2.resize(sky_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return sky_small.astype(bool)


def resolve_model_source(repo_root: Path | None = None, model_dir: str | Path | None = None) -> str:
    if model_dir:
        return str(Path(model_dir))
    local = (repo_root or Path(__file__).resolve().parent) / "models" / "mask2former-swin-large-ade-semantic"
    if local.exists():
        return str(local)
    if DEFAULT_LOCAL_MODEL_DIR.exists():
        return str(DEFAULT_LOCAL_MODEL_DIR)
    return DEFAULT_MODEL_ID


def bgr_to_rgb8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        converted = image
    elif np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        converted = np.clip(np.rint(image.astype(np.float32) * 255.0 / max_value), 0, 255).astype(np.uint8)
    else:
        converted = np.clip(image, 0.0, 1.0)
        converted = np.rint(converted * 255.0).astype(np.uint8)

    if converted.ndim == 2:
        return cv2.cvtColor(converted, cv2.COLOR_GRAY2RGB)
    if converted.ndim == 3 and converted.shape[2] == 4:
        return cv2.cvtColor(converted, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(converted, cv2.COLOR_BGR2RGB)


def iter_image_files(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTS:
        return [input_path]
    if not input_path.is_dir():
        return []
    return sorted(
        (path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: str(path).lower(),
    )


def mask_output_path_for_image(image_path: Path, images_root: Path, masks_dir: Path, *, add_ext: bool = False) -> Path:
    if images_root.is_file():
        rel_parent = Path()
    else:
        rel_parent = image_path.resolve().relative_to(images_root.resolve()).parent
    name = f"{image_path.name}.png" if add_ext else f"{image_path.stem}.png"
    return masks_dir / rel_parent / name


def detect_sky_mask(image: np.ndarray, segmenter: Any, options: SkyMaskOptions) -> np.ndarray:
    sky = np.zeros(image.shape[:2], dtype=bool)
    mode = options.mode
    if options.projection != "equirect" and mode in {"top", "hybrid"}:
        mode = "direct"

    if mode in {"direct", "hybrid"}:
        sky |= segmenter.detect_sky(image, options)

    if mode in {"top", "hybrid"}:
        h, w = image.shape[:2]
        view_size = options.view_size or auto_view_size(w, h)
        top_view = get_top_from_pano(image, view_size)
        top_sky = segmenter.detect_sky(top_view, options)
        sky |= back_to_pano_from_top(top_sky.astype(np.uint8) * 255, w, h) > 0

    sky = postprocess_sky_components(
        sky,
        min_area_ratio=options.min_area_ratio,
        expand_px=options.expand_px,
        top_connected=options.top_connected,
    )
    return np.where(sky, 0, 255).astype(np.uint8)


def auto_view_size(width: int, height: int) -> int:
    return max(512, min(2048, int(width) // 4 if width > 0 else int(height)))


def get_top_from_pano(pano_img: np.ndarray, size: int) -> np.ndarray:
    h, w = pano_img.shape[:2]
    key = (w, h, int(size))
    cached = _top_extract_cache.get(key)
    if cached is None:
        u = np.linspace(-1, 1, int(size), dtype=np.float32)
        v = np.linspace(-1, 1, int(size), dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)
        x = u_grid
        y = v_grid
        z = np.ones_like(u_grid)
        lon = np.arctan2(y, x)
        lat = np.arctan2(z, np.sqrt(x**2 + y**2))
        map_x = ((lon + np.pi) / (2 * np.pi) * (w - 1)).astype(np.float32)
        map_y = ((np.pi / 2 - lat) / np.pi * (h - 1)).astype(np.float32)
        _top_extract_cache[key] = (map_x, map_y)
    else:
        map_x, map_y = cached
    return cv2.remap(pano_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def back_to_pano_from_top(top_mask: np.ndarray, pano_width: int, pano_height: int) -> np.ndarray:
    size = top_mask.shape[0]
    key = (int(pano_width), int(pano_height), int(size))
    cached = _top_back_cache.get(key)
    if cached is None:
        lon = np.linspace(-np.pi, np.pi, pano_width, dtype=np.float32)
        lat = np.linspace(np.pi / 2, -np.pi / 2, pano_height, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.cos(lat_grid) * np.sin(lon_grid)
        z = np.sin(lat_grid)
        abs_z = np.abs(z)
        is_top = (z > 0) & (abs_z >= np.abs(x)) & (abs_z >= np.abs(y))
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[is_top] = x[is_top] / z[is_top]
        v[is_top] = y[is_top] / z[is_top]
        map_x = ((u + 1) / 2 * (size - 1)).astype(np.float32)
        map_y = ((v + 1) / 2 * (size - 1)).astype(np.float32)
        _top_back_cache[key] = (map_x, map_y, is_top)
    else:
        map_x, map_y, is_top = cached

    mapped = cv2.remap(top_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = np.zeros((pano_height, pano_width), dtype=np.uint8)
    result[is_top] = mapped[is_top]
    return result


def postprocess_sky_components(
    sky: np.ndarray,
    *,
    min_area_ratio: float,
    expand_px: int,
    top_connected: bool,
) -> np.ndarray:
    mask = sky.astype(np.uint8)
    total_area = max(1, mask.shape[0] * mask.shape[1])
    min_area = max(1, int(total_area * max(0.0, float(min_area_ratio))))
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        if top_connected and not bool(component[0, :].any()):
            continue
        filtered[component] = 1

    if expand_px != 0 and filtered.any():
        radius = abs(int(expand_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        if expand_px > 0:
            filtered = cv2.dilate(filtered, kernel)
        else:
            filtered = cv2.erode(filtered, kernel)
    return filtered.astype(bool)


def merge_with_existing(mask_out: Path, new_mask: np.ndarray) -> np.ndarray:
    existing = imread_unicode(mask_out, cv2.IMREAD_GRAYSCALE) if mask_out.is_file() else None
    if existing is None:
        return new_mask
    if existing.shape != new_mask.shape:
        existing = cv2.resize(existing, (new_mask.shape[1], new_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(existing, new_mask)


def process_image(
    image_path: Path,
    images_root: Path,
    masks_dir: Path,
    segmenter: Any,
    options: SkyMaskOptions,
) -> str | None:
    image = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return f"Skipped (read error): {image_path.name}"
    sky_mask = detect_sky_mask(image, segmenter, options)
    mask_out = mask_output_path_for_image(image_path, images_root, masks_dir, add_ext=options.add_ext)
    merged = merge_with_existing(mask_out, sky_mask)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(mask_out, merged):
        return f"Skipped (write error): {mask_out.name}"
    return None


def run(
    images: str | Path,
    masks_dir: str | Path,
    *,
    model_dir: str | Path | None = None,
    device: str = "auto",
    options: SkyMaskOptions | None = None,
) -> SkyMaskRunResult:
    images_path = Path(images)
    masks_path = Path(masks_dir)
    options = options or SkyMaskOptions()
    image_files = iter_image_files(images_path)
    result = SkyMaskRunResult(total=len(image_files))
    if not image_files:
        print(f"No images found in {images_path}", flush=True)
        return result

    model_source = resolve_model_source(model_dir=model_dir)
    print(f"Sky model: {model_source}", flush=True)
    print(
        "Sky settings:",
        f"projection={options.projection}",
        f"mode={options.mode}",
        f"inference_size={options.inference_size}",
        f"view_size={options.view_size or 'auto'}",
        f"min_score={options.min_score:g}",
        f"min_area_ratio={options.min_area_ratio:g}",
        f"expand={options.expand_px}",
        f"top_connected={options.top_connected}",
        flush=True,
    )
    segmenter = SkySegmenter(model_source, device=device)

    print(f"[progress] 0/{len(image_files)}", flush=True)
    for done, image_path in enumerate(image_files, start=1):
        try:
            error = process_image(image_path, images_path, masks_path, segmenter, options)
        except Exception as e:  # noqa: BLE001 - keep batch processing alive.
            error = f"Failed: {image_path.name}: {e}"
        if error is None:
            result.applied += 1
        else:
            result.skipped += 1
            result.add_message(error)
        print(f"Processed: {image_path.name}", flush=True)
        print(f"[progress] {done}/{len(image_files)}", flush=True)

    for message in result.messages or []:
        print(message, flush=True)
    print(f"Done: {result.applied} applied, {result.skipped} skipped, {result.failed} failed", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect sky regions with Mask2Former ADE20K and merge masks.")
    parser.add_argument("images", help="Source image file or directory")
    parser.add_argument("masks_dir", help="Mask output directory")
    parser.add_argument("--model-dir", default=None, help="Local Mask2Former model directory override")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device")
    parser.add_argument("--projection", choices=("equirect", "normal"), default="equirect")
    parser.add_argument("--mode", choices=("direct", "top", "hybrid"), default=DEFAULT_MODE)
    parser.add_argument("--inference-size", type=int, default=DEFAULT_INFERENCE_SIZE)
    parser.add_argument("--view-size", type=int, default=0, help="Top-view face size for equirect mode (0=auto)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-area-ratio", type=float, default=DEFAULT_MIN_AREA_RATIO)
    parser.add_argument("--expand", type=int, default=DEFAULT_EXPAND, help="Expand sky exclusion mask in pixels")
    parser.add_argument("--no-top-connected", action="store_true", help="Keep all sky components, not only top-connected")
    parser.add_argument("--add-ext", action="store_true", help="Append .png to the original filename")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inference_size < 384 or args.inference_size > 2048:
        print("Error: --inference-size must be between 384 and 2048", flush=True)
        return 1
    if args.view_size < 0:
        print("Error: --view-size must be 0 or a positive integer", flush=True)
        return 1
    if args.min_score < 0.0:
        print("Error: --min-score must be >= 0", flush=True)
        return 1
    if args.min_area_ratio < 0.0 or args.min_area_ratio > 0.25:
        print("Error: --min-area-ratio must be between 0 and 0.25", flush=True)
        return 1

    options = SkyMaskOptions(
        projection=args.projection,
        mode=args.mode,
        inference_size=int(args.inference_size),
        view_size=int(args.view_size) if int(args.view_size) > 0 else None,
        min_score=float(args.min_score),
        min_area_ratio=float(args.min_area_ratio),
        expand_px=int(args.expand),
        top_connected=not bool(args.no_top_connected),
        add_ext=bool(args.add_ext),
    )
    try:
        result = run(args.images, args.masks_dir, model_dir=args.model_dir, device=args.device, options=options)
    except Exception as e:  # noqa: BLE001 - CLI should report concise errors to the GUI log.
        print(f"Error: {e}", flush=True)
        return 1
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
