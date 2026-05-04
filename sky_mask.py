"""Detect sky regions and merge them into masks.

Mask convention: white means keep, black means exclude. Sky pixels are written
as black and, unless --replace is used, AND-merged with any existing mask for
the same source image.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from image_io import imread_unicode, imwrite_unicode
from mask_view_recipes import (
    DEFAULT_QUALITY,
    QUALITY_CHOICES,
    PROJECTION_EQUIRECT,
    PROJECTION_NORMAL,
    back_project_bottom_mask as shared_back_project_bottom_mask,
    back_project_top_mask as shared_back_project_top_mask,
    extract_bottom_view as shared_extract_bottom_view,
    extract_top_view as shared_extract_top_view,
    iter_tile_regions,
    normalize_quality,
    recipe_for,
    auto_view_size as shared_auto_view_size,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BACKEND_MASK2FORMER = "mask2former"
BACKEND_SAM31 = "sam31"
SUPPORTED_BACKENDS = (BACKEND_MASK2FORMER, BACKEND_SAM31)
DEFAULT_BACKEND = BACKEND_MASK2FORMER
DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
DEFAULT_MASK2FORMER_MODEL_ID = DEFAULT_MODEL_ID
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "mask2former-swin-large-ade-semantic"
DEFAULT_MASK2FORMER_LOCAL_MODEL_DIR = DEFAULT_LOCAL_MODEL_DIR
DEFAULT_SAM31_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "sam3.1"
DEFAULT_SAM31_CHECKPOINT_NAME = "sam3.1_multiplex.pt"
DEFAULT_SAM31_PROMPT = "sky"
DEFAULT_SAM31_SCORE = 0.5
DEFAULT_SAM31_INFERENCE_SIZE = 1008
DEFAULT_INFERENCE_SIZE = 768
DEFAULT_MODE = ""
SUPPORTED_MODES = ("direct", "top", "bottom", "hybrid", "full")
DEFAULT_EXPAND = 0
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MIN_AREA_RATIO = 0.0005
DEFAULT_MASK2FORMER_LABELS = ("sky",)

_top_extract_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
_top_back_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_bottom_extract_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
_bottom_back_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class SkyMaskOptions:
    projection: str = PROJECTION_EQUIRECT
    quality: str = DEFAULT_QUALITY
    mode: str = DEFAULT_MODE
    inference_size: int = DEFAULT_INFERENCE_SIZE
    view_size: int | None = None
    min_score: float = DEFAULT_MIN_SCORE
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    expand_px: int = DEFAULT_EXPAND
    top_connected: bool = True
    add_ext: bool = False
    replace: bool = False
    sam_prompt: str = DEFAULT_SAM31_PROMPT
    labels: tuple[str, ...] = DEFAULT_MASK2FORMER_LABELS
    sam_prompts: tuple[str, ...] = ()


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


class Mask2FormerSkySegmenter:
    """Thin wrapper around a Mask2Former semantic segmentation model."""

    def __init__(self, model_source: str | Path, *, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as e:
            raise RuntimeError(
                "Mask2Former sky masking requires transformers and safetensors. "
                "Run setup_windows.bat or update_venv.bat."
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
        self.label_name_to_id = {
            str(label).strip().lower(): int(idx)
            for idx, label in getattr(self.model.config, "id2label", {}).items()
        }

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
        return self.detect_labels(bgr, options, labels=options.labels)

    def _resolve_label_ids(self, labels: tuple[str, ...]) -> list[int]:
        resolved: list[int] = []
        available = getattr(self.model.config, "id2label", {})
        for label in labels or DEFAULT_MASK2FORMER_LABELS:
            text = str(label).strip()
            if not text:
                continue
            if text.isdigit():
                idx = int(text)
                if idx not in {int(k) for k in available.keys()}:
                    raise RuntimeError(f"Mask2Former model does not expose label id {idx}.")
                resolved.append(idx)
                continue
            idx = self.label_name_to_id.get(text.lower())
            if idx is None:
                raise RuntimeError(f"Mask2Former model does not expose label '{text}'.")
            resolved.append(idx)
        if not resolved:
            resolved.append(self.sky_label_id)
        return sorted(set(resolved))

    def detect_labels(self, bgr: np.ndarray, options: SkyMaskOptions, *, labels: tuple[str, ...]) -> np.ndarray:
        rgb = bgr_to_rgb8(bgr)
        pil = Image.fromarray(rgb)
        label_ids = self._resolve_label_ids(labels)
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
            predicted_labels = scores.argmax(dim=0)
            selected_ids = self.torch.tensor(label_ids, device=predicted_labels.device)
            selected = (predicted_labels[..., None] == selected_ids).any(dim=-1)
            selected_scores = scores[label_ids].amax(dim=0)
            if options.min_score > 0.0:
                selected = selected & (selected_scores >= float(options.min_score))
            sky_small = selected.detach().to("cpu").numpy().astype(np.uint8)
        h, w = bgr.shape[:2]
        if sky_small.shape != (h, w):
            sky_small = cv2.resize(sky_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return sky_small.astype(bool)


class Sam31SkySegmenter:
    """Thin wrapper around Meta SAM3.1 text-prompted image segmentation."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:
            raise RuntimeError(
                "SAM3.1 sky masking requires Meta's sam3 Python package. "
                "Install it in the current venv, then place sam3.1_multiplex.pt under models/sam3.1/."
            ) from e

        checkpoint = Path(checkpoint_path)
        if checkpoint.is_dir():
            checkpoint = checkpoint / DEFAULT_SAM31_CHECKPOINT_NAME
        if not checkpoint.is_file():
            raise RuntimeError(f"SAM3.1 checkpoint not found: {checkpoint}")

        self.torch = torch
        self.processor_class = Sam3Processor
        self.device = self._resolve_device(device)
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=self.device,
        )
        self.model.eval()
        self._processor_key: tuple[int, float] | None = None
        self._processor: Any | None = None

    def _resolve_device(self, device: str) -> str:
        value = str(device).strip().lower()
        if value == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if value == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for SAM3.1 sky masking, but CUDA is not available.")
        if value not in {"cpu", "cuda"}:
            raise ValueError("--device must be auto, cpu, or cuda")
        return value

    def _processor_for_options(self, options: SkyMaskOptions) -> Any:
        resolution = int(options.inference_size)
        if resolution != DEFAULT_SAM31_INFERENCE_SIZE:
            raise RuntimeError(
                f"SAM3.1 backend requires --inference-size {DEFAULT_SAM31_INFERENCE_SIZE} "
                "with the current Meta image processor."
            )
        threshold = float(options.min_score) if float(options.min_score) > 0.0 else DEFAULT_SAM31_SCORE
        threshold = max(0.0, min(1.0, threshold))
        key = (resolution, threshold)
        if self._processor is None or self._processor_key != key:
            self._processor = self.processor_class(
                self.model,
                resolution=resolution,
                device=self.device,
                confidence_threshold=threshold,
            )
            self._processor_key = key
        return self._processor

    def detect_sky(self, bgr: np.ndarray, options: SkyMaskOptions) -> np.ndarray:
        prompts = options.sam_prompts or (options.sam_prompt.strip() or DEFAULT_SAM31_PROMPT,)
        return self.detect_prompts(bgr, options, prompts=tuple(prompts))

    def detect_prompts(self, bgr: np.ndarray, options: SkyMaskOptions, *, prompts: tuple[str, ...]) -> np.ndarray:
        rgb = bgr_to_rgb8(bgr)
        pil = Image.fromarray(rgb)
        processor = self._processor_for_options(options)
        cleaned_prompts = tuple(dict.fromkeys(prompt.strip() for prompt in prompts if prompt.strip()))
        if not cleaned_prompts:
            cleaned_prompts = (DEFAULT_SAM31_PROMPT,)
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        with self.torch.inference_mode(), autocast:
            state = processor.set_image(pil)
            sky = np.zeros(bgr.shape[:2], dtype=bool)
            for prompt in cleaned_prompts:
                output = processor.set_text_prompt(state=state, prompt=prompt)
                masks = output.get("masks")
                if masks is None or masks.numel() == 0:
                    continue
                masks = masks.detach().to("cpu")
                if masks.ndim == 4:
                    masks = masks[:, 0]
                if masks.ndim == 2:
                    prompt_mask = masks.bool().numpy()
                elif masks.ndim == 3:
                    prompt_mask = masks.bool().any(dim=0).numpy()
                else:
                    continue
                if prompt_mask.shape != bgr.shape[:2]:
                    prompt_mask = cv2.resize(
                        prompt_mask.astype(np.uint8),
                        (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                sky |= prompt_mask
        return sky.astype(bool)


SkySegmenter = Mask2FormerSkySegmenter


def resolve_model_source(
    repo_root: Path | None = None,
    model_dir: str | Path | None = None,
    *,
    backend: str = DEFAULT_BACKEND,
) -> str:
    backend = normalize_backend(backend)
    if model_dir:
        path = Path(model_dir)
        if backend == BACKEND_SAM31 and path.is_dir():
            checkpoint = path / DEFAULT_SAM31_CHECKPOINT_NAME
            return str(checkpoint if checkpoint.exists() else path)
        return str(path)

    root = repo_root or Path(__file__).resolve().parent
    if backend == BACKEND_SAM31:
        local = root / "models" / "sam3.1"
        checkpoint = local / DEFAULT_SAM31_CHECKPOINT_NAME
        if checkpoint.exists():
            return str(checkpoint)
        default_checkpoint = DEFAULT_SAM31_LOCAL_MODEL_DIR / DEFAULT_SAM31_CHECKPOINT_NAME
        if default_checkpoint.exists():
            return str(default_checkpoint)
        return str(checkpoint)

    local = root / "models" / "mask2former-swin-large-ade-semantic"
    if local.exists():
        return str(local)
    if DEFAULT_MASK2FORMER_LOCAL_MODEL_DIR.exists():
        return str(DEFAULT_MASK2FORMER_LOCAL_MODEL_DIR)
    return DEFAULT_MASK2FORMER_MODEL_ID


def normalize_backend(backend: str) -> str:
    value = str(backend).strip().lower().replace("-", "").replace(".", "")
    if value in {"mask2former", "m2f"}:
        return BACKEND_MASK2FORMER
    if value in {"sam31", "sam3", "sam"}:
        return BACKEND_SAM31
    raise ValueError(f"--backend must be one of: {', '.join(SUPPORTED_BACKENDS)}")


def create_sky_segmenter(backend: str, model_source: str | Path, *, device: str = "auto") -> Any:
    backend = normalize_backend(backend)
    if backend == BACKEND_MASK2FORMER:
        return Mask2FormerSkySegmenter(model_source, device=device)
    if backend == BACKEND_SAM31:
        return Sam31SkySegmenter(model_source, device=device)
    raise ValueError(f"Unsupported sky backend: {backend}")


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
    h, w = image.shape[:2]
    mode = str(options.mode or "").strip().lower()
    if mode:
        if options.projection != PROJECTION_EQUIRECT and mode in {"top", "bottom", "hybrid", "full"}:
            mode = "direct"

        if mode in {"direct", "hybrid", "full"}:
            sky |= segmenter.detect_sky(image, options)

        if mode in {"top", "hybrid", "full"}:
            view_size = options.view_size or shared_auto_view_size(w, h)
            top_view = shared_extract_top_view(image, view_size)
            top_sky = segmenter.detect_sky(top_view, options)
            sky |= shared_back_project_top_mask(top_sky.astype(np.uint8) * 255, w, h) > 0

        if mode in {"bottom", "full"}:
            view_size = options.view_size or shared_auto_view_size(w, h)
            bottom_view = shared_extract_bottom_view(image, view_size)
            bottom_sky = segmenter.detect_sky(bottom_view, options)
            sky |= shared_back_project_bottom_mask(bottom_sky.astype(np.uint8) * 255, w, h) > 0
    else:
        recipe = recipe_for(options.quality, options.projection)
        if recipe.direct:
            sky |= segmenter.detect_sky(image, options)

        for region in iter_tile_regions(w, h, recipe.tile_spec):
            tile = image[region.y1:region.y2, region.x1:region.x2]
            tile_sky = segmenter.detect_sky(tile, options)
            sky[region.y1:region.y2, region.x1:region.x2] |= tile_sky

        if recipe.top_view:
            view_size = options.view_size or shared_auto_view_size(w, h)
            top_view = shared_extract_top_view(image, view_size)
            top_sky = segmenter.detect_sky(top_view, options)
            sky |= shared_back_project_top_mask(top_sky.astype(np.uint8) * 255, w, h) > 0

        if recipe.bottom_view:
            view_size = options.view_size or shared_auto_view_size(w, h)
            bottom_view = shared_extract_bottom_view(image, view_size)
            bottom_sky = segmenter.detect_sky(bottom_view, options)
            sky |= shared_back_project_bottom_mask(bottom_sky.astype(np.uint8) * 255, w, h) > 0

    sky = postprocess_sky_components(
        sky,
        min_area_ratio=options.min_area_ratio,
        expand_px=options.expand_px,
        top_connected=options.top_connected,
    )
    return np.where(sky, 0, 255).astype(np.uint8)


def auto_view_size(width: int, height: int) -> int:
    return shared_auto_view_size(width, height)


def get_top_from_pano(pano_img: np.ndarray, size: int) -> np.ndarray:
    return shared_extract_top_view(pano_img, int(size))


def get_bottom_from_pano(pano_img: np.ndarray, size: int) -> np.ndarray:
    return shared_extract_bottom_view(pano_img, int(size))


def get_cube_pole_from_pano(pano_img: np.ndarray, size: int, *, pole: str) -> np.ndarray:
    h, w = pano_img.shape[:2]
    key = (w, h, int(size))
    cache = _top_extract_cache if pole == "top" else _bottom_extract_cache
    cached = cache.get(key)
    if cached is None:
        u = np.linspace(-1, 1, int(size), dtype=np.float32)
        v = np.linspace(-1, 1, int(size), dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)
        x = u_grid
        y = v_grid
        z = np.ones_like(u_grid) if pole == "top" else -np.ones_like(u_grid)
        lon = np.arctan2(y, x)
        lat = np.arctan2(z, np.sqrt(x**2 + y**2))
        map_x = ((lon + np.pi) / (2 * np.pi) * (w - 1)).astype(np.float32)
        map_y = ((np.pi / 2 - lat) / np.pi * (h - 1)).astype(np.float32)
        cache[key] = (map_x, map_y)
    else:
        map_x, map_y = cached
    return cv2.remap(pano_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def back_to_pano_from_top(top_mask: np.ndarray, pano_width: int, pano_height: int) -> np.ndarray:
    return shared_back_project_top_mask(top_mask, pano_width, pano_height)


def back_to_pano_from_bottom(bottom_mask: np.ndarray, pano_width: int, pano_height: int) -> np.ndarray:
    return shared_back_project_bottom_mask(bottom_mask, pano_width, pano_height)


def back_to_pano_from_cube_pole(pole_mask: np.ndarray, pano_width: int, pano_height: int, *, pole: str) -> np.ndarray:
    sign = 1.0 if pole == "top" else -1.0
    size = pole_mask.shape[0]
    key = (int(pano_width), int(pano_height), int(size))
    cache = _top_back_cache if pole == "top" else _bottom_back_cache
    cached = cache.get(key)
    if cached is None:
        lon = np.linspace(-np.pi, np.pi, pano_width, dtype=np.float32)
        lat = np.linspace(np.pi / 2, -np.pi / 2, pano_height, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.cos(lat_grid) * np.sin(lon_grid)
        z = np.sin(lat_grid)
        abs_z = np.abs(z)
        is_pole = (sign * z > 0) & (abs_z >= np.abs(x)) & (abs_z >= np.abs(y))
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[is_pole] = x[is_pole] / abs_z[is_pole]
        v[is_pole] = y[is_pole] / abs_z[is_pole]
        map_x = ((u + 1) / 2 * (size - 1)).astype(np.float32)
        map_y = ((v + 1) / 2 * (size - 1)).astype(np.float32)
        cache[key] = (map_x, map_y, is_pole)
    else:
        map_x, map_y, is_pole = cached

    mapped = cv2.remap(pole_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = np.zeros((pano_height, pano_width), dtype=np.uint8)
    result[is_pole] = mapped[is_pole]
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


def merge_with_existing(mask_out: Path, new_mask: np.ndarray, *, replace: bool = False) -> np.ndarray:
    if replace:
        return new_mask
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
    merged = merge_with_existing(mask_out, sky_mask, replace=options.replace)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(mask_out, merged):
        return f"Skipped (write error): {mask_out.name}"
    return None


def run(
    images: str | Path,
    masks_dir: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
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

    backend = normalize_backend(backend)
    model_source = resolve_model_source(model_dir=model_dir, backend=backend)
    print(f"Mask backend: {backend}", flush=True)
    print(f"Mask model: {model_source}", flush=True)
    print(
        "Mask settings:",
        f"projection={options.projection}",
        f"quality={options.quality}",
        f"mode={options.mode}",
        f"inference_size={options.inference_size}",
        f"view_size={options.view_size or 'auto'}",
        f"min_score={options.min_score:g}",
        f"min_area_ratio={options.min_area_ratio:g}",
        f"expand={options.expand_px}",
        f"top_connected={options.top_connected}",
        f"replace={options.replace}",
        f"labels={','.join(options.labels)}",
        f"sam_prompts={','.join(options.sam_prompts or (options.sam_prompt,))}",
        flush=True,
    )
    segmenter = create_sky_segmenter(backend, model_source, device=device)

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
    parser = argparse.ArgumentParser(description="Detect sky regions or SAM3.1 prompt regions and merge masks.")
    parser.add_argument("images", help="Source image file or directory")
    parser.add_argument("masks_dir", help="Mask output directory")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=DEFAULT_BACKEND, help="Segmentation backend")
    parser.add_argument("--model-dir", default=None, help="Local model directory or checkpoint override")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device")
    parser.add_argument("--projection", choices=(PROJECTION_EQUIRECT, PROJECTION_NORMAL), default=PROJECTION_EQUIRECT)
    parser.add_argument("--quality", choices=QUALITY_CHOICES, default=DEFAULT_QUALITY)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=DEFAULT_MODE)
    parser.add_argument("--inference-size", type=int, default=DEFAULT_INFERENCE_SIZE)
    parser.add_argument("--view-size", type=int, default=0, help="Top-view face size for equirect mode (0=auto)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-area-ratio", type=float, default=DEFAULT_MIN_AREA_RATIO)
    parser.add_argument("--expand", type=int, default=DEFAULT_EXPAND, help="Expand sky exclusion mask in pixels")
    parser.add_argument("--no-top-connected", action="store_true", help="Keep all sky components, not only top-connected")
    parser.add_argument("--add-ext", action="store_true", help="Append .png to the original filename")
    parser.add_argument("--replace", action="store_true", help="Ignore existing masks and write sky-only masks")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_MASK2FORMER_LABELS),
        help="Comma-separated Mask2Former label names or ids",
    )
    parser.add_argument(
        "--sam-prompt",
        action="append",
        default=None,
        help="Text prompt for the SAM3.1 backend; can be passed multiple times",
    )
    return parser.parse_args()


def split_csv_values(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


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
        quality=normalize_quality(args.quality),
        mode=args.mode,
        inference_size=int(args.inference_size),
        view_size=int(args.view_size) if int(args.view_size) > 0 else None,
        min_score=float(args.min_score),
        min_area_ratio=float(args.min_area_ratio),
        expand_px=int(args.expand),
        top_connected=not bool(args.no_top_connected),
        add_ext=bool(args.add_ext),
        replace=bool(args.replace),
        sam_prompt=(args.sam_prompt[0] if args.sam_prompt else DEFAULT_SAM31_PROMPT),
        sam_prompts=tuple(args.sam_prompt or ()),
        labels=split_csv_values(str(args.labels)),
    )
    try:
        result = run(
            args.images,
            args.masks_dir,
            backend=args.backend,
            model_dir=args.model_dir,
            device=args.device,
            options=options,
        )
    except Exception as e:  # noqa: BLE001 - CLI should report concise errors to the GUI log.
        print(f"Error: {e}", flush=True)
        return 1
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
