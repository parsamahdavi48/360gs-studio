"""Detect sky regions and merge them into masks.

Mask convention: white means keep, black means exclude. Sky pixels are written
as black and, unless --replace is used, AND-merged with any existing mask for
the same source image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.image_io import imread_unicode, imwrite_unicode
from core.mask_targets import MaskTarget, collect_image_targets
from core.mask_view_recipes import (
    DEFAULT_QUALITY,
    PROJECTION_EQUIRECT,
    PROJECTION_NORMAL,
    QUALITY_CHOICES,
    iter_tile_regions,
    normalize_quality,
    recipe_for,
)
from core.mask_view_recipes import (
    auto_view_size as shared_auto_view_size,
)
from core.mask_view_recipes import (
    back_project_bottom_mask as shared_back_project_bottom_mask,
)
from core.mask_view_recipes import (
    back_project_top_mask as shared_back_project_top_mask,
)
from core.mask_view_recipes import (
    extract_bottom_view as shared_extract_bottom_view,
)
from core.mask_view_recipes import (
    extract_top_view as shared_extract_top_view,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BACKEND_MASK2FORMER = "mask2former"
BACKEND_SAM31 = "sam31"
SUPPORTED_BACKENDS = (BACKEND_MASK2FORMER, BACKEND_SAM31)
DEFAULT_BACKEND = BACKEND_MASK2FORMER
DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
DEFAULT_MASK2FORMER_MODEL_ID = DEFAULT_MODEL_ID
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL_DIR = REPO_ROOT / "models" / "mask2former-swin-large-ade-semantic"
DEFAULT_MASK2FORMER_LOCAL_MODEL_DIR = DEFAULT_LOCAL_MODEL_DIR
DEFAULT_SAM31_LOCAL_MODEL_DIR = REPO_ROOT / "models" / "sam3.1"
DEFAULT_SAM31_CHECKPOINT_NAME = "sam3.1_multiplex.pt"
DEFAULT_SAM31_PROMPT = "sky"
DEFAULT_SAM31_SCORE = 0.5
DEFAULT_SAM31_INFERENCE_SIZE = 1008
RESUME_STATE_FILENAME = ".sky_mask_run_state.json"
RESUME_STATE_VERSION = 1
OOM_MARKER = "[sam31-oom]"
MEMORY_MARKER = "[sam31-memory]"
MASK_MERGE_REPLACE = "replace"
MASK_MERGE_ADD = "add"
MASK_MERGE_SUBTRACT = "subtract"
SUPPORTED_MERGE_MODES = (MASK_MERGE_REPLACE, MASK_MERGE_ADD, MASK_MERGE_SUBTRACT)
DEFAULT_INFERENCE_SIZE = 768
DEFAULT_MODE = ""
SUPPORTED_MODES = ("direct", "top", "bottom", "hybrid", "full")
DEFAULT_EXPAND = 0
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MIN_AREA_RATIO = 0.0
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
    top_connected: bool = False
    add_ext: bool = False
    replace: bool = False
    merge_mode: str = MASK_MERGE_ADD
    sam_prompt: str = DEFAULT_SAM31_PROMPT
    labels: tuple[str, ...] = DEFAULT_MASK2FORMER_LABELS
    sam_prompts: tuple[str, ...] = ()
    sam_subtract_prompts: tuple[str, ...] = ()


@dataclass
class DetectedRegionMasks:
    sky: np.ndarray
    other: np.ndarray

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> DetectedRegionMasks:
        return cls(np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))

    @property
    def combined(self) -> np.ndarray:
        return self.sky | self.other

    def merge(self, other: DetectedRegionMasks) -> None:
        self.sky |= other.sky
        self.other |= other.other


@dataclass
class SkyMaskRunResult:
    total: int = 0
    applied: int = 0
    resumed: int = 0
    skipped: int = 0
    failed: int = 0
    fatal_error: str | None = None
    messages: list[str] | None = None

    @property
    def ok(self) -> bool:
        return self.fatal_error is None and self.failed == 0 and (self.total == 0 or self.applied + self.resumed > 0)

    def add_message(self, message: str) -> None:
        if self.messages is None:
            self.messages = []
        self.messages.append(message)


class MaskOutOfMemoryError(RuntimeError):
    """Fatal CUDA memory pressure while running a mask backend."""


class Mask2FormerSkySegmenter:
    """Thin wrapper around a Mask2Former semantic segmentation model."""

    def __init__(self, model_source: str | Path, *, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as e:
            raise RuntimeError(
                "Mask2Former sky masking requires transformers and safetensors. "
                "Run setup_windows.bat or update.bat."
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
            str(label).strip().lower(): int(idx) for idx, label in getattr(self.model.config, "id2label", {}).items()
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
        return self._resolve_label_ids_impl(labels, default_to_sky=True)

    def _resolve_label_ids_impl(self, labels: tuple[str, ...], *, default_to_sky: bool) -> list[int]:
        resolved: list[int] = []
        available = getattr(self.model.config, "id2label", {})
        label_source = labels if labels else (DEFAULT_MASK2FORMER_LABELS if default_to_sky else ())
        for label in label_source:
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
        if not resolved and default_to_sky:
            resolved.append(self.sky_label_id)
        return sorted(set(resolved))

    def _label_name_for_id(self, idx: int) -> str:
        labels = getattr(self.model.config, "id2label", {})
        return str(labels.get(int(idx), "")).strip().lower()

    def is_sky_label(self, label: str) -> bool:
        text = str(label).strip()
        if not text:
            return False
        if text.isdigit():
            return self._label_name_for_id(int(text)) == "sky"
        return text.lower() == "sky"

    def split_labels(self, labels: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        sky_labels: list[str] = []
        other_labels: list[str] = []
        for label in labels or DEFAULT_MASK2FORMER_LABELS:
            if self.is_sky_label(label):
                sky_labels.append(label)
            else:
                other_labels.append(label)
        return tuple(sky_labels), tuple(other_labels)

    def detect_labels(self, bgr: np.ndarray, options: SkyMaskOptions, *, labels: tuple[str, ...]) -> np.ndarray:
        masks = self.detect_label_masks(bgr, options, sky_labels=labels, other_labels=())
        return masks.sky

    def detect_label_masks(
        self,
        bgr: np.ndarray,
        options: SkyMaskOptions,
        *,
        sky_labels: tuple[str, ...],
        other_labels: tuple[str, ...],
    ) -> DetectedRegionMasks:
        rgb = bgr_to_rgb8(bgr)
        pil = Image.fromarray(rgb)
        sky_label_ids = self._resolve_label_ids_impl(sky_labels, default_to_sky=False)
        other_label_ids = self._resolve_label_ids_impl(other_labels, default_to_sky=False)
        label_ids = sorted(set(sky_label_ids + other_label_ids))
        if not label_ids:
            label_ids = [self.sky_label_id]
            sky_label_ids = [self.sky_label_id]
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
            sky_small = self._select_label_mask(scores, predicted_labels, sky_label_ids, options)
            other_small = self._select_label_mask(scores, predicted_labels, other_label_ids, options)
        h, w = bgr.shape[:2]
        if sky_small.shape != (h, w):
            sky_small = cv2.resize(sky_small, (w, h), interpolation=cv2.INTER_NEAREST)
        if other_small.shape != (h, w):
            other_small = cv2.resize(other_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return DetectedRegionMasks(sky_small.astype(bool), other_small.astype(bool))

    def _select_label_mask(
        self,
        scores: Any,
        predicted_labels: Any,
        label_ids: list[int],
        options: SkyMaskOptions,
    ) -> np.ndarray:
        if not label_ids:
            return np.zeros(tuple(int(v) for v in predicted_labels.shape), dtype=np.uint8)
        selected_ids = self.torch.tensor(label_ids, device=predicted_labels.device)
        selected = (predicted_labels[..., None] == selected_ids).any(dim=-1)
        selected_scores = scores[label_ids].amax(dim=0)
        if options.min_score > 0.0:
            selected = selected & (selected_scores >= float(options.min_score))
        return selected.detach().to("cpu").numpy().astype(np.uint8)


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
        masks = self.detect_prompt_masks(bgr, options, sky_prompts=prompts, other_prompts=())
        return masks.sky

    def detect_prompt_masks(
        self,
        bgr: np.ndarray,
        options: SkyMaskOptions,
        *,
        sky_prompts: tuple[str, ...],
        other_prompts: tuple[str, ...],
    ) -> DetectedRegionMasks:
        rgb = bgr_to_rgb8(bgr)
        pil = Image.fromarray(rgb)
        processor = self._processor_for_options(options)
        cleaned_sky_prompts = tuple(dict.fromkeys(prompt.strip() for prompt in sky_prompts if prompt.strip()))
        cleaned_other_prompts = tuple(dict.fromkeys(prompt.strip() for prompt in other_prompts if prompt.strip()))
        if not cleaned_sky_prompts and not cleaned_other_prompts:
            cleaned_sky_prompts = (DEFAULT_SAM31_PROMPT,)
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        with self.torch.inference_mode(), autocast:
            state = processor.set_image(pil)
            sky = np.zeros(bgr.shape[:2], dtype=bool)
            other = np.zeros(bgr.shape[:2], dtype=bool)
            for prompt in cleaned_sky_prompts:
                sky |= self._detect_prompt_mask(state, processor, bgr.shape[:2], prompt)
            for prompt in cleaned_other_prompts:
                other |= self._detect_prompt_mask(state, processor, bgr.shape[:2], prompt)
        return DetectedRegionMasks(sky.astype(bool), other.astype(bool))

    def _detect_prompt_mask(self, state: Any, processor: Any, shape: tuple[int, int], prompt: str) -> np.ndarray:
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks = output.get("masks")
        if masks is None or masks.numel() == 0:
            return np.zeros(shape, dtype=bool)
        masks = masks.detach().to("cpu")
        if masks.ndim == 4:
            masks = masks[:, 0]
        if masks.ndim == 2:
            prompt_mask = masks.bool().numpy()
        elif masks.ndim == 3:
            prompt_mask = masks.bool().any(dim=0).numpy()
        else:
            return np.zeros(shape, dtype=bool)
        if prompt_mask.shape != shape:
            prompt_mask = cv2.resize(
                prompt_mask.astype(np.uint8),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return prompt_mask


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

    root = repo_root or REPO_ROOT
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


def resume_state_path(masks_dir: str | Path) -> Path:
    return Path(masks_dir) / RESUME_STATE_FILENAME


def _relative_key(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _source_identity(path: Path, root: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": _relative_key(path, root),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _settings_fingerprint(
    *,
    backend: str,
    model_source: str | Path,
    device: str,
    options: SkyMaskOptions,
) -> str:
    model_path = Path(model_source)
    model_identity: dict[str, int | str] = {"path": str(model_source)}
    if model_path.exists() and model_path.is_file():
        stat = model_path.stat()
        model_identity.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    payload = {
        "backend": normalize_backend(backend),
        "model": model_identity,
        "device": str(device),
        "projection": options.projection,
        "quality": normalize_quality(options.quality),
        "mode": options.mode,
        "inference_size": int(options.inference_size),
        "view_size": int(options.view_size) if options.view_size else None,
        "min_score": float(options.min_score),
        "min_area_ratio": float(options.min_area_ratio),
        "expand_px": int(options.expand_px),
        "top_connected": bool(options.top_connected),
        "add_ext": bool(options.add_ext),
        "replace": bool(options.replace),
        "merge_mode": normalize_merge_mode(options.merge_mode),
        "sam_prompt": options.sam_prompt,
        "labels": tuple(options.labels),
        "sam_prompts": tuple(options.sam_prompts),
        "sam_subtract_prompts": tuple(options.sam_subtract_prompts),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_resume_state(settings_hash: str) -> dict[str, object]:
    return {
        "version": RESUME_STATE_VERSION,
        "settings_hash": settings_hash,
        "completed": {},
    }


def _load_resume_state(path: Path, settings_hash: str) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _new_resume_state(settings_hash)
    if not isinstance(data, dict):
        return _new_resume_state(settings_hash)
    if data.get("version") != RESUME_STATE_VERSION or data.get("settings_hash") != settings_hash:
        return _new_resume_state(settings_hash)
    completed = data.get("completed")
    if not isinstance(completed, dict):
        data["completed"] = {}
    return data


def _write_json_atomic(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _completed_record_matches(
    state: dict[str, object],
    image_path: Path,
    images_root: Path,
    masks_dir: Path,
    options: SkyMaskOptions,
    mask_path: Path | None = None,
) -> bool:
    completed = state.get("completed")
    if not isinstance(completed, dict):
        return False
    key = _relative_key(image_path, images_root)
    record = completed.get(key)
    if not isinstance(record, dict):
        return False
    try:
        source = _source_identity(image_path, images_root)
    except OSError:
        return False
    if record.get("source") != source:
        return False
    mask_out = mask_path or mask_output_path_for_image(image_path, images_root, masks_dir, add_ext=options.add_ext)
    return mask_out.is_file()


def _mark_resume_completed(
    state_path: Path,
    state: dict[str, object],
    image_path: Path,
    images_root: Path,
    masks_dir: Path,
    options: SkyMaskOptions,
    mask_path: Path | None = None,
) -> None:
    completed = state.setdefault("completed", {})
    if not isinstance(completed, dict):
        completed = {}
        state["completed"] = completed
    key = _relative_key(image_path, images_root)
    mask_out = mask_path or mask_output_path_for_image(image_path, images_root, masks_dir, add_ext=options.add_ext)
    completed[key] = {
        "source": _source_identity(image_path, images_root),
        "mask": _relative_key(mask_out, masks_dir),
    }
    _write_json_atomic(state_path, state)


def _valid_completed_count(
    targets: list[MaskTarget],
    images_root: Path,
    masks_dir: Path,
    state: dict[str, object],
    options: SkyMaskOptions,
) -> int:
    return sum(
        1
        for target in targets
        if _completed_record_matches(state, target.image_path, images_root, masks_dir, options, target.mask_path)
    )


def imwrite_unicode_atomic(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        if not imwrite_unicode(temp, image):
            return False
        os.replace(temp, path)
    except OSError:
        return False
    finally:
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass
    return True


def _is_sky_prompt(prompt: str) -> bool:
    value = str(prompt).strip().lower()
    return value in {"sky", "the sky"} or value.endswith(" sky")


def _split_sam_prompts(
    options: SkyMaskOptions,
    prompts: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prompt_values = prompts if prompts is not None else options.sam_prompts
    prompt_values = prompt_values or (options.sam_prompt.strip() or DEFAULT_SAM31_PROMPT,)
    sky_prompts: list[str] = []
    other_prompts: list[str] = []
    for prompt in prompt_values:
        if _is_sky_prompt(prompt):
            sky_prompts.append(prompt)
        else:
            other_prompts.append(prompt)
    return tuple(sky_prompts), tuple(other_prompts)


def detect_region_masks(
    image: np.ndarray,
    segmenter: Any,
    options: SkyMaskOptions,
    *,
    sam_prompts: tuple[str, ...] | None = None,
) -> DetectedRegionMasks:
    if hasattr(segmenter, "split_labels") and hasattr(segmenter, "detect_label_masks"):
        sky_labels, other_labels = segmenter.split_labels(options.labels)
        return segmenter.detect_label_masks(
            image,
            options,
            sky_labels=sky_labels,
            other_labels=other_labels,
        )
    if hasattr(segmenter, "detect_prompt_masks"):
        sky_prompts, other_prompts = _split_sam_prompts(options, prompts=sam_prompts)
        return segmenter.detect_prompt_masks(
            image,
            options,
            sky_prompts=sky_prompts,
            other_prompts=other_prompts,
        )
    return DetectedRegionMasks(
        segmenter.detect_sky(image, options).astype(bool),
        np.zeros(image.shape[:2], dtype=bool),
    )


def expand_mask(mask: np.ndarray, expand_px: int) -> np.ndarray:
    expanded = mask.astype(np.uint8)
    if expand_px != 0 and expanded.any():
        radius = abs(int(expand_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        if expand_px > 0:
            expanded = cv2.dilate(expanded, kernel)
        else:
            expanded = cv2.erode(expanded, kernel)
    return expanded.astype(bool)


def detect_exclusion_regions(
    image: np.ndarray,
    segmenter: Any,
    options: SkyMaskOptions,
    *,
    sam_prompts: tuple[str, ...] | None = None,
) -> np.ndarray:
    regions = DetectedRegionMasks.empty(image.shape[:2])
    h, w = image.shape[:2]
    mode = str(options.mode or "").strip().lower()
    if mode:
        if options.projection != PROJECTION_EQUIRECT and mode in {"top", "bottom", "hybrid", "full"}:
            mode = "direct"

        if mode in {"direct", "hybrid", "full"}:
            regions.merge(detect_region_masks(image, segmenter, options, sam_prompts=sam_prompts))

        if mode in {"top", "hybrid", "full"}:
            view_size = options.view_size or shared_auto_view_size(w, h)
            top_view = shared_extract_top_view(image, view_size)
            top_regions = detect_region_masks(top_view, segmenter, options, sam_prompts=sam_prompts)
            regions.sky |= shared_back_project_top_mask(top_regions.sky.astype(np.uint8) * 255, w, h) > 0
            regions.other |= shared_back_project_top_mask(top_regions.other.astype(np.uint8) * 255, w, h) > 0

        if mode in {"bottom", "full"}:
            view_size = options.view_size or shared_auto_view_size(w, h)
            bottom_view = shared_extract_bottom_view(image, view_size)
            bottom_regions = detect_region_masks(bottom_view, segmenter, options, sam_prompts=sam_prompts)
            regions.sky |= shared_back_project_bottom_mask(bottom_regions.sky.astype(np.uint8) * 255, w, h) > 0
            regions.other |= shared_back_project_bottom_mask(bottom_regions.other.astype(np.uint8) * 255, w, h) > 0
    else:
        recipe = recipe_for(options.quality, options.projection)
        if recipe.direct:
            regions.merge(detect_region_masks(image, segmenter, options, sam_prompts=sam_prompts))

        for region in iter_tile_regions(w, h, recipe.tile_spec):
            tile = image[region.y1 : region.y2, region.x1 : region.x2]
            tile_regions = detect_region_masks(tile, segmenter, options, sam_prompts=sam_prompts)
            regions.sky[region.y1 : region.y2, region.x1 : region.x2] |= tile_regions.sky
            regions.other[region.y1 : region.y2, region.x1 : region.x2] |= tile_regions.other

        if recipe.top_view:
            view_size = options.view_size or shared_auto_view_size(w, h)
            top_view = shared_extract_top_view(image, view_size)
            top_regions = detect_region_masks(top_view, segmenter, options, sam_prompts=sam_prompts)
            regions.sky |= shared_back_project_top_mask(top_regions.sky.astype(np.uint8) * 255, w, h) > 0
            regions.other |= shared_back_project_top_mask(top_regions.other.astype(np.uint8) * 255, w, h) > 0

        if recipe.bottom_view:
            view_size = options.view_size or shared_auto_view_size(w, h)
            bottom_view = shared_extract_bottom_view(image, view_size)
            bottom_regions = detect_region_masks(bottom_view, segmenter, options, sam_prompts=sam_prompts)
            regions.sky |= shared_back_project_bottom_mask(bottom_regions.sky.astype(np.uint8) * 255, w, h) > 0
            regions.other |= shared_back_project_bottom_mask(bottom_regions.other.astype(np.uint8) * 255, w, h) > 0

    sky = postprocess_sky_components(
        regions.sky,
        min_area_ratio=options.min_area_ratio,
        expand_px=0,
        top_connected=options.top_connected,
    )
    return expand_mask(sky | regions.other, options.expand_px)


def detect_sky_mask(image: np.ndarray, segmenter: Any, options: SkyMaskOptions) -> np.ndarray:
    detected = detect_exclusion_regions(image, segmenter, options)
    subtract_prompts = tuple(prompt.strip() for prompt in options.sam_subtract_prompts if prompt.strip())
    if subtract_prompts and hasattr(segmenter, "detect_prompt_masks"):
        subtract = detect_exclusion_regions(image, segmenter, options, sam_prompts=subtract_prompts)
        detected &= ~subtract
    return np.where(detected, 0, 255).astype(np.uint8)


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

    return expand_mask(filtered, expand_px)


def normalize_merge_mode(value: str) -> str:
    mode = str(value or MASK_MERGE_ADD).strip().lower().replace("_", "-")
    if mode in {"and", "merge"}:
        return MASK_MERGE_ADD
    if mode not in SUPPORTED_MERGE_MODES:
        raise ValueError(f"--merge-mode must be one of: {', '.join(SUPPORTED_MERGE_MODES)}")
    return mode


def merge_with_existing(
    mask_out: Path,
    new_mask: np.ndarray,
    *,
    replace: bool = False,
    merge_mode: str = MASK_MERGE_ADD,
) -> np.ndarray:
    mode = MASK_MERGE_REPLACE if replace else normalize_merge_mode(merge_mode)
    if mode == MASK_MERGE_REPLACE:
        return new_mask
    existing = imread_unicode(mask_out, cv2.IMREAD_GRAYSCALE) if mask_out.is_file() else None
    if existing is None:
        if mode == MASK_MERGE_SUBTRACT:
            return np.full_like(new_mask, 255, dtype=np.uint8)
        return new_mask
    if existing.shape != new_mask.shape:
        existing = cv2.resize(existing, (new_mask.shape[1], new_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mode == MASK_MERGE_SUBTRACT:
        merged = existing.copy()
        merged[new_mask == 0] = 255
        return merged
    return cv2.bitwise_and(existing, new_mask)


def is_out_of_memory_error(exc: BaseException) -> bool:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__
    for item in chain:
        class_name = item.__class__.__name__.lower()
        text = str(item).lower()
        if "outofmemory" in class_name:
            return True
        if "out of memory" in text and any(marker in text for marker in ("cuda", "gpu", "cudnn")):
            return True
    return False


def _segmenter_torch(segmenter: Any) -> Any | None:
    torch_module = getattr(segmenter, "torch", None)
    if torch_module is not None:
        return torch_module
    model = getattr(segmenter, "model", None)
    return getattr(model, "torch", None)


def _cuda_memory_stats(segmenter: Any) -> dict[str, int] | None:
    torch_module = _segmenter_torch(segmenter)
    if torch_module is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    try:
        free, total = cuda.mem_get_info()
        return {
            "free": int(free),
            "total": int(total),
            "allocated": int(cuda.memory_allocated()),
            "reserved": int(cuda.memory_reserved()),
            "peak_reserved": int(cuda.max_memory_reserved()),
        }
    except Exception:
        return None


def _reset_cuda_peak(segmenter: Any) -> None:
    torch_module = _segmenter_torch(segmenter)
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    if cuda is None or not cuda.is_available():
        return
    try:
        cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _empty_cuda_cache(segmenter: Any) -> None:
    torch_module = _segmenter_torch(segmenter)
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    if cuda is None or not cuda.is_available():
        return
    try:
        cuda.empty_cache()
    except Exception:
        pass


def _format_bytes(value: int) -> str:
    gib = float(value) / (1024.0**3)
    return f"{gib:.1f} GiB"


def _print_memory_stats(prefix: str, stats: dict[str, int] | None) -> None:
    if not stats:
        return
    print(
        MEMORY_MARKER,
        prefix,
        f"free={_format_bytes(stats['free'])}",
        f"total={_format_bytes(stats['total'])}",
        f"reserved={_format_bytes(stats['reserved'])}",
        f"peak_reserved={_format_bytes(stats['peak_reserved'])}",
        flush=True,
    )


def process_image(
    image_path: Path,
    images_root: Path,
    masks_dir: Path,
    segmenter: Any,
    options: SkyMaskOptions,
    mask_path: Path | None = None,
) -> str | None:
    image = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return f"Skipped (read error): {image_path.name}"
    sky_mask = detect_sky_mask(image, segmenter, options)
    mask_out = mask_path or mask_output_path_for_image(image_path, images_root, masks_dir, add_ext=options.add_ext)
    merged = merge_with_existing(mask_out, sky_mask, replace=options.replace, merge_mode=options.merge_mode)
    if not imwrite_unicode_atomic(mask_out, merged):
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
    resume_state: bool = False,
    max_images: int | None = None,
    progress_offset: int | None = None,
    progress_total: int | None = None,
    image_list: str | Path | None = None,
) -> SkyMaskRunResult:
    images_path = Path(images)
    masks_path = Path(masks_dir)
    options = options or SkyMaskOptions()
    images_root, targets = collect_image_targets(
        images_path,
        masks_path,
        add_ext=options.add_ext,
        image_list=image_list,
    )
    result = SkyMaskRunResult(total=len(targets))
    if not targets:
        print(f"No images found in {images_path}", flush=True)
        return result

    backend = normalize_backend(backend)
    model_source = resolve_model_source(model_dir=model_dir, backend=backend)
    settings_hash = _settings_fingerprint(
        backend=backend,
        model_source=model_source,
        device=device,
        options=options,
    )
    state_path = resume_state_path(masks_path)
    state: dict[str, object] | None = None
    if resume_state:
        state = _load_resume_state(state_path, settings_hash)
        _write_json_atomic(state_path, state)

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
        f"merge_mode={normalize_merge_mode(options.merge_mode)}",
        f"labels={','.join(options.labels)}",
        f"sam_prompts={','.join(options.sam_prompts or (options.sam_prompt,))}",
        f"sam_subtract_prompts={','.join(options.sam_subtract_prompts)}",
        flush=True,
    )
    segmenter = create_sky_segmenter(backend, model_source, device=device)
    if backend == BACKEND_SAM31:
        _print_memory_stats("before", _cuda_memory_stats(segmenter))
        _reset_cuda_peak(segmenter)

    pending_targets: list[MaskTarget] = []
    for target in targets:
        if state is not None and _completed_record_matches(
            state,
            target.image_path,
            images_root,
            masks_path,
            options,
            target.mask_path,
        ):
            result.resumed += 1
            continue
        pending_targets.append(target)
    if max_images is not None and max_images > 0:
        pending_targets = pending_targets[: int(max_images)]

    total_for_progress = int(progress_total) if progress_total and progress_total > 0 else len(targets)
    done_base = int(progress_offset) if progress_offset is not None and progress_offset >= 0 else result.resumed
    print(f"[progress] {done_base}/{total_for_progress}", flush=True)
    printed_first_memory = False
    for local_done, target in enumerate(pending_targets, start=1):
        image_path = target.image_path
        try:
            error = process_image(image_path, images_root, masks_path, segmenter, options, target.mask_path)
        except Exception as e:  # noqa: BLE001 - keep batch processing alive.
            if is_out_of_memory_error(e):
                _empty_cuda_cache(segmenter)
                message = (
                    f"{OOM_MARKER} CUDA/GPU memory ran out while processing {image_path.name}. "
                    "Completed masks remain saved; rerun with the same settings to resume unfinished images."
                )
                print(message, flush=True)
                result.failed += 1
                result.fatal_error = message
                result.add_message(message)
                break
            error = f"Failed: {image_path.name}: {e}"
        if error is None:
            result.applied += 1
            if state is not None:
                _mark_resume_completed(
                    state_path, state, image_path, images_root, masks_path, options, target.mask_path
                )
            if backend == BACKEND_SAM31 and not printed_first_memory:
                _print_memory_stats("after_first_image", _cuda_memory_stats(segmenter))
                printed_first_memory = True
        else:
            result.skipped += 1
            result.add_message(error)
        print(f"Processed: {image_path.name}", flush=True)
        print(f"[progress] {done_base + local_done}/{total_for_progress}", flush=True)

    for message in result.messages or []:
        print(message, flush=True)
    if state is not None and result.fatal_error is None:
        valid_completed = _valid_completed_count(targets, images_root, masks_path, state, options)
        if valid_completed >= len(targets):
            try:
                state_path.unlink()
            except OSError:
                pass
    print(
        f"Done: {result.applied} applied, {result.resumed} resumed, {result.skipped} skipped, {result.failed} failed",
        flush=True,
    )
    return result


def _build_child_args(
    args: argparse.Namespace,
    *,
    max_images: int,
    progress_offset: int,
    progress_total: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "core.sky_mask",
        str(args.images),
        str(args.masks_dir),
        "--backend",
        str(args.backend),
        "--device",
        str(args.device),
        "--projection",
        str(args.projection),
        "--quality",
        str(args.quality),
        "--inference-size",
        str(args.inference_size),
        "--view-size",
        str(args.view_size),
        "--min-score",
        str(args.min_score),
        "--min-area-ratio",
        str(args.min_area_ratio),
        "--expand",
        str(args.expand),
        "--merge-mode",
        str(args.merge_mode),
        "--labels",
        str(args.labels),
        "--resume-state",
        "--max-images",
        str(max_images),
        "--progress-offset",
        str(progress_offset),
        "--progress-total",
        str(progress_total),
    ]
    if args.model_dir:
        cmd.extend(["--model-dir", str(args.model_dir)])
    if args.top_connected:
        cmd.append("--top-connected")
    if args.no_top_connected:
        cmd.append("--no-top-connected")
    if args.add_ext:
        cmd.append("--add-ext")
    if args.replace:
        cmd.append("--replace")
    image_list = getattr(args, "image_list", None)
    if image_list:
        cmd.extend(["--image-list", str(image_list)])
    for prompt in args.sam_prompt or []:
        cmd.extend(["--sam-prompt", str(prompt)])
    for prompt in args.subtract_sam_prompt or []:
        cmd.extend(["--subtract-sam-prompt", str(prompt)])
    return cmd


def _run_child_and_stream(cmd: list[str]) -> tuple[int, bool]:
    oom = False
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(  # noqa: S603 - command is built from current interpreter/script and parsed args.
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if OOM_MARKER in line or "CUDA out of memory" in line or "cuda out of memory" in line.lower():
            oom = True
        print(line, end="", flush=True)
    return proc.wait(), oom


def _safe_batch_completed_count(
    args: argparse.Namespace,
    options: SkyMaskOptions,
    *,
    settings_hash: str,
) -> tuple[int, int, dict[str, object], list[MaskTarget], Path, Path, Path]:
    images_path = Path(args.images)
    masks_path = Path(args.masks_dir)
    images_root, targets = collect_image_targets(
        images_path,
        masks_path,
        add_ext=options.add_ext,
        image_list=getattr(args, "image_list", None),
    )
    state_path = resume_state_path(masks_path)
    state = _load_resume_state(state_path, settings_hash)
    _write_json_atomic(state_path, state)
    completed = _valid_completed_count(targets, images_root, masks_path, state, options)
    return completed, len(targets), state, targets, images_root, masks_path, state_path


def run_safe_batch(args: argparse.Namespace, options: SkyMaskOptions) -> int:
    backend = normalize_backend(args.backend)
    images_path = Path(args.images)
    if backend != BACKEND_SAM31 or not images_path.is_dir():
        result = run(
            args.images,
            args.masks_dir,
            backend=args.backend,
            model_dir=args.model_dir,
            device=args.device,
            options=options,
            resume_state=bool(args.resume_state),
            max_images=int(args.max_images) if int(args.max_images) > 0 else None,
            progress_offset=int(args.progress_offset) if int(args.progress_offset) >= 0 else None,
            progress_total=int(args.progress_total) if int(args.progress_total) > 0 else None,
            image_list=getattr(args, "image_list", None),
        )
        return 0 if result.ok else 1

    model_source = resolve_model_source(model_dir=args.model_dir, backend=backend)
    settings_hash = _settings_fingerprint(
        backend=backend,
        model_source=model_source,
        device=args.device,
        options=options,
    )
    completed, total, _state, _targets, _images_root, _masks_path, state_path = _safe_batch_completed_count(
        args,
        options,
        settings_hash=settings_hash,
    )
    if total == 0:
        print(f"No images found in {images_path}", flush=True)
        return 1

    print("SAM3.1 safe batch: enabled", flush=True)
    print(f"[progress] {completed}/{total}", flush=True)
    if completed > 0:
        print(f"Resume: {completed}/{total} masks are already saved for the current settings.", flush=True)

    chunk_size = max(1, total - completed)
    while completed < total:
        remaining = total - completed
        current_chunk = max(1, min(chunk_size, remaining))
        print(f"SAM3.1 safe batch: processing up to {current_chunk} remaining image(s)", flush=True)
        cmd = _build_child_args(
            args,
            max_images=current_chunk,
            progress_offset=completed,
            progress_total=total,
        )
        exit_code, oom = _run_child_and_stream(cmd)
        previous_completed = completed
        completed, total, _state, _targets, _images_root, _masks_path, state_path = _safe_batch_completed_count(
            args,
            options,
            settings_hash=settings_hash,
        )
        if exit_code == 0:
            if current_chunk >= remaining or completed >= total:
                print(f"[progress] {total}/{total}", flush=True)
                try:
                    state_path.unlink()
                except OSError:
                    pass
                return 0
            if completed <= previous_completed:
                print("Error: SAM3.1 safe batch made no progress.", flush=True)
                return 1
            chunk_size = min(total - completed, max(current_chunk, chunk_size))
            continue

        if not oom:
            return exit_code or 1

        if completed > previous_completed:
            print(
                f"SAM3.1 safe batch: saved progress through {completed}/{total}; retrying unfinished images.",
                flush=True,
            )
        if current_chunk <= 1 and completed <= previous_completed:
            print(
                "Error: SAM3.1 could not process the next image with the current hardware/settings. "
                "Completed masks remain saved; lower quality, close other GPU apps, or use another backend.",
                flush=True,
            )
            return 1
        chunk_size = 1 if current_chunk <= 2 else max(1, current_chunk // 2)
        print(f"SAM3.1 safe batch: reducing retry chunk size to {chunk_size}.", flush=True)

    try:
        state_path.unlink()
    except OSError:
        pass
    print(f"[progress] {total}/{total}", flush=True)
    return 0


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
    parser.add_argument(
        "--top-connected", action="store_true", help="Keep only sky components connected to the top edge"
    )
    parser.add_argument("--no-top-connected", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--add-ext", action="store_true", help="Append .png to the original filename")
    parser.add_argument("--replace", action="store_true", help="Ignore existing masks and write sky-only masks")
    parser.add_argument(
        "--merge-mode",
        choices=SUPPORTED_MERGE_MODES,
        default=MASK_MERGE_ADD,
        help="How to apply detected regions to existing masks when --replace is not used",
    )
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
    parser.add_argument(
        "--subtract-sam-prompt",
        action="append",
        default=None,
        help="SAM3.1 prompt to subtract from detected prompt masks; can be passed multiple times",
    )
    parser.add_argument(
        "--safe-batch",
        action="store_true",
        help="Run SAM3.1 directory processing through a resumable parent process",
    )
    parser.add_argument("--resume-state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-images", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--progress-offset", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--progress-total", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--image-list", default=None, help="JSON or JSONL list of images to process")
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
        top_connected=bool(args.top_connected) and not bool(args.no_top_connected),
        add_ext=bool(args.add_ext),
        replace=bool(args.replace),
        merge_mode=MASK_MERGE_REPLACE if bool(args.replace) else normalize_merge_mode(args.merge_mode),
        sam_prompt=(args.sam_prompt[0] if args.sam_prompt else DEFAULT_SAM31_PROMPT),
        sam_prompts=tuple(args.sam_prompt or ()),
        sam_subtract_prompts=tuple(args.subtract_sam_prompt or ()),
        labels=split_csv_values(str(args.labels)),
    )
    if args.safe_batch:
        return run_safe_batch(args, options)
    try:
        result = run(
            args.images,
            args.masks_dir,
            backend=args.backend,
            model_dir=args.model_dir,
            device=args.device,
            options=options,
            resume_state=bool(args.resume_state),
            max_images=int(args.max_images) if int(args.max_images) > 0 else None,
            progress_offset=int(args.progress_offset) if int(args.progress_offset) >= 0 else None,
            progress_total=int(args.progress_total) if int(args.progress_total) > 0 else None,
            image_list=getattr(args, "image_list", None),
        )
    except Exception as e:  # noqa: BLE001 - CLI should report concise errors to the GUI log.
        print(f"Error: {e}", flush=True)
        return 1
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
