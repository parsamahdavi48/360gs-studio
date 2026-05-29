from __future__ import annotations

import os

import cv2
import numpy as np

from core.image_io import imread_unicode, imwrite_unicode

RAW_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
ALPHA_CAPABLE_EXTS = {".png", ".tif", ".tiff", ".webp"}
HIGH_BIT_EXTS = {".png", ".tif", ".tiff"}


def split_filename_for_output(input_file: str) -> tuple[str, str, str]:
    basename, ext = os.path.splitext(os.path.basename(input_file))
    ext2 = ""
    lower = basename.lower()
    if lower.endswith(tuple(RAW_IMAGE_EXTS)):
        basename, ext2 = os.path.splitext(basename)
    return basename, ext2, ext


def resolve_output_ext(input_ext: str, output_format: str | None) -> str:
    if not output_format or output_format.lower() == "auto":
        ext = input_ext.lower()
        if ext == ".jpeg":
            return ".jpg"
        if ext in RAW_IMAGE_EXTS:
            return ext
        return ".jpg"
    fmt = output_format.lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        return ".jpg"
    if fmt in {"png", "tif", "tiff", "webp", "bmp"}:
        return f".{fmt}"
    raise ValueError(f"Unsupported output format: {output_format}")


def load_equirect(path: str) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Cannot read image: {path}")
    return img


def max_value_for_dtype(dtype: np.dtype) -> int:
    if dtype == np.uint16:
        return 65535
    return 255


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        max_value = np.iinfo(arr.dtype).max
        if max_value <= 0:
            return arr.astype(np.uint8)
        return np.clip(np.rint(arr.astype(np.float64) * 255.0 / max_value), 0, 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.nanmax(finite)) <= 1.0:
            finite = finite * 255.0
        return np.clip(np.rint(finite), 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def remap_with_channels(
    arr: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    alpha_interpolation: int | None = None,
) -> np.ndarray:
    alpha_interpolation = interpolation if alpha_interpolation is None else alpha_interpolation
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[..., :3])
        alpha = np.ascontiguousarray(arr[..., 3])
        remapped_color = cv2.remap(color, map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_WRAP)
        remapped_alpha = cv2.remap(
            alpha,
            map_x,
            map_y,
            interpolation=alpha_interpolation,
            borderMode=cv2.BORDER_WRAP,
        )
        return np.dstack([remapped_color, remapped_alpha])
    return cv2.remap(arr, map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_WRAP)


def save_image(arr: np.ndarray, path: str, jpg_quality: int = 95, force_8bit: bool = False) -> None:
    ext = os.path.splitext(path)[1].lower()
    out = arr

    if force_8bit:
        out = to_uint8_image(out)

    if ext not in ALPHA_CAPABLE_EXTS and out.ndim == 3 and out.shape[2] == 4:
        out = out[..., :3]

    if ext not in HIGH_BIT_EXTS and out.dtype != np.uint8:
        out = to_uint8_image(out)

    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    elif ext == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(jpg_quality)]
    else:
        params = []

    ok = imwrite_unicode(path, out, params)
    if not ok:
        raise OSError(f"Failed to write image: {path}")
