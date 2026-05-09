"""Printable AprilTag target generation for synthetic and physical validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FAMILY_TO_DICT = {
    "tag16h5": "DICT_APRILTAG_16H5",
    "tag25h9": "DICT_APRILTAG_25H9",
    "tag36h10": "DICT_APRILTAG_36H10",
    "tag36h11": "DICT_APRILTAG_36H11",
}


@dataclass(frozen=True)
class PrintableTarget:
    marker_png: Path
    a4_png: Path
    a4_pdf: Path
    spec_json: Path
    page_pixels: tuple[int, int]
    marker_pixels: tuple[int, int]


def _dictionary_for_family(family: str) -> cv2.aruco.Dictionary:
    key = family.strip().lower()
    attr = _FAMILY_TO_DICT.get(key)
    if attr is None:
        raise ValueError(f"Unsupported AprilTag family: {family}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, attr))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothR.ttc", "C:/Windows/Fonts/msgothic.ttc"):
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _centered_text(draw: ImageDraw.ImageDraw, y: int, text: str, font: ImageFont.ImageFont, width: int) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    x = (width - (box[2] - box[0])) // 2
    draw.text((x, y), text, fill="black", font=font)
    return y + (box[3] - box[1]) + 18


def _draw_crop_marks(draw: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    mark_len = 72
    gap = 28
    width = 4
    left = x - gap
    right = x + size + gap
    top = y - gap
    bottom = y + size + gap

    draw.line([(left - mark_len, top), (left, top)], fill="black", width=width)
    draw.line([(left, top - mark_len), (left, top)], fill="black", width=width)

    draw.line([(right, top), (right + mark_len, top)], fill="black", width=width)
    draw.line([(right, top - mark_len), (right, top)], fill="black", width=width)

    draw.line([(left - mark_len, bottom), (left, bottom)], fill="black", width=width)
    draw.line([(left, bottom), (left, bottom + mark_len)], fill="black", width=width)

    draw.line([(right, bottom), (right + mark_len, bottom)], fill="black", width=width)
    draw.line([(right, bottom), (right, bottom + mark_len)], fill="black", width=width)


def create_printable_target(
    output_dir: Path,
    *,
    family: str = "tag36h11",
    tag_id: int = 7,
    tag_size_m: float = 0.160,
    dpi: int = 300,
) -> PrintableTarget:
    if tag_size_m <= 0.0:
        raise ValueError("tag_size_m must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    tag_size_mm = float(tag_size_m) * 1000.0
    page_w_mm = 210.0
    page_h_mm = 297.0
    page_w_px = round(page_w_mm / 25.4 * dpi)
    page_h_px = round(page_h_mm / 25.4 * dpi)
    tag_px = round(tag_size_mm / 25.4 * dpi)
    if tag_px >= min(page_w_px, page_h_px):
        raise ValueError("tag_size_m is too large for an A4 target at the requested DPI")

    dictionary = _dictionary_for_family(family)
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), int(tag_px))
    marker_image = Image.fromarray(marker).convert("RGB")
    size_label = f"{tag_size_mm:.0f}mm" if abs(tag_size_mm - round(tag_size_mm)) < 1e-6 else f"{tag_size_mm:.1f}mm"
    stem = f"apriltag_{family}_id{int(tag_id)}_{size_label}_{int(dpi)}dpi"
    marker_png = output_dir / f"{stem}_marker.png"
    a4_png = output_dir / f"{stem}_A4.png"
    a4_pdf = output_dir / f"{stem}_A4.pdf"
    spec_json = output_dir / f"{stem}_spec.json"
    marker_image.save(marker_png, dpi=(dpi, dpi))

    page = Image.new("RGB", (page_w_px, page_h_px), "white")
    x = (page_w_px - tag_px) // 2
    y = max(260, int(page_h_px * 0.12))
    page.paste(marker_image, (x, y))
    draw = ImageDraw.Draw(page)
    _draw_crop_marks(draw, x, y, tag_px)

    text_y = y + tag_px + 140
    text_y = _centered_text(draw, text_y, f"AprilTag {family} / ID {int(tag_id)}", _font(42), page_w_px)
    text_y = _centered_text(
        draw,
        text_y,
        f"Detected tag square: {tag_size_mm:.1f} mm  (use tag_size_m = {tag_size_m:.3f})",
        _font(30),
        page_w_px,
    )
    _centered_text(draw, text_y, f"A4 {dpi} DPI. Print at actual size / 100%.", _font(30), page_w_px)
    page.save(a4_png, dpi=(dpi, dpi))
    page.save(a4_pdf, "PDF", resolution=dpi)

    spec = {
        "schema_version": 1,
        "family": family,
        "tag_id": int(tag_id),
        "tag_size_m": float(tag_size_m),
        "detected_tag_square_mm": tag_size_mm,
        "dpi": int(dpi),
        "page": "A4 portrait",
        "page_pixels": [page_w_px, page_h_px],
        "marker_pixels": [tag_px, tag_px],
        "print_scaling": "actual size / 100%",
        "note": "PnP tag size is the black detected marker square, not the whole A4 page.",
    }
    spec_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return PrintableTarget(
        marker_png=marker_png,
        a4_png=a4_png,
        a4_pdf=a4_pdf,
        spec_json=spec_json,
        page_pixels=(page_w_px, page_h_px),
        marker_pixels=(tag_px, tag_px),
    )


def marker_image_array(family: str, tag_id: int, size_px: int) -> np.ndarray:
    dictionary = _dictionary_for_family(family)
    return cv2.aruco.generateImageMarker(dictionary, int(tag_id), int(size_px))
