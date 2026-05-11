"""Printable AprilTag target generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.apriltag_markers import (
    DEFAULT_APRILTAG_FAMILY,
    DEFAULT_APRILTAG_ID,
    DEFAULT_APRILTAG_SIZE_M,
    marker_image_array,
)

_PAGE_SIZES_MM = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "Letter": (215.9, 279.4),
}


@dataclass(frozen=True)
class PrintableTarget:
    marker_png: Path
    page_png: Path
    page_pdf: Path
    spec_json: Path
    page: str
    page_pixels: tuple[int, int]
    marker_pixels: tuple[int, int]

    @property
    def a4_png(self) -> Path:
        """Backward-compatible alias for dev tools tests."""
        return self.page_png

    @property
    def a4_pdf(self) -> Path:
        """Backward-compatible alias for dev tools tests."""
        return self.page_pdf


def available_pages() -> tuple[str, ...]:
    return tuple(_PAGE_SIZES_MM)


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


def _draw_crop_marks(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, *, dpi: int) -> None:
    mark_len = max(24, round(6.0 / 25.4 * dpi))
    gap = max(10, round(2.5 / 25.4 * dpi))
    width = max(1, round(0.35 / 25.4 * dpi))
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


def _page_size_pixels(page: str, dpi: int) -> tuple[int, int, float, float]:
    try:
        page_w_mm, page_h_mm = _PAGE_SIZES_MM[page]
    except KeyError as exc:
        raise ValueError(f"Unsupported page size: {page}") from exc
    return (
        round(page_w_mm / 25.4 * dpi),
        round(page_h_mm / 25.4 * dpi),
        page_w_mm,
        page_h_mm,
    )


def _size_label_mm(tag_size_mm: float) -> str:
    return f"{tag_size_mm:.0f}mm" if abs(tag_size_mm - round(tag_size_mm)) < 1e-6 else f"{tag_size_mm:.1f}mm"


def create_printable_target(
    output_dir: Path,
    *,
    family: str = DEFAULT_APRILTAG_FAMILY,
    tag_id: int = DEFAULT_APRILTAG_ID,
    tag_size_m: float = DEFAULT_APRILTAG_SIZE_M,
    page: str = "A4",
    dpi: int = 300,
) -> PrintableTarget:
    if tag_size_m <= 0.0:
        raise ValueError("tag_size_m must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    page_w_px, page_h_px, page_w_mm, page_h_mm = _page_size_pixels(page, dpi)
    tag_size_mm = float(tag_size_m) * 1000.0
    tag_px = round(tag_size_mm / 25.4 * dpi)
    margin_px = round(14.0 / 25.4 * dpi)
    text_block_px = round(34.0 / 25.4 * dpi)
    max_tag_px = min(page_w_px - margin_px * 2, page_h_px - margin_px * 2 - text_block_px)
    if tag_px <= 0 or tag_px > max_tag_px:
        raise ValueError(f"tag_size_m is too large for a {page} target at the requested DPI")

    marker = marker_image_array(family, tag_id, int(tag_px))
    marker_image = Image.fromarray(marker).convert("RGB")
    size_label = _size_label_mm(tag_size_mm)
    stem = f"apriltag_{family}_id{int(tag_id)}_{size_label}_{page}_{int(dpi)}dpi"
    marker_png = output_dir / f"{stem}_marker.png"
    page_png = output_dir / f"{stem}.png"
    page_pdf = output_dir / f"{stem}.pdf"
    spec_json = output_dir / f"{stem}_spec.json"
    marker_image.save(marker_png, dpi=(dpi, dpi))

    page_image = Image.new("RGB", (page_w_px, page_h_px), "white")
    x = (page_w_px - tag_px) // 2
    y = max(260, int(page_h_px * 0.12))
    if y + tag_px + text_block_px > page_h_px - margin_px:
        y = max(margin_px, page_h_px - margin_px - text_block_px - tag_px)
    page_image.paste(marker_image, (x, y))
    draw = ImageDraw.Draw(page_image)
    _draw_crop_marks(draw, x, y, tag_px, dpi=dpi)

    text_y = y + tag_px + round(12.0 / 25.4 * dpi)
    text_y = _centered_text(draw, text_y, f"AprilTag {family} / ID {int(tag_id)}", _font(42), page_w_px)
    text_y = _centered_text(
        draw,
        text_y,
        f"Detected tag square: {tag_size_mm:.1f} mm  (tag_size_m = {tag_size_m:.3f})",
        _font(30),
        page_w_px,
    )
    _centered_text(draw, text_y, f"{page} {dpi} DPI. Print at actual size / 100%.", _font(30), page_w_px)
    page_image.save(page_png, dpi=(dpi, dpi))
    page_image.save(page_pdf, "PDF", resolution=dpi)

    spec = {
        "schema_version": 1,
        "family": family,
        "tag_id": int(tag_id),
        "tag_size_m": float(tag_size_m),
        "detected_tag_square_mm": tag_size_mm,
        "dpi": int(dpi),
        "page": page,
        "page_size_mm": [page_w_mm, page_h_mm],
        "page_pixels": [page_w_px, page_h_px],
        "marker_pixels": [tag_px, tag_px],
        "print_scaling": "actual size / 100%",
        "note": "PnP tag size is the black detected marker square, not the whole paper page.",
    }
    spec_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return PrintableTarget(
        marker_png=marker_png,
        page_png=page_png,
        page_pdf=page_pdf,
        spec_json=spec_json,
        page=page,
        page_pixels=(page_w_px, page_h_px),
        marker_pixels=(tag_px, tag_px),
    )
