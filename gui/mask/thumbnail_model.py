"""Mask thumbnail rendering plugged into the shared async thumbnail model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode
from overexposure_mask import read_image_preserve_depth
from PySide6.QtCore import QObject, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from gui import theme
from gui.common.thumbnail_list_model import (
    AsyncThumbnailModel,
    DEFAULT_GRID_SIZE,
    DEFAULT_THUMB_SIZE,
    ThumbnailItem,
)
from gui.mask.mask_files import mask_candidates_for_image


@dataclass(frozen=True)
class ThumbnailRenderConfig:
    images_dir: str = ""
    masks_dir: str = ""
    opacity: int = 45
    icon_size: QSize = DEFAULT_THUMB_SIZE


class MaskThumbnailModel(AsyncThumbnailModel):
    """Mask-specific adapter for the shared thumbnail model."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = ThumbnailRenderConfig()

    def set_sources(
        self,
        images: list[Path],
        *,
        images_dir: str,
        masks_dir: str,
        opacity: int,
        force: bool = False,
    ) -> None:
        config = ThumbnailRenderConfig(
            images_dir=images_dir,
            masks_dir=masks_dir,
            opacity=max(0, min(100, int(opacity))),
            icon_size=DEFAULT_THUMB_SIZE,
        )
        self._config = config
        items = [
            ThumbnailItem(
                path=path,
                label=path.name,
                tooltip=str(path),
                cache_key=(_first_mask_signature(path, config), int(config.opacity)),
            )
            for path in images
        ]

        def renderer(item: ThumbnailItem, size: QSize) -> QImage:
            sized_config = ThumbnailRenderConfig(
                images_dir=config.images_dir,
                masks_dir=config.masks_dir,
                opacity=config.opacity,
                icon_size=size,
            )
            return render_mask_thumbnail(item.path, sized_config)

        self.set_items(
            items,
            renderer,
            renderer_key=(config.images_dir, config.masks_dir, int(config.opacity)),
            force=force,
            icon_size=DEFAULT_THUMB_SIZE,
            grid_size=DEFAULT_GRID_SIZE,
        )

    def image_at(self, row: int) -> Path | None:
        item = self.item_at(row)
        return item.path if item is not None else None


def render_mask_thumbnail(image_path: Path, config: ThumbnailRenderConfig) -> QImage:
    """Build a small image+mask overlay thumbnail without touching GUI widgets."""

    source = read_image_preserve_depth(str(image_path))
    display = _display_bgr8(source)
    if display is None:
        return _error_image(config.icon_size)
    display = _fit_for_thumbnail(display, config.icon_size)

    mask = _load_existing_mask(image_path, config, display.shape[:2])
    if mask is not None:
        display = _overlay_excluded_mask(display, mask, config.opacity)

    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
    return qimg.scaled(
        config.icon_size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


def _load_existing_mask(
    image_path: Path,
    config: ThumbnailRenderConfig,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    for candidate in mask_candidates_for_image(image_path, config.images_dir, config.masks_dir):
        if not candidate.is_file():
            continue
        mask = imread_unicode(candidate, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    return None


def _display_bgr8(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.dtype == np.uint8:
        display = image
    elif np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        display = np.clip(np.rint(image.astype(np.float32) * 255.0 / max_value), 0, 255).astype(np.uint8)
    else:
        display = np.clip(image, 0.0, 1.0)
        display = np.rint(display * 255.0).astype(np.uint8)

    if display.ndim == 2:
        return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    if display.ndim == 3 and display.shape[2] == 4:
        return cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
    return display


def _overlay_excluded_mask(image: np.ndarray, mask: np.ndarray, opacity: int) -> np.ndarray:
    alpha = max(0.0, min(1.0, float(opacity) / 100.0))
    if alpha <= 0:
        return image
    excluded = mask < 128
    if not np.any(excluded):
        return image

    out = image.copy()
    overlay = np.zeros_like(out)
    overlay[:, :, 2] = 255
    out[excluded] = (
        out[excluded].astype(np.float32) * (1.0 - alpha)
        + overlay[excluded].astype(np.float32) * alpha
    ).astype(np.uint8)
    return out


def _fit_for_thumbnail(image: np.ndarray, size: QSize) -> np.ndarray:
    max_width = max(1, int(size.width()) * 3)
    max_height = max(1, int(size.height()) * 3)
    h, w = image.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _error_image(size: QSize) -> QImage:
    image = QImage(size, QImage.Format_ARGB32)
    image.fill(QColor(theme.BG_INPUT))
    painter = QPainter(image)
    painter.setPen(QPen(QColor(theme.DANGER), 2))
    painter.drawLine(8, 8, size.width() - 8, size.height() - 8)
    painter.drawLine(size.width() - 8, 8, 8, size.height() - 8)
    painter.end()
    return image


def _file_signature(path: Path) -> tuple:
    try:
        st = path.stat()
        return (str(path.resolve()).lower(), int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return (str(path).lower(), -1, -1)


def _first_mask_signature(image_path: Path, config: ThumbnailRenderConfig) -> tuple:
    for candidate in mask_candidates_for_image(image_path, config.images_dir, config.masks_dir):
        if not candidate.is_file():
            continue
        return _file_signature(candidate)
    return ("", 0, 0)
