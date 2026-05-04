"""Virtual thumbnail model for mask review."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode
from overexposure_mask import read_image_preserve_depth
from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap

from gui import theme
from gui.mask.mask_files import mask_candidates_for_image

_THUMB_CACHE_LIMIT = 256
_DEFAULT_THUMB_SIZE = QSize(176, 88)
_DEFAULT_GRID_SIZE = QSize(196, 124)


@dataclass(frozen=True)
class ThumbnailRenderConfig:
    images_dir: str = ""
    masks_dir: str = ""
    opacity: int = 45
    icon_size: QSize = _DEFAULT_THUMB_SIZE


class _ThumbnailSignals(QObject):
    ready = Signal(int, int, tuple, object)


class _ThumbnailJob(QRunnable):
    def __init__(
        self,
        row: int,
        generation: int,
        key: tuple,
        image_path: Path,
        config: ThumbnailRenderConfig,
    ) -> None:
        super().__init__()
        self.row = row
        self.generation = generation
        self.key = key
        self.image_path = image_path
        self.config = config
        self.signals = _ThumbnailSignals()

    def run(self) -> None:
        image = render_mask_thumbnail(self.image_path, self.config)
        self.signals.ready.emit(self.row, self.generation, self.key, image)


class MaskThumbnailModel(QAbstractListModel):
    """Lazy QAbstractListModel backing a QListView IconMode thumbnail grid."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._images: list[Path] = []
        self._config = ThumbnailRenderConfig()
        self._cache: OrderedDict[tuple, QIcon] = OrderedDict()
        self._pending: set[tuple[int, tuple]] = set()
        self._jobs: dict[tuple[int, tuple], _ThumbnailJob] = {}
        self._generation = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, min(2, QThreadPool.globalInstance().maxThreadCount())))
        self._placeholder = _placeholder_icon(_DEFAULT_THUMB_SIZE)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid():
            return 0
        return len(self._images)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN001, N802
        if not index.isValid() or not (0 <= index.row() < len(self._images)):
            return None

        path = self._images[index.row()]
        if role == Qt.DisplayRole:
            return path.name
        if role == Qt.ToolTipRole:
            return str(path)
        if role == Qt.DecorationRole:
            key = self._thumbnail_key(path)
            icon = self._cache.get(key)
            if icon is not None:
                self._cache.move_to_end(key)
                return icon
            self._schedule_thumbnail(index.row(), path, key)
            return self._placeholder
        return None

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
            icon_size=_DEFAULT_THUMB_SIZE,
        )
        if not force and images == self._images and config == self._config:
            return
        self.beginResetModel()
        self._images = list(images)
        self._config = config
        self._cache.clear()
        self._pending.clear()
        self._generation += 1
        self._placeholder = _placeholder_icon(config.icon_size)
        self.endResetModel()

    def image_at(self, row: int) -> Path | None:
        if not (0 <= row < len(self._images)):
            return None
        return self._images[row]

    @staticmethod
    def icon_size() -> QSize:
        return QSize(_DEFAULT_THUMB_SIZE)

    @staticmethod
    def grid_size() -> QSize:
        return QSize(_DEFAULT_GRID_SIZE)

    def _thumbnail_key(self, image_path: Path) -> tuple:
        image_sig = _file_signature(image_path)
        mask_sig = _first_mask_signature(image_path, self._config)
        return (
            image_sig,
            mask_sig,
            int(self._config.opacity),
            int(self._config.icon_size.width()),
            int(self._config.icon_size.height()),
        )

    def _schedule_thumbnail(self, row: int, path: Path, key: tuple) -> None:
        pending_key = (self._generation, key)
        if pending_key in self._pending:
            return
        self._pending.add(pending_key)
        job = _ThumbnailJob(row, self._generation, key, path, self._config)
        job.signals.ready.connect(self._on_thumbnail_ready)
        self._jobs[pending_key] = job
        self._pool.start(job)

    def _on_thumbnail_ready(self, row: int, generation: int, key: tuple, image: object) -> None:
        pending_key = (generation, key)
        self._pending.discard(pending_key)
        self._jobs.pop(pending_key, None)
        if generation != self._generation:
            return
        if not isinstance(image, QImage) or image.isNull():
            icon = self._placeholder
        else:
            icon = QIcon(QPixmap.fromImage(image))
        self._cache[key] = icon
        self._cache.move_to_end(key)
        while len(self._cache) > _THUMB_CACHE_LIMIT:
            self._cache.popitem(last=False)
        if 0 <= row < len(self._images):
            model_index = self.index(row, 0)
            self.dataChanged.emit(model_index, model_index, [Qt.DecorationRole])


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


def _placeholder_icon(size: QSize) -> QIcon:
    image = QImage(size, QImage.Format_ARGB32)
    image.fill(QColor(theme.BG_INPUT))
    painter = QPainter(image)
    painter.setPen(QPen(QColor(theme.BORDER), 1))
    painter.drawRect(image.rect().adjusted(0, 0, -1, -1))
    painter.end()
    return QIcon(QPixmap.fromImage(image))


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
