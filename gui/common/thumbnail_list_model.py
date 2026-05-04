"""Reusable async thumbnail list model for preview grids."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from PySide6.QtCore import QPoint, QAbstractListModel, QModelIndex, QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap

from gui import theme

DEFAULT_THUMB_SIZE = QSize(176, 88)
DEFAULT_GRID_SIZE = QSize(196, 124)
THUMB_CACHE_MIN_ENTRIES = 512
THUMB_CACHE_MAX_ENTRIES = 10_000
THUMB_CACHE_MEMORY_BUDGET_BYTES = 1024 * 1024 * 1024
SHUTDOWN_WAIT_MS = 3000


@dataclass(frozen=True)
class ThumbnailItem:
    path: Path
    label: str
    tooltip: str = ""
    cache_key: tuple = ()


ThumbnailRenderer = Callable[[ThumbnailItem, QSize], QImage]


class _ThumbnailSignals(QObject):
    ready = Signal(int, int, tuple, object)


class _ThumbnailJob(QRunnable):
    def __init__(
        self,
        row: int,
        generation: int,
        key: tuple,
        item: ThumbnailItem,
        renderer: ThumbnailRenderer,
        icon_size: QSize,
        cancel_event: Event,
    ) -> None:
        super().__init__()
        self.row = row
        self.generation = generation
        self.key = key
        self.item = item
        self.renderer = renderer
        self.icon_size = QSize(icon_size)
        self.cancel_event = cancel_event
        self.started = False
        self.signals = _ThumbnailSignals()

    def run(self) -> None:
        self.started = True
        if self.cancel_event.is_set():
            return
        image = self.renderer(self.item, self.icon_size)
        if self.cancel_event.is_set():
            return
        try:
            self.signals.ready.emit(self.row, self.generation, self.key, image)
        except RuntimeError:
            # The application may already be closing and the QObject signal source
            # can be gone before a worker reaches its final emit.
            return


class AsyncThumbnailModel(QAbstractListModel):
    """QListView IconMode model that lazily renders thumbnails off the UI thread."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[ThumbnailItem] = []
        self._renderer: ThumbnailRenderer = _default_renderer
        self._renderer_key: tuple = ()
        self._icon_size = QSize(DEFAULT_THUMB_SIZE)
        self._grid_size = QSize(DEFAULT_GRID_SIZE)
        self._cache: OrderedDict[tuple, QIcon] = OrderedDict()
        self._pending: set[tuple[int, tuple]] = set()
        self._jobs: dict[tuple[int, tuple], _ThumbnailJob] = {}
        self._generation = 0
        self._cancel_event = Event()
        self._shutdown = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, min(2, QThreadPool.globalInstance().maxThreadCount())))
        self._placeholder = _placeholder_icon(self._icon_size)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: ANN001, N802
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None

        item = self._items[index.row()]
        if role == Qt.DisplayRole:
            return item.label
        if role == Qt.ToolTipRole:
            return item.tooltip or str(item.path)
        if role == Qt.DecorationRole:
            key = self._thumbnail_key(item)
            icon = self._cache.get(key)
            if icon is not None:
                self._cache.move_to_end(key)
                return icon
            if self._shutdown:
                return self._placeholder
            self._schedule_thumbnail(index.row(), item, key)
            return self._placeholder
        return None

    def set_items(
        self,
        items: list[ThumbnailItem],
        renderer: ThumbnailRenderer,
        *,
        renderer_key: tuple = (),
        force: bool = False,
        icon_size: QSize = DEFAULT_THUMB_SIZE,
        grid_size: QSize = DEFAULT_GRID_SIZE,
    ) -> None:
        icon_size = QSize(icon_size)
        grid_size = QSize(grid_size)
        if self._shutdown:
            return
        renderer_key = tuple(renderer_key)
        if (
            not force
            and items == self._items
            and renderer_key == self._renderer_key
            and icon_size == self._icon_size
            and grid_size == self._grid_size
        ):
            return
        old_items = self._items
        old_renderer_key = self._renderer_key
        old_icon_size = self._icon_size
        old_grid_size = self._grid_size
        same_paths = (
            len(items) == len(old_items)
            and all(new.path == old.path for new, old in zip(items, old_items))
        )
        reset_model = not same_paths or grid_size != old_grid_size
        self._generation += 1
        self._drop_queued_jobs()

        if reset_model:
            self.beginResetModel()
            self._items = list(items)
            self._renderer = renderer
            self._renderer_key = renderer_key
            self._icon_size = icon_size
            self._grid_size = grid_size
            self._placeholder = _placeholder_icon(self._icon_size)
            self.endResetModel()
            return

        changed_rows = (
            list(range(len(items)))
            if force or renderer_key != old_renderer_key or icon_size != old_icon_size
            else [idx for idx, (old, new) in enumerate(zip(old_items, items)) if old != new]
        )
        self._items = list(items)
        self._renderer = renderer
        self._renderer_key = renderer_key
        self._icon_size = icon_size
        self._grid_size = grid_size
        self._placeholder = _placeholder_icon(self._icon_size)
        self._emit_data_changed_ranges(
            changed_rows,
            [Qt.DisplayRole, Qt.ToolTipRole, Qt.DecorationRole],
        )

    def item_at(self, row: int) -> ThumbnailItem | None:
        if not (0 <= row < len(self._items)):
            return None
        return self._items[row]

    def set_item(self, row: int, item: ThumbnailItem) -> bool:
        if not (0 <= row < len(self._items)):
            return False
        old_item = self._items[row]
        if item == old_item:
            return False

        self._items[row] = item
        model_index = self.index(row, 0)
        self.dataChanged.emit(
            model_index,
            model_index,
            [Qt.DisplayRole, Qt.ToolTipRole, Qt.DecorationRole],
        )
        return True

    def icon_size(self) -> QSize:
        return QSize(self._icon_size)

    def grid_size(self) -> QSize:
        return QSize(self._grid_size)

    def wait_for_done(self, timeout_ms: int = 5000) -> bool:
        return self._pool.waitForDone(timeout_ms)

    def shutdown(self, wait_ms: int = SHUTDOWN_WAIT_MS) -> bool:
        self._shutdown = True
        self._cancel_event.set()
        self._pool.clear()
        self._pending.clear()
        self._jobs.clear()
        return self._pool.waitForDone(max(0, int(wait_ms)))

    def prioritize_rows(self, rows: list[int] | set[int], *, prefetch: int = 128) -> None:
        if self._shutdown or not rows:
            return
        visible_rows = sorted({row for row in rows if 0 <= row < len(self._items)})
        if not visible_rows:
            return

        self._drop_queued_jobs()
        for row in visible_rows:
            self._schedule_row(row, priority=100)

        start = max(0, visible_rows[0] - max(0, int(prefetch)))
        end = min(len(self._items), visible_rows[-1] + max(0, int(prefetch)) + 1)
        for row in range(start, end):
            if row not in visible_rows:
                self._schedule_row(row, priority=10)

    def _thumbnail_key(self, item: ThumbnailItem) -> tuple:
        return (
            _file_signature(item.path),
            tuple(item.cache_key),
            self._renderer_key,
            int(self._icon_size.width()),
            int(self._icon_size.height()),
        )

    def _schedule_row(self, row: int, *, priority: int = 0) -> None:
        if not (0 <= row < len(self._items)):
            return
        item = self._items[row]
        key = self._thumbnail_key(item)
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        self._schedule_thumbnail(row, item, key, priority=priority)

    def _schedule_thumbnail(self, row: int, item: ThumbnailItem, key: tuple, *, priority: int = 0) -> None:
        if self._shutdown:
            return
        pending_key = (self._generation, key)
        if pending_key in self._pending:
            return
        self._pending.add(pending_key)
        job = _ThumbnailJob(row, self._generation, key, item, self._renderer, self._icon_size, self._cancel_event)
        job.signals.ready.connect(self._on_thumbnail_ready)
        self._jobs[pending_key] = job
        self._pool.start(job, priority)

    def _on_thumbnail_ready(self, row: int, generation: int, key: tuple, image: object) -> None:
        if self._shutdown:
            return
        pending_key = (generation, key)
        self._pending.discard(pending_key)
        self._jobs.pop(pending_key, None)
        if generation != self._generation:
            return
        if not isinstance(image, QImage) or image.isNull():
            icon = self._placeholder
        else:
            icon = QIcon(QPixmap.fromImage(image))
        self._store_cache_icon(key, icon)
        if 0 <= row < len(self._items):
            model_index = self.index(row, 0)
            self.dataChanged.emit(model_index, model_index, [Qt.DecorationRole])

    def _store_cache_icon(self, key: tuple, icon: QIcon) -> None:
        self._cache[key] = icon
        self._cache.move_to_end(key)
        self._prune_cache()

    def _prune_cache(self) -> None:
        approx_bytes = max(1, int(self._icon_size.width()) * int(self._icon_size.height()) * 4)
        memory_limit = max(THUMB_CACHE_MIN_ENTRIES, THUMB_CACHE_MEMORY_BUDGET_BYTES // approx_bytes)
        limit = max(THUMB_CACHE_MIN_ENTRIES, min(THUMB_CACHE_MAX_ENTRIES, int(memory_limit)))
        while len(self._cache) > limit:
            self._cache.popitem(last=False)

    def _cancel_pending_jobs(self) -> None:
        self._cancel_event.set()
        self._pool.clear()
        self._pending.clear()
        self._jobs.clear()
        self._cancel_event = Event()

    def _drop_queued_jobs(self) -> None:
        self._pool.clear()
        for pending_key, job in list(self._jobs.items()):
            if job.started:
                continue
            self._pending.discard(pending_key)
            self._jobs.pop(pending_key, None)

    def _emit_data_changed_ranges(self, rows: list[int], roles: list[int]) -> None:
        if not rows:
            return
        rows = sorted(set(rows))
        start = rows[0]
        prev = rows[0]
        for row in rows[1:]:
            if row == prev + 1:
                prev = row
                continue
            self.dataChanged.emit(self.index(start, 0), self.index(prev, 0), roles)
            start = prev = row
        self.dataChanged.emit(self.index(start, 0), self.index(prev, 0), roles)


def _default_renderer(_item: ThumbnailItem, size: QSize) -> QImage:
    return _error_image(size)


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


def visible_rows_for_view(view) -> list[int]:  # noqa: ANN001
    model = view.model()
    if model is None or model.rowCount() <= 0:
        return []

    viewport = view.viewport()
    rect = viewport.rect()
    grid_size = view.gridSize()
    step_x = max(1, int(grid_size.width()) or DEFAULT_GRID_SIZE.width())
    step_y = max(1, int(grid_size.height()) or DEFAULT_GRID_SIZE.height())
    rows: set[int] = set()

    xs = list(range(rect.left() + step_x // 2, rect.right() + 1, step_x))
    ys = list(range(rect.top() + step_y // 2, rect.bottom() + 1, step_y))
    xs.extend([rect.left() + 2, max(rect.left(), rect.right() - 2)])
    ys.extend([rect.top() + 2, max(rect.top(), rect.bottom() - 2)])

    for y in ys:
        for x in xs:
            index = view.indexAt(QPoint(x, y))
            if index.isValid():
                rows.add(index.row())

    current = view.currentIndex()
    if current.isValid():
        rows.add(current.row())
    return sorted(row for row in rows if 0 <= row < model.rowCount())
