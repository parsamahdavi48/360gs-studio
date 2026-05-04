import os
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from gui.common.thumbnail_list_model import AsyncThumbnailModel, ThumbnailItem


def _app():
    return QApplication.instance() or QApplication([])


def test_thumbnail_model_shutdown_cancels_running_job_without_late_signal(tmp_path: Path) -> None:
    app = _app()
    model = AsyncThumbnailModel()
    started = Event()
    release = Event()
    changed_rows: list[int] = []

    def renderer(_item: ThumbnailItem, size) -> QImage:  # noqa: ANN001
        started.set()
        assert release.wait(2.0)
        image = QImage(size, QImage.Format_ARGB32)
        image.fill(Qt.red)
        return image

    model.set_items([ThumbnailItem(path=tmp_path / "frame.png", label="frame.png")], renderer)
    model.dataChanged.connect(lambda top_left, _bottom_right, _roles: changed_rows.append(top_left.row()))

    index = model.index(0, 0)
    model.data(index, Qt.DecorationRole)
    assert started.wait(2.0)

    model.shutdown(wait_ms=10)
    release.set()
    assert model.wait_for_done(2000)
    app.processEvents()

    assert changed_rows == []
    assert model._pending == set()
    assert model._jobs == {}


def test_thumbnail_model_reuses_cached_icon_after_forced_resync(tmp_path: Path) -> None:
    app = _app()
    model = AsyncThumbnailModel()
    calls = 0
    item = ThumbnailItem(path=tmp_path / "frame.png", label="frame.png")

    def renderer(_item: ThumbnailItem, size) -> QImage:  # noqa: ANN001
        nonlocal calls
        calls += 1
        image = QImage(size, QImage.Format_ARGB32)
        image.fill(Qt.red)
        return image

    model.set_items([item], renderer)
    model.data(model.index(0, 0), Qt.DecorationRole)
    assert model.wait_for_done(2000)
    app.processEvents()
    assert calls == 1

    model.set_items([item], renderer, force=True)
    model.data(model.index(0, 0), Qt.DecorationRole)
    assert model.wait_for_done(2000)
    app.processEvents()

    assert calls == 1


def test_thumbnail_model_only_rerenders_changed_cache_key(tmp_path: Path) -> None:
    app = _app()
    model = AsyncThumbnailModel()
    calls: list[str] = []
    items = [
        ThumbnailItem(path=tmp_path / "a.png", label="a", cache_key=("keep",)),
        ThumbnailItem(path=tmp_path / "b.png", label="b", cache_key=("keep",)),
    ]

    def renderer(item: ThumbnailItem, size) -> QImage:  # noqa: ANN001
        calls.append(item.label)
        image = QImage(size, QImage.Format_ARGB32)
        image.fill(Qt.red)
        return image

    model.set_items(items, renderer)
    for row in range(2):
        model.data(model.index(row, 0), Qt.DecorationRole)
    assert model.wait_for_done(2000)
    app.processEvents()
    assert calls == ["a", "b"]

    updated = [items[0], ThumbnailItem(path=tmp_path / "b.png", label="b", cache_key=("drop",))]
    model.set_items(updated, renderer)
    for row in range(2):
        model.data(model.index(row, 0), Qt.DecorationRole)
    assert model.wait_for_done(2000)
    app.processEvents()

    assert calls == ["a", "b", "b"]


def test_thumbnail_model_prioritizes_current_rows_over_stale_queue(tmp_path: Path) -> None:
    app = _app()
    model = AsyncThumbnailModel()
    model._pool.setMaxThreadCount(1)
    first_started = Event()
    release_first = Event()
    calls: list[str] = []
    items = [ThumbnailItem(path=tmp_path / f"{idx}.png", label=str(idx)) for idx in range(5)]

    def renderer(item: ThumbnailItem, size) -> QImage:  # noqa: ANN001
        calls.append(item.label)
        if item.label == "0":
            first_started.set()
            assert release_first.wait(2.0)
        image = QImage(size, QImage.Format_ARGB32)
        image.fill(Qt.red)
        return image

    model.set_items(items, renderer)
    for row in range(5):
        model.data(model.index(row, 0), Qt.DecorationRole)
    assert first_started.wait(2.0)

    model.prioritize_rows([4], prefetch=0)
    release_first.set()
    assert model.wait_for_done(2000)
    app.processEvents()

    assert calls == ["0", "4"]
