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
