import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui.common.zoomable_image_label import ZoomableImageLabel


def _app():
    return QApplication.instance() or QApplication([])


def test_zoomable_image_label_accepts_source_pixmap() -> None:
    _app()
    label = ZoomableImageLabel("empty")
    pixmap = QPixmap(320, 160)

    label.set_source_pixmap(pixmap)

    assert label._source_pixmap is pixmap
    assert label.text() == ""


def test_zoomable_image_label_text_clears_source_pixmap() -> None:
    _app()
    label = ZoomableImageLabel()
    label.set_source_pixmap(QPixmap(320, 160))

    label.setText("No preview")

    assert label._source_pixmap is None
    assert label.text() == "No preview"


def test_zoomable_image_label_look_drag_emits_delta_without_panning() -> None:
    _app()
    label = ZoomableImageLabel()
    label.resize(160, 120)
    label.set_source_pixmap(QPixmap(320, 160))
    label.set_drag_mode("look")
    deltas: list[tuple[float, float]] = []
    label.look_dragged.connect(lambda dx, dy: deltas.append((dx, dy)))

    QTest.mousePress(label, Qt.LeftButton, pos=QPoint(20, 20))
    QTest.mouseMove(label, QPoint(40, 26))
    QTest.mouseRelease(label, Qt.LeftButton, pos=QPoint(40, 26))

    assert deltas
    assert label._pan.x() == 0.0
    assert label._pan.y() == 0.0
