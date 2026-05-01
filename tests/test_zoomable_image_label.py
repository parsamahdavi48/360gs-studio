import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPixmap
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
