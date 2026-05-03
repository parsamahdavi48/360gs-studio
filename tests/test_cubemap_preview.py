import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QSpinBox

from gui import i18n
from gui.cubemap.preview_renderer import PreviewWidget, _overlay_draw_order


def _app():
    return QApplication.instance() or QApplication([])


def test_cubemap_preview_removes_manual_image_and_mask_controls() -> None:
    _app()
    widget = PreviewWidget()

    button_texts = {button.text() for button in widget.findChildren(QPushButton)}

    assert i18n.BROWSE not in button_texts
    assert i18n.t("AUTO") not in button_texts
    assert i18n.t("RELOAD") not in button_texts
    assert i18n.t("CLEAR") not in button_texts
    assert not widget.findChildren(QLineEdit)
    assert not widget.findChildren(QSpinBox)


def test_cubemap_preview_timeline_label_keeps_current_filename(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "frame_000001.png"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    widget = PreviewWidget()

    widget.set_scene_dir(str(tmp_path))

    assert widget.current_image_path() == image_path
    assert widget.tl_label.text() == i18n.t("PREVIEW_IMAGE_POSITION_FORMAT").format(
        seq=1,
        total=1,
        name=image_path.name,
    )


def test_cubemap_preview_resolves_mask_from_mask_folder(tmp_path: Path) -> None:
    _app()
    image_path = tmp_path / "frame_000001.jpg"
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    mask_path = mask_dir / "frame_000001.png"
    image_path.write_bytes(b"image")
    mask_path.write_bytes(b"mask")
    widget = PreviewWidget()

    assert widget._resolve_mask(image_path, str(mask_dir)) == mask_path


def test_cubemap_preview_draws_disabled_view_boxes_first() -> None:
    views = [
        {"name": "enabled_a", "enabled": True},
        {"name": "disabled_a", "enabled": False},
        {"name": "enabled_b", "enabled": True},
        {"name": "disabled_b", "enabled": False},
    ]

    ordered = _overlay_draw_order(views)

    assert [view["name"] for view in ordered] == [
        "disabled_a",
        "disabled_b",
        "enabled_a",
        "enabled_b",
    ]
