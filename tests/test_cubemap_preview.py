import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QSpinBox

from gui import i18n
from gui.cubemap.preview_renderer import (
    PreviewWidget,
    _box_overlap_area,
    _layout_view_labels,
    _overlay_draw_order,
    _pitch_color_map,
    _point_inside_view,
)


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


def test_cubemap_preview_draws_highlighted_view_last() -> None:
    views = [
        {"name": "enabled", "enabled": True},
        {"name": "highlighted_disabled", "enabled": False, "highlighted": True},
        {"name": "disabled", "enabled": False},
        {"name": "highlighted_enabled", "enabled": True, "highlighted": True},
    ]

    ordered = _overlay_draw_order(views)

    assert [view["name"] for view in ordered] == [
        "disabled",
        "enabled",
        "highlighted_disabled",
        "highlighted_enabled",
    ]


def test_cubemap_preview_pitch_palette_uses_five_distinct_colors() -> None:
    views = [{"pitch": pitch} for pitch in [-60.0, -30.0, 0.0, 30.0, 60.0]]

    colors = _pitch_color_map(views)

    assert list(colors) == [-60.0, -30.0, 0.0, 30.0, 60.0]
    assert len(set(colors.values())) == 5
    assert colors[-60.0] != colors[-30.0]
    assert colors[-30.0] != colors[0.0]
    assert colors[0.0] != colors[30.0]
    assert colors[30.0] != colors[60.0]


def test_cubemap_preview_label_layout_avoids_overlap_for_full_custom_grid() -> None:
    views = []
    for pitch in [-60.0, -30.0, 0.0, 30.0, 60.0]:
        for slot in range(8):
            views.append({
                "label": f"p{pitch:g}/s{slot}",
                "yaw": slot * 45.0,
                "pitch": pitch,
                "enabled": True,
            })
    colors = _pitch_color_map(views)

    labels = _layout_view_labels(views, 1800, 900, colors)

    assert len(labels) == 40
    for idx, label in enumerate(labels):
        box = label["box"]
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        view = label["view"]
        assert _point_inside_view(center_x, center_y, 1800, 900, view["yaw"], view["pitch"])
        for other in labels[idx + 1:]:
            assert _box_overlap_area(box, other["box"]) == 0


def test_cubemap_preview_label_layout_handles_pole_views_without_overlap() -> None:
    views = [
        {"label": f"top{s}", "yaw": s * 45.0, "pitch": 90.0, "enabled": True}
        for s in range(8)
    ]
    colors = _pitch_color_map(views)

    labels = _layout_view_labels(views, 1800, 900, colors)

    assert len(labels) == 8
    for idx, label in enumerate(labels):
        box = label["box"]
        assert 0 <= box[0] < box[2] < 1800
        assert 0 <= box[1] < box[3] < 900
        for other in labels[idx + 1:]:
            assert _box_overlap_area(box, other["box"]) == 0
