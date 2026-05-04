import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QAbstractItemView, QLineEdit, QPushButton, QSpinBox
from PySide6.QtCore import QItemSelectionModel

from gui import i18n
from gui.mask.mask_preview import MaskPreviewConfig, MaskPreviewWidget
from gui.mask.thumbnail_model import ThumbnailRenderConfig, render_mask_thumbnail
from gui.steps.step3_mask import MaskStep, _yolo_preview_output_name


def _app():
    return QApplication.instance() or QApplication([])


def test_yolo_preview_output_name_matches_yolo_mask_script() -> None:
    image = Path("frame_000001.jpg")

    assert _yolo_preview_output_name(image) == "frame_000001.png"


def test_mask_preview_removes_manual_image_picker_and_opacity_spinbox() -> None:
    _app()
    widget = MaskPreviewWidget()

    button_texts = {button.text() for button in widget.findChildren(QPushButton)}

    assert i18n.BROWSE not in button_texts
    assert i18n.t("AUTO") not in button_texts
    assert i18n.t("RELOAD") not in button_texts
    assert i18n.t("YOLO_PREVIEW_BUTTON") in button_texts
    assert i18n.t("MASK_REPROCESS_CURRENT_BUTTON") in button_texts
    assert not widget.findChildren(QLineEdit)
    assert not widget.findChildren(QSpinBox)


def test_mask_preview_uses_temporary_yolo_preview_mask(tmp_path: Path) -> None:
    _app()
    image_path = tmp_path / "frame_000001.png"
    mask_path = tmp_path / "preview_mask.png"
    cv2.imwrite(str(image_path), np.full((32, 64, 3), 180, dtype=np.uint8))
    mask = np.full((32, 64), 255, dtype=np.uint8)
    mask[:, :32] = 0
    cv2.imwrite(str(mask_path), mask)

    widget = MaskPreviewWidget()
    widget.set_current_image_path(image_path)

    assert widget.set_yolo_preview_mask(image_path, mask_path)
    widget.render(MaskPreviewConfig(use_yolo=True))

    assert i18n.t("MASK_PREVIEW_YOLO_TEMP") in widget.status_label.text()
    assert widget.image_label._source_pixmap is not None


def test_mask_preview_resizes_overexposure_mask_for_large_preview(tmp_path: Path) -> None:
    _app()
    image_path = tmp_path / "large_frame.png"
    image = np.full((1200, 2400, 3), 180, dtype=np.uint8)
    image[100:180, 200:280] = 255
    cv2.imwrite(str(image_path), image)

    widget = MaskPreviewWidget()
    widget.set_current_image_path(image_path)

    widget.render(MaskPreviewConfig(use_yolo=False, use_overexposure=True))

    assert i18n.t("MASK_PREVIEW_OVEREXP_STATUS").format(threshold=254, dilate=1) in widget.status_label.text()
    assert widget.image_label._source_pixmap is not None


def test_mask_preview_does_not_scan_cwd_without_images_dir(tmp_path: Path, monkeypatch) -> None:
    _app()
    monkeypatch.chdir(tmp_path)
    cv2.imwrite(str(tmp_path / "frame_000001.png"), np.full((16, 32, 3), 180, dtype=np.uint8))
    widget = MaskPreviewWidget()

    widget.refresh_image_list(prefer_current=False)

    assert widget.preview_images == []
    assert widget.current_image_path() is None


def test_mask_preview_timeline_label_shows_current_filename(tmp_path: Path) -> None:
    _app()
    image_path = tmp_path / "frame_000001.png"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    widget = MaskPreviewWidget()

    widget.set_images_dir(str(tmp_path))

    assert widget.timeline_label.text() == i18n.t("PREVIEW_IMAGE_POSITION_FORMAT").format(
        seq=1,
        total=1,
        name=image_path.name,
    )


def test_mask_preview_thumbnail_mode_tracks_images_and_selection(tmp_path: Path) -> None:
    _app()
    image_paths: list[Path] = []
    for idx in range(3):
        image_path = tmp_path / f"frame_{idx:06d}.png"
        cv2.imwrite(str(image_path), np.full((16, 32, 3), 120 + idx, dtype=np.uint8))
        image_paths.append(image_path)
    widget = MaskPreviewWidget()

    widget.set_images_dir(str(tmp_path))
    widget.set_preview_mode("thumbnails")
    widget.thumbnail_view.setCurrentIndex(widget.thumbnail_model.index(2, 0))

    assert widget.preview_mode() == "thumbnails"
    assert widget.thumbnail_model.rowCount() == 3
    assert widget.current_image_path() == image_paths[2]
    assert widget.slider.value() == 2
    assert widget.timeline_label.text() == i18n.t("PREVIEW_IMAGE_POSITION_FORMAT").format(
        seq=3,
        total=3,
        name=image_paths[2].name,
    )


def test_mask_preview_thumbnail_click_does_not_scroll_resync(tmp_path: Path) -> None:
    _app()
    image_paths: list[Path] = []
    for idx in range(4):
        image_path = tmp_path / f"frame_{idx:06d}.png"
        cv2.imwrite(str(image_path), np.full((16, 32, 3), 120 + idx, dtype=np.uint8))
        image_paths.append(image_path)
    widget = MaskPreviewWidget()

    widget.set_images_dir(str(tmp_path))
    widget.set_preview_mode("thumbnails")
    sync_calls: list[int] = []

    def fake_sync(idx: int, *, scroll: bool = False) -> None:
        sync_calls.append(idx)

    widget._sync_thumbnail_selection = fake_sync  # type: ignore[method-assign]

    widget.thumbnail_view.setCurrentIndex(widget.thumbnail_model.index(2, 0))

    assert sync_calls == []
    assert widget.current_image_path() == image_paths[2]
    assert widget.slider.value() == 2


def test_mask_preview_thumbnail_mode_uses_extended_selection(tmp_path: Path) -> None:
    _app()
    image_paths: list[Path] = []
    for idx in range(4):
        image_path = tmp_path / f"frame_{idx:06d}.png"
        cv2.imwrite(str(image_path), np.full((16, 32, 3), 120, dtype=np.uint8))
        image_paths.append(image_path)
    widget = MaskPreviewWidget()

    widget.set_images_dir(str(tmp_path))
    widget.set_preview_mode("thumbnails")
    selection = widget.thumbnail_view.selectionModel()
    selection.select(widget.thumbnail_model.index(1, 0), QItemSelectionModel.ClearAndSelect)
    selection.select(widget.thumbnail_model.index(3, 0), QItemSelectionModel.Select)

    assert widget.thumbnail_view.selectionMode() == QAbstractItemView.ExtendedSelection
    assert widget.selected_reprocess_image_paths() == [image_paths[1], image_paths[3]]
    assert widget.reprocess_current_btn.text() == i18n.t("MASK_REPROCESS_SELECTED_BUTTON").format(count=2)


def test_mask_preview_thumbnail_render_preserves_multi_selection(tmp_path: Path) -> None:
    _app()
    image_paths: list[Path] = []
    for idx in range(4):
        image_path = tmp_path / f"frame_{idx:06d}.png"
        cv2.imwrite(str(image_path), np.full((16, 32, 3), 120, dtype=np.uint8))
        image_paths.append(image_path)
    widget = MaskPreviewWidget()

    widget.set_images_dir(str(tmp_path))
    widget.set_preview_mode("thumbnails")
    selection = widget.thumbnail_view.selectionModel()
    selection.select(widget.thumbnail_model.index(1, 0), QItemSelectionModel.ClearAndSelect)
    selection.select(widget.thumbnail_model.index(3, 0), QItemSelectionModel.Select)

    widget.render(MaskPreviewConfig())

    assert widget.selected_reprocess_image_paths() == [image_paths[1], image_paths[3]]


def test_mask_preview_thumbnail_mode_skips_large_detail_render(tmp_path: Path) -> None:
    _app()
    image_path = tmp_path / "frame_000001.png"
    cv2.imwrite(str(image_path), np.full((16, 32, 3), 180, dtype=np.uint8))
    widget = MaskPreviewWidget()

    widget.set_images_dir(str(tmp_path))
    widget.set_preview_mode("thumbnails")
    widget.render(MaskPreviewConfig(use_yolo=True))

    assert widget.image_label._source_pixmap is None
    assert widget.status_label.text() == i18n.t("MASK_PREVIEW_THUMBNAIL_STATUS").format(count=1)


def test_render_mask_thumbnail_overlays_existing_mask(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "frame_000001.png"
    mask_path = masks / "frame_000001.png"
    cv2.imwrite(str(image_path), np.full((40, 80, 3), 180, dtype=np.uint8))
    mask = np.full((40, 80), 255, dtype=np.uint8)
    mask[:, :40] = 0
    cv2.imwrite(str(mask_path), mask)

    thumb = render_mask_thumbnail(
        image_path,
        ThumbnailRenderConfig(images_dir=str(images), masks_dir=str(masks), opacity=60),
    )

    excluded_pixel = thumb.pixelColor(20, thumb.height() // 2)
    kept_pixel = thumb.pixelColor(thumb.width() - 20, thumb.height() // 2)
    assert excluded_pixel.red() > kept_pixel.red()
    assert excluded_pixel.green() < kept_pixel.green()


def test_mask_step_uses_conservative_manual_yolo_expand_by_default() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step._yolo_expand_arg() == "2"
    assert step.yolo_expand_edit.isEnabled()

    step.yolo_expand_edit.setValue(7)

    assert step._yolo_expand_arg() == "7"
