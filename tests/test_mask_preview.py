import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui import i18n
from gui.mask.mask_preview import MaskPreviewConfig, MaskPreviewWidget
from gui.steps.step3_mask import MaskStep, _yolo_preview_output_name


def _app():
    return QApplication.instance() or QApplication([])


def test_yolo_preview_output_name_matches_yolo_mask_script() -> None:
    image = Path("frame_000001.jpg")

    assert _yolo_preview_output_name(image, add_ext=False) == "frame_000001.png"
    assert _yolo_preview_output_name(image, add_ext=True) == "frame_000001.jpg.png"


def test_mask_preview_uses_temporary_yolo_preview_mask(tmp_path: Path) -> None:
    _app()
    image_path = tmp_path / "frame_000001.png"
    mask_path = tmp_path / "preview_mask.png"
    cv2.imwrite(str(image_path), np.full((32, 64, 3), 180, dtype=np.uint8))
    mask = np.full((32, 64), 255, dtype=np.uint8)
    mask[:, :32] = 0
    cv2.imwrite(str(mask_path), mask)

    widget = MaskPreviewWidget()
    widget.sample_edit.setText(str(image_path))

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
    widget.sample_edit.setText(str(image_path))

    widget.render(MaskPreviewConfig(use_yolo=False, use_overexposure=True))

    assert i18n.t("MASK_PREVIEW_OVEREXP_STATUS").format(threshold=254, dilate=1) in widget.status_label.text()
    assert widget.image_label._source_pixmap is not None


def test_mask_step_uses_conservative_manual_yolo_expand_by_default() -> None:
    _app()
    step = MaskStep(Path.cwd())

    assert step._yolo_expand_arg() == "2"
    assert step.yolo_expand_edit.isEnabled()

    step.yolo_expand_edit.setValue(7)

    assert step._yolo_expand_arg() == "7"
