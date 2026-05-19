"""Selected camera image view for scene preview."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QVBoxLayout, QWidget

from core.image_io import imread_unicode
from core.scene_preview import ScenePreviewCamera
from gui import i18n
from gui.common.perspective_image_view import PerspectiveImageView, bgr_to_qimage
from gui.common.perspective_preview import PerspectiveParams, params_from_drag

MASK_OVERLAY_ALPHA = 0.45


class SceneCameraImageView(QWidget):
    """Display the image linked to the selected preview camera."""

    perspective_params_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._camera: ScenePreviewCamera | None = None
        self._perspective_params: PerspectiveParams | None = None
        self._view = PerspectiveImageView(i18n.t("SCENE_PREVIEW_NO_CAMERA"))
        self._view.set_drag_mode("pan")
        self._view.look_dragged.connect(self._on_look_dragged)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def perspective_params(self) -> PerspectiveParams | None:
        return self._perspective_params

    def set_message(self, text: str) -> None:
        self._camera = None
        self._perspective_params = None
        self._view.setText(text)

    def set_camera(
        self,
        camera: ScenePreviewCamera | None,
        *,
        reconstructed_bgr: np.ndarray | None = None,
        perspective_params: PerspectiveParams | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        self._camera = camera
        if camera is None:
            self._perspective_params = None
            self._view.setText(i18n.t("SCENE_PREVIEW_NO_CAMERA"))
            return
        if reconstructed_bgr is not None:
            reconstructed_bgr = apply_mask_overlay(reconstructed_bgr, mask)
            shown = self._set_perspective_image(
                reconstructed_bgr,
                perspective_params or PerspectiveParams(fov_deg=90.0),
            )
            if shown:
                return
            self._view.setText(i18n.t("SCENE_PREVIEW_CUBEMAP_VIEW_FAILED"))
            return
        if camera.image_path is None:
            self._perspective_params = None
            self._view.setText(i18n.t("SCENE_PREVIEW_CAMERA_IMAGE_MISSING"))
            return
        image = imread_unicode(camera.image_path)
        if image is None:
            self._perspective_params = None
            self._view.setText(i18n.t("SCENE_PREVIEW_CAMERA_IMAGE_LOAD_FAILED").format(path=str(camera.image_path)))
            return
        image = apply_mask_overlay(image, mask)
        if camera.projection == "equirectangular":
            shown = self._set_perspective_image(image, PerspectiveParams(fov_deg=90.0))
            if shown:
                return
            self._view.setText(i18n.t("SCENE_PREVIEW_CAMERA_IMAGE_LOAD_FAILED").format(path=str(camera.image_path)))
            return
        self._perspective_params = None
        self._view.set_drag_mode("pan")
        self._view.set_source_pixmap(QPixmap.fromImage(bgr_to_qimage(image)))

    def _set_perspective_image(self, image_bgr: np.ndarray, params: PerspectiveParams) -> bool:
        self._perspective_params = params
        self._view.set_drag_mode("look")
        return self._view.set_perspective_image_bgr(
            image_bgr,
            params,
            logical_size=QSize(768, 768),
        )

    def _on_look_dragged(self, delta_x: float, delta_y: float) -> None:
        if self._perspective_params is None:
            return
        self._perspective_params = params_from_drag(self._perspective_params, delta_x, delta_y)
        self._view.set_perspective_params(self._perspective_params)
        self.perspective_params_changed.emit(self._perspective_params)


def apply_mask_overlay(image_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    image = _as_bgr_uint8(image_bgr)
    if mask is None or mask.size == 0:
        return image
    mask_gray = _as_mask_gray(mask)
    if mask_gray.shape[:2] != image.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    excluded = mask_gray < 128
    if not np.any(excluded):
        return image
    overlay = np.zeros_like(image)
    overlay[:, :, 2] = 255
    image[excluded] = (
        (1.0 - MASK_OVERLAY_ALPHA) * image[excluded].astype(np.float32)
        + MASK_OVERLAY_ALPHA * overlay[excluded].astype(np.float32)
    ).astype(np.uint8)
    return image


def _as_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] >= 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] < 3:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    result = np.ascontiguousarray(image[:, :, :3])
    if result.dtype == np.uint8:
        return result.copy()
    return np.clip(result, 0, 255).astype(np.uint8)


def _as_mask_gray(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        gray = mask
    elif mask.ndim == 3 and mask.shape[2] >= 3:
        gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif mask.ndim == 3 and mask.shape[2] == 1:
        gray = mask[:, :, 0]
    else:
        gray = np.full((1, 1), 255, dtype=np.uint8)
    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)
    return np.clip(gray, 0, 255).astype(np.uint8)
