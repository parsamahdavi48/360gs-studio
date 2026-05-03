"""Mask preview for Step 3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.zoomable_image_label import ZoomableImageLabel
from overexposure_mask import detect_overexposure, read_image_preserve_depth
from stitch_mask import boundary_width_to_limit_angle, create_angular_stitched_mask

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskPreviewConfig:
    use_yolo: bool = True
    use_stitch: bool = False
    use_overexposure: bool = False
    stitch_boundary_width_deg: float | None = 5.0
    overexposure_threshold: int = 254
    overexposure_dilate: int = 1
    masks_dir: str = ""


class MaskPreviewWidget(QWidget):
    """Preview the currently selected mask layers over an equirectangular frame."""

    yolo_preview_requested = Signal()
    current_image_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._images_dir = ""
        self._pixmap: QPixmap | None = None
        self.preview_images: list[Path] = []
        self._slider_sync = False
        self._current_image_path = ""
        self._yolo_preview_image_key = ""
        self._yolo_preview_mask: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.image_label = ZoomableImageLabel(i18n.t("NO_PREVIEW"))
        self.image_label.setMinimumSize(640, 280)
        self.image_label.setStyleSheet("border: 1px solid palette(mid);")
        layout.addWidget(self.image_label, stretch=1)

        timeline_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip(i18n.tip("PREVIEW_SLIDER"))
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        timeline_row.addWidget(self.slider, stretch=1)
        self.timeline_label = QLabel("0 / 0")
        timeline_row.addWidget(self.timeline_label)
        layout.addLayout(timeline_row)

        overlay_row = QHBoxLayout()
        self.yolo_preview_btn = QPushButton(i18n.t("YOLO_PREVIEW_BUTTON"))
        self.yolo_preview_btn.setToolTip(i18n.tip("YOLO_PREVIEW_BUTTON"))
        self.yolo_preview_btn.clicked.connect(self.yolo_preview_requested.emit)
        overlay_row.addWidget(self.yolo_preview_btn)

        overlay_row.addWidget(QLabel(i18n.t("MASK_OPACITY_LABEL")))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setToolTip(i18n.tip("MASK_OPACITY"))
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.setMaximumWidth(160)
        overlay_row.addWidget(self.opacity_slider)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        overlay_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(overlay_row)

    def set_images_dir(self, images_dir: str) -> None:
        if images_dir == self._images_dir:
            return
        self._images_dir = images_dir
        self.refresh_image_list(prefer_current=False)

    def render(self, config: MaskPreviewConfig) -> None:
        sample = self._current_image_path.strip()
        if not sample:
            self.image_label.setText(i18n.t("NO_PREVIEW"))
            self.status_label.setText("")
            self._pixmap = None
            return

        image_path = Path(sample)
        if not image_path.exists() or not image_path.is_file():
            self.image_label.setText(i18n.t("NO_PREVIEW_FOUND"))
            self.status_label.setText("")
            self._pixmap = None
            return

        source_img = read_image_preserve_depth(str(image_path))
        img = _display_bgr8(source_img)
        if img is None:
            self.image_label.setText(i18n.t("PREVIEW_LOAD_FAIL"))
            self.status_label.setText("")
            self._pixmap = None
            return

        max_w = 1900
        if img.shape[1] > max_w:
            scale = max_w / float(img.shape[1])
            img = cv2.resize(
                img,
                (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        h, w = img.shape[:2]
        combined = np.full((h, w), 255, dtype=np.uint8)
        status_parts: list[str] = []

        if config.use_yolo:
            yolo_mask = self._load_yolo_preview_mask(image_path)
            yolo_status = i18n.t("MASK_PREVIEW_YOLO_TEMP") if yolo_mask is not None else ""
            if yolo_mask is None:
                yolo_mask = self._load_existing_mask(image_path, config)
                yolo_status = i18n.t("MASK_PREVIEW_YOLO_EXISTING") if yolo_mask is not None else ""
            if yolo_mask is None:
                status_parts.append(i18n.t("MASK_PREVIEW_YOLO_PENDING"))
            else:
                if yolo_mask.shape != combined.shape:
                    yolo_mask = cv2.resize(yolo_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                combined = cv2.bitwise_and(combined, yolo_mask)
                status_parts.append(yolo_status)

        if config.use_stitch:
            if config.stitch_boundary_width_deg is None:
                status_parts.append(i18n.t("MASK_PREVIEW_INVALID_STITCH_WIDTH"))
            else:
                stitch = create_angular_stitched_mask(
                    w,
                    h,
                    boundary_width_to_limit_angle(config.stitch_boundary_width_deg),
                )
                combined = cv2.bitwise_and(combined, stitch)
                status_parts.append(
                    i18n.t("MASK_PREVIEW_STITCH_STATUS").format(
                        width=config.stitch_boundary_width_deg,
                    )
                )

        if config.use_overexposure:
            overexp = detect_overexposure(
                source_img,
                threshold=int(config.overexposure_threshold),
                dilate_px=int(config.overexposure_dilate),
            )
            if overexp.shape != combined.shape:
                overexp = cv2.resize(overexp, (w, h), interpolation=cv2.INTER_NEAREST)
            combined = cv2.bitwise_and(combined, overexp)
            status_parts.append(
                i18n.t("MASK_PREVIEW_OVEREXP_STATUS").format(
                    threshold=config.overexposure_threshold,
                    dilate=config.overexposure_dilate,
                )
            )

        excluded = combined < 128
        alpha = float(self.opacity_slider.value()) / 100.0
        if alpha > 0 and np.any(excluded):
            overlay = np.zeros_like(img)
            overlay[:, :, 2] = 255
            img[excluded] = (
                (1.0 - alpha) * img[excluded].astype(np.float32)
                + alpha * overlay[excluded].astype(np.float32)
            ).astype(np.uint8)

        if np.any(excluded):
            contours, _ = cv2.findContours(
                excluded.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(img, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        self.status_label.setText(
            " / ".join(status_parts) if status_parts else i18n.t("MASK_PREVIEW_NO_ACTIVE_MASK")
        )

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()

    def refresh_image_list(self, prefer_current: bool = True) -> None:
        current = self._current_image_path.strip()
        self.preview_images = self._iter_images()
        total = len(self.preview_images)
        self.slider.setEnabled(total > 0)
        self.slider.setRange(0, max(0, total - 1))

        if total <= 0:
            self.slider.setValue(0)
            self.timeline_label.setText("0 / 0")
            self._set_current_image_path("", emit=True)
            return

        target = 0
        if prefer_current and current:
            try:
                current_key = str(Path(current).resolve()).lower()
                for i, p in enumerate(self.preview_images):
                    if str(p.resolve()).lower() == current_key:
                        target = i
                        break
            except Exception:
                pass
        self._set_index(target)

    def _iter_images(self) -> list[Path]:
        if not self._images_dir:
            return []
        root = Path(self._images_dir)
        roots = [root] if root.is_dir() else []
        result, seen = [], set()
        for r in roots:
            for p in r.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
                    continue
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    result.append(p)
        result.sort(key=lambda x: str(x).lower())
        return result

    def _set_index(self, idx: int) -> None:
        if not self.preview_images:
            self.timeline_label.setText("0 / 0")
            self._set_current_image_path("", emit=True)
            return
        idx = max(0, min(idx, len(self.preview_images) - 1))
        self._slider_sync = True
        self.slider.setValue(idx)
        self._slider_sync = False
        self._set_current_image_path(str(self.preview_images[idx]), emit=True)
        self.timeline_label.setText(
            i18n.t("PREVIEW_IMAGE_POSITION_FORMAT").format(
                seq=idx + 1,
                total=len(self.preview_images),
                name=self.preview_images[idx].name,
            )
        )

    def set_current_image_path(self, image_path: str | Path) -> None:
        self._set_current_image_path(str(image_path), emit=False)

    def _set_current_image_path(self, image_path: str, emit: bool) -> None:
        if image_path == self._current_image_path:
            return
        self._current_image_path = image_path
        if emit:
            self.current_image_changed.emit()

    def _load_existing_mask(self, image_path: Path, config: MaskPreviewConfig) -> np.ndarray | None:
        masks_root = Path(config.masks_dir) if config.masks_dir else None
        if masks_root is None or not masks_root.is_dir():
            return None

        rel_parent = Path()
        try:
            images_root = Path(self._images_dir).resolve()
            rel_parent = image_path.resolve().relative_to(images_root).parent
        except Exception:
            pass

        candidates: list[Path] = []
        candidates.append(masks_root / rel_parent / f"{image_path.stem}.png")
        candidates.append(masks_root / rel_parent / f"{image_path.name}.png")
        candidates.append(masks_root / f"{image_path.name}.png")
        candidates.append(masks_root / f"{image_path.stem}.png")

        for candidate in candidates:
            if not candidate.is_file():
                continue
            mask = imread_unicode(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            return mask
        return None

    def current_image_path(self) -> Path | None:
        sample = self._current_image_path.strip()
        if not sample:
            return None
        image_path = Path(sample)
        if not image_path.exists() or not image_path.is_file():
            return None
        return image_path

    def clear_yolo_preview_mask(self, image_path: Path | None = None) -> None:
        if image_path is not None and self._yolo_preview_image_key != _path_key(image_path):
            return
        self._yolo_preview_image_key = ""
        self._yolo_preview_mask = None

    def set_yolo_preview_mask(self, image_path: Path, mask_path: Path) -> bool:
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        self._yolo_preview_image_key = _path_key(image_path)
        self._yolo_preview_mask = mask
        return True

    def set_yolo_preview_running(self, running: bool) -> None:
        self.yolo_preview_btn.setEnabled(not running)
        self.yolo_preview_btn.setText(
            i18n.t("MASK_PREVIEW_YOLO_RUNNING") if running else i18n.t("YOLO_PREVIEW_BUTTON")
        )

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(text)

    def _load_yolo_preview_mask(self, image_path: Path) -> np.ndarray | None:
        if self._yolo_preview_image_key != _path_key(image_path):
            return None
        return self._yolo_preview_mask

    def _on_slider_changed(self, idx: int) -> None:
        if self._slider_sync:
            return
        if 0 <= idx < len(self.preview_images):
            self._set_index(idx)

    def _update_pixmap(self) -> None:
        self.image_label.set_source_pixmap(self._pixmap)


def _display_bgr8(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.dtype == np.uint8:
        display = image
    elif np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        display = np.clip(np.rint(image.astype(np.float32) * 255.0 / max_value), 0, 255).astype(np.uint8)
    else:
        display = np.clip(image, 0.0, 1.0)
        display = np.rint(display * 255.0).astype(np.uint8)

    if display.ndim == 2:
        return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    if display.ndim == 3 and display.shape[2] == 4:
        return cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
    return display


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()
