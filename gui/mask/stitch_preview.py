"""スティッチ境界マスクのエクイレクタングラープレビュー."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.drag_spinbox import DragSpinBox
from stitch_mask import (
    boundary_width_to_fov,
    boundary_width_to_limit_angle,
    create_angular_stitched_mask,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class StitchPreviewWidget(QWidget):
    """抽出済みフレーム上に、現在のスティッチ境界マスク範囲を重ねて表示する."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._images_dir = ""
        self._pixmap: QPixmap | None = None
        self.preview_images: list[Path] = []
        self._slider_sync = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.image_label = QLabel(i18n.t("NO_PREVIEW"))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 280)
        self.image_label.setStyleSheet("border: 1px solid palette(mid);")
        layout.addWidget(self.image_label, stretch=1)

        img_row = QHBoxLayout()
        img_row.addWidget(QLabel(i18n.t("PREVIEW_IMAGE_LABEL")))
        self.sample_edit = QLineEdit()
        self.sample_edit.setToolTip(i18n.tip("PREVIEW_SAMPLE"))
        img_row.addWidget(self.sample_edit, stretch=1)

        browse_btn = QPushButton(i18n.BROWSE)
        browse_btn.setToolTip(i18n.tip("PREVIEW_BROWSE"))
        browse_btn.clicked.connect(self._browse_sample)
        img_row.addWidget(browse_btn)

        auto_btn = QPushButton(i18n.t("AUTO"))
        auto_btn.setToolTip(i18n.tip("PREVIEW_AUTO"))
        auto_btn.clicked.connect(self._auto_select)
        img_row.addWidget(auto_btn)

        reload_btn = QPushButton(i18n.t("RELOAD"))
        reload_btn.setToolTip(i18n.tip("PREVIEW_RELOAD"))
        reload_btn.clicked.connect(lambda: self.refresh_image_list(prefer_current=False))
        img_row.addWidget(reload_btn)
        layout.addLayout(img_row)

        tl_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip(i18n.tip("PREVIEW_SLIDER"))
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        tl_row.addWidget(self.slider, stretch=1)
        self.tl_label = QLabel("0 / 0")
        tl_row.addWidget(self.tl_label)
        layout.addLayout(tl_row)

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel(i18n.t("MASK_OPACITY_LABEL")))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setToolTip(i18n.tip("MASK_OPACITY"))
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.setMaximumWidth(160)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        overlay_row.addWidget(self.opacity_slider)

        self.opacity_spin = DragSpinBox(
            minimum=0,
            maximum=100,
            step=5,
            value=45,
            suffix=" %",
            drag_pixels_per_step=6.0,
        )
        self.opacity_spin.setToolTip(i18n.tip("MASK_OPACITY"))
        self.opacity_spin.setFixedWidth(76)
        self.opacity_spin.valueChanged.connect(self._on_opacity_spin_changed)
        overlay_row.addWidget(self.opacity_spin)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        overlay_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(overlay_row)

    def set_images_dir(self, images_dir: str) -> None:
        if images_dir == self._images_dir:
            return
        self._images_dir = images_dir
        self.refresh_image_list(prefer_current=False)

    def render(self, boundary_width_deg: float | None) -> None:
        sample = self.sample_edit.text().strip()
        if not sample:
            self.image_label.setText(i18n.t("NO_PREVIEW"))
            self.status_label.setText("")
            self._pixmap = None
            return

        p = Path(sample)
        if not p.exists() or not p.is_file():
            self.image_label.setText(i18n.t("NO_PREVIEW_FOUND"))
            self.status_label.setText("")
            self._pixmap = None
            return

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
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

        if boundary_width_deg is None:
            self.status_label.setText(i18n.t("STITCH_PREVIEW_INVALID_WIDTH"))
        else:
            h, w = img.shape[:2]
            mask = create_angular_stitched_mask(w, h, boundary_width_to_limit_angle(boundary_width_deg))
            excluded = mask < 128
            alpha = float(self.opacity_slider.value()) / 100.0
            if alpha > 0 and np.any(excluded):
                overlay = np.zeros_like(img)
                overlay[:, :, 2] = 255
                img[excluded] = (
                    (1.0 - alpha) * img[excluded].astype(np.float32)
                    + alpha * overlay[excluded].astype(np.float32)
                ).astype(np.uint8)

            contours, _ = cv2.findContours(excluded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            self.status_label.setText(
                i18n.t("STITCH_PREVIEW_STATUS_FORMAT").format(
                    width=boundary_width_deg,
                    fov=boundary_width_to_fov(boundary_width_deg),
                )
            )

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()

    def refresh_image_list(self, prefer_current: bool = True) -> None:
        current = self.sample_edit.text().strip()
        self.preview_images = self._iter_images()
        total = len(self.preview_images)
        self.slider.setEnabled(total > 0)
        self.slider.setRange(0, max(0, total - 1))

        if total <= 0:
            self.slider.setValue(0)
            self.tl_label.setText("0 / 0")
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
        root = Path(self._images_dir) if self._images_dir else Path(".")
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
            self.sample_edit.setText("")
            self.tl_label.setText("0 / 0")
            return
        idx = max(0, min(idx, len(self.preview_images) - 1))
        self._slider_sync = True
        self.slider.setValue(idx)
        self._slider_sync = False
        self.sample_edit.setText(str(self.preview_images[idx]))
        self.tl_label.setText(f"{idx + 1} / {len(self.preview_images)} : {self.preview_images[idx].name}")

    def _on_slider_changed(self, idx: int) -> None:
        if self._slider_sync:
            return
        if 0 <= idx < len(self.preview_images):
            self._set_index(idx)

    def _browse_sample(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "",
            self.sample_edit.text(),
            "画像 (*.jpg *.jpeg *.png);;すべて (*.*)",
        )
        if path:
            self.sample_edit.setText(path)

    def _auto_select(self) -> None:
        if not self.preview_images:
            self.refresh_image_list(prefer_current=False)
            return
        self._set_index(0)

    def _on_opacity_slider_changed(self, value: int) -> None:
        with QSignalBlocker(self.opacity_spin):
            self.opacity_spin.setValue(value)

    def _on_opacity_spin_changed(self, value: int) -> None:
        with QSignalBlocker(self.opacity_slider):
            self.opacity_slider.setValue(value)

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()
