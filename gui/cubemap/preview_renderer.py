"""エクイレクタングラープレビュー描画（ビュー境界オーバーレイ付き）"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
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
from gui.common.collapsible_section import CollapsibleSection


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ], dtype=np.float64)
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ], dtype=np.float64)
    r = ry @ rx
    r[np.abs(r) < 1e-10] = 0.0
    return r


def _view_boundary_segments(
    width: int, height: int,
    yaw_deg: float, pitch_deg: float,
    fov_deg: float, samples_per_edge: int = 120,
) -> list[np.ndarray]:
    t = np.linspace(-1.0, 1.0, samples_per_edge)
    u = np.concatenate([t, np.ones_like(t), t[::-1], -np.ones_like(t), np.array([-1.0])])
    v = np.concatenate([np.ones_like(t), t[::-1], -np.ones_like(t), t, np.array([1.0])])

    focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    rays = np.stack([u, v, np.full_like(u, focal)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    r = _rotation_matrix(yaw_deg, pitch_deg)
    rays = rays @ r.T

    lon = np.arctan2(rays[:, 0], rays[:, 2])
    lat = np.arcsin(np.clip(rays[:, 1], -1.0, 1.0))

    x = (lon / np.pi + 1.0) * 0.5 * width
    y = (0.5 - lat / np.pi) * height
    points = np.stack([x, y], axis=-1)

    segments: list[np.ndarray] = []
    current: list[np.ndarray] = [points[0]]
    threshold = width * 0.45
    for p in points[1:]:
        if abs(float(p[0] - current[-1][0])) > threshold:
            if len(current) >= 2:
                segments.append(np.array(current, dtype=np.float32))
            current = [p]
        else:
            current.append(p)
    if len(current) >= 2:
        segments.append(np.array(current, dtype=np.float32))
    return segments


class PreviewWidget(QWidget):
    """プレビュー画像 + マスクオーバーレイ + タイムラインスライダー"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.preview_images: list[Path] = []
        self._slider_sync = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel(i18n.t("NO_PREVIEW"))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 320)
        self.image_label.setStyleSheet("border: 1px solid palette(mid);")
        layout.addWidget(self.image_label, stretch=1)

        # サンプル画像
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel(i18n.t("PREVIEW_IMAGE_LABEL")))
        self.sample_edit = QLineEdit()
        self.sample_edit.setToolTip(i18n.tip("PREVIEW_SAMPLE"))
        self.sample_edit.textChanged.connect(self._on_sample_changed)
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

        # タイムライン
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

        # マスクオーバーレイ設定（折りたたみ）
        mask_section = CollapsibleSection(i18n.t("PREVIEW_OVERLAY_SECTION"), expanded=False)
        mask_inner = QHBoxLayout()

        mask_inner.addWidget(QLabel(i18n.t("MASK_OPACITY_LABEL")))
        self.mask_slider = QSlider(Qt.Horizontal)
        self.mask_slider.setToolTip(i18n.tip("MASK_OPACITY"))
        self.mask_slider.setRange(0, 100)
        self.mask_slider.setValue(35)
        mask_inner.addWidget(self.mask_slider)

        mask_inner.addWidget(QLabel(i18n.t("MASK_IMAGE_LABEL")))
        self.mask_edit = QLineEdit()
        self.mask_edit.setToolTip(i18n.tip("MASK_IMAGE"))
        mask_inner.addWidget(self.mask_edit)
        mask_browse = QPushButton(i18n.BROWSE)
        mask_browse.setToolTip(i18n.tip("MASK_IMAGE_BROWSE"))
        mask_browse.clicked.connect(self._browse_mask)
        mask_inner.addWidget(mask_browse)
        mask_clear = QPushButton(i18n.t("CLEAR"))
        mask_clear.setToolTip(i18n.tip("MASK_IMAGE_CLEAR"))
        mask_clear.clicked.connect(lambda: self.mask_edit.setText(""))
        mask_inner.addWidget(mask_clear)

        w = QWidget()
        w.setLayout(mask_inner)
        mask_section.content_layout.addWidget(w)
        layout.addWidget(mask_section)

    # -- public --

    def set_scene_dir(self, scene_dir: str) -> None:
        self._scene_dir = scene_dir
        self.refresh_image_list(prefer_current=False)

    def render(self, views: list[dict], mask_dir: str = "") -> None:
        sample = self.sample_edit.text().strip()
        if not sample:
            self.image_label.setText(i18n.t("NO_PREVIEW"))
            self._pixmap = None
            return

        p = Path(sample)
        if not p.exists() or not p.is_file():
            self.image_label.setText(i18n.t("NO_PREVIEW_FOUND"))
            self._pixmap = None
            return

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            self.image_label.setText(i18n.t("PREVIEW_LOAD_FAIL"))
            self._pixmap = None
            return

        max_w = 1900
        if img.shape[1] > max_w:
            scale = max_w / float(img.shape[1])
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)

        # マスクオーバーレイ
        mask_path = self._resolve_mask(p, mask_dir)
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                alpha = float(self.mask_slider.value()) / 100.0
                if alpha > 0:
                    masked = mask < 128
                    overlay = np.zeros_like(img)
                    overlay[:, :, 2] = 255
                    img[masked] = (
                        (1.0 - alpha) * img[masked].astype(np.float32)
                        + alpha * overlay[masked].astype(np.float32)
                    ).astype(np.uint8)

        # ビュー境界描画
        h, w = img.shape[:2]
        for view in views:
            color = (0, 220, 255) if view["enabled"] else (128, 128, 128)
            thickness = 2 if view["enabled"] else 1
            segments = _view_boundary_segments(w, h, view["yaw"], view["pitch"], 90.0)
            all_pts: list[np.ndarray] = []
            for seg in segments:
                if len(seg) < 2:
                    continue
                pts = np.round(seg).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)
                all_pts.append(seg)

            if view["enabled"] and all_pts:
                merged = np.concatenate(all_pts, axis=0)
                cx = int(np.clip(np.mean(merged[:, 0]), 0, w - 1))
                cy = int(np.clip(np.mean(merged[:, 1]), 0, h - 1))
                label = str(view.get("label", ""))
                cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()

    # -- internal --

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _resolve_mask(self, sample_path: Path, mask_dir: str) -> Path | None:
        manual = self.mask_edit.text().strip()
        if manual:
            mp = Path(manual)
            if mp.exists():
                return mp

        md = Path(mask_dir) if mask_dir else None
        if md is None or not md.is_dir():
            return None

        candidates = [
            md / sample_path.name,
            md / f"{sample_path.name}.png",
            md / f"{sample_path.stem}.png",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _iter_images(self) -> list[Path]:
        scene_dir = Path(getattr(self, "_scene_dir", "."))
        images_dir = scene_dir / "images"
        roots = [images_dir] if images_dir.is_dir() else [scene_dir]
        exts = {".jpg", ".jpeg", ".png"}
        result, seen = [], set()
        for root in roots:
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    result.append(p)
        result.sort(key=lambda x: str(x).lower())
        return result

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
                ck = str(Path(current).resolve()).lower()
                for i, p in enumerate(self.preview_images):
                    if str(p.resolve()).lower() == ck:
                        target = i
                        break
            except Exception:
                pass
        self._set_index(target)

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

    def _on_sample_changed(self, _text: str) -> None:
        pass  # 外部からrender()呼び出しで更新

    def _browse_sample(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "", "", "画像 (*.jpg *.jpeg *.png);;すべて (*.*)")
        if path:
            self.sample_edit.setText(path)

    def _auto_select(self) -> None:
        if not self.preview_images:
            self.refresh_image_list(prefer_current=False)
            return
        self._set_index(0)

    def _browse_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "", "", "画像 (*.png *.jpg);;すべて (*.*)")
        if path:
            self.mask_edit.setText(path)
