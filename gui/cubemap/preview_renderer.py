"""エクイレクタングラープレビュー描画（ビュー境界オーバーレイ付き）"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.zoomable_image_label import ZoomableImageLabel

_PITCH_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 159, 230),    # orange
    (233, 180, 86),   # sky blue
    (115, 158, 0),    # bluish green
    (167, 121, 204),  # reddish purple
    (66, 228, 240),   # yellow
)
_LINE_OUTER = (0, 0, 0)
_LINE_MID = (245, 245, 245)
_DISABLED_LINE = (150, 150, 150)
_DISABLED_OUTER = (25, 25, 25)


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


def _pitch_color_map(views: list[dict]) -> dict[float, tuple[int, int, int]]:
    pitches = sorted({round(float(view.get("pitch", 0.0)), 6) for view in views})
    if not pitches:
        return {}
    if len(pitches) == 1:
        return {pitches[0]: _PITCH_PALETTE_BGR[1]}
    return {pitch: _PITCH_PALETTE_BGR[idx % len(_PITCH_PALETTE_BGR)] for idx, pitch in enumerate(pitches)}


def _overlay_draw_order(views: list[dict]) -> list[dict]:
    """Draw disabled gray view boxes first so enabled colored boxes stay on top."""
    return sorted(views, key=lambda view: bool(view.get("enabled", False)))


def _draw_view_polyline(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], *, enabled: bool) -> None:
    if enabled:
        # Black/white/color halo keeps the line readable on both dark and bright footage.
        cv2.polylines(img, [pts], False, _LINE_OUTER, 7, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], False, _LINE_MID, 5, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], False, color, 3, lineType=cv2.LINE_AA)
        return
    cv2.polylines(img, [pts], False, _DISABLED_OUTER, 4, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], False, _DISABLED_LINE, 2, lineType=cv2.LINE_AA)


def _draw_view_label(img: np.ndarray, label: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    x, y = pos
    x = int(np.clip(x, 4, max(4, img.shape[1] - tw - 8)))
    y = int(np.clip(y, th + 8, max(th + 8, img.shape[0] - 6)))
    pad_x, pad_y = 4, 3
    x1, y1 = max(0, x - pad_x), max(0, y - th - pad_y)
    x2, y2 = min(img.shape[1] - 1, x + tw + pad_x), min(img.shape[0] - 1, y + baseline + pad_y)

    roi = img[y1:y2, x1:x2]
    if roi.size:
        bg = np.zeros_like(roi)
        img[y1:y2, x1:x2] = cv2.addWeighted(bg, 0.68, roi, 0.32, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), _LINE_MID, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (x, y), font, scale, _LINE_OUTER, 3, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)


class PreviewWidget(QWidget):
    """プレビュー画像 + マスクオーバーレイ + タイムラインスライダー"""

    current_image_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.preview_images: list[Path] = []
        self._slider_sync = False
        self._current_image_path = ""

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = ZoomableImageLabel(i18n.t("NO_PREVIEW"))
        self.image_label.setMinimumSize(640, 320)
        self.image_label.setStyleSheet("border: 1px solid palette(mid);")
        layout.addWidget(self.image_label, stretch=1)

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

        mask_row = QHBoxLayout()
        mask_row.addWidget(QLabel(i18n.t("MASK_OPACITY_LABEL")))
        self.mask_slider = QSlider(Qt.Horizontal)
        self.mask_slider.setToolTip(i18n.tip("MASK_OPACITY"))
        self.mask_slider.setRange(0, 100)
        self.mask_slider.setValue(35)
        self.mask_slider.setMaximumWidth(160)
        mask_row.addWidget(self.mask_slider)
        mask_row.addStretch()
        layout.addLayout(mask_row)

    # -- public --

    def set_scene_dir(self, scene_dir: str) -> None:
        self._scene_dir = scene_dir
        self.refresh_image_list(prefer_current=False)

    def render(self, views: list[dict], mask_dir: str = "") -> None:
        sample = self._current_image_path.strip()
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
        pitch_colors = _pitch_color_map(views)
        for view in _overlay_draw_order(views):
            pitch_key = round(float(view.get("pitch", 0.0)), 6)
            color = pitch_colors.get(pitch_key, (90, 240, 120))
            enabled = bool(view["enabled"])
            segments = _view_boundary_segments(w, h, view["yaw"], view["pitch"], 90.0)
            all_pts: list[np.ndarray] = []
            for seg in segments:
                if len(seg) < 2:
                    continue
                pts = np.round(seg).astype(np.int32).reshape((-1, 1, 2))
                _draw_view_polyline(img, pts, color, enabled=enabled)
                all_pts.append(seg)

            if enabled and all_pts:
                merged = np.concatenate(all_pts, axis=0)
                cx = int(np.clip(np.mean(merged[:, 0]), 0, w - 1))
                cy = int(np.clip(np.mean(merged[:, 1]), 0, h - 1))
                label = str(view.get("label", ""))
                _draw_view_label(img, label, (cx, cy), color)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()

    # -- internal --

    def _update_pixmap(self) -> None:
        self.image_label.set_source_pixmap(self._pixmap)

    def _resolve_mask(self, sample_path: Path, mask_dir: str) -> Path | None:
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
        current = self._current_image_path.strip()
        self.preview_images = self._iter_images()
        total = len(self.preview_images)
        self.slider.setEnabled(total > 0)
        self.slider.setRange(0, max(0, total - 1))

        if total <= 0:
            self.slider.setValue(0)
            self.tl_label.setText("0 / 0")
            self._set_current_image_path("", emit=True)
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
            self.tl_label.setText("0 / 0")
            self._set_current_image_path("", emit=True)
            return
        idx = max(0, min(idx, len(self.preview_images) - 1))
        self._slider_sync = True
        self.slider.setValue(idx)
        self._slider_sync = False
        self._set_current_image_path(str(self.preview_images[idx]), emit=True)
        self.tl_label.setText(
            i18n.t("PREVIEW_IMAGE_POSITION_FORMAT").format(
                seq=idx + 1,
                total=len(self.preview_images),
                name=self.preview_images[idx].name,
            )
        )

    def current_image_path(self) -> Path | None:
        sample = self._current_image_path.strip()
        if not sample:
            return None
        p = Path(sample)
        if not p.exists() or not p.is_file():
            return None
        return p

    def _set_current_image_path(self, image_path: str, emit: bool) -> None:
        if image_path == self._current_image_path:
            return
        self._current_image_path = image_path
        if emit:
            self.current_image_changed.emit()

    def _on_slider_changed(self, idx: int) -> None:
        if self._slider_sync:
            return
        if 0 <= idx < len(self.preview_images):
            self._set_index(idx)
