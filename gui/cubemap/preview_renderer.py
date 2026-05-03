"""エクイレクタングラープレビュー描画（ビュー境界オーバーレイ付き）"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode
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
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.45
_LABEL_THICKNESS = 1
_LABEL_PAD_X = 4
_LABEL_PAD_Y = 3
_HIGHLIGHT_FILL_ALPHA = 0.18


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


@lru_cache(maxsize=4)
def _equirect_unit_rays(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = (np.arange(width, dtype=np.float32) + 0.5) / max(width, 1)
    ys = (np.arange(height, dtype=np.float32) + 0.5) / max(height, 1)
    lon = (xs * 2.0 - 1.0) * np.pi
    lat = (0.5 - ys) * np.pi
    cos_lat = np.cos(lat).astype(np.float32)
    world_x = cos_lat[:, None] * np.sin(lon).astype(np.float32)[None, :]
    world_y = np.sin(lat).astype(np.float32)[:, None] * np.ones((1, width), dtype=np.float32)
    world_z = cos_lat[:, None] * np.cos(lon).astype(np.float32)[None, :]
    return world_x, world_y, world_z


def _view_fill_mask(width: int, height: int, yaw_deg: float, pitch_deg: float, fov_deg: float = 90.0) -> np.ndarray:
    world_x, world_y, world_z = _equirect_unit_rays(width, height)
    r = _rotation_matrix(yaw_deg, pitch_deg).astype(np.float32)
    local_x = world_x * r[0, 0] + world_y * r[1, 0] + world_z * r[2, 0]
    local_y = world_x * r[0, 1] + world_y * r[1, 1] + world_z * r[2, 1]
    local_z = world_x * r[0, 2] + world_y * r[1, 2] + world_z * r[2, 2]
    limit = np.tan(np.deg2rad(fov_deg) / 2.0) * 1.01
    front = local_z > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        x_over_z = np.divide(local_x, local_z, out=np.full_like(local_x, np.inf), where=front)
        y_over_z = np.divide(local_y, local_z, out=np.full_like(local_y, np.inf), where=front)
    return front & (np.abs(x_over_z) <= limit) & (np.abs(y_over_z) <= limit)


def _apply_view_fill(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = _HIGHLIGHT_FILL_ALPHA,
) -> None:
    if not np.any(mask):
        return
    alpha = max(0.0, min(1.0, float(alpha)))
    color_arr = np.array(color, dtype=np.float32)
    img[mask] = (
        img[mask].astype(np.float32) * (1.0 - alpha)
        + color_arr[None, :] * alpha
    ).astype(np.uint8)


def _pitch_color_map(views: list[dict]) -> dict[float, tuple[int, int, int]]:
    pitches = sorted({round(float(view.get("pitch", 0.0)), 6) for view in views})
    if not pitches:
        return {}
    if len(pitches) == 1:
        return {pitches[0]: _PITCH_PALETTE_BGR[1]}
    return {pitch: _PITCH_PALETTE_BGR[idx % len(_PITCH_PALETTE_BGR)] for idx, pitch in enumerate(pitches)}


def _overlay_draw_order(views: list[dict]) -> list[dict]:
    """Draw disabled gray view boxes first so enabled colored boxes stay on top."""
    return sorted(
        views,
        key=lambda view: (
            bool(view.get("highlighted", False)),
            bool(view.get("enabled", False)),
        ),
    )


def _draw_view_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    *,
    enabled: bool,
    highlighted: bool,
) -> None:
    if highlighted:
        overlay = img.copy()
        cv2.polylines(overlay, [pts], False, color, 16, lineType=cv2.LINE_AA)
        img[:] = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
        cv2.polylines(img, [pts], False, _LINE_OUTER, 10, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], False, _LINE_MID, 7, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], False, color, 5, lineType=cv2.LINE_AA)
        return
    if enabled:
        # Black/white/color halo keeps the line readable on both dark and bright footage.
        cv2.polylines(img, [pts], False, _LINE_OUTER, 7, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], False, _LINE_MID, 5, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], False, color, 3, lineType=cv2.LINE_AA)
        return
    cv2.polylines(img, [pts], False, _DISABLED_OUTER, 4, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], False, _DISABLED_LINE, 2, lineType=cv2.LINE_AA)


def _view_center_point(width: int, height: int, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
    center_ray = np.array([[0.0, 0.0, 1.0]], dtype=np.float64) @ _rotation_matrix(yaw_deg, pitch_deg).T
    x, y = _ray_to_equirect(center_ray[0], width, height)
    return x, y


def _ray_to_equirect(ray: np.ndarray, width: int, height: int) -> tuple[float, float]:
    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
    lon = np.arctan2(ray[0], ray[2])
    lat = np.arcsin(np.clip(ray[1], -1.0, 1.0))
    x = ((lon / np.pi + 1.0) * 0.5 * width) % width
    y = np.clip((0.5 - lat / np.pi) * height, 0.0, max(0.0, height - 1.0))
    return float(x), float(y)


def _point_inside_view(
    x: float,
    y: float,
    width: int,
    height: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float = 90.0,
) -> bool:
    lon = ((float(x) / max(width, 1)) * 2.0 - 1.0) * np.pi
    lat = (0.5 - float(y) / max(height, 1)) * np.pi
    world = np.array([
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
        np.cos(lat) * np.cos(lon),
    ], dtype=np.float64)
    local = world @ _rotation_matrix(yaw_deg, pitch_deg)
    if local[2] <= 1e-6:
        return False
    limit = np.tan(np.deg2rad(fov_deg) / 2.0) * 1.04
    return abs(local[0] / local[2]) <= limit and abs(local[1] / local[2]) <= limit


def _label_metrics(label: str) -> tuple[int, int, int]:
    (tw, th), baseline = cv2.getTextSize(label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS)
    return tw, th, baseline


def _label_box_for_center(
    center: tuple[float, float],
    metrics: tuple[int, int, int],
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    tw, th, baseline = metrics
    box_w = tw + _LABEL_PAD_X * 2
    box_h = th + baseline + _LABEL_PAD_Y * 2
    cx, cy = center
    x1 = int(round(cx - box_w / 2.0))
    y1 = int(round(cy - box_h / 2.0))
    x1 = max(0, min(max(0, width - box_w - 1), x1))
    y1 = max(0, min(max(0, height - box_h - 1), y1))
    x2 = min(width - 1, x1 + box_w)
    y2 = min(height - 1, y1 + box_h)
    origin = (x1 + _LABEL_PAD_X, y1 + _LABEL_PAD_Y + th)
    return (x1, y1, x2, y2), origin


def _box_overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _label_candidate_centers(anchor: tuple[float, float], width: int, height: int) -> list[tuple[float, float]]:
    centers = [anchor]
    min_dim = max(1.0, float(min(width, height)))
    for radius in [min_dim * v for v in (0.035, 0.06, 0.09, 0.13, 0.18, 0.24, 0.31, 0.39)]:
        for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            centers.append((anchor[0] + np.cos(angle) * radius, anchor[1] + np.sin(angle) * radius))
    return centers


def _layout_view_labels(
    views: list[dict],
    width: int,
    height: int,
    pitch_colors: dict[float, tuple[int, int, int]],
) -> list[dict]:
    labels: list[dict] = []
    occupied: list[tuple[int, int, int, int]] = []
    enabled_views = [view for view in views if view.get("enabled", False) and str(view.get("label", ""))]
    enabled_views.sort(key=lambda view: bool(view.get("highlighted", False)))

    for view in enabled_views:
        label = str(view.get("label", ""))
        yaw = float(view.get("yaw", 0.0))
        pitch = float(view.get("pitch", 0.0))
        anchor = _view_center_point(width, height, yaw, pitch)
        metrics = _label_metrics(label)
        best: tuple[float, tuple[int, int, int, int], tuple[int, int], tuple[float, float]] | None = None
        for center in _label_candidate_centers(anchor, width, height):
            cx = float(np.clip(center[0], 0.0, max(0.0, width - 1.0)))
            cy = float(np.clip(center[1], 0.0, max(0.0, height - 1.0)))
            box, origin = _label_box_for_center((cx, cy), metrics, width, height)
            overlap = sum(_box_overlap_area(box, existing) for existing in occupied)
            in_view = _point_inside_view(
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
                width,
                height,
                yaw,
                pitch,
            )
            distance = float(np.hypot(cx - anchor[0], cy - anchor[1]))
            edge_penalty = 20.0 if box[0] == 0 or box[1] == 0 or box[2] >= width - 1 or box[3] >= height - 1 else 0.0
            score = overlap * 10000.0 + distance + edge_penalty
            if not in_view:
                score += 500000.0
            if best is None or score < best[0]:
                best = (score, box, origin, (cx, cy))
                if overlap == 0 and in_view and distance < 1.0:
                    break
        if best is None:
            continue
        _, box, origin, center = best
        occupied.append(box)
        pitch_key = round(pitch, 6)
        labels.append({
            "label": label,
            "box": box,
            "origin": origin,
            "center": center,
            "color": pitch_colors.get(pitch_key, (90, 240, 120)),
            "highlighted": bool(view.get("highlighted", False)),
            "view": view,
        })
    return labels


def _draw_view_label_box(
    img: np.ndarray,
    label: str,
    box: tuple[int, int, int, int],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    highlighted: bool = False,
) -> None:
    if not label:
        return
    x1, y1, x2, y2 = box

    roi = img[y1:y2, x1:x2]
    if roi.size:
        bg = np.zeros_like(roi)
        img[y1:y2, x1:x2] = cv2.addWeighted(bg, 0.70, roi, 0.30, 0)
    border = color if highlighted else _LINE_MID
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, label, origin, _LABEL_FONT, _LABEL_SCALE, _LINE_OUTER, 3, lineType=cv2.LINE_AA)
    cv2.putText(img, label, origin, _LABEL_FONT, _LABEL_SCALE, color, _LABEL_THICKNESS, lineType=cv2.LINE_AA)


class PreviewWidget(QWidget):
    """プレビュー画像 + マスクオーバーレイ + タイムラインスライダー"""

    current_image_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.preview_images: list[Path] = []
        self._slider_sync = False
        self._current_image_path = ""
        self._scene_dir = ""

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

        img = imread_unicode(p, cv2.IMREAD_COLOR)
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
            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
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
        draw_order = _overlay_draw_order(views)
        for view in draw_order:
            if not view.get("highlighted", False):
                continue
            pitch_key = round(float(view.get("pitch", 0.0)), 6)
            color = pitch_colors.get(pitch_key, (90, 240, 120))
            mask = _view_fill_mask(w, h, float(view["yaw"]), float(view["pitch"]), 90.0)
            _apply_view_fill(img, mask, color)

        for view in draw_order:
            pitch_key = round(float(view.get("pitch", 0.0)), 6)
            color = pitch_colors.get(pitch_key, (90, 240, 120))
            enabled = bool(view["enabled"])
            highlighted = bool(view.get("highlighted", False))
            segments = _view_boundary_segments(w, h, view["yaw"], view["pitch"], 90.0)
            for seg in segments:
                if len(seg) < 2:
                    continue
                pts = np.round(seg).astype(np.int32).reshape((-1, 1, 2))
                _draw_view_polyline(img, pts, color, enabled=enabled, highlighted=highlighted)

        for item in _layout_view_labels(draw_order, w, h, pitch_colors):
            _draw_view_label_box(
                img,
                item["label"],
                item["box"],
                item["origin"],
                item["color"],
                highlighted=bool(item.get("highlighted", False)),
            )

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
        if not self._scene_dir:
            return []
        scene_dir = Path(self._scene_dir)
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
