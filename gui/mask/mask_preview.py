"""Mask preview for Step 3."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from custom_mask import load_custom_mask
from image_io import imread_unicode
from PySide6.QtCore import QItemSelectionModel, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import i18n
from gui.common.icons import mask_overlay_off_icon, mask_overlay_on_icon
from gui.common.preview_mode_toolbar import (
    PREVIEW_MODE_SINGLE,
    PREVIEW_MODE_THUMBNAILS,
    PreviewModeToolbar,
)
from gui.common.thumbnail_list_model import visible_rows_for_view
from gui.common.zoomable_image_label import ZoomableImageLabel
from gui.mask.mask_files import iter_image_files, mask_candidates_for_image, path_key
from gui.mask.thumbnail_delegate import MaskThumbnailDelegate
from gui.mask.thumbnail_model import MaskThumbnailModel
from overexposure_mask import detect_overexposure, read_image_preserve_depth
from stitch_mask import boundary_width_to_limit_angle, create_angular_stitched_mask

_IMAGE_CACHE_LIMIT = 2
_LAYER_CACHE_LIMIT = 4
_MASK_OVERLAY_OPACITY = 45
_STATUS_LABEL_MIN_WIDTH = 72


@dataclass(frozen=True)
class MaskPreviewConfig:
    use_yolo: bool = True
    use_stitch: bool = False
    use_overexposure: bool = False
    use_sky: bool = False
    use_custom: bool = False
    stitch_boundary_width_deg: float | None = 5.0
    overexposure_threshold: int = 254
    overexposure_dilate: int = 1
    masks_dir: str = ""
    custom_mask_path: str = ""
    settings_key: tuple = ()


class ElidedStatusLabel(QLabel):
    """One-line status label that preserves full text for code and tooltips."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self._full_text = ""
        self.setWordWrap(False)
        self.setMinimumWidth(_STATUS_LABEL_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full_text = text
        self.setToolTip(text)
        self._apply_elide()

    def text(self) -> str:  # noqa: N802 - Qt API
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt API
        super().resizeEvent(event)
        self._apply_elide()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(_STATUS_LABEL_MIN_WIDTH, QLabel.sizeHint(self).height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(_STATUS_LABEL_MIN_WIDTH, QLabel.minimumSizeHint(self).height())

    def _apply_elide(self) -> None:
        width = max(0, self.contentsRect().width())
        display_text = self._full_text
        if width > 0 and self._full_text:
            display_text = QFontMetrics(self.font()).elidedText(self._full_text, Qt.ElideRight, width)
        if QLabel.text(self) != display_text:
            QLabel.setText(self, display_text)


class MaskPreviewWidget(QWidget):
    """Preview the currently selected mask layers over an equirectangular frame."""

    mask_preview_requested = Signal()
    current_reprocess_requested = Signal()
    current_image_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._images_dir = ""
        self._pixmap: QPixmap | None = None
        self.preview_images: list[Path] = []
        self._slider_sync = False
        self._thumbnail_sync = False
        self._thumbnail_model_signature: tuple | None = None
        self._preview_mode = PREVIEW_MODE_SINGLE
        self._last_config = MaskPreviewConfig()
        self._current_image_path = ""
        self._mask_overlay_visible = True
        self._temporary_preview_image_key = ""
        self._temporary_preview_config_key: tuple | None = None
        self._temporary_preview_mask: np.ndarray | None = None
        self._image_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._mask_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._stitch_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._overexp_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._thumbnail_priority_timer = QTimer(self)
        self._thumbnail_priority_timer.setSingleShot(True)
        self._thumbnail_priority_timer.setInterval(0)
        self._thumbnail_priority_timer.timeout.connect(self._prioritize_visible_thumbnails)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.mode_toolbar = PreviewModeToolbar()
        self.mode_toolbar.mode_changed.connect(self.set_preview_mode)
        self.single_preview_btn = self.mode_toolbar.single_preview_btn
        self.thumbnail_preview_btn = self.mode_toolbar.thumbnail_preview_btn

        self.preview_stack = QStackedWidget()

        self.image_label = ZoomableImageLabel(i18n.t("NO_PREVIEW"))
        self.image_label.setMinimumSize(640, 280)
        self.image_label.setStyleSheet("border: 1px solid palette(mid);")
        self.preview_stack.addWidget(self.image_label)

        self.thumbnail_model = MaskThumbnailModel(self)
        self.thumbnail_view = QListView()
        self.thumbnail_view.setModel(self.thumbnail_model)
        self.thumbnail_delegate = MaskThumbnailDelegate(self.thumbnail_view)
        self.thumbnail_view.setItemDelegate(self.thumbnail_delegate)
        self.thumbnail_view.setViewMode(QListView.IconMode)
        self.thumbnail_view.setResizeMode(QListView.Adjust)
        self.thumbnail_view.setMovement(QListView.Static)
        self.thumbnail_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.thumbnail_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thumbnail_view.setUniformItemSizes(True)
        self.thumbnail_view.setWrapping(True)
        self.thumbnail_view.setWordWrap(True)
        self.thumbnail_view.setSpacing(6)
        self.thumbnail_view.setIconSize(self.thumbnail_model.icon_size())
        self.thumbnail_view.setGridSize(self.thumbnail_model.grid_size())
        self.thumbnail_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbnail_view.setToolTip(i18n.tip("MASK_PREVIEW_MODE_THUMBNAILS"))
        self.thumbnail_view.doubleClicked.connect(self._on_thumbnail_double_clicked)
        self.thumbnail_view.selectionModel().currentChanged.connect(self._on_thumbnail_current_changed)
        self.thumbnail_view.selectionModel().selectionChanged.connect(
            lambda _selected, _deselected: self._update_reprocess_button_text()
        )
        self.thumbnail_view.verticalScrollBar().valueChanged.connect(lambda _value: self._queue_thumbnail_priority())
        self.thumbnail_view.horizontalScrollBar().valueChanged.connect(lambda _value: self._queue_thumbnail_priority())
        self.preview_stack.addWidget(self.thumbnail_view)

        layout.addWidget(self.preview_stack, stretch=1)

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
        self.yolo_preview_btn = QPushButton(i18n.t("MASK_PREVIEW_BUTTON"))
        self.yolo_preview_btn.setToolTip(i18n.tip("MASK_PREVIEW_BUTTON"))
        self.yolo_preview_btn.clicked.connect(self.mask_preview_requested.emit)
        overlay_row.addWidget(self.yolo_preview_btn)

        self.reprocess_current_btn = QPushButton(i18n.t("MASK_REPROCESS_CURRENT_BUTTON"))
        self.reprocess_current_btn.setToolTip(i18n.tip("MASK_REPROCESS_CURRENT_BUTTON"))
        self.reprocess_current_btn.clicked.connect(self.current_reprocess_requested.emit)
        overlay_row.addWidget(self.reprocess_current_btn)

        self.mask_overlay_btn = QToolButton()
        self.mask_overlay_btn.setObjectName("iconToolButton")
        self.mask_overlay_btn.setCheckable(True)
        self.mask_overlay_btn.setChecked(True)
        self.mask_overlay_btn.setIcon(mask_overlay_on_icon())
        self.mask_overlay_btn.setAccessibleName(i18n.t("MASK_OVERLAY_TOGGLE"))
        self.mask_overlay_btn.setToolTip(i18n.tip("MASK_OVERLAY_TOGGLE"))
        self.mask_overlay_btn.setFixedSize(28, 28)
        self.mask_overlay_btn.toggled.connect(self._on_mask_overlay_toggled)
        overlay_row.addWidget(self.mask_overlay_btn)

        self.status_label = ElidedStatusLabel("")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        overlay_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(overlay_row)
        self._update_reprocess_button_text()

    def set_images_dir(self, images_dir: str) -> None:
        if images_dir == self._images_dir:
            return
        self._images_dir = images_dir
        self.refresh_image_list(prefer_current=False)

    def render(self, config: MaskPreviewConfig) -> None:
        self._last_config = config
        self._sync_thumbnail_model(config, scroll=False)
        if self._preview_mode == PREVIEW_MODE_THUMBNAILS:
            self._update_mask_preview_button_text()
            self.status_label.setText(
                i18n.t("MASK_PREVIEW_THUMBNAIL_STATUS").format(count=len(self.preview_images))
            )
            self._queue_thumbnail_priority()
            return

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

        loaded = self._read_source_and_display(image_path)
        if loaded is None:
            self.image_label.setText(i18n.t("PREVIEW_LOAD_FAIL"))
            self.status_label.setText("")
            self._pixmap = None
            return
        source_img, img = loaded

        h, w = img.shape[:2]
        combined = np.full((h, w), 255, dtype=np.uint8)
        status_parts: list[str] = []

        temporary_mask = self._load_temporary_preview_mask(image_path, config)
        if temporary_mask is not None:
            if temporary_mask.shape != combined.shape:
                temporary_mask = cv2.resize(temporary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined = temporary_mask
            status_parts.append(i18n.t("MASK_PREVIEW_TEMP"))

        if temporary_mask is None and config.use_yolo:
            yolo_mask = self._load_yolo_preview_mask(image_path)
            yolo_status = i18n.t("MASK_PREVIEW_YOLO_TEMP") if yolo_mask is not None else ""
            if yolo_mask is None:
                yolo_mask = self._load_existing_mask(image_path, config, combined.shape)
                yolo_status = i18n.t("MASK_PREVIEW_YOLO_EXISTING") if yolo_mask is not None else ""
            if yolo_mask is None:
                status_parts.append(i18n.t("MASK_PREVIEW_YOLO_PENDING"))
            else:
                if yolo_mask.shape != combined.shape:
                    yolo_mask = cv2.resize(yolo_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                combined = cv2.bitwise_and(combined, yolo_mask)
                status_parts.append(yolo_status)

        if temporary_mask is None and config.use_stitch:
            if config.stitch_boundary_width_deg is None:
                status_parts.append(i18n.t("MASK_PREVIEW_INVALID_STITCH_WIDTH"))
            else:
                stitch = self._stitch_mask(w, h, config.stitch_boundary_width_deg)
                combined = cv2.bitwise_and(combined, stitch)
                status_parts.append(
                    i18n.t("MASK_PREVIEW_STITCH_STATUS").format(
                        width=config.stitch_boundary_width_deg,
                    )
                )

        if temporary_mask is None and config.use_overexposure:
            overexp = self._overexposure_mask(
                image_path,
                source_img,
                combined.shape,
                int(config.overexposure_threshold),
                int(config.overexposure_dilate),
            )
            combined = cv2.bitwise_and(combined, overexp)
            status_parts.append(
                i18n.t("MASK_PREVIEW_OVEREXP_STATUS").format(
                    threshold=config.overexposure_threshold,
                    dilate=config.overexposure_dilate,
                    )
                )

        if temporary_mask is None and config.use_sky:
            sky_mask = self._load_existing_mask(image_path, config, combined.shape)
            if sky_mask is None:
                status_parts.append(i18n.t("MASK_PREVIEW_SKY_PENDING"))
            else:
                if sky_mask.shape != combined.shape:
                    sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                combined = cv2.bitwise_and(combined, sky_mask)
                status_parts.append(i18n.t("MASK_PREVIEW_SKY_EXISTING"))

        if temporary_mask is None and config.use_custom:
            custom = self._custom_mask_for_preview(config, source_img.shape[:2], combined.shape)
            if custom is None:
                status_parts.append(i18n.t("MASK_PREVIEW_CUSTOM_INVALID"))
            else:
                combined = cv2.bitwise_and(combined, custom)
                status_parts.append(i18n.t("MASK_PREVIEW_CUSTOM_STATUS"))

        excluded = combined < 128
        alpha = self._mask_overlay_alpha()
        if alpha > 0 and np.any(excluded):
            overlay = np.zeros_like(img)
            overlay[:, :, 2] = 255
            img[excluded] = (
                (1.0 - alpha) * img[excluded].astype(np.float32)
                + alpha * overlay[excluded].astype(np.float32)
            ).astype(np.uint8)

        if alpha > 0 and np.any(excluded):
            contours, _ = cv2.findContours(
                excluded.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(img, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        self.status_label.setText(
            " / ".join(status_parts) if status_parts else i18n.t("MASK_PREVIEW_NO_ACTIVE_MASK")
        )
        self._update_mask_preview_button_text()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()

    def refresh_image_list(self, prefer_current: bool = True, *, force_thumbnails: bool = False) -> None:
        current = self._current_image_path.strip()
        old_images = self.preview_images
        self.preview_images = self._iter_images()
        force = force_thumbnails or tuple(old_images) != tuple(self.preview_images)
        self._sync_thumbnail_model(self._last_config, force=force, scroll=False)
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
        self._set_index(target, scroll_thumbnail=False)
        self._queue_thumbnail_priority()

    def _iter_images(self) -> list[Path]:
        return iter_image_files(self._images_dir)

    def _set_index(self, idx: int, *, sync_thumbnail: bool = True, scroll_thumbnail: bool = False) -> None:
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
        if sync_thumbnail:
            self._sync_thumbnail_selection(idx, scroll=scroll_thumbnail)

    def set_current_image_path(self, image_path: str | Path) -> None:
        self._set_current_image_path(str(image_path), emit=False)

    def _set_current_image_path(self, image_path: str, emit: bool) -> None:
        if image_path == self._current_image_path:
            return
        self._current_image_path = image_path
        if emit:
            self.current_image_changed.emit()

    def _load_existing_mask(
        self,
        image_path: Path,
        config: MaskPreviewConfig,
        target_shape: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        for candidate in mask_candidates_for_image(image_path, self._images_dir, config.masks_dir):
            if not candidate.is_file():
                continue
            mask = self._read_mask(candidate, target_shape)
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

    def clear_temporary_preview_mask(self, image_path: Path | None = None) -> None:
        if image_path is not None and self._temporary_preview_image_key != _path_key(image_path):
            return
        self._temporary_preview_image_key = ""
        self._temporary_preview_config_key = None
        self._temporary_preview_mask = None
        self._update_mask_preview_button_text()

    def clear_yolo_preview_mask(self, image_path: Path | None = None) -> None:
        self.clear_temporary_preview_mask(image_path)

    def set_temporary_preview_mask(self, image_path: Path, mask_path: Path, config: MaskPreviewConfig) -> bool:
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        self._temporary_preview_image_key = _path_key(image_path)
        self._temporary_preview_config_key = _preview_config_key(config)
        self._temporary_preview_mask = mask
        return True

    def set_yolo_preview_mask(self, image_path: Path, mask_path: Path) -> bool:
        return self.set_temporary_preview_mask(image_path, mask_path, MaskPreviewConfig(use_yolo=True))

    def set_mask_preview_running(self, running: bool) -> None:
        self.yolo_preview_btn.setEnabled(not running)
        self._update_mask_preview_button_text(running=running)

    def set_yolo_preview_running(self, running: bool) -> None:
        self.set_mask_preview_running(running)

    def set_current_reprocess_running(self, running: bool) -> None:
        self.reprocess_current_btn.setEnabled(not running)
        self.yolo_preview_btn.setEnabled(not running)
        self.thumbnail_view.setEnabled(not running)
        if running:
            self.reprocess_current_btn.setText(i18n.t("MASK_REPROCESS_CURRENT_RUNNING"))
        else:
            self._update_reprocess_button_text()

    def wait_for_thumbnail_rendering(self, timeout_ms: int = 3000) -> bool:
        return self.thumbnail_model.wait_for_done(timeout_ms)

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(text)

    def has_active_temporary_preview(self, config: MaskPreviewConfig | None = None) -> bool:
        if self._preview_mode != PREVIEW_MODE_SINGLE:
            return False
        image_path = self.current_image_path()
        if image_path is None:
            return False
        return self._load_temporary_preview_mask(image_path, config or self._last_config) is not None

    def _load_yolo_preview_mask(self, image_path: Path) -> np.ndarray | None:
        return None

    def _load_temporary_preview_mask(self, image_path: Path, config: MaskPreviewConfig) -> np.ndarray | None:
        if self._temporary_preview_image_key != _path_key(image_path):
            return None
        if self._temporary_preview_config_key != _preview_config_key(config):
            return None
        return self._temporary_preview_mask

    def _on_slider_changed(self, idx: int) -> None:
        if self._slider_sync:
            return
        if 0 <= idx < len(self.preview_images):
            self._set_index(idx, scroll_thumbnail=True)

    def set_preview_mode(self, mode: str) -> None:
        if mode not in {PREVIEW_MODE_SINGLE, PREVIEW_MODE_THUMBNAILS}:
            return
        if mode == self._preview_mode:
            return
        self._preview_mode = mode
        self.preview_stack.setCurrentIndex(1 if mode == PREVIEW_MODE_THUMBNAILS else 0)
        self.mode_toolbar.set_mode(mode)
        self._update_reprocess_button_text()
        self.render(self._last_config)
        if mode == PREVIEW_MODE_THUMBNAILS:
            self.thumbnail_view.setFocus(Qt.OtherFocusReason)

    def preview_mode(self) -> str:
        return self._preview_mode

    def _on_mask_overlay_toggled(self, checked: bool) -> None:
        self._mask_overlay_visible = checked
        self.mask_overlay_btn.setIcon(mask_overlay_on_icon() if checked else mask_overlay_off_icon())
        self.thumbnail_delegate.set_overlay_visible(checked)
        if self._preview_mode == PREVIEW_MODE_THUMBNAILS:
            self.thumbnail_view.viewport().update()
            self.thumbnail_view.setFocus(Qt.OtherFocusReason)
            return
        self.render(self._last_config)

    def selected_reprocess_image_paths(self) -> list[Path]:
        if self._preview_mode != PREVIEW_MODE_THUMBNAILS:
            current = self.current_image_path()
            return [current] if current is not None else []

        rows = sorted({index.row() for index in self.thumbnail_view.selectionModel().selectedIndexes()})
        selected = [self.thumbnail_model.image_at(row) for row in rows]
        paths = [path for path in selected if path is not None and path.exists() and path.is_file()]
        if paths:
            return paths

        current = self.current_image_path()
        return [current] if current is not None else []

    def _on_thumbnail_current_changed(self, current, _previous) -> None:  # noqa: ANN001
        if self._thumbnail_sync or not current.isValid():
            return
        self._set_index(current.row(), sync_thumbnail=False)

    def _on_thumbnail_double_clicked(self, index) -> None:  # noqa: ANN001
        if not index.isValid():
            return
        self._set_index(index.row(), sync_thumbnail=False)
        self.set_preview_mode(PREVIEW_MODE_SINGLE)

    def _sync_thumbnail_model(
        self,
        config: MaskPreviewConfig,
        *,
        force: bool = False,
        scroll: bool = False,
    ) -> None:
        signature = (
            tuple(str(path) for path in self.preview_images),
            self._images_dir,
            config.masks_dir,
        )
        if force or signature != self._thumbnail_model_signature:
            self.thumbnail_model.set_sources(
                self.preview_images,
                images_dir=self._images_dir,
                masks_dir=config.masks_dir,
                force=force,
            )
            self._thumbnail_model_signature = signature
        self._sync_thumbnail_selection(self.slider.value(), scroll=scroll)
        self._queue_thumbnail_priority()

    def _sync_thumbnail_selection(self, idx: int, *, scroll: bool = False) -> None:
        if not (0 <= idx < len(self.preview_images)):
            return
        model_index = self.thumbnail_model.index(idx, 0)
        if not model_index.isValid():
            return
        self._thumbnail_sync = True
        try:
            selected_rows = {
                index.row()
                for index in self.thumbnail_view.selectionModel().selectedIndexes()
                if index.isValid()
            }
            flags = (
                QItemSelectionModel.NoUpdate
                if selected_rows
                else QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Current
            )
            self.thumbnail_view.selectionModel().setCurrentIndex(model_index, flags)
            if scroll and self._preview_mode == PREVIEW_MODE_THUMBNAILS:
                self.thumbnail_view.scrollTo(model_index, QAbstractItemView.EnsureVisible)
        finally:
            self._thumbnail_sync = False

    def _update_reprocess_button_text(self) -> None:
        if self._preview_mode == PREVIEW_MODE_THUMBNAILS:
            count = len(self.thumbnail_view.selectionModel().selectedIndexes())
            if count > 0:
                self.reprocess_current_btn.setText(
                    i18n.t("MASK_REPROCESS_SELECTED_BUTTON").format(count=count)
                )
                return
            self.reprocess_current_btn.setText(i18n.t("MASK_REPROCESS_SELECTED_FALLBACK_BUTTON"))
            return
        self.reprocess_current_btn.setText(i18n.t("MASK_REPROCESS_CURRENT_BUTTON"))

    def _update_mask_preview_button_text(self, *, running: bool = False) -> None:
        if running:
            self.yolo_preview_btn.setText(i18n.t("MASK_PREVIEW_RUNNING"))
            self.yolo_preview_btn.setToolTip(i18n.tip("MASK_PREVIEW_BUTTON"))
            return
        if self.has_active_temporary_preview(self._last_config):
            self.yolo_preview_btn.setText(i18n.t("MASK_PREVIEW_CLEAR_BUTTON"))
            self.yolo_preview_btn.setToolTip(i18n.tip("MASK_PREVIEW_CLEAR_BUTTON"))
            return
        self.yolo_preview_btn.setText(i18n.t("MASK_PREVIEW_BUTTON"))
        self.yolo_preview_btn.setToolTip(i18n.tip("MASK_PREVIEW_BUTTON"))

    def _update_pixmap(self) -> None:
        self.image_label.set_source_pixmap(self._pixmap)

    def _mask_overlay_opacity(self) -> int:
        return _MASK_OVERLAY_OPACITY if self._mask_overlay_visible else 0

    def _mask_overlay_alpha(self) -> float:
        return float(self._mask_overlay_opacity()) / 100.0

    def invalidate_thumbnail_images(self, images: list[Path] | set[Path]) -> None:
        self.thumbnail_model.invalidate_images(images)
        self._queue_thumbnail_priority()

    def _queue_thumbnail_priority(self) -> None:
        if self._preview_mode != PREVIEW_MODE_THUMBNAILS:
            return
        self._thumbnail_priority_timer.start()

    def _prioritize_visible_thumbnails(self) -> None:
        if self._preview_mode != PREVIEW_MODE_THUMBNAILS:
            return
        try:
            rows = visible_rows_for_view(self.thumbnail_view)
        except RuntimeError:
            return
        self.thumbnail_model.prioritize_rows(rows, prefetch=192)

    def shutdown(self) -> None:
        self._thumbnail_priority_timer.stop()
        self.thumbnail_model.shutdown()

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt API
        self.shutdown()
        super().closeEvent(event)

    def _cache_key(self, path: Path, *extra: object) -> tuple | None:
        try:
            st = path.stat()
            return (str(path.resolve()).lower(), int(st.st_size), int(st.st_mtime_ns), *extra)
        except OSError:
            return None

    @staticmethod
    def _store_cache(cache: OrderedDict, key: tuple, value, limit: int = _LAYER_CACHE_LIMIT) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _read_source_and_display(self, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        max_w = 1900
        key = self._cache_key(path, "image", max_w)
        if key is not None and key in self._image_cache:
            self._image_cache.move_to_end(key)
            source, display = self._image_cache[key]
            return source.copy(), display.copy()

        source = read_image_preserve_depth(str(path))
        display = _display_bgr8(source)
        if source is None or display is None:
            return None
        if display.shape[1] > max_w:
            scale = max_w / float(display.shape[1])
            display = cv2.resize(
                display,
                (max(1, int(display.shape[1] * scale)), max(1, int(display.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        if key is not None:
            self._store_cache(self._image_cache, key, (source, display), _IMAGE_CACHE_LIMIT)
        return source.copy(), display.copy()

    def _read_mask(self, path: Path, target_shape: tuple[int, int] | None) -> np.ndarray | None:
        extra: tuple[object, ...] = ("mask",)
        if target_shape is not None:
            extra = ("mask", int(target_shape[1]), int(target_shape[0]))
        key = self._cache_key(path, *extra)
        if key is not None and key in self._mask_cache:
            self._mask_cache.move_to_end(key)
            return self._mask_cache[key].copy()

        mask = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if target_shape is not None and mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        if key is not None:
            self._store_cache(self._mask_cache, key, mask)
        return mask.copy()

    def _stitch_mask(self, width: int, height: int, boundary_width_deg: float) -> np.ndarray:
        limit_angle = boundary_width_to_limit_angle(boundary_width_deg)
        key = ("stitch", int(width), int(height), round(float(limit_angle), 6))
        if key in self._stitch_cache:
            self._stitch_cache.move_to_end(key)
            return self._stitch_cache[key].copy()
        mask = create_angular_stitched_mask(width, height, limit_angle)
        self._store_cache(self._stitch_cache, key, mask)
        return mask.copy()

    def _overexposure_mask(
        self,
        path: Path,
        source_img: np.ndarray,
        target_shape: tuple[int, int],
        threshold: int,
        dilate_px: int,
    ) -> np.ndarray:
        key = self._cache_key(path, "overexp", int(threshold), int(dilate_px), target_shape[1], target_shape[0])
        if key is not None and key in self._overexp_cache:
            self._overexp_cache.move_to_end(key)
            return self._overexp_cache[key].copy()

        mask = detect_overexposure(source_img, threshold=threshold, dilate_px=dilate_px)
        if mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        if key is not None:
            self._store_cache(self._overexp_cache, key, mask)
        return mask.copy()

    def _custom_mask_for_preview(
        self,
        config: MaskPreviewConfig,
        source_shape: tuple[int, int],
        display_shape: tuple[int, int],
    ) -> np.ndarray | None:
        custom_path_text = config.custom_mask_path.strip()
        if not custom_path_text:
            return None
        loaded_custom, _load_error = load_custom_mask(custom_path_text)
        if loaded_custom is None:
            return None
        custom = loaded_custom.mask
        if custom.shape == display_shape:
            return custom
        if custom.shape == source_shape:
            return cv2.resize(
                custom,
                (display_shape[1], display_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return None


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
    return path_key(path)


def _preview_config_key(config: MaskPreviewConfig) -> tuple:
    return (
        bool(config.use_yolo),
        bool(config.use_stitch),
        bool(config.use_overexposure),
        bool(config.use_sky),
        bool(config.use_custom),
        None if config.stitch_boundary_width_deg is None else round(float(config.stitch_boundary_width_deg), 6),
        int(config.overexposure_threshold),
        int(config.overexposure_dilate),
        str(config.custom_mask_path),
        tuple(config.settings_key),
    )
