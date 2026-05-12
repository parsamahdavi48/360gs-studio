#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

try:
    from PySide6.QtCore import QItemSelectionModel, QSize, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QHBoxLayout,
        QLabel,
        QListView,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QStackedWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:  # pragma: no cover - environment-dependent import
    QItemSelectionModel = None
    QSize = None
    Qt = None
    QTimer = None
    Signal = None
    QColor = None
    QIcon = None
    QImage = None
    QKeySequence = None
    QPainter = None
    QPen = None
    QPixmap = None
    QShortcut = None
    QAbstractItemView = None
    QApplication = None
    QHBoxLayout = None
    QLabel = None
    QListView = None
    QMainWindow = None
    QMessageBox = None
    QPushButton = None
    QSlider = None
    QStackedWidget = None
    QToolButton = None
    QVBoxLayout = None
    QWidget = None
    _PYSIDE_IMPORT_ERROR = e
else:
    _PYSIDE_IMPORT_ERROR = None

# i18n は PySide6 に依存しないので無条件 import
from core.apply_frame_decisions import pending_drop_image_paths as find_pending_drop_image_paths
from core.scene_layout import selected_frames_path
from gui import i18n

if _PYSIDE_IMPORT_ERROR is None:
    import cv2
    import numpy as np

    from core.image_io import imread_unicode
    from gui.common.perspective_image_view import PerspectiveImageView
    from gui.common.perspective_preview import (
        PREVIEW_PROJECTION_EQUIRECT,
        PREVIEW_PROJECTION_PERSPECTIVE,
        PerspectiveParams,
        equirect_to_perspective,
        params_from_drag,
    )
    from gui.common.preview_mode_toolbar import (
        PREVIEW_MODE_PERSPECTIVE,
        PREVIEW_MODE_SINGLE,
        PREVIEW_MODE_THUMBNAILS,
        PreviewModeToolbar,
    )
    from gui.common.thumbnail_delegate import ThumbnailSelectionDelegate
    from gui.common.thumbnail_list_model import AsyncThumbnailModel, ThumbnailItem, visible_rows_for_view
else:  # pragma: no cover - PySide6 missing
    PREVIEW_MODE_PERSPECTIVE = "perspective"
    PREVIEW_MODE_SINGLE = "single"
    PREVIEW_MODE_THUMBNAILS = "thumbnails"
    PREVIEW_PROJECTION_EQUIRECT = "equirect"
    PREVIEW_PROJECTION_PERSPECTIVE = "perspective"
    PreviewModeToolbar = None
    ThumbnailSelectionDelegate = None
    AsyncThumbnailModel = None
    ThumbnailItem = None
    visible_rows_for_view = None
    PerspectiveImageView = None


_ICON_DIR = Path(__file__).resolve().parents[1] / "gui" / "assets" / "icons"
_PIXMAP_CACHE_LIMIT = 3


def _review_icon(name: str) -> QIcon:
    return QIcon(str(_ICON_DIR / f"{name}.svg"))


def _review_thumbnail_image(item: ThumbnailItem, size: QSize) -> QImage:
    decision = str(item.cache_key[0]) if len(item.cache_key) >= 1 else "keep"
    advisory_fg = str(item.cache_key[3]) if len(item.cache_key) >= 4 and item.cache_key[3] else "#e5e7eb"
    advisory_bg = str(item.cache_key[4]) if len(item.cache_key) >= 5 and item.cache_key[4] else "#14532d"
    advisory_short = str(item.cache_key[5]) if len(item.cache_key) >= 6 and item.cache_key[5] else ""
    ribbon = QColor(advisory_bg)
    text = advisory_short or (i18n.t("REVIEW_DECISION_DROP") if decision == "drop" else i18n.t("REVIEW_DECISION_KEEP"))

    canvas = QImage(size, QImage.Format_ARGB32)
    canvas.fill(QColor("#101316"))
    painter = QPainter(canvas)

    image = QImage(str(item.path))
    if image.isNull():
        painter.setPen(QPen(QColor("#ef4444"), 2))
        painter.drawLine(8, 8, size.width() - 8, size.height() - 8)
        painter.drawLine(size.width() - 8, 8, 8, size.height() - 8)
    else:
        scaled = image.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = max(0, (size.width() - scaled.width()) // 2)
        y = max(0, (size.height() - scaled.height()) // 2)
        painter.drawImage(x, y, scaled)

    painter.fillRect(0, size.height() - 18, size.width(), 18, ribbon)
    painter.setPen(QColor(advisory_fg))
    elided_text = painter.fontMetrics().elidedText(text, Qt.ElideRight, max(1, size.width() - 12))
    painter.drawText(6, size.height() - 4, elided_text)
    painter.end()
    return canvas


def _bgr_to_pixmap(img) -> QPixmap:  # noqa: ANN001 - numpy type is optional at import time
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _perspective_pixmap_for_review(image_path: Path, params: PerspectiveParams) -> QPixmap:
    img = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return QPixmap()
    output_size = max(1, min(950, img.shape[0], img.shape[1]))
    return _bgr_to_pixmap(equirect_to_perspective(img, params, output_size=output_size))


def _perspective_bgr_for_review(image_path: Path) -> np.ndarray | None:
    img = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    max_w = 1900
    if img.shape[1] > max_w:
        scale = max_w / float(img.shape[1])
        img = cv2.resize(
            img,
            (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


if QMainWindow is not None:

    class ReviewWidget(QWidget):
        decisions_changed = Signal()

        def __init__(self, scene_dir: Path, csv_path: Path) -> None:
            super().__init__()
            self.scene_dir = scene_dir
            self.csv_path = csv_path
            self._all_rows = self._load_rows(csv_path)
            if not self._all_rows:
                raise RuntimeError(f"No rows found in {csv_path}")
            self._initial_decisions = [row["decision"] for row in self._all_rows]
            self._source_filter_key = "all"
            self._visible_indices = list(range(len(self._all_rows)))
            self.rows = list(self._all_rows)
            self.problem_indices = self._collect_problem_indices()

            self.index = 0
            self._slider_sync = False
            self._thumbnail_sync = False
            self._preview_mode = PREVIEW_MODE_SINGLE
            self._preview_projection = PREVIEW_PROJECTION_EQUIRECT
            self._perspective_params = PerspectiveParams()
            self.current_pixmap: QPixmap | None = None
            self._pixmap_cache: OrderedDict[tuple, QPixmap] = OrderedDict()
            self._thumbnail_priority_timer = QTimer(self)
            self._thumbnail_priority_timer.setSingleShot(True)
            self._thumbnail_priority_timer.setInterval(0)
            self._thumbnail_priority_timer.timeout.connect(self._prioritize_visible_thumbnails)

            self._build_ui()
            self._bind_shortcuts()
            self._render_current()

        def source_filter_options(self) -> list[dict[str, str]]:
            options = [
                {
                    "key": "all",
                    "label": i18n.t("REVIEW_SOURCE_FILTER_ALL").format(n=len(self._all_rows)),
                }
            ]
            groups: dict[str, dict[str, object]] = {}
            for idx, row in enumerate(self._all_rows):
                session = row.get("source_session", "").strip()
                video = row.get("source_video", "").strip()
                if session:
                    key = f"session:{session}"
                elif video:
                    key = f"video:{video}"
                else:
                    key = "unassigned"
                group = groups.setdefault(
                    key,
                    {
                        "count": 0,
                        "name": Path(video).name if video else i18n.t("REVIEW_SOURCE_FILTER_UNASSIGNED"),
                        "first_index": idx,
                    },
                )
                group["count"] = int(group["count"]) + 1

            for key, group in sorted(groups.items(), key=lambda item: int(item[1]["first_index"])):
                label = i18n.t("REVIEW_SOURCE_FILTER_ITEM").format(
                    name=str(group["name"]),
                    n=int(group["count"]),
                )
                options.append({"key": key, "label": label})
            return options

        def set_source_filter(self, key: str) -> None:
            key = key or "all"
            if key == self._source_filter_key:
                return
            self._source_filter_key = key
            if key == "all":
                self._visible_indices = list(range(len(self._all_rows)))
            elif key == "unassigned":
                self._visible_indices = [
                    idx
                    for idx, row in enumerate(self._all_rows)
                    if not row.get("source_session", "").strip() and not row.get("source_video", "").strip()
                ]
            elif key.startswith("session:"):
                session = key.split(":", 1)[1]
                self._visible_indices = [
                    idx for idx, row in enumerate(self._all_rows) if row.get("source_session", "").strip() == session
                ]
            elif key.startswith("video:"):
                video = key.split(":", 1)[1]
                self._visible_indices = [
                    idx for idx, row in enumerate(self._all_rows) if row.get("source_video", "").strip() == video
                ]
            else:
                self._visible_indices = list(range(len(self._all_rows)))
            if not self._visible_indices:
                self._visible_indices = list(range(len(self._all_rows)))
            self.rows = [self._all_rows[idx] for idx in self._visible_indices]
            self.problem_indices = self._collect_problem_indices()
            self.index = 0
            self._pixmap_cache.clear()
            self._slider_sync = True
            self.frame_slider.setRange(0, max(0, len(self.rows) - 1))
            self.frame_slider.setEnabled(len(self.rows) > 1)
            self.frame_slider.setValue(0)
            self._slider_sync = False
            self._sync_thumbnail_model(force=True)
            self._render_current()

        def _all_index(self, visible_index: int) -> int:
            if not (0 <= visible_index < len(self._visible_indices)):
                return visible_index
            return self._visible_indices[visible_index]

        def _load_rows(self, path: Path) -> list[dict[str, str]]:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for row in rows:
                decision = row.get("decision", "keep").strip().lower()
                row["decision"] = "drop" if decision == "drop" else "keep"
            return rows

        def _is_problem_row(self, row: dict[str, str]) -> bool:
            status = row.get("status", "").strip().lower()
            return status not in {"", "ok"}

        def _collect_problem_indices(self) -> list[int]:
            return [i for i, row in enumerate(self.rows) if self._is_problem_row(row)]

        def _build_ui(self) -> None:
            layout = QVBoxLayout(self)

            top_row = QHBoxLayout()
            self.title_label = QLabel()
            self.title_label.setStyleSheet("font-weight: 700;")
            top_row.addWidget(self.title_label)

            top_row.addStretch(1)

            self.decision_label = QLabel()
            self.decision_label.setStyleSheet("font-weight: 700;")
            top_row.addWidget(self.decision_label)

            self.flag_button = QToolButton()
            self.flag_button.setCheckable(True)
            self.flag_button.setText("")
            self.flag_button.setToolTip(i18n.t("REVIEW_FLAG_TIP"))
            self.flag_button.setIconSize(QSize(20, 20))
            self.flag_button.setFixedSize(36, 32)
            self.flag_button.clicked.connect(lambda _checked=False: self.toggle_decision())
            top_row.addWidget(self.flag_button)

            self.reset_decision_button = QToolButton()
            self.reset_decision_button.setText("")
            self.reset_decision_button.setIcon(_review_icon("rotate-ccw"))
            self.reset_decision_button.setToolTip(i18n.t("REVIEW_RESET_DECISION_TIP"))
            self.reset_decision_button.setIconSize(QSize(20, 20))
            self.reset_decision_button.setFixedSize(36, 32)
            self.reset_decision_button.clicked.connect(lambda _checked=False: self.reset_decision())
            top_row.addWidget(self.reset_decision_button)

            self.mode_toolbar = PreviewModeToolbar(
                single_text_key="REVIEW_PREVIEW_MODE_SINGLE",
                thumbnail_text_key="REVIEW_PREVIEW_MODE_THUMBNAILS",
                single_tip_key="REVIEW_PREVIEW_MODE_SINGLE",
                thumbnail_tip_key="REVIEW_PREVIEW_MODE_THUMBNAILS",
                include_perspective=True,
            )
            self.mode_toolbar.mode_changed.connect(self.set_preview_mode)
            self.projection_toggle_btn = self.mode_toolbar.perspective_preview_btn
            if self.projection_toggle_btn is None:
                raise RuntimeError("Perspective preview button was not created")
            top_row.addWidget(self.mode_toolbar)
            layout.addLayout(top_row)

            self.preview_stack = QStackedWidget()
            self.image_view = PerspectiveImageView()
            self.image_view.setMinimumHeight(260)
            self.image_view.setStyleSheet("border: 1px solid palette(mid);")
            self.image_view.look_dragged.connect(self._on_perspective_dragged)
            self.image_view.gpu_failed.connect(self._render_current)
            self.preview_stack.addWidget(self.image_view)

            self.thumbnail_model = AsyncThumbnailModel(self)
            self.thumbnail_view = QListView()
            self.thumbnail_view.setModel(self.thumbnail_model)
            self.thumbnail_view.setItemDelegate(ThumbnailSelectionDelegate(self.thumbnail_view))
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
            self.thumbnail_view.setToolTip(i18n.tip("REVIEW_PREVIEW_MODE_THUMBNAILS"))
            self.thumbnail_view.selectionModel().currentChanged.connect(self._on_thumbnail_current_changed)
            self.thumbnail_view.selectionModel().selectionChanged.connect(
                lambda _selected, _deselected: self._on_thumbnail_selection_changed()
            )
            self.thumbnail_view.doubleClicked.connect(self._on_thumbnail_double_clicked)
            self.thumbnail_view.verticalScrollBar().valueChanged.connect(
                lambda _value: self._queue_thumbnail_priority()
            )
            self.thumbnail_view.horizontalScrollBar().valueChanged.connect(
                lambda _value: self._queue_thumbnail_priority()
            )
            self.preview_stack.addWidget(self.thumbnail_view)

            layout.addWidget(self.preview_stack, stretch=1)

            timeline_row = QHBoxLayout()
            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setToolTip(i18n.t("REVIEW_FRAME_SLIDER_TIP"))
            self.frame_slider.setRange(0, max(0, len(self.rows) - 1))
            self.frame_slider.setEnabled(len(self.rows) > 1)
            self.frame_slider.valueChanged.connect(self._on_slider_changed)
            timeline_row.addWidget(self.frame_slider, stretch=1)

            self.frame_position_label = QLabel()
            timeline_row.addWidget(self.frame_position_label)
            layout.addLayout(timeline_row)

            # アドバイザリー（このフレームが要注意な理由を自動表示）
            self.advisory_label = QLabel()
            self.advisory_label.setWordWrap(False)
            self.advisory_label.setStyleSheet(
                "padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 10pt;"
            )
            self.advisory_label.setFixedHeight(32)
            layout.addWidget(self.advisory_label)

            self.info_label = QLabel()
            self.info_label.setWordWrap(True)
            layout.addWidget(self.info_label)

            self.problem_summary_label = QLabel()
            self.problem_summary_label.setWordWrap(False)
            self.problem_summary_label.setStyleSheet("color: palette(text); font-weight: 500;")
            layout.addWidget(self.problem_summary_label)

            btn_row = QHBoxLayout()
            self.prev_problem_button = QPushButton(i18n.t("REVIEW_BTN_PREV_PROBLEM"))
            self.prev_problem_button.setToolTip(i18n.t("REVIEW_BTN_PROBLEM_TIP"))
            self.prev_problem_button.clicked.connect(self.prev_problem)
            btn_row.addWidget(self.prev_problem_button)

            self.next_problem_button = QPushButton(i18n.t("REVIEW_BTN_NEXT_PROBLEM"))
            self.next_problem_button.setToolTip(i18n.t("REVIEW_BTN_PROBLEM_TIP"))
            self.next_problem_button.clicked.connect(self.next_problem)
            btn_row.addWidget(self.next_problem_button)

            btn_row.addStretch(1)
            layout.addLayout(btn_row)

        def _bind_shortcuts(self) -> None:
            self.prev_row_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_row)
            self.next_row_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_row)
            QShortcut(QKeySequence("F"), self, activated=self.next_problem)
            QShortcut(QKeySequence("Shift+F"), self, activated=self.prev_problem)
            QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_decision)
            QShortcut(QKeySequence("Q"), self, activated=self._close_review_window)
            QShortcut(QKeySequence("0"), self, activated=self.reset_zoom)

        def _close_review_window(self) -> None:
            window = self.window()
            if isinstance(window, ReviewWindow):
                window.close()

        def set_preview_mode(self, mode: str) -> None:
            if mode not in {PREVIEW_MODE_PERSPECTIVE, PREVIEW_MODE_SINGLE, PREVIEW_MODE_THUMBNAILS}:
                return
            if mode == self._preview_mode:
                return
            self._preview_mode = mode
            projection = (
                PREVIEW_PROJECTION_PERSPECTIVE if mode == PREVIEW_MODE_PERSPECTIVE else PREVIEW_PROJECTION_EQUIRECT
            )
            self._set_preview_projection(projection)
            self.preview_stack.setCurrentIndex(1 if mode == PREVIEW_MODE_THUMBNAILS else 0)
            self.mode_toolbar.set_mode(mode)
            thumbnail_mode = mode == PREVIEW_MODE_THUMBNAILS
            self.prev_row_shortcut.setEnabled(not thumbnail_mode)
            self.next_row_shortcut.setEnabled(not thumbnail_mode)
            self._render_current()
            if thumbnail_mode:
                self._focus_thumbnail_view_if_active()

        def preview_mode(self) -> str:
            return self._preview_mode

        def preview_projection(self) -> str:
            return self._preview_projection

        def _set_preview_projection(self, projection: str) -> None:
            if projection == self._preview_projection:
                self._update_projection_button()
                return
            self._preview_projection = projection
            if projection == PREVIEW_PROJECTION_PERSPECTIVE:
                self._perspective_params = PerspectiveParams()
            self.image_view.set_drag_mode("look" if projection == PREVIEW_PROJECTION_PERSPECTIVE else "pan")
            self.image_view.reset_view()
            self._update_projection_button()

        def _update_projection_button(self) -> None:
            perspective = self._preview_projection == PREVIEW_PROJECTION_PERSPECTIVE
            self.projection_toggle_btn.blockSignals(True)
            try:
                self.projection_toggle_btn.setChecked(perspective)
            finally:
                self.projection_toggle_btn.blockSignals(False)
            self.projection_toggle_btn.setToolTip(i18n.tip("PREVIEW_PROJECTION_TOGGLE"))

        def _on_perspective_dragged(self, delta_x: float, delta_y: float) -> None:
            if self._preview_projection != PREVIEW_PROJECTION_PERSPECTIVE:
                return
            self._perspective_params = params_from_drag(self._perspective_params, delta_x, delta_y)
            if self.image_view.set_perspective_params(self._perspective_params):
                return
            self._render_current()

        def _focus_thumbnail_view_if_active(self) -> None:
            if self._preview_mode == PREVIEW_MODE_THUMBNAILS:
                self.thumbnail_view.setFocus(Qt.OtherFocusReason)

        def _set_index(
            self,
            idx: int,
            *,
            sync_thumbnail: bool = True,
            scroll_thumbnail: bool = False,
        ) -> None:
            self.index = max(0, min(idx, len(self.rows) - 1))
            self._render_current(sync_thumbnail=sync_thumbnail, scroll_thumbnail=scroll_thumbnail)

        def _on_thumbnail_current_changed(self, current, _previous) -> None:  # noqa: ANN001
            if self._thumbnail_sync or not current.isValid():
                return
            self._set_index(current.row(), sync_thumbnail=False)

        def _on_thumbnail_selection_changed(self) -> None:
            if self._thumbnail_sync:
                return
            self._update_decision_buttons(self._current_row().get("decision", "keep"))

        def _on_thumbnail_double_clicked(self, index) -> None:  # noqa: ANN001
            if not index.isValid():
                return
            self._set_index(index.row(), sync_thumbnail=False)
            self.set_preview_mode(PREVIEW_MODE_SINGLE)

        def _thumbnail_item_for_row(self, idx: int) -> ThumbnailItem:
            row = self.rows[idx]
            rel = row.get("output_file", "")
            path = self.scene_dir / rel
            decision = row.get("decision", "keep")
            status = row.get("status", "")
            pipeline = row.get("analysis_pipeline", "")
            adv_text, adv_short, adv_fg, adv_bg = self._advisory_for_row(row, idx)
            seq = row.get("seq", str(idx + 1))
            name = Path(rel).name
            return ThumbnailItem(
                path=path,
                label=name,
                tooltip=f"{seq}: {name} / {adv_text} / {self._decision_text(decision)}",
                cache_key=(decision, status, pipeline, adv_fg, adv_bg, adv_short),
            )

        def _sync_thumbnail_model(self, *, force: bool = False) -> None:
            if not force and self.thumbnail_model.rowCount() == len(self.rows):
                return

            items = [self._thumbnail_item_for_row(idx) for idx in range(len(self.rows))]
            self.thumbnail_model.set_items(
                items,
                _review_thumbnail_image,
                renderer_key=("review",),
                force=force,
            )
            self._queue_thumbnail_priority()

        def _refresh_thumbnail_row(self, idx: int) -> None:
            if self.thumbnail_model.rowCount() != len(self.rows):
                self._sync_thumbnail_model(force=True)
                return
            self.thumbnail_model.set_item(idx, self._thumbnail_item_for_row(idx))

        def _sync_thumbnail_selection(self, idx: int, *, scroll: bool = False) -> None:
            if not (0 <= idx < len(self.rows)):
                return
            model_index = self.thumbnail_model.index(idx, 0)
            if not model_index.isValid():
                return
            self._thumbnail_sync = True
            try:
                selected_rows = self._selected_thumbnail_rows()
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

        def _selected_thumbnail_rows(self) -> list[int]:
            rows = {
                index.row()
                for index in self.thumbnail_view.selectionModel().selectedIndexes()
                if index.isValid() and 0 <= index.row() < len(self.rows)
            }
            return sorted(rows)

        def _decision_action_indices(self) -> list[int]:
            if self._preview_mode == PREVIEW_MODE_THUMBNAILS:
                selected_rows = self._selected_thumbnail_rows()
                if selected_rows:
                    return selected_rows
            return [self.index]

        def _current_row(self) -> dict[str, str]:
            return self.rows[self.index]

        def _decision_color(self, decision: str) -> str:
            return "#b00020" if decision == "drop" else "#1b7f3b"

        def _decision_text(self, decision: str) -> str:
            if decision == "drop":
                return i18n.t("REVIEW_DECISION_DROP")
            return i18n.t("REVIEW_DECISION_KEEP")

        def _advisory_for_row(self, row: dict[str, str], idx: int) -> tuple[str, str, str, str]:
            """Return (detail label, thumbnail label, fg, bg) for the current user action.

            フレーム抽出時に代表フレーム選択は済んでいるので、
            低品質のまま残ったフレームだけを強く確認対象として扱う。
            「ブレ top X%」のような相対順位 advisory は出さない（誤判定の元）。
            """
            status = row.get("status", "ok").strip().lower()
            pipeline = row.get("analysis_pipeline", "").strip().lower()
            decision = row.get("decision", "keep").strip().lower()

            drop_fg, drop_bg = "#991b1b", "#fee2e2"
            warning_fg, warning_bg = "#92400e", "#fef3c7"
            added_fg, added_bg = "#1e40af", "#dbeafe"
            quick_fg, quick_bg = "#5b21b6", "#ede9fe"
            external_fg, external_bg = "#0f766e", "#ccfbf1"
            ok_fg, ok_bg = "#166534", "#dcfce7"

            if decision == "drop":
                if "motion_blur" in status:
                    return (
                        i18n.t("REVIEW_ADVISORY_DROP_BLUR"),
                        i18n.t("REVIEW_ADVISORY_SHORT_DROP_BLUR"),
                        drop_fg,
                        drop_bg,
                    )
                if "redundant_drop" in status:
                    return (
                        i18n.t("REVIEW_ADVISORY_DROP_REDUNDANT"),
                        i18n.t("REVIEW_ADVISORY_SHORT_DROP_REDUNDANT"),
                        drop_fg,
                        drop_bg,
                    )
                return (
                    i18n.t("REVIEW_ADVISORY_DROP_MANUAL"),
                    i18n.t("REVIEW_ADVISORY_SHORT_DROP_MANUAL"),
                    drop_fg,
                    drop_bg,
                )

            if "motion_blur" in status:
                return (
                    i18n.t("REVIEW_ADVISORY_MOTION_BLUR"),
                    i18n.t("REVIEW_ADVISORY_SHORT_MOTION_BLUR"),
                    warning_fg,
                    warning_bg,
                )

            if "borderline_blur" in status:
                return (
                    i18n.t("REVIEW_ADVISORY_BORDERLINE_BLUR"),
                    i18n.t("REVIEW_ADVISORY_SHORT_BORDERLINE_BLUR"),
                    warning_fg,
                    warning_bg,
                )

            if "low_texture" in status:
                return (
                    i18n.t("REVIEW_ADVISORY_LOW_TEXTURE"),
                    i18n.t("REVIEW_ADVISORY_SHORT_LOW_TEXTURE"),
                    warning_fg,
                    warning_bg,
                )

            if "weak_match" in status:
                return (
                    i18n.t("REVIEW_ADVISORY_WEAK_MATCH"),
                    i18n.t("REVIEW_ADVISORY_SHORT_WEAK_MATCH"),
                    warning_fg,
                    warning_bg,
                )

            if "blur_replacement" in status:
                return (
                    i18n.t("REVIEW_ADVISORY_BLUR_REPLACEMENT"),
                    i18n.t("REVIEW_ADVISORY_SHORT_BLUR_REPLACEMENT"),
                    added_fg,
                    added_bg,
                )

            if "gap_forced" in status:
                return (
                    i18n.t("REVIEW_ADVISORY_GAP_FORCED"),
                    i18n.t("REVIEW_ADVISORY_SHORT_GAP_FORCED"),
                    added_fg,
                    added_bg,
                )

            if "novelty_added" in status:
                return (
                    i18n.t("REVIEW_ADVISORY_NOVELTY_ADDED"),
                    i18n.t("REVIEW_ADVISORY_SHORT_NOVELTY_ADDED"),
                    added_fg,
                    added_bg,
                )

            if pipeline == "quick":
                return (
                    i18n.t("REVIEW_ADVISORY_QUICK"),
                    i18n.t("REVIEW_ADVISORY_SHORT_QUICK"),
                    quick_fg,
                    quick_bg,
                )

            if pipeline == "external_import":
                return (
                    i18n.t("REVIEW_ADVISORY_EXTERNAL_IMPORT"),
                    i18n.t("REVIEW_ADVISORY_SHORT_EXTERNAL_IMPORT"),
                    external_fg,
                    external_bg,
                )

            # 緑: 通常品質
            return (
                i18n.t("REVIEW_ADVISORY_NORMAL"),
                i18n.t("REVIEW_ADVISORY_SHORT_NORMAL"),
                ok_fg,
                ok_bg,
            )

        def _format_metric_value(self, value: str | None, decimals: int = 3) -> str:
            if value in (None, ""):
                return "-"
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return "-"

        def _pair_info_summary(self, row: dict[str, str], ts_str: str) -> str:
            return i18n.t("REVIEW_PAIR_INFO_FORMAT").format(
                ts=ts_str,
                gap=self._format_metric_value(row.get("gap_sec"), 2),
                residual=self._format_metric_value(row.get("residual_score"), 4),
                yaw=self._format_metric_value(row.get("yaw_shift_deg"), 1),
                tracks=row.get("track_count") or "-",
                confidence=self._format_metric_value(row.get("match_confidence"), 2),
                blur=self._format_metric_value(row.get("blur_score_final"), 1),
                sharpness_ratio=self._format_metric_value(row.get("sharpness_ratio"), 2),
            )

        def _render_current(self, *, sync_thumbnail: bool = True, scroll_thumbnail: bool = False) -> None:
            row = self._current_row()
            seq = int(row.get("seq", self.index + 1))
            total = len(self.rows)

            image_rel = row.get("output_file", "")
            image_path = self.scene_dir / image_rel

            self.title_label.setText(f"{seq}/{total}  {image_rel}")
            self._slider_sync = True
            self.frame_slider.setValue(self.index)
            self._slider_sync = False
            self.frame_position_label.setText(
                i18n.t("REVIEW_FRAME_POSITION_FORMAT").format(
                    seq=seq,
                    total=total,
                    name=Path(image_rel).name,
                )
            )

            decision = row.get("decision", "keep")
            self.decision_label.setText(f"{i18n.t('REVIEW_DECISION_PREFIX')}{self._decision_text(decision)}")
            self.decision_label.setStyleSheet(f"font-weight: 700; color: {self._decision_color(decision)};")
            self._update_decision_buttons(decision)

            # アドバイザリー (このフレームが要注意な理由)
            adv_text, _adv_short, adv_fg, adv_bg = self._advisory_for_row(row, self.index)
            self.advisory_label.setText(adv_text)
            self.advisory_label.setStyleSheet(
                f"padding: 6px 10px; border-radius: 4px; font-weight: 600; font-size: 10pt; "
                f"color: {adv_fg}; background-color: {adv_bg};"
            )

            problem_count = len(self.problem_indices)
            current_problem = i18n.t("REVIEW_INFO_YES") if self._is_problem_row(row) else i18n.t("REVIEW_INFO_NO")
            novelty_count = sum(1 for r in self.rows if "novelty_added" in r.get("status", "").strip().lower())
            redundant_count = sum(1 for r in self.rows if "redundant_drop" in r.get("status", "").strip().lower())
            gap_count = sum(1 for r in self.rows if "gap_forced" in r.get("status", "").strip().lower())
            blur_count = sum(1 for r in self.rows if "motion_blur" in r.get("status", "").strip().lower())
            borderline_blur_count = sum(
                1 for r in self.rows if "borderline_blur" in r.get("status", "").strip().lower()
            )
            texture_count = sum(1 for r in self.rows if "low_texture" in r.get("status", "").strip().lower())
            weak_count = sum(1 for r in self.rows if "weak_match" in r.get("status", "").strip().lower())
            self.problem_summary_label.setText(
                i18n.t("REVIEW_PAIR_PROBLEMS_FORMAT").format(
                    n=problem_count,
                    a=novelty_count,
                    d=redundant_count,
                    g=gap_count,
                    b=blur_count,
                    bb=borderline_blur_count,
                    l=texture_count,
                    w=weak_count,
                    cur=current_problem,
                )
            )

            # 動画内位置
            ts_raw = row.get("timestamp_sec", "-")
            try:
                ts_str = f"{float(ts_raw):.2f}s"
            except (ValueError, TypeError):
                ts_str = ts_raw

            if row.get("analysis_pipeline") == "pair":
                info_text = self._pair_info_summary(row, ts_str)
            else:
                info_text = i18n.t("REVIEW_INFO_FORMAT").format(ts=ts_str)
            self.info_label.setText(info_text)
            self._sync_thumbnail_model()
            if sync_thumbnail:
                self._sync_thumbnail_selection(self.index, scroll=scroll_thumbnail)

            if self._preview_mode == PREVIEW_MODE_THUMBNAILS:
                self._queue_thumbnail_priority()
                return

            if not image_path.exists():
                self.current_pixmap = None
                self.image_view.setText(i18n.t("REVIEW_IMAGE_NOT_FOUND").format(path=image_path))
                return

            if self._preview_projection == PREVIEW_PROJECTION_PERSPECTIVE:
                img = _perspective_bgr_for_review(image_path)
                if img is not None and self.image_view.set_perspective_image_bgr(img, self._perspective_params):
                    self.current_pixmap = _perspective_pixmap_for_review(image_path, self._perspective_params)
                    return

            pixmap = self._pixmap_for(image_path)
            if pixmap.isNull():
                self.current_pixmap = None
                self.image_view.setText(i18n.t("REVIEW_IMAGE_LOAD_FAILED").format(path=image_path))
                return

            self.current_pixmap = pixmap
            self.image_view.set_source_pixmap(pixmap)
            self._prefetch_neighbor_pixmaps()

        def _pixmap_cache_key(self, image_path: Path, *extra: object) -> tuple | None:
            try:
                st = image_path.stat()
                return (str(image_path.resolve()).lower(), int(st.st_size), int(st.st_mtime_ns), *extra)
            except OSError:
                return None

        def _pixmap_for(self, image_path: Path) -> QPixmap:
            extra: tuple[object, ...] = (self._preview_projection,)
            if self._preview_projection == PREVIEW_PROJECTION_PERSPECTIVE:
                extra = (
                    self._preview_projection,
                    round(float(self._perspective_params.yaw_deg), 3),
                    round(float(self._perspective_params.pitch_deg), 3),
                    round(float(self._perspective_params.fov_deg), 3),
                )
            key = self._pixmap_cache_key(image_path, *extra)
            if key is not None and key in self._pixmap_cache:
                self._pixmap_cache.move_to_end(key)
                return self._pixmap_cache[key]

            if self._preview_projection == PREVIEW_PROJECTION_PERSPECTIVE:
                pixmap = _perspective_pixmap_for_review(image_path, self._perspective_params)
            else:
                pixmap = QPixmap(str(image_path))
            if not pixmap.isNull() and key is not None:
                self._pixmap_cache[key] = pixmap
                self._pixmap_cache.move_to_end(key)
                while len(self._pixmap_cache) > _PIXMAP_CACHE_LIMIT:
                    self._pixmap_cache.popitem(last=False)
            return pixmap

        def _prefetch_neighbor_pixmaps(self) -> None:
            if self._preview_projection == PREVIEW_PROJECTION_PERSPECTIVE:
                return
            for idx in (self.index - 1, self.index + 1):
                if idx < 0 or idx >= len(self.rows):
                    continue
                rel = self.rows[idx].get("output_file", "")
                if not rel:
                    continue
                path = self.scene_dir / rel
                if path.exists() and path.is_file():
                    self._pixmap_for(path)

        def reset_zoom(self) -> None:
            self.image_view.reset_view()

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

        def _on_slider_changed(self, value: int) -> None:
            if self._slider_sync:
                return
            if 0 <= value < len(self.rows):
                self._set_index(value, scroll_thumbnail=True)

        def _update_decision_buttons(self, decision: str) -> None:
            keep = decision != "drop"
            self.flag_button.setChecked(keep)
            self.flag_button.setIcon(_review_icon("flag-keep" if keep else "flag-drop"))
            self.flag_button.setStyleSheet(
                "QToolButton {"
                f"background: {'#14532d' if keep else '#3b1717'};"
                f"border: 1px solid {'#22c55e' if keep else '#991b1b'};"
                "border-radius: 4px;"
                "}"
            )
            reset_enabled = any(
                self.rows[idx].get("decision", "keep") != self._initial_decisions[self._all_index(idx)]
                for idx in self._decision_action_indices()
            )
            self.reset_decision_button.setEnabled(reset_enabled)
            self.reset_decision_button.setStyleSheet(
                "QToolButton { border-radius: 4px; }QToolButton:disabled { opacity: 0.45; }"
            )

        def has_decision_changes(self) -> bool:
            return any(
                row.get("decision", "keep") != initial
                for row, initial in zip(self._all_rows, self._initial_decisions, strict=False)
            )

        def pending_drop_image_paths(self) -> list[Path]:
            return find_pending_drop_image_paths(self.scene_dir, str(self.csv_path))

        def has_pending_finalize(self) -> bool:
            return self.has_decision_changes() or bool(self.pending_drop_image_paths())

        def _write_rows(self) -> None:
            fieldnames = list(self._all_rows[0].keys())
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._all_rows)

        def _set_decisions(self, decisions_by_index: dict[int, str]) -> None:
            changes: dict[int, str] = {}
            for idx, decision in decisions_by_index.items():
                if not (0 <= idx < len(self.rows)):
                    continue
                normalized = "drop" if decision == "drop" else "keep"
                if self.rows[idx].get("decision", "keep") != normalized:
                    changes[idx] = normalized

            if not changes:
                self._render_current()
                return

            old_decisions = {idx: self.rows[idx].get("decision", "keep") for idx in changes}
            for idx, decision in changes.items():
                self.rows[idx]["decision"] = decision
            try:
                self._write_rows()
            except Exception as e:
                for idx, old_decision in old_decisions.items():
                    self.rows[idx]["decision"] = old_decision
                QMessageBox.critical(
                    self,
                    i18n.t("REVIEW_SAVE_FAILED_HEADER"),
                    i18n.t("REVIEW_SAVE_FAILED_BODY").format(error=e),
                )
                self._render_current()
                return
            for idx in changes:
                self._refresh_thumbnail_row(idx)
            self._render_current()
            self.decisions_changed.emit()

        def _set_current_decision(self, decision: str) -> None:
            self._set_decisions({self.index: decision})

        def prev_row(self) -> None:
            if self.index > 0:
                self._set_index(self.index - 1, scroll_thumbnail=True)

        def next_row(self) -> None:
            if self.index < len(self.rows) - 1:
                self._set_index(self.index + 1, scroll_thumbnail=True)

        def next_problem(self) -> None:
            if not self.problem_indices:
                QMessageBox.information(self, i18n.t("REVIEW_INFO_HEADER"), i18n.t("REVIEW_NO_PROBLEMS"))
                return

            for idx in self.problem_indices:
                if idx > self.index:
                    self._set_index(idx, scroll_thumbnail=True)
                    return

            self._set_index(self.problem_indices[0], scroll_thumbnail=True)

        def prev_problem(self) -> None:
            if not self.problem_indices:
                QMessageBox.information(self, i18n.t("REVIEW_INFO_HEADER"), i18n.t("REVIEW_NO_PROBLEMS"))
                return

            for idx in reversed(self.problem_indices):
                if idx < self.index:
                    self._set_index(idx, scroll_thumbnail=True)
                    return

            self._set_index(self.problem_indices[-1], scroll_thumbnail=True)

        def toggle_decision(self) -> None:
            indices = self._decision_action_indices()
            next_decision = (
                "drop" if all(self.rows[idx].get("decision", "keep") != "drop" for idx in indices) else "keep"
            )
            self._set_decisions({idx: next_decision for idx in indices})
            self._focus_thumbnail_view_if_active()

        def reset_decision(self) -> None:
            self._set_decisions(
                {idx: self._initial_decisions[self._all_index(idx)] for idx in self._decision_action_indices()}
            )
            self._focus_thumbnail_view_if_active()

        def shutdown(self) -> None:
            self._thumbnail_priority_timer.stop()
            self.thumbnail_model.shutdown()

        def closeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt API
            self.shutdown()
            super().closeEvent(event)

    class ReviewWindow(QMainWindow):
        def __init__(self, scene_dir: Path, csv_path: Path) -> None:
            super().__init__()
            self.setWindowTitle(i18n.t("REVIEW_TITLE"))
            self.resize(1280, 860)
            self.review_widget = ReviewWidget(scene_dir, csv_path)
            self.setCentralWidget(self.review_widget)

        def closeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt API
            self.review_widget.shutdown()
            super().closeEvent(event)

else:

    class ReviewWidget:  # pragma: no cover - placeholder when PySide6 missing
        pass

    class ReviewWindow:  # pragma: no cover - placeholder when PySide6 missing
        pass


def ensure_gui_deps() -> None:
    if QApplication is None:
        raise RuntimeError(f"PySide6 is required to run this GUI: {_PYSIDE_IMPORT_ERROR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review extracted frames and edit keep/drop decisions.")
    parser.add_argument(
        "scene_dir",
        nargs="?",
        default=".",
        help="Scene directory containing _stechdrive/frames/selected_frames.csv and images/",
    )
    parser.add_argument(
        "--csv",
        default="selected_frames.csv",
        help="CSV filename under scene_dir/_stechdrive/frames, or an absolute path (default=selected_frames.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).resolve()
    csv_path = selected_frames_path(scene_dir, args.csv)

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    try:
        ensure_gui_deps()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    try:
        window = ReviewWindow(scene_dir, csv_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
