"""Step 1: フレーム抽出"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from extract_sessions import load_manifest, matching_video_sessions, sanitize_filename_prefix
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.common.icons import delete_icon, plus_icon, reset_icon
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)
from scene_layout import APP_DIR_NAME, source_videos_path
from scene_project import infer_video_projection, load_json, remove_source_videos, source_video_record, upsert_source_videos

_FIXED_INTERVAL_MIN = 0.05
_FIXED_INTERVAL_MAX = 60.0
_CHANGE_GAP_MIN = 0.05
_CHANGE_GAP_MAX = 60.0
_GAP_SPINBOX_WIDTH = 112
_JPEG_QUALITY_MIN = 1
_JPEG_QUALITY_MAX = 31
_JPEG_QUALITY_DEFAULT = 2
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
_VIDEO_SCAN_EXCLUDED_DIRS = {
    APP_DIR_NAME.casefold(),
    ".git",
    ".venv",
    "__pycache__",
    "images",
    "masks",
    "output",
    "outputs",
}


def _detect_binary(name: str) -> str:
    """PATH から実行ファイルを検出。見つからなければ素の名前を返す（PATH 解決に委ねる）。"""
    found = shutil.which(name)
    return found if found else name


def _row_label(text: str, tooltip: str | None = None) -> QLabel:
    label = QLabel(text)
    if tooltip:
        label.setToolTip(tooltip)
    return label


class ExtractStep(BaseStepWidget):
    scene_dir_suggested = Signal(str)
    input_videos_cleared = Signal()

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)

        self.video_info: dict | None = None
        self.video_infos: dict[str, dict] = {}
        self.video_info_failures: dict[str, str] = {}
        self.last_estimate_summary: dict | None = None
        self.instant_estimate_text = "-"
        self._syncing_gap_fields = False
        self._smart_before_quick: bool | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        settings_scroll = QScrollArea()
        configure_settings_scroll(settings_scroll)

        settings = QWidget()
        settings.setObjectName("settingsPane")
        layout = QVBoxLayout(settings)
        layout.setContentsMargins(*SETTINGS_PANE_MARGINS)
        layout.setSpacing(8)

        work_pane = QWidget()
        work_pane.setObjectName("workPane")
        work_layout = QVBoxLayout(work_pane)
        work_layout.setContentsMargins(12, 12, 12, 12)
        work_layout.setSpacing(10)

        # ===== 基本設定 =====
        basic = QFormLayout()
        basic.setSpacing(6)

        self.video_browse = BrowseWidget(
            self,
            mode="files",
            filter_str=i18n.t("VIDEO_FILE_FILTER"),
            placeholder=i18n.t("INPUT_VIDEO_PLACEHOLDER"),
        )
        self.video_browse.setToolTip(i18n.tip("INPUT_VIDEO"))
        self.video_browse.path_changed.connect(self._on_video_changed)
        self.video_browse.hide()

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.setToolTip(i18n.tip("EXTRACT_OUTPUT_MODE"))
        self.output_mode_combo.addItem(i18n.t("EXTRACT_OUTPUT_APPEND"), "append")
        self.output_mode_combo.addItem(i18n.t("EXTRACT_OUTPUT_REPLACE_VIDEO"), "replace-video")
        self.output_mode_combo.setFixedWidth(180)
        self.output_mode_combo.currentIndexChanged.connect(lambda _: self._on_output_mode_changed())
        add_tooltip_row(basic, i18n.t("EXTRACT_OUTPUT_MODE"), self.output_mode_combo, i18n.tip("EXTRACT_OUTPUT_MODE"))

        self.images_path_label = QLabel("-")
        self.images_path_label.setToolTip(i18n.tip("IMAGES_DIR"))
        self.images_path_label.setWordWrap(True)
        self.images_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add_tooltip_row(basic, i18n.IMAGES_DIR, self.images_path_label, i18n.tip("IMAGES_DIR"))
        layout.addLayout(basic)

        self.interval_edit = DragDoubleSpinBox(
            minimum=_FIXED_INTERVAL_MIN,
            maximum=_FIXED_INTERVAL_MAX,
            step=0.05,
            decimals=2,
            value=1.0,
            suffix=f" {i18n.t('SECONDS_SUFFIX')}",
            drag_pixels_per_step=6.0,
        )
        self.interval_edit.setToolTip(i18n.tip("INTERVAL"))
        self.interval_edit.setFixedWidth(86)
        self.interval_edit.valueChanged.connect(self._mark_estimate_stale)

        self.min_gap_edit = DragDoubleSpinBox(
            minimum=_CHANGE_GAP_MIN,
            maximum=_CHANGE_GAP_MAX,
            step=0.05,
            decimals=2,
            value=0.5,
            suffix=f" {i18n.t('SECONDS_SUFFIX')}",
            drag_pixels_per_step=6.0,
        )
        self.min_gap_edit.setToolTip(i18n.tip("MIN_GAP"))
        self.min_gap_edit.setFixedWidth(_GAP_SPINBOX_WIDTH)
        self.min_gap_edit.valueChanged.connect(lambda _: self._clamp_gap_order("min"))
        self.min_gap_edit.valueChanged.connect(self._mark_estimate_stale)

        self.max_gap_edit = DragDoubleSpinBox(
            minimum=_CHANGE_GAP_MIN,
            maximum=_CHANGE_GAP_MAX,
            step=0.05,
            decimals=2,
            value=2.0,
            suffix=f" {i18n.t('SECONDS_SUFFIX')}",
            drag_pixels_per_step=6.0,
        )
        self.max_gap_edit.setToolTip(i18n.tip("MAX_GAP"))
        self.max_gap_edit.setFixedWidth(_GAP_SPINBOX_WIDTH)
        self.max_gap_edit.valueChanged.connect(lambda _: self._clamp_gap_order("max"))
        self.max_gap_edit.valueChanged.connect(self._mark_estimate_stale)

        self.smart_fixed_cb = QCheckBox(i18n.t("FIXED_SMART"))
        self.smart_fixed_cb.setToolTip(i18n.tip("FIXED_SMART"))
        self.smart_fixed_cb.setChecked(True)
        self.smart_fixed_cb.toggled.connect(self._update_mode_widgets)
        self.smart_fixed_cb.toggled.connect(self._mark_estimate_stale)

        self.quick_extract_cb = QCheckBox(i18n.t("QUICK_EXTRACT"))
        self.quick_extract_cb.setToolTip(i18n.tip("QUICK_EXTRACT"))
        self.quick_extract_cb.toggled.connect(self._on_quick_extract_toggled)
        self.quick_extract_cb.toggled.connect(self._mark_estimate_stale)

        mode_panel = QWidget()
        mode_layout = QVBoxLayout(mode_panel)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)

        self.extract_interval_label = _row_label(f"{i18n.t('EXTRACT_INTERVAL')}:", i18n.tip("INTERVAL"))
        self.min_gap_label = _row_label(f"{i18n.t('MIN_GAP_SHORT')}:", i18n.tip("MIN_GAP"))
        self.max_gap_label = _row_label(f"{i18n.t('MAX_GAP_SHORT')}:", i18n.tip("MAX_GAP"))

        self.mode_panel = mode_panel

        fixed_row_widget = QWidget()
        fixed_row = QHBoxLayout(fixed_row_widget)
        fixed_row.setContentsMargins(0, 0, 0, 0)
        fixed_row.setSpacing(4)
        fixed_row.addWidget(self.extract_interval_label)
        fixed_row.addWidget(self.interval_edit)
        fixed_row.addSpacing(8)
        fixed_row.addWidget(self.quick_extract_cb)
        fixed_row.addStretch()
        mode_layout.addWidget(fixed_row_widget)

        smart_row_widget = QWidget()
        smart_row = QHBoxLayout(smart_row_widget)
        smart_row.setContentsMargins(0, 0, 0, 0)
        smart_row.setSpacing(4)
        smart_row.addWidget(self.smart_fixed_cb)
        smart_row.addWidget(self.min_gap_label)
        smart_row.addWidget(self.min_gap_edit)
        smart_row.addWidget(self.max_gap_label)
        smart_row.addWidget(self.max_gap_edit)
        smart_row.addStretch()
        mode_layout.addWidget(smart_row_widget)

        self.fixed_interval_row = fixed_row_widget
        self.smart_interval_row = smart_row_widget
        layout.addWidget(mode_panel)

        # 画像形式
        format_form = QFormLayout()
        format_form.setSpacing(6)
        fmt_row = QHBoxLayout()
        self.image_ext_combo = QComboBox()
        self.image_ext_combo.setToolTip(i18n.tip("IMAGE_FORMAT"))
        self.image_ext_combo.addItems(["jpg", "png"])
        self.image_ext_combo.setFixedWidth(80)
        self.image_ext_combo.currentIndexChanged.connect(self._mark_estimate_stale)
        fmt_row.addWidget(self.image_ext_combo)
        jpg_quality_label = QLabel(i18n.JPEG_QUALITY + ":")
        jpg_quality_label.setToolTip(i18n.tip("JPEG_QUALITY"))
        fmt_row.addWidget(jpg_quality_label)
        self.jpg_quality_edit = DragSpinBox(
            minimum=_JPEG_QUALITY_MIN,
            maximum=_JPEG_QUALITY_MAX,
            step=1,
            value=_JPEG_QUALITY_DEFAULT,
            drag_pixels_per_step=6.0,
        )
        self.jpg_quality_edit.setToolTip(i18n.tip("JPEG_QUALITY"))
        self.jpg_quality_edit.setFixedWidth(54)
        self.jpg_quality_edit.valueChanged.connect(self._mark_estimate_stale)
        fmt_row.addWidget(self.jpg_quality_edit)
        fmt_row.addStretch()
        add_tooltip_row(format_form, i18n.IMAGE_FORMAT, fmt_row, i18n.tip("IMAGE_FORMAT"))
        layout.addLayout(format_form)

        # ===== 詳細設定 (折りたたみ) =====
        advanced = CollapsibleSection(i18n.t("ADVANCED_SETTINGS"), expanded=False)
        advanced.layout().removeWidget(advanced.toggle_button)
        advanced.layout().setContentsMargins(0, 0, 0, 0)
        advanced.toggle_button.setParent(None)

        # ===== 詳細設定ボタン =====
        info_row_widget = QWidget()
        info_box = QHBoxLayout(info_row_widget)
        info_box.setContentsMargins(0, 0, 0, 0)
        info_box.setSpacing(8)

        info_box.addWidget(advanced.toggle_button)
        info_box.addStretch()
        self.extract_action_row = info_row_widget
        layout.addWidget(info_row_widget)

        queue_header = QHBoxLayout()
        queue_header.setContentsMargins(0, 0, 0, 0)
        queue_header.setSpacing(6)
        queue_header.addWidget(QLabel(i18n.t("VIDEO_QUEUE_SECTION")))
        self.video_queue_summary_label = QLabel("")
        self.video_queue_summary_label.setObjectName("videoQueueSummary")
        self.video_queue_summary_label.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self.video_queue_summary_label.setWordWrap(False)
        queue_header.addWidget(self.video_queue_summary_label, stretch=1)
        queue_header.addStretch()
        self.add_video_btn = QToolButton()
        self.add_video_btn.setObjectName("iconToolButton")
        self.add_video_btn.setIcon(plus_icon())
        self.add_video_btn.setToolTip(i18n.tip("ADD_INPUT_VIDEO"))
        self.add_video_btn.setAccessibleName(i18n.t("ADD_INPUT_VIDEO"))
        self.add_video_btn.setFixedSize(32, 32)
        self.add_video_btn.clicked.connect(self._add_input_videos)
        queue_header.addWidget(self.add_video_btn)
        self.remove_video_btn = QToolButton()
        self.remove_video_btn.setObjectName("iconToolButton")
        self.remove_video_btn.setIcon(delete_icon())
        self.remove_video_btn.setToolTip(i18n.tip("REMOVE_INPUT_VIDEO"))
        self.remove_video_btn.setAccessibleName(i18n.t("REMOVE_INPUT_VIDEO"))
        self.remove_video_btn.setFixedSize(32, 32)
        self.remove_video_btn.clicked.connect(self._remove_selected_input_videos)
        queue_header.addWidget(self.remove_video_btn)
        self.clear_video_btn = QToolButton()
        self.clear_video_btn.setObjectName("iconToolButton")
        self.clear_video_btn.setIcon(reset_icon())
        self.clear_video_btn.setToolTip(i18n.t("CLEAR_INPUT_VIDEO_HINT"))
        self.clear_video_btn.setAccessibleName(i18n.t("CLEAR_INPUT_VIDEO"))
        self.clear_video_btn.setFixedSize(32, 32)
        self.clear_video_btn.clicked.connect(self._clear_input_videos)
        queue_header.addWidget(self.clear_video_btn)
        work_layout.addLayout(queue_header)

        self.video_queue_list = QListWidget()
        self.video_queue_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.video_queue_list.setMinimumHeight(150)
        self.video_queue_list.setToolTip(i18n.tip("VIDEO_QUEUE_SECTION"))
        self.video_queue_list.itemSelectionChanged.connect(self._update_video_queue_buttons)
        work_layout.addWidget(self.video_queue_list)

        work_layout.addWidget(QLabel(i18n.t("EXTRACT_READY_SECTION")))
        self.ready_status_label = QLabel()
        self.ready_status_label.setWordWrap(True)
        work_layout.addWidget(self.ready_status_label)

        self.video_info_label = QLabel(i18n.t("VIDEO_LABEL_DEFAULT"))
        self.video_info_label.setStyleSheet("color: #8888aa;")
        self.video_info_label.setWordWrap(True)
        work_layout.addWidget(self.video_info_label)
        self.video_info_label.hide()

        work_layout.addWidget(QLabel(i18n.FRAME_ESTIMATE))
        self.estimate_label = QLabel()
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self._refresh_estimate_label()
        work_layout.addWidget(self.estimate_label)
        work_layout.addStretch()

        adv_form = QFormLayout()
        adv_form.setSpacing(6)

        self.analysis_width_edit = QLineEdit("1920")
        self.analysis_width_edit.setToolTip(i18n.tip("ANALYSIS_WIDTH"))
        self.analysis_width_edit.setFixedWidth(80)
        self.analysis_width_edit.textChanged.connect(self._mark_estimate_stale)
        add_tooltip_row(adv_form, i18n.ANALYSIS_WIDTH, self.analysis_width_edit, i18n.tip("ANALYSIS_WIDTH"))
        self.analysis_width_label = adv_form.labelForField(self.analysis_width_edit)

        self.pair_motion_profile_combo = QComboBox()
        self.pair_motion_profile_combo.setToolTip(i18n.tip("PAIR_MOTION_PROFILE"))
        self.pair_motion_profile_combo.addItem(i18n.t("PAIR_PROFILE_WALK"), "walk")
        self.pair_motion_profile_combo.addItem(i18n.t("PAIR_PROFILE_DRONE"), "drone")
        self.pair_motion_profile_combo.setFixedWidth(120)
        self.pair_motion_profile_combo.currentIndexChanged.connect(self._mark_estimate_stale)
        add_tooltip_row(
            adv_form,
            i18n.t("PAIR_MOTION_PROFILE"),
            self.pair_motion_profile_combo,
            i18n.tip("PAIR_MOTION_PROFILE"),
        )
        self.pair_motion_profile_label = adv_form.labelForField(self.pair_motion_profile_combo)
        advanced.content_layout.addLayout(adv_form)

        path_form = QFormLayout()
        path_form.setSpacing(6)

        # ffmpeg / ffprobe: PATH から自動検出して初期値にセット。参照ボタンで上書き可能
        ffmpeg_filter = "Executable (*.exe);;すべて (*.*)" if sys.platform == "win32" else "すべて (*.*)"
        self.ffmpeg_browse = BrowseWidget(mode="file", filter_str=ffmpeg_filter,
                                          placeholder="ffmpeg (PATH から自動検出)")
        self.ffmpeg_browse.set_text(_detect_binary("ffmpeg"))
        self.ffmpeg_browse.setToolTip(i18n.tip("FFMPEG_PATH"))
        add_tooltip_row(path_form, i18n.FFMPEG_PATH, self.ffmpeg_browse, i18n.tip("FFMPEG_PATH"))

        self.ffprobe_browse = BrowseWidget(mode="file", filter_str=ffmpeg_filter,
                                           placeholder="ffprobe (PATH から自動検出)")
        self.ffprobe_browse.set_text(_detect_binary("ffprobe"))
        self.ffprobe_browse.setToolTip(i18n.tip("FFPROBE_PATH"))
        self.ffprobe_browse.path_changed.connect(lambda _path: self._reload_video_info_if_selected())
        add_tooltip_row(path_form, i18n.FFPROBE_PATH, self.ffprobe_browse, i18n.tip("FFPROBE_PATH"))

        self.prefix_edit = QLineEdit("")
        self.prefix_edit.setToolTip(i18n.tip("FILENAME_PREFIX"))
        self.prefix_edit.setPlaceholderText(i18n.t("AUTO_PREFIX_HINT"))
        self.prefix_edit.textChanged.connect(lambda _: self._update_ready_status())
        add_tooltip_row(path_form, i18n.FILENAME_PREFIX, self.prefix_edit, i18n.tip("FILENAME_PREFIX"))

        advanced.content_layout.addLayout(path_form)
        layout.addWidget(advanced)

        layout.addStretch()
        settings_scroll.setWidget(settings)
        splitter.addWidget(settings_scroll)
        splitter.addWidget(work_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        root_layout.addWidget(splitter)
        self._update_mode_widgets()
        self._refresh_video_queue_list()
        self._update_ready_status()

    # -- シーンディレクトリ --

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        self._prune_missing_selected_videos()
        self._autoload_videos_from_scene_if_empty()
        self._update_images_path_label()
        self._update_video_info_label()
        self._update_instant_estimate()
        self._update_ready_status()

    def _update_images_path_label(self) -> None:
        if self.scene_dir:
            self.images_path_label.setText(str(Path(self.scene_dir) / "images"))
        else:
            self.images_path_label.setText("-")

    def primary_action_text(self) -> str:
        return i18n.RUN

    def primary_action_tooltip(self) -> str:
        ready, reason = self._readiness()
        return i18n.tip("RUN") if ready else reason

    def primary_action_enabled(self) -> bool:
        ready, _reason = self._readiness()
        return ready

    def _on_quick_extract_toggled(self, checked: bool) -> None:
        if checked:
            self._smart_before_quick = self.smart_fixed_cb.isChecked()
            if self.smart_fixed_cb.isChecked():
                self.smart_fixed_cb.setChecked(False)
        elif self._smart_before_quick is not None:
            restore_smart = self._smart_before_quick
            self._smart_before_quick = None
            if self.smart_fixed_cb.isChecked() != restore_smart:
                self.smart_fixed_cb.setChecked(restore_smart)
        self._update_mode_widgets()

    def _update_mode_widgets(self) -> None:
        quick_enabled = self.quick_extract_cb.isChecked()
        smart_enabled = self.smart_fixed_cb.isChecked() and not quick_enabled
        for widget in (self.extract_interval_label, self.interval_edit, self.quick_extract_cb):
            widget.setEnabled(True)
        for widget in (
            self.smart_fixed_cb,
            self.min_gap_label,
            self.min_gap_edit,
            self.max_gap_label,
            self.max_gap_edit,
        ):
            widget.setEnabled((smart_enabled or widget is self.smart_fixed_cb) and not quick_enabled)
        for widget in (
            self.analysis_width_label,
            self.analysis_width_edit,
            self.pair_motion_profile_label,
            self.pair_motion_profile_combo,
        ):
            if widget is not None:
                widget.setEnabled(not quick_enabled)

    def _clamp_gap_order(self, changed: str) -> None:
        if self._syncing_gap_fields:
            return
        min_gap = self.min_gap_edit.value()
        max_gap = self.max_gap_edit.value()
        if min_gap <= max_gap:
            return
        self._syncing_gap_fields = True
        try:
            if changed == "min":
                self.max_gap_edit.setValue(min_gap)
            else:
                self.min_gap_edit.setValue(max_gap)
        finally:
            self._syncing_gap_fields = False

    def _analysis_width_valid(self) -> bool:
        text = self.analysis_width_edit.text().strip()
        try:
            return int(text) >= 0
        except ValueError:
            return False

    def _selected_video_paths(self) -> list[Path]:
        text = self.video_browse.text()
        if not text:
            return []
        raw_paths = [part.strip().strip('"') for part in text.split(";")]
        return [Path(part) for part in raw_paths if part]

    def _set_video_queue_paths(self, videos: list[Path]) -> None:
        unique: list[Path] = []
        seen: set[str] = set()
        for video in videos:
            key = str(video).replace("\\", "/").casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(video)
        self.video_browse.set_text("; ".join(str(video) for video in unique))
        if not unique:
            self._refresh_video_queue_list()

    def _queue_dialog_start_path(self) -> str:
        videos = self._selected_video_paths()
        for video in videos:
            if video.is_file():
                return str(video.parent)
        if self.scene_dir:
            return self.scene_dir
        return ""

    def _add_input_videos(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            i18n.t("ADD_INPUT_VIDEO"),
            self._queue_dialog_start_path(),
            i18n.t("VIDEO_FILE_FILTER"),
        )
        if not paths:
            return
        self._set_video_queue_paths([*self._selected_video_paths(), *(Path(path) for path in paths)])

    def _remove_selected_input_videos(self) -> None:
        selected_paths = {str(item.data(Qt.UserRole)) for item in self.video_queue_list.selectedItems()}
        if not selected_paths:
            return
        videos = self._selected_video_paths()
        removed = [video for video in videos if str(video) in selected_paths]
        self._forget_source_videos(removed)
        self._set_video_queue_paths([video for video in videos if str(video) not in selected_paths])

    def _video_info_for_queue_item(self, video: Path) -> dict | None:
        info = self.video_infos.get(self._video_key(video))
        if info is None and len(self._selected_video_paths()) == 1:
            info = self.video_info
        return info if isinstance(info, dict) else None

    def _video_queue_item_text(self, video: Path) -> str:
        key = self._video_key(video)
        info = self._video_info_for_queue_item(video)
        if key in self.video_info_failures:
            status = i18n.t("VIDEO_QUEUE_STATUS_ERROR")
        else:
            status = self._video_queue_status_text(video)
        if info is not None:
            return i18n.t("VIDEO_QUEUE_ITEM_INFO_FORMAT").format(
                name=video.name or str(video),
                status=status,
                projection=self._video_projection_text(info),
                width=info["width"],
                height=info["height"],
                fps=info["fps"],
                duration=self._format_duration(float(info.get("duration_sec", 0))),
                frames=self._format_number(self._estimated_total_frames(info)),
                folder=str(video.parent),
            )
        return i18n.t("VIDEO_QUEUE_ITEM_FORMAT").format(
            name=video.name or str(video),
            status=status,
            projection=self._video_projection_text(info),
            folder=str(video.parent),
        )

    def _update_video_queue_summary_label(self) -> None:
        if not hasattr(self, "video_queue_summary_label"):
            return
        videos = self._selected_video_paths()
        if not videos:
            self.video_queue_summary_label.setText(i18n.t("NO_VIDEO"))
            return
        queued, skipped = self._queued_selected_videos()
        probed = sum(1 for video in videos if self._video_info_for_queue_item(video) is not None)
        failed = sum(1 for video in videos if self._video_key(video) in self.video_info_failures)
        text = i18n.t("VIDEO_QUEUE_SUMMARY_FORMAT").format(
            total=len(videos),
            queued=len(queued),
            skipped=skipped,
            probed=probed,
        )
        if failed:
            text += i18n.t("VIDEO_INFO_FAILED_SUFFIX").format(failed=failed)
        self.video_queue_summary_label.setText(text)

    def _refresh_video_queue_list(self) -> None:
        if not hasattr(self, "video_queue_list"):
            return
        selected_paths = {str(item.data(Qt.UserRole)) for item in self.video_queue_list.selectedItems()}
        self.video_queue_list.blockSignals(True)
        try:
            self.video_queue_list.clear()
            for video in self._selected_video_paths():
                item = QListWidgetItem(self._video_queue_item_text(video))
                item.setData(Qt.UserRole, str(video))
                item.setToolTip(str(video))
                self.video_queue_list.addItem(item)
                if str(video) in selected_paths:
                    item.setSelected(True)
        finally:
            self.video_queue_list.blockSignals(False)
        self._update_video_queue_summary_label()
        self._update_video_queue_buttons()

    def _update_video_queue_buttons(self) -> None:
        if not hasattr(self, "video_queue_list"):
            return
        has_videos = bool(self._selected_video_paths())
        self.remove_video_btn.setEnabled(bool(self.video_queue_list.selectedItems()))
        self.clear_video_btn.setEnabled(has_videos)

    def _is_multi_video_input(self) -> bool:
        return len(self._selected_video_paths()) > 1

    def _extract_output_mode(self) -> str:
        data = self.output_mode_combo.currentData()
        return str(data or "append")

    def _on_output_mode_changed(self) -> None:
        self._update_video_info_label()
        self._update_instant_estimate()
        self._update_ready_status()

    def _matching_video_sessions_for_path(self, video: Path) -> list[dict]:
        if not self.scene_dir:
            return []
        if not video.is_file():
            return []
        return matching_video_sessions(Path(self.scene_dir), video)

    def _matching_video_sessions(self) -> list[dict]:
        videos = self._selected_video_paths()
        if not videos:
            return []
        return self._matching_video_sessions_for_path(videos[0])

    def _autoload_videos_from_scene_if_empty(self) -> None:
        if not self.scene_dir or self.video_browse.text():
            return
        scene = Path(self.scene_dir)
        if not scene.is_dir():
            return
        videos = self._source_video_paths_from_project(scene)
        if not videos:
            videos = self._source_video_paths_from_extract_manifest(scene)
        if not videos:
            videos = self._scan_video_paths_under_scene(scene)
        if videos:
            self.video_browse.set_text("; ".join(str(video) for video in videos))

    def _prune_missing_selected_videos(self) -> bool:
        videos = self._selected_video_paths()
        if not videos:
            return False
        existing = [video for video in videos if video.is_file()]
        if len(existing) == len(videos):
            return False
        missing_keys = {self._video_key(video) for video in videos if not video.is_file()}
        self.video_infos = {key: value for key, value in self.video_infos.items() if key not in missing_keys}
        self.video_info_failures = {
            key: value for key, value in self.video_info_failures.items() if key not in missing_keys
        }
        if not existing:
            self.video_info = None
        self.video_browse.set_text("; ".join(str(video) for video in existing))
        if not existing:
            self._autoload_videos_from_scene_if_empty()
        return True

    @staticmethod
    def _resolve_scene_or_absolute_path(scene: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else scene / path

    @staticmethod
    def _is_supported_video_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS

    def _unique_existing_video_paths(self, scene: Path, values: list[str]) -> list[Path]:
        videos: list[Path] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            path = self._resolve_scene_or_absolute_path(scene, value)
            if not self._is_supported_video_file(path):
                continue
            try:
                key = str(path.resolve()).casefold()
            except OSError:
                key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            videos.append(path)
        return videos

    def _source_video_paths_from_project(self, scene: Path) -> list[Path]:
        data = load_json(source_videos_path(scene), {"videos": []})
        records = data.get("videos")
        if not isinstance(records, list):
            return []
        values: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            source = record.get("source")
            if not isinstance(source, dict):
                continue
            values.append(str(source.get("path") or ""))
        return self._unique_existing_video_paths(scene, values)

    def _source_video_paths_from_extract_manifest(self, scene: Path) -> list[Path]:
        values: list[str] = []
        for session in load_manifest(scene).get("sessions", []):
            if not isinstance(session, dict):
                continue
            source = session.get("source_video")
            if isinstance(source, dict):
                values.append(str(source.get("path") or ""))
            else:
                values.append(str(session.get("source_video_path") or ""))
        return self._unique_existing_video_paths(scene, values)

    def _scan_video_paths_under_scene(self, scene: Path) -> list[Path]:
        videos: list[Path] = []
        try:
            for root, dirs, files in os.walk(scene):
                dirs[:] = sorted(
                    [
                        name
                        for name in dirs
                        if name.casefold() not in _VIDEO_SCAN_EXCLUDED_DIRS and not name.startswith(".")
                    ],
                    key=str.lower,
                )
                for name in sorted(files, key=str.lower):
                    path = Path(root) / name
                    if self._is_supported_video_file(path):
                        videos.append(path)
        except OSError:
            return []
        return videos

    def _video_queue_status_key(self, video: Path) -> str:
        if not video.is_file():
            return "missing"
        matching = self._matching_video_sessions_for_path(video)
        if self._extract_output_mode() == "replace-video":
            return "reextract" if matching else "new"
        return "skip" if matching else "new"

    def _video_queue_status_text(self, video: Path) -> str:
        key = self._video_queue_status_key(video)
        if key == "skip":
            return i18n.t("VIDEO_QUEUE_STATUS_SKIP")
        if key == "reextract":
            return i18n.t("VIDEO_QUEUE_STATUS_REEXTRACT")
        if key == "missing":
            return i18n.t("VIDEO_QUEUE_STATUS_MISSING")
        return i18n.t("VIDEO_QUEUE_STATUS_NEW")

    def _video_projection_text(self, info: dict | None) -> str:
        if not isinstance(info, dict):
            return i18n.t("VIDEO_PROJECTION_UNKNOWN")
        detected = infer_video_projection(info)
        projection = str(detected.get("projection") or "")
        if projection == "equirectangular":
            return i18n.t("VIDEO_PROJECTION_EQUIRECT")
        if projection == "normal":
            return i18n.t("VIDEO_PROJECTION_NORMAL")
        return i18n.t("VIDEO_PROJECTION_UNKNOWN")

    def _queued_selected_videos(self) -> tuple[list[Path], int]:
        videos = self._selected_video_paths()
        mode = self._extract_output_mode()
        if mode == "replace-video":
            return videos, 0
        queued: list[Path] = []
        skipped = 0
        for video in videos:
            if self._matching_video_sessions_for_path(video):
                skipped += 1
            else:
                queued.append(video)
        return queued, skipped

    def _effective_filename_prefix(self, video_path: Path | None = None) -> str:
        prefix = sanitize_filename_prefix(self.prefix_edit.text())
        if prefix and not self._is_multi_video_input():
            return prefix
        if video_path is not None:
            prefix = sanitize_filename_prefix(video_path.stem)
        else:
            video = self.video_browse.text()
            if video:
                prefix = sanitize_filename_prefix(Path(video).stem)
        return prefix or "frame"

    def _prefix_in_use(self, prefix: str) -> bool:
        if not self.scene_dir:
            return False
        scene = Path(self.scene_dir)
        manifest = load_manifest(scene)
        for session in manifest.get("sessions", []):
            if isinstance(session, dict) and session.get("filename_prefix") == prefix:
                return True
        images = scene / "images"
        if images.exists():
            return any(images.glob(f"{prefix}_*"))
        return False

    def _unique_prefix(self, base: str, used_prefixes: set[str]) -> str:
        if base not in used_prefixes and not self._prefix_in_use(base):
            used_prefixes.add(base)
            return base
        for index in range(2, 1000):
            candidate = f"{base}_session{index}"
            if candidate not in used_prefixes and not self._prefix_in_use(candidate):
                used_prefixes.add(candidate)
                return candidate
        used_prefixes.add(f"{base}_session")
        return f"{base}_session"

    def _prefix_for_video(self, video_path: Path, used_prefixes: set[str]) -> str:
        mode = self._extract_output_mode()
        matching = self._matching_video_sessions_for_path(video_path)
        if not self._is_multi_video_input():
            prefix = sanitize_filename_prefix(self.prefix_edit.text())
            if prefix:
                return prefix
        base = self._effective_filename_prefix(video_path)
        if mode == "replace-video" and matching:
            prefix = str(matching[0].get("filename_prefix") or base)
            if prefix not in used_prefixes:
                used_prefixes.add(prefix)
                return prefix
            return self._unique_prefix(prefix, used_prefixes)
        return self._unique_prefix(base, used_prefixes)

    def _readiness(self) -> tuple[bool, str]:
        videos = self._selected_video_paths()
        if len(videos) > 1:
            missing = [video for video in videos if not video.is_file()]
            if missing:
                return False, i18n.t("EXTRACT_READY_VIDEO_NOT_FOUND")
            if not self.scene_dir:
                return False, i18n.t("EXTRACT_READY_NO_SCENE")
            if not self.quick_extract_cb.isChecked() and not self._analysis_width_valid():
                return False, i18n.t("EXTRACT_READY_BAD_ANALYSIS_WIDTH")
            queued, skipped = self._queued_selected_videos()
            mode = self._extract_output_mode()
            if mode == "append" and not queued:
                return False, i18n.t("EXTRACT_READY_QUEUE_ALL_DUPLICATE").format(n=len(videos))
            if mode == "append" and skipped:
                return True, i18n.t("EXTRACT_READY_QUEUE_PARTIAL").format(n=len(queued), skipped=skipped)
            if mode == "replace-video":
                replace_count = sum(1 for video in videos if self._matching_video_sessions_for_path(video))
                return True, i18n.t("EXTRACT_READY_QUEUE_REPLACE").format(n=len(videos), replace=replace_count)
            return True, i18n.t("EXTRACT_READY_QUEUE_OK").format(n=len(videos))

        if not videos:
            return False, i18n.t("EXTRACT_READY_NO_VIDEO")
        if not videos[0].is_file():
            return False, i18n.t("EXTRACT_READY_VIDEO_NOT_FOUND")
        if not self.scene_dir:
            return False, i18n.t("EXTRACT_READY_NO_SCENE")
        if not self.quick_extract_cb.isChecked() and not self._analysis_width_valid():
            return False, i18n.t("EXTRACT_READY_BAD_ANALYSIS_WIDTH")
        if not self.video_info:
            return False, i18n.t("EXTRACT_READY_NO_VIDEO_INFO")
        matching_sessions = self._matching_video_sessions()
        output_mode = self._extract_output_mode()
        if matching_sessions and output_mode == "append":
            return False, i18n.t("EXTRACT_READY_DUPLICATE_VIDEO").format(n=len(matching_sessions))
        if matching_sessions and output_mode == "replace-video":
            return True, i18n.t("EXTRACT_READY_DUPLICATE_REPLACE").format(n=len(matching_sessions))
        return True, i18n.t("EXTRACT_READY_OK")

    def _update_ready_status(self) -> None:
        ready, reason = self._readiness()
        self.ready_status_label.setText(reason)
        if ready:
            self.ready_status_label.setStyleSheet(
                "padding: 8px 10px; border-radius: 4px; color: #dcfce7; background-color: #14532d;"
            )
        else:
            self.ready_status_label.setStyleSheet(
                "padding: 8px 10px; border-radius: 4px; color: #fef3c7; background-color: #713f12;"
            )
        self.primary_action_state_changed.emit()

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        videos = self._selected_video_paths()
        missing = [video for video in videos if not video.is_file()]
        if missing:
            preview = ", ".join(str(video) for video in missing[:3])
            raise ValueError(f"{i18n.t('EXTRACT_READY_VIDEO_NOT_FOUND')}\n{preview}")
        if not self._is_multi_video_input():
            return [("extract", self._build_extract_cmd())]

        videos, _skipped = self._queued_selected_videos()
        if not videos:
            raise ValueError(i18n.t("EXTRACT_READY_QUEUE_ALL_DUPLICATE").format(n=len(self._selected_video_paths())))

        used_prefixes: set[str] = set()
        return [
            (f"extract: {video.name}", self._build_extract_cmd_for_video(video, used_prefixes))
            for video in videos
        ]

    def _build_extract_cmd(self) -> list[str]:
        videos = self._selected_video_paths()
        if not videos:
            raise ValueError("入力動画が指定されていません")
        video = videos[0]
        if not video.is_file():
            raise ValueError(f"入力動画が見つかりません: {video}")
        if not self.scene_dir:
            raise ValueError("シーンフォルダが指定されていません")

        return self._build_extract_cmd_for_video(video, set())

    def _build_extract_cmd_for_video(self, video_path: Path, used_prefixes: set[str]) -> list[str]:
        if not video_path.is_file():
            raise ValueError(f"入力動画が見つかりません: {video_path}")
        if not self.scene_dir:
            raise ValueError("シーンフォルダが指定されていません")

        script = self.base_dir / "extract_frames.py"
        if not script.exists():
            raise FileNotFoundError(f"extract_frames.py が見つかりません: {script}")

        output_mode = self._extract_output_mode()
        prefix = self._prefix_for_video(video_path, used_prefixes)
        quick_extract = self.quick_extract_cb.isChecked()

        cmd = [
            sys.executable, "-u", str(script),
            str(video_path), self.scene_dir,
            "--image-ext", self.image_ext_combo.currentText(),
            "--jpg-quality", str(self.jpg_quality_edit.value()),
            "--ffmpeg", self.ffmpeg_browse.text() or "ffmpeg",
            "--ffprobe", self.ffprobe_browse.text() or "ffprobe",
            "--output-mode", output_mode,
        ]
        if not quick_extract:
            cmd.extend([
                "--pair-motion-profile", str(self.pair_motion_profile_combo.currentData() or "walk"),
                "--analysis-width", self.analysis_width_edit.text().strip(),
            ])
        if prefix:
            cmd.extend(["--filename-prefix", prefix])

        cmd.extend(["--interval-sec", f"{self.interval_edit.value():g}"])
        if quick_extract:
            cmd.append("--quick-extract")
        elif self.smart_fixed_cb.isChecked():
            cmd.extend([
                "--fixed-smart",
                "--min-gap-sec", f"{self.min_gap_edit.value():g}",
                "--max-gap-sec", f"{self.max_gap_edit.value():g}",
            ])

        return cmd

    # -- プログレス解析 --

    def on_line(self, line: str) -> tuple[int, int] | None:
        progress_prefix = "[progress] "
        if line.startswith(progress_prefix):
            text = line[len(progress_prefix):]
            match = re.search(r"(\d+)/(\d+)", text)
            if match:
                return int(match.group(1)), int(match.group(2))

        if line.startswith("SUMMARY_JSON:"):
            payload = line[len("SUMMARY_JSON:"):]
            try:
                summary = json.loads(payload)
                self.last_estimate_summary = summary
                self._apply_summary(summary)
            except Exception:
                pass
        return None

    def on_queue_finished(self, success: bool) -> None:
        if success:
            self._save_source_video_registry()
            self._refresh_finished_run_state(revalidate_video_info=False)
        else:
            self._refresh_finished_run_state(revalidate_video_info=True)

    def _refresh_finished_run_state(self, *, revalidate_video_info: bool) -> None:
        videos = self._selected_video_paths()
        if self._prune_missing_selected_videos():
            return
        if revalidate_video_info and videos:
            self._load_video_info(show_error=False)
            return
        self._update_video_info_label()
        self._update_ready_status()

    # -- 動画情報 --

    def _clear_input_videos(self) -> None:
        self.last_estimate_summary = None
        videos = self._selected_video_paths()
        if self.video_browse.text():
            self._forget_source_videos(videos)
            self.video_browse.set_text("")
        else:
            self.video_info = None
            self.video_infos.clear()
            self.video_info_failures.clear()
            self._update_video_info_label()
            self._update_instant_estimate()
            self._update_ready_status()
        self.input_videos_cleared.emit()

    def _forget_source_videos(self, videos: list[Path]) -> None:
        if not self.scene_dir or not videos:
            return
        remove_source_videos(Path(self.scene_dir), videos)

    def _on_video_changed(self, path: str) -> None:
        videos = self._selected_video_paths()
        self._refresh_video_queue_list()
        self._suggest_scene_dir_from_videos(videos)
        if len(videos) == 1 and videos[0].is_file():
            self._load_video_info(show_error=False)
        elif len(videos) > 1:
            self._load_video_info(show_error=False)
        else:
            self.video_info = None
            self._prune_video_info_cache(videos)
            self._update_video_info_label()
            self._update_instant_estimate()
            self._update_ready_status()

    def _suggest_scene_dir_from_videos(self, videos: list[Path]) -> None:
        if self.scene_dir or not videos:
            return
        if any(not video.is_file() for video in videos):
            return
        try:
            parents = {video.parent.resolve() for video in videos}
        except OSError:
            return
        if len(parents) != 1:
            return
        self.scene_dir_suggested.emit(str(next(iter(parents))))

    @staticmethod
    def _parse_fraction(value: str) -> float:
        if not value:
            return 0.0
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(value)

    @staticmethod
    def _format_duration(sec: float) -> str:
        whole = int(max(0, sec))
        h, m, s = whole // 3600, (whole % 3600) // 60, whole % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _format_number(value: int) -> str:
        return f"{int(value):,}"

    @staticmethod
    def _video_key(video: Path) -> str:
        return str(video)

    @staticmethod
    def _estimated_total_frames(info: dict) -> int:
        total = int(info.get("total_frames", 0))
        if total > 0:
            return total
        dur = float(info.get("duration_sec", 0))
        fps = float(info.get("fps", 0))
        if dur > 0 and fps > 0:
            return max(1, int(round(dur * fps)))
        return 0

    def _fixed_estimate_count(self, info: dict) -> int:
        total = self._estimated_total_frames(info)
        fps = float(info.get("fps", 0))
        if total <= 0 or fps <= 0:
            return 0
        iv = self.interval_edit.value()
        if iv <= 0:
            return 0
        step = max(1, int(round(iv * fps)))
        indices = list(range(0, max(total, 1), step))
        last_index = max(total - 1, 0)
        if indices[-1] != last_index:
            indices.append(last_index)
        return len(indices)

    def _prune_video_info_cache(self, videos: list[Path]) -> None:
        keys = {self._video_key(video) for video in videos}
        self.video_infos = {key: value for key, value in self.video_infos.items() if key in keys}
        self.video_info_failures = {key: value for key, value in self.video_info_failures.items() if key in keys}

    def _load_video_info(self, show_error: bool = True) -> bool:
        videos = self._selected_video_paths()
        self._prune_video_info_cache(videos)
        if len(videos) > 1:
            return self._load_multi_video_info(videos, show_error=show_error)
        try:
            self.video_info = self._probe_video_info()
            if videos:
                key = self._video_key(videos[0])
                self.video_infos[key] = self.video_info
                self.video_info_failures.pop(key, None)
            self._update_video_info_label()
            self._mark_estimate_stale()
            self._update_ready_status()
            return True
        except Exception as e:
            self.video_info = None
            for video in videos[:1]:
                self.video_info_failures[self._video_key(video)] = str(e)
            self._update_video_info_label()
            self.instant_estimate_text = "-"
            self._refresh_estimate_label()
            self._update_ready_status()
            if show_error:
                QMessageBox.critical(self, i18n.INVALID_INPUT, str(e))
            return False

    def _load_multi_video_info(self, videos: list[Path], show_error: bool = True) -> bool:
        self.video_info = None
        self._prune_video_info_cache(videos)
        failures: list[str] = []
        for video in videos:
            key = self._video_key(video)
            try:
                info = self._probe_video_info_for_path(video)
            except Exception as e:
                self.video_infos.pop(key, None)
                self.video_info_failures[key] = str(e)
                failures.append(f"{video.name}: {e}")
                continue
            self.video_infos[key] = info
            self.video_info_failures.pop(key, None)

        self._update_video_info_label()
        self._mark_estimate_stale()
        self._update_ready_status()
        if failures and show_error:
            QMessageBox.warning(self, i18n.INVALID_INPUT, "\n".join(failures))
        return bool(self.video_infos)

    def _reload_video_info_if_selected(self) -> None:
        videos = self._selected_video_paths()
        if videos:
            self._load_video_info(show_error=False)

    def _probe_video_info(self) -> dict:
        videos = self._selected_video_paths()
        if not videos:
            raise ValueError("入力動画が指定されていません")
        return self._probe_video_info_for_path(videos[0])

    def _probe_video_info_for_path(self, video_path: Path) -> dict:
        video = str(video_path)
        ffprobe = self.ffprobe_browse.text() or "ffprobe"
        if not video_path.exists():
            raise ValueError(f"入力動画が見つかりません: {video}")

        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            video,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffprobe 失敗")

        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError("動画ストリームが見つかりません")

        s = streams[0]
        fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
        w, h = int(s.get("width", 0)), int(s.get("height", 0))
        fps = self._parse_fraction(s.get("avg_frame_rate", "0"))
        if fps <= 0:
            fps = self._parse_fraction(s.get("r_frame_rate", "0"))
        dur = float(s.get("duration") or fmt.get("duration") or 0.0)
        nb = int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else 0

        if fps <= 0 and dur > 0 and nb > 0:
            fps = nb / dur
        if fps <= 0:
            raise RuntimeError("FPSを取得できません")
        if dur <= 0 and nb > 0:
            dur = nb / fps
        if nb <= 0 and dur > 0:
            nb = max(1, int(round(dur * fps)))

        return {
            "width": w,
            "height": h,
            "fps": fps,
            "duration_sec": dur,
            "total_frames": nb,
            "tags": s.get("tags") if isinstance(s.get("tags"), dict) else {},
            "format_tags": fmt.get("tags") if isinstance(fmt.get("tags"), dict) else {},
            "side_data_list": s.get("side_data_list") if isinstance(s.get("side_data_list"), list) else [],
        }

    def _save_source_video_registry(self) -> None:
        if not self.scene_dir:
            return
        records: list[dict] = []
        for video in self._selected_video_paths():
            if not video.is_file():
                continue
            info = self.video_infos.get(self._video_key(video))
            if info is None and len(self._selected_video_paths()) == 1:
                info = self.video_info
            if not isinstance(info, dict):
                continue
            try:
                records.append(source_video_record(video, info))
            except OSError:
                continue
        if records:
            upsert_source_videos(Path(self.scene_dir), records)

    def _update_video_info_label(self) -> None:
        self._refresh_video_queue_list()
        if self._is_multi_video_input():
            videos = self._selected_video_paths()
            queued, skipped = self._queued_selected_videos()
            info_rows = [(video, self.video_infos[self._video_key(video)]) for video in videos if self._video_key(video) in self.video_infos]
            if info_rows:
                failed = len([video for video in videos if self._video_key(video) in self.video_info_failures])
                lines = [
                    i18n.t("VIDEO_INFO_MULTI_HEADER_FORMAT").format(
                        total=len(videos),
                        queued=len(queued),
                        skipped=skipped,
                        probed=len(info_rows),
                    )
                ]
                for video, info in info_rows:
                    lines.append(
                        i18n.t("VIDEO_INFO_MULTI_ITEM_FORMAT").format(
                            name=video.name,
                            status=self._video_queue_status_text(video),
                            projection=self._video_projection_text(info),
                            width=info["width"],
                            height=info["height"],
                            fps=info["fps"],
                            duration=self._format_duration(float(info.get("duration_sec", 0))),
                            frames=self._format_number(self._estimated_total_frames(info)),
                        )
                    )
                if failed:
                    lines[0] += i18n.t("VIDEO_INFO_FAILED_SUFFIX").format(failed=failed)
                self.video_info_label.setText("\n".join(lines))
                return
            self.video_info_label.setText(
                i18n.t("VIDEO_QUEUE_LABEL_FORMAT").format(
                    total=len(videos),
                    queued=len(queued),
                    skipped=skipped,
                )
            )
            return
        if not self.video_info:
            self.video_info_label.setText(i18n.t("VIDEO_LABEL_DEFAULT"))
            return
        i = self.video_info
        d = self._format_duration(float(i["duration_sec"]))
        videos = self._selected_video_paths()
        status = self._video_queue_status_text(videos[0]) if videos else i18n.t("VIDEO_QUEUE_STATUS_NEW")
        self.video_info_label.setText(
            i18n.t("VIDEO_INFO_SINGLE_FORMAT").format(
                status=status,
                projection=self._video_projection_text(i),
                width=i["width"],
                height=i["height"],
                fps=i["fps"],
                duration=d,
                frames=self._format_number(self._estimated_total_frames(i)),
            )
        )

    # -- フレーム数推定 --

    def _refresh_estimate_label(self) -> None:
        self.estimate_label.setText(f"{i18n.INSTANT_ESTIMATE}: {self.instant_estimate_text}")

    def _mark_estimate_stale(self, *_args) -> None:
        self.last_estimate_summary = None
        self._update_instant_estimate()
        self._update_ready_status()

    def _update_instant_estimate(self) -> None:
        if self._is_multi_video_input():
            queued, skipped = self._queued_selected_videos()
            info_rows = [(video, self.video_infos[self._video_key(video)]) for video in queued if self._video_key(video) in self.video_infos]
            if info_rows:
                counts = [(video, self._fixed_estimate_count(info)) for video, info in info_rows]
                total_estimated = sum(count for _video, count in counts)
                missing = max(0, len(queued) - len(info_rows))
                lines = [
                    i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_HEADER_FORMAT").format(
                        interval=f"{self.interval_edit.value():g}",
                    )
                ]
                for video, count in counts:
                    lines.append(
                        i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_ITEM_FORMAT").format(
                            name=video.name,
                            count=self._format_number(count),
                        )
                    )
                total_line = i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_TOTAL_FORMAT").format(
                    count=self._format_number(total_estimated),
                    videos=len(info_rows),
                )
                if self.quick_extract_cb.isChecked():
                    total_line += f" ({i18n.t('QUICK_EXTRACT_ESTIMATE')})"
                elif self.smart_fixed_cb.isChecked():
                    total_line += f" ({i18n.t('FIXED_SMART_ESTIMATE')})"
                if missing:
                    total_line += i18n.t("ESTIMATE_MISSING_INFO_SUFFIX").format(missing=missing)
                lines.append(total_line)
                self.instant_estimate_text = "\n".join(lines)
                self._refresh_estimate_label()
                return
            self.instant_estimate_text = i18n.t("QUEUE_ESTIMATE_FORMAT").format(
                queued=len(queued),
                skipped=skipped,
            )
            self._refresh_estimate_label()
            return
        if not self.video_info:
            self.instant_estimate_text = "-"
            self._refresh_estimate_label()
            return
        dur = float(self.video_info.get("duration_sec", 0))
        fps = float(self.video_info.get("fps", 0))
        total = int(self.video_info.get("total_frames", 0))

        try:
            iv = self.interval_edit.value()
            if iv <= 0:
                raise ValueError
            if total <= 0 and dur > 0 and fps > 0:
                total = max(1, int(round(dur * fps)))
            info = dict(self.video_info)
            info["total_frames"] = total
            estimated = self._fixed_estimate_count(info)
            text = i18n.t("FIXED_INTERVAL_ESTIMATE_FORMAT").format(
                interval=f"{iv:g}",
                count=self._format_number(estimated),
            )
            if self.quick_extract_cb.isChecked():
                text += f" ({i18n.t('QUICK_EXTRACT_ESTIMATE')})"
            elif self.smart_fixed_cb.isChecked():
                text += f" ({i18n.t('FIXED_SMART_ESTIMATE')})"
            self.instant_estimate_text = text
        except Exception:
            self.instant_estimate_text = "-"
        self._refresh_estimate_label()

    def _apply_summary(self, summary: dict) -> None:
        video = summary.get("video", {})
        if video:
            self.video_info = {
                "width": int(video.get("width", 0)),
                "height": int(video.get("height", 0)),
                "fps": float(video.get("fps", 0.0)),
                "duration_sec": float(video.get("duration_sec", 0.0)),
                "total_frames": int(video.get("total_frames", 0)),
            }
            self._update_video_info_label()

        result = summary.get("result", {})
        selected = int(result.get("selected_count", 0))
        total_f = int(video.get("total_frames", 0))
        ratio = (selected / total_f * 100.0) if total_f > 0 else 0.0
        parts = [f"{selected} {i18n.t('FRAMES_UNIT')} ({ratio:.1f}%)"]
        if result.get("novelty_added_count"):
            parts.append(f"+{int(result['novelty_added_count'])}")
        if result.get("dropped_count"):
            parts.append(f"-{int(result['dropped_count'])}")
        self.instant_estimate_text = " ".join(parts)
        self._refresh_estimate_label()
