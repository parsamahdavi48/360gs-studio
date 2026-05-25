"""Step 1: フレーム抽出"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.input_sources import (
    SOURCE_KIND_IMAGE_SEQUENCE,
    SOURCE_KIND_VIDEO,
    InputSource,
)
from core.scene_layout import scene_images_dir
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox, DragSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.common.icons import delete_icon, image_folder_source_icon, reset_icon, video_source_icon
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)
from gui.steps.step1_execution import Step1ExecutionMixin
from gui.steps.step1_input_sources import Step1InputSourcesMixin
from gui.steps.step1_video_info import Step1VideoInfoMixin

__all__ = ["ExtractStep"]

_FIXED_INTERVAL_MIN = 0.05
_FIXED_INTERVAL_MAX = 60.0
_CHANGE_GAP_MIN = 0.05
_CHANGE_GAP_MAX = 60.0
_GAP_SPINBOX_WIDTH = 112
_DEFAULT_CAPTURE_PROFILE = "walk_standard"
_CAPTURE_PROFILE_PRESETS: dict[str, tuple[float, float, float]] = {
    "walk_standard": (1.5, 0.8, 4.0),
    "walk_close": (1.0, 0.5, 2.5),
    "walk_wide": (3.0, 1.5, 7.0),
    "drone_distant": (3.0, 1.5, 8.0),
}
_JPEG_QUALITY_MIN = 1
_JPEG_QUALITY_MAX = 31
_JPEG_QUALITY_DEFAULT = 2
_SOURCE_MODE_VIDEO = "video"
_SOURCE_MODE_IMAGE_SEQUENCE = "image_sequence"
_SOURCE_KIND_VIDEO = SOURCE_KIND_VIDEO
_SOURCE_KIND_IMAGE_SEQUENCE = SOURCE_KIND_IMAGE_SEQUENCE


def _detect_binary(name: str) -> str:
    """PATH から実行ファイルを検出。見つからなければ素の名前を返す（PATH 解決に委ねる）。"""
    found = shutil.which(name)
    return found if found else name


def _row_label(text: str, tooltip: str | None = None) -> QLabel:
    label = QLabel(text)
    if tooltip:
        label.setToolTip(tooltip)
    return label


class ExtractStep(Step1ExecutionMixin, Step1VideoInfoMixin, Step1InputSourcesMixin, BaseStepWidget):
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
        self._input_sources: list[InputSource] = []
        self._syncing_input_source_widgets = False

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

        self.source_mode_combo = QComboBox()
        self.source_mode_combo.setToolTip(i18n.tip("INPUT_SOURCE_MODE"))
        self.source_mode_combo.addItem(i18n.t("INPUT_SOURCE_VIDEO"), _SOURCE_MODE_VIDEO)
        self.source_mode_combo.addItem(i18n.t("INPUT_SOURCE_IMAGE_SEQUENCE"), _SOURCE_MODE_IMAGE_SEQUENCE)
        self.source_mode_combo.setFixedWidth(180)
        self.source_mode_combo.currentIndexChanged.connect(lambda _: self._on_source_mode_changed())
        self.source_mode_combo.hide()

        self.video_browse = BrowseWidget(
            self,
            mode="files",
            filter_str=i18n.t("VIDEO_FILE_FILTER"),
            placeholder=i18n.t("INPUT_VIDEO_PLACEHOLDER"),
        )
        self.video_browse.setToolTip(i18n.tip("INPUT_VIDEO"))
        self.video_browse.path_changed.connect(self._on_video_changed)
        self.video_browse.hide()

        self.image_sequence_browse = BrowseWidget(
            self,
            mode="dir",
            placeholder=i18n.t("INPUT_IMAGE_SEQUENCE_PLACEHOLDER"),
        )
        self.image_sequence_browse.setToolTip(i18n.tip("INPUT_IMAGE_SEQUENCE"))
        self.image_sequence_browse.path_changed.connect(self._on_image_sequence_changed)
        self.image_sequence_browse.hide()
        self.image_sequence_label = None

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

        self.pair_motion_profile_combo = QComboBox()
        self.pair_motion_profile_combo.setToolTip(i18n.tip("PAIR_MOTION_PROFILE"))
        self.pair_motion_profile_combo.addItem(i18n.t("PAIR_PROFILE_WALK_STANDARD"), "walk_standard")
        self.pair_motion_profile_combo.addItem(i18n.t("PAIR_PROFILE_WALK_CLOSE"), "walk_close")
        self.pair_motion_profile_combo.addItem(i18n.t("PAIR_PROFILE_WALK_WIDE"), "walk_wide")
        self.pair_motion_profile_combo.addItem(i18n.t("PAIR_PROFILE_DRONE_DISTANT"), "drone_distant")
        self.pair_motion_profile_combo.setFixedWidth(170)
        add_tooltip_row(
            basic,
            i18n.t("PAIR_MOTION_PROFILE"),
            self.pair_motion_profile_combo,
            i18n.tip("PAIR_MOTION_PROFILE"),
        )
        self.pair_motion_profile_label = basic.labelForField(self.pair_motion_profile_combo)
        layout.addLayout(basic)

        default_interval, default_min_gap, default_max_gap = _CAPTURE_PROFILE_PRESETS[_DEFAULT_CAPTURE_PROFILE]
        self.interval_edit = DragDoubleSpinBox(
            minimum=_FIXED_INTERVAL_MIN,
            maximum=_FIXED_INTERVAL_MAX,
            step=0.05,
            decimals=2,
            value=default_interval,
            suffix=f" {i18n.t('SECONDS_SUFFIX')}",
            drag_pixels_per_step=6.0,
        )
        self.interval_edit.setToolTip(i18n.tip("INTERVAL"))
        self.interval_edit.setFixedWidth(86)
        self.interval_edit.valueChanged.connect(lambda _: self._clamp_interval_constraints("interval"))
        self.interval_edit.valueChanged.connect(self._mark_estimate_stale)

        self.min_gap_edit = DragDoubleSpinBox(
            minimum=_CHANGE_GAP_MIN,
            maximum=_CHANGE_GAP_MAX,
            step=0.05,
            decimals=2,
            value=default_min_gap,
            suffix=f" {i18n.t('SECONDS_SUFFIX')}",
            drag_pixels_per_step=6.0,
        )
        self.min_gap_edit.setToolTip(i18n.tip("MIN_GAP"))
        self.min_gap_edit.setFixedWidth(_GAP_SPINBOX_WIDTH)
        self.min_gap_edit.valueChanged.connect(lambda _: self._clamp_interval_constraints("min"))
        self.min_gap_edit.valueChanged.connect(self._mark_estimate_stale)

        self.max_gap_edit = DragDoubleSpinBox(
            minimum=_CHANGE_GAP_MIN,
            maximum=_CHANGE_GAP_MAX,
            step=0.05,
            decimals=2,
            value=default_max_gap,
            suffix=f" {i18n.t('SECONDS_SUFFIX')}",
            drag_pixels_per_step=6.0,
        )
        self.max_gap_edit.setToolTip(i18n.tip("MAX_GAP"))
        self.max_gap_edit.setFixedWidth(_GAP_SPINBOX_WIDTH)
        self.max_gap_edit.valueChanged.connect(lambda _: self._clamp_interval_constraints("max"))
        self.max_gap_edit.valueChanged.connect(self._mark_estimate_stale)

        self.smart_fixed_cb = QCheckBox(i18n.t("FIXED_SMART"))
        self.smart_fixed_cb.setToolTip(i18n.tip("FIXED_SMART"))
        self.smart_fixed_cb.setChecked(True)
        self.smart_fixed_cb.toggled.connect(self._update_mode_widgets)
        self.smart_fixed_cb.toggled.connect(self._mark_estimate_stale)
        self.pair_motion_profile_combo.currentIndexChanged.connect(self._on_pair_motion_profile_changed)

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

        self.video_queue_header_widget = QWidget()
        queue_header = QHBoxLayout(self.video_queue_header_widget)
        queue_header.setContentsMargins(0, 0, 0, 0)
        queue_header.setSpacing(6)
        queue_header.addWidget(QLabel(i18n.t("INPUT_SOURCE_QUEUE_SECTION")))
        self.video_queue_summary_label = QLabel("")
        self.video_queue_summary_label.setObjectName("videoQueueSummary")
        self.video_queue_summary_label.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self.video_queue_summary_label.setWordWrap(False)
        queue_header.addWidget(self.video_queue_summary_label, stretch=1)
        queue_header.addStretch()
        self.add_video_btn = QToolButton()
        self.add_video_btn.setObjectName("iconToolButton")
        self.add_video_btn.setIcon(video_source_icon())
        self.add_video_btn.setToolTip(i18n.tip("ADD_INPUT_VIDEO"))
        self.add_video_btn.setAccessibleName(i18n.t("ADD_INPUT_VIDEO"))
        self.add_video_btn.setFixedSize(32, 32)
        self.add_video_btn.clicked.connect(self._add_input_videos)
        queue_header.addWidget(self.add_video_btn)
        self.add_image_sequence_btn = QToolButton()
        self.add_image_sequence_btn.setObjectName("iconToolButton")
        self.add_image_sequence_btn.setIcon(image_folder_source_icon())
        self.add_image_sequence_btn.setToolTip(i18n.tip("ADD_INPUT_IMAGE_SEQUENCE"))
        self.add_image_sequence_btn.setAccessibleName(i18n.t("ADD_INPUT_IMAGE_SEQUENCE"))
        self.add_image_sequence_btn.setFixedSize(32, 32)
        self.add_image_sequence_btn.clicked.connect(self._add_input_image_sequence)
        queue_header.addWidget(self.add_image_sequence_btn)
        self.remove_video_btn = QToolButton()
        self.remove_video_btn.setObjectName("iconToolButton")
        self.remove_video_btn.setIcon(delete_icon())
        self.remove_video_btn.setToolTip(i18n.tip("REMOVE_INPUT_SOURCE"))
        self.remove_video_btn.setAccessibleName(i18n.t("REMOVE_INPUT_SOURCE"))
        self.remove_video_btn.setFixedSize(32, 32)
        self.remove_video_btn.clicked.connect(self._remove_selected_input_videos)
        queue_header.addWidget(self.remove_video_btn)
        self.clear_video_btn = QToolButton()
        self.clear_video_btn.setObjectName("iconToolButton")
        self.clear_video_btn.setIcon(reset_icon())
        self.clear_video_btn.setToolTip(i18n.t("CLEAR_INPUT_SOURCES_HINT"))
        self.clear_video_btn.setAccessibleName(i18n.t("CLEAR_INPUT_SOURCES"))
        self.clear_video_btn.setFixedSize(32, 32)
        self.clear_video_btn.clicked.connect(self._clear_input_videos)
        queue_header.addWidget(self.clear_video_btn)
        work_layout.addWidget(self.video_queue_header_widget)

        self.video_queue_list = QListWidget()
        self.video_queue_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.video_queue_list.setMinimumHeight(150)
        self.video_queue_list.setToolTip(i18n.tip("INPUT_SOURCE_QUEUE_SECTION"))
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
        advanced.content_layout.addLayout(adv_form)

        path_form = QFormLayout()
        path_form.setSpacing(6)

        # ffmpeg / ffprobe: PATH から自動検出して初期値にセット。参照ボタンで上書き可能
        ffmpeg_filter = "Executable (*.exe);;すべて (*.*)" if sys.platform == "win32" else "すべて (*.*)"
        self.ffmpeg_browse = BrowseWidget(
            mode="file", filter_str=ffmpeg_filter, placeholder="ffmpeg (PATH から自動検出)"
        )
        self.ffmpeg_browse.set_text(_detect_binary("ffmpeg"))
        self.ffmpeg_browse.setToolTip(i18n.tip("FFMPEG_PATH"))
        add_tooltip_row(path_form, i18n.FFMPEG_PATH, self.ffmpeg_browse, i18n.tip("FFMPEG_PATH"))

        self.ffprobe_browse = BrowseWidget(
            mode="file", filter_str=ffmpeg_filter, placeholder="ffprobe (PATH から自動検出)"
        )
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
        self._update_source_mode_widgets()
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
            self.images_path_label.setText(str(scene_images_dir(Path(self.scene_dir))))
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

    def _source_mode(self) -> str:
        return str(self.source_mode_combo.currentData() or _SOURCE_MODE_VIDEO)

    def _is_image_sequence_mode(self) -> bool:
        return self._source_mode() == _SOURCE_MODE_IMAGE_SEQUENCE

    def _on_source_mode_changed(self) -> None:
        if self._syncing_input_source_widgets:
            return
        self.video_info = None
        self.video_infos.clear()
        self.video_info_failures.clear()
        if self._is_image_sequence_mode():
            folders = self._image_sequence_dirs_from_text()
            if folders:
                self._set_input_sources([InputSource(_SOURCE_KIND_IMAGE_SEQUENCE, folder) for folder in folders])
                return
        else:
            videos = self._video_paths_from_text()
            if videos:
                self._set_input_sources([InputSource(_SOURCE_KIND_VIDEO, video) for video in videos])
                return
        self._update_source_mode_widgets()
        self._update_video_info_label()
        self._update_instant_estimate()
        self._update_ready_status()

    def _update_source_mode_widgets(self) -> None:
        self.image_sequence_browse.setVisible(False)
        if self.image_sequence_label is not None:
            self.image_sequence_label.setVisible(False)
        self.video_queue_header_widget.setVisible(True)
        self.video_queue_list.setVisible(True)
        self.output_mode_combo.setEnabled(True)
        self.mode_panel.setEnabled(True)
        for widget in (self.image_ext_combo, self.jpg_quality_edit):
            widget.setEnabled(True)
        self.ffmpeg_browse.setEnabled(True)
        self.ffprobe_browse.setEnabled(True)

    def _on_image_sequence_changed(self, _path: str) -> None:
        if self._syncing_input_source_widgets:
            return
        folders = self._image_sequence_dirs_from_text()
        self._set_input_sources([InputSource(_SOURCE_KIND_IMAGE_SEQUENCE, folder) for folder in folders])

    def _on_pair_motion_profile_changed(self, *_args) -> None:
        profile = str(self.pair_motion_profile_combo.currentData() or _DEFAULT_CAPTURE_PROFILE)
        preset = _CAPTURE_PROFILE_PRESETS.get(profile)
        if preset is not None:
            interval_sec, min_gap_sec, max_gap_sec = preset
            self._syncing_gap_fields = True
            try:
                self.interval_edit.setValue(interval_sec)
                self.min_gap_edit.setValue(min_gap_sec)
                self.max_gap_edit.setValue(max_gap_sec)
            finally:
                self._syncing_gap_fields = False
        self._mark_estimate_stale()

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
        self._clamp_interval_constraints("interval")

    def _clamp_interval_constraints(self, changed: str) -> None:
        if self._syncing_gap_fields:
            return
        if self.quick_extract_cb.isChecked() or not self.smart_fixed_cb.isChecked():
            return
        interval = self.interval_edit.value()
        min_gap = self.min_gap_edit.value()
        max_gap = self.max_gap_edit.value()
        self._syncing_gap_fields = True
        try:
            if changed == "interval":
                if min_gap > interval:
                    self.min_gap_edit.setValue(interval)
                if max_gap < interval:
                    self.max_gap_edit.setValue(interval)
            elif changed == "min":
                if min_gap > interval:
                    self.interval_edit.setValue(min_gap)
                if min_gap > max_gap:
                    self.max_gap_edit.setValue(min_gap)
            elif changed == "max":
                if max_gap < interval:
                    self.interval_edit.setValue(max_gap)
                if max_gap < min_gap:
                    self.min_gap_edit.setValue(max_gap)
        finally:
            self._syncing_gap_fields = False

    def _analysis_width_valid(self) -> bool:
        text = self.analysis_width_edit.text().strip()
        try:
            return int(text) >= 0
        except ValueError:
            return False
