"""Step 4: 視点画像書き出し (Metashape / COLMAP modes)."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QSize, Qt, QTimer
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from colmap_rig_export import pinhole_camera_params
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.cubemap.preview_renderer import PreviewWidget
from gui.cubemap.view_config import _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS, ViewConfigWidget
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
    configure_settings_scroll,
)
from gui.steps.cubemap_commands import (
    ColmapExportCommand,
    ColmapSfmCommand,
    CubemapConversionCommand,
    MetashapePreprocessCommand,
    SphereSfmCommand,
    build_colmap_export_cmd,
    build_colmap_sfm_commands,
    build_cubemap_conversion_cmd,
    build_metashape_preprocess_cmd,
    build_spheresfm_commands,
    views_config_payload,
    write_views_config,
)
from gui.user_settings import load_user_settings_section, update_user_settings_section
from gui.version import APP_VERSION

_CONVERT_RE = re.compile(r"^Converting\s+(\d+)\s+(?:images|files)\.\.\.$")
_PROGRESS_RE = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)")
_COLMAP_FEATURE_RE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_COLMAP_MATCH_IMAGE_RE = re.compile(r"Matching image \[(\d+)/(\d+)\]")
_COLMAP_MATCH_BLOCK_RE = re.compile(r"Matching block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]")
_COLMAP_GLOBAL_BA_FIXED_RE = re.compile(
    r"Global bundle adjustment iteration\s+(\d+)\s*/\s*(\d+),\s*fixed-rotation stage finished"
)
_COLMAP_GLOBAL_BA_DONE_RE = re.compile(r"Global bundle adjustment iteration\s+(\d+)\s*/\s*(\d+)\s+finished")
_COLMAP_RETRIANGULATION_START_RE = re.compile(r"=== Running iterative retriangulation and refinement ===")
_COLMAP_RETRIANGULATION_DONE_RE = re.compile(r"Iterative retriangulation and refinement done")
_COLMAP_RECONSTRUCTION_DONE_RE = re.compile(r"Reconstruction done")
_PROFILE_POSTSHOT = "postshot"
_PROFILE_BRUSH = "brush"
_PROFILE_LICHTFELD = "lichtfeld"
_PROFILE_CUSTOM = "custom"
_METHOD_METASHAPE = "metashape"
_METHOD_COLMAP = "colmap"
_METHOD_SPHERESFM = "spheresfm"
_OUTPUT_SHAPE_PROJECTED = "projected"
_OUTPUT_SHAPE_EQUIRECT_3DGUT = "equirect_3dgut"
_COLMAP_MAPPER_INCREMENTAL = "incremental"
_COLMAP_MAPPER_GLOBAL = "global"
_COLMAP_MAPPER_GLOMAP = "glomap"
_COLMAP_MATCHER_SEQUENTIAL = "sequential"
_COLMAP_MATCHER_EXHAUSTIVE = "exhaustive"
_SPHERESFM_MATCHER_SPATIAL = "spatial"
_SPHERESFM_FEATURE_FAST = "fast"
_SPHERESFM_FEATURE_STANDARD = "standard"
_SPHERESFM_FEATURE_ROBUST = "robust"
_AXIS_POSTSHOT = "postshot"
_AXIS_BRUSH = "brush"
_AXIS_NONE = "none"
_NORMAL_OUTPUT_SCALE = 2.0 / math.pi
_EXPORT_SETTINGS_NAME = "stechdrive_export_settings.json"
_COLMAP_PROJECT_MANIFEST_NAME = "stechdrive_colmap_project.json"
_SPHERESFM_PROJECT_MANIFEST_NAME = "stechdrive_spheresfm_project.json"
_USER_SETTINGS_SECTION = "step4_colmap"
_LICHTFELD_FINAL_CORRECTION = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class ElidedPathLabel(QLabel):
    """Keep long paths on one line while preserving the full path internally."""

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = text or "-"
        self._apply_elide()

    def full_text(self) -> str:
        return self._full_text

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        return QSize(0, base.height())

    def minimumSizeHint(self) -> QSize:
        base = super().minimumSizeHint()
        return QSize(0, base.height())

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        width = max(0, self.contentsRect().width())
        text = self._full_text
        if width > 0:
            text = QFontMetrics(self.font()).elidedText(text, Qt.ElideMiddle, width)
        QLabel.setText(self, text)


class CubemapStep(BaseStepWidget):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._converted_total = 0
        self._processed = 0
        self._explicit_progress = False
        self._colmap_ba_iterations = 0
        self._syncing_profile_controls = False
        self._syncing_output_shape_controls = False
        self._syncing_user_preferences = False
        self._user_preferences_enabled = False
        self._export_method_value = _METHOD_METASHAPE
        self._saved_projected_export_targets: tuple[bool, bool] | None = None
        self._input_image_count = 0
        self._preview_render_pending = False
        self._preview_render_timer = QTimer(self)
        self._preview_render_timer.setSingleShot(True)
        self._preview_render_timer.setInterval(50)
        self._preview_render_timer.timeout.connect(self._flush_scheduled_render_preview)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 左パネル: 設定 (スクロール可能) + 固定サマリー
        left_pane = QWidget()
        left_pane.setFixedWidth(SETTINGS_PANE_WIDTH)
        left_pane_layout = QVBoxLayout(left_pane)
        left_pane_layout.setContentsMargins(0, 0, 0, 0)
        left_pane_layout.setSpacing(0)

        top_scroll = QScrollArea()
        configure_settings_scroll(top_scroll)
        top = QWidget()
        top.setObjectName("settingsPane")
        top.setMinimumWidth(0)
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(*SETTINGS_PANE_MARGINS)
        top_layout.setSpacing(8)
        left_layout = top_layout  # 既存コードとの互換用エイリアス

        output_dir_label = QLabel(i18n.OUTPUT_DIR)
        output_dir_label.setToolTip(i18n.tip("OUTPUT_DIR_CUBEMAP"))
        self.output_path_label = ElidedPathLabel("-")
        self.output_path_label.setToolTip(i18n.tip("OUTPUT_DIR_CUBEMAP"))
        output_row = QWidget()
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.setSpacing(6)
        output_row_layout.addWidget(output_dir_label)
        output_row_layout.addWidget(self.output_path_label, stretch=1)
        left_layout.addWidget(output_row)

        self.export_method_label = QLabel(i18n.t("EXPORT_METHOD_COMPACT"))
        self.export_method_label.setToolTip(i18n.tip("EXPORT_METHOD"))
        self.export_method_row = QWidget()
        method_row = QHBoxLayout(self.export_method_row)
        method_row.setContentsMargins(0, 0, 0, 0)
        method_row.setSpacing(6)
        method_row.addWidget(self.export_method_label)
        self.export_method_group = QButtonGroup(self)
        self.export_method_group.setExclusive(True)
        self.export_method_buttons: dict[str, QPushButton] = {}
        for method, label, tip_key in [
            (_METHOD_METASHAPE, i18n.t("METHOD_METASHAPE_IMPORT"), "METHOD_METASHAPE_IMPORT"),
            (_METHOD_COLMAP, i18n.t("METHOD_COLMAP_EXPORT"), "METHOD_COLMAP_EXPORT"),
            (_METHOD_SPHERESFM, i18n.t("METHOD_SPHERESFM"), "METHOD_SPHERESFM"),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("segmentedOption")
            btn.setCheckable(True)
            btn.setToolTip(i18n.tip(tip_key))
            btn.clicked.connect(lambda _checked=False, m=method: self._set_export_method(m))
            method_row.addWidget(btn, stretch=1)
            self.export_method_group.addButton(btn)
            self.export_method_buttons[method] = btn
        left_layout.addWidget(self.export_method_row)

        export_targets_form = QFormLayout()
        export_targets_form.setContentsMargins(0, 0, 0, 0)
        export_targets_form.setSpacing(6)
        self.export_targets_row = QWidget()
        export_targets_layout = QHBoxLayout(self.export_targets_row)
        export_targets_layout.setContentsMargins(0, 0, 0, 0)
        export_targets_layout.setSpacing(12)
        self.export_images_cb = QCheckBox(i18n.t("EXPORT_IMAGES"))
        self.export_images_cb.setToolTip(i18n.tip("EXPORT_IMAGES"))
        self.export_images_cb.setChecked(True)
        export_targets_layout.addWidget(self.export_images_cb)
        self.export_masks_cb = QCheckBox(i18n.t("EXPORT_MASKS"))
        self.export_masks_cb.setToolTip(i18n.tip("EXPORT_MASKS"))
        self.export_masks_cb.setChecked(True)
        export_targets_layout.addWidget(self.export_masks_cb)
        export_targets_layout.addStretch()
        add_tooltip_row(
            export_targets_form,
            i18n.t("EXPORT_TARGETS"),
            self.export_targets_row,
            i18n.tip("EXPORT_TARGETS"),
        )
        left_layout.addLayout(export_targets_form)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("step4SettingsTabs")

        colmap_section = QWidget()
        self.colmap_section = colmap_section
        colmap_section_layout = QVBoxLayout(colmap_section)
        colmap_section_layout.setContentsMargins(8, 8, 8, 8)
        colmap_section_layout.setSpacing(6)
        colmap_form = QFormLayout()
        colmap_form.setSpacing(6)

        self.run_colmap_cb = QCheckBox(i18n.t("RUN_COLMAP_SFM"))
        self.run_colmap_cb.setToolTip(i18n.tip("RUN_COLMAP_SFM"))
        self.run_colmap_cb.toggled.connect(self._on_colmap_run_toggled)
        colmap_form.addRow("", self.run_colmap_cb)

        exe_filter = "Executable (*.exe);;All (*.*)" if os.name == "nt" else "All (*)"
        self.colmap_exec_browse = BrowseWidget(
            mode="file",
            filter_str=exe_filter,
            placeholder="colmap.exe" if os.name == "nt" else "colmap",
        )
        self.colmap_exec_browse.setToolTip(i18n.tip("COLMAP_EXECUTABLE"))
        add_tooltip_row(
            colmap_form,
            i18n.t("COLMAP_EXECUTABLE"),
            self.colmap_exec_browse,
            i18n.tip("COLMAP_EXECUTABLE"),
        )

        self.colmap_pipeline_row = QWidget()
        pipeline_layout = QHBoxLayout(self.colmap_pipeline_row)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.setSpacing(8)
        self.colmap_matcher_combo = QComboBox()
        self.colmap_matcher_combo.setToolTip(i18n.tip("COLMAP_MATCHER"))
        self.colmap_matcher_combo.addItem(i18n.t("COLMAP_MATCHER_SEQUENTIAL"), _COLMAP_MATCHER_SEQUENTIAL)
        self.colmap_matcher_combo.addItem(i18n.t("COLMAP_MATCHER_EXHAUSTIVE"), _COLMAP_MATCHER_EXHAUSTIVE)
        self.colmap_matcher_combo.setFixedWidth(120)
        self.colmap_mapper_combo = QComboBox()
        self.colmap_mapper_combo.setToolTip(i18n.tip("COLMAP_MAPPER"))
        self.colmap_mapper_combo.addItem(i18n.t("COLMAP_MAPPER_GLOBAL"), _COLMAP_MAPPER_GLOBAL)
        self.colmap_mapper_combo.addItem(i18n.t("COLMAP_MAPPER_INCREMENTAL"), _COLMAP_MAPPER_INCREMENTAL)
        self.colmap_mapper_combo.addItem(i18n.t("COLMAP_MAPPER_GLOMAP"), _COLMAP_MAPPER_GLOMAP)
        self.colmap_mapper_combo.setFixedWidth(150)
        self.colmap_mapper_combo.currentIndexChanged.connect(self._on_colmap_mapper_changed)
        pipeline_layout.addWidget(QLabel(i18n.t("COLMAP_MATCHER_COMPACT")))
        pipeline_layout.addWidget(self.colmap_matcher_combo)
        pipeline_layout.addWidget(QLabel(i18n.t("COLMAP_MAPPER_COMPACT")))
        pipeline_layout.addWidget(self.colmap_mapper_combo)
        pipeline_layout.addStretch()
        colmap_form.addRow(self.colmap_pipeline_row)

        self.glomap_exec_browse = BrowseWidget(
            mode="file",
            filter_str=exe_filter,
            placeholder="glomap.exe" if os.name == "nt" else "glomap",
        )
        self.glomap_exec_browse.setToolTip(i18n.tip("GLOMAP_EXECUTABLE"))
        self.glomap_exec_row_label = QLabel(i18n.t("GLOMAP_EXECUTABLE"))
        self.glomap_exec_row_label.setToolTip(i18n.tip("GLOMAP_EXECUTABLE"))
        colmap_form.addRow(self.glomap_exec_row_label, self.glomap_exec_browse)

        colmap_section_layout.addLayout(colmap_form)
        colmap_section_layout.addStretch()

        spheresfm_section = QWidget()
        self.spheresfm_section = spheresfm_section
        spheresfm_layout = QVBoxLayout(spheresfm_section)
        spheresfm_layout.setContentsMargins(8, 8, 8, 8)
        spheresfm_layout.setSpacing(6)
        spheresfm_form = QFormLayout()
        spheresfm_form.setSpacing(6)

        self.spheresfm_exec_browse = BrowseWidget(
            mode="file",
            filter_str=exe_filter,
            placeholder="colmap.exe" if os.name == "nt" else "colmap",
        )
        self.spheresfm_exec_browse.setToolTip(i18n.tip("SPHERESFM_EXECUTABLE"))
        add_tooltip_row(
            spheresfm_form,
            i18n.t("SPHERESFM_EXECUTABLE"),
            self.spheresfm_exec_browse,
            i18n.tip("SPHERESFM_EXECUTABLE"),
        )

        self.spheresfm_use_masks_cb = QCheckBox(i18n.t("SPHERESFM_USE_MASKS"))
        self.spheresfm_use_masks_cb.setToolTip(i18n.tip("SPHERESFM_USE_MASKS"))
        self.spheresfm_use_masks_cb.setChecked(True)
        spheresfm_form.addRow("", self.spheresfm_use_masks_cb)

        self.spheresfm_matcher_combo = QComboBox()
        self.spheresfm_matcher_combo.setToolTip(i18n.tip("SPHERESFM_MATCHER"))
        self.spheresfm_matcher_combo.addItem(i18n.t("COLMAP_MATCHER_SEQUENTIAL"), _COLMAP_MATCHER_SEQUENTIAL)
        self.spheresfm_matcher_combo.addItem(i18n.t("COLMAP_MATCHER_EXHAUSTIVE"), _COLMAP_MATCHER_EXHAUSTIVE)
        self.spheresfm_matcher_combo.addItem(i18n.t("SPHERESFM_MATCHER_SPATIAL"), _SPHERESFM_MATCHER_SPATIAL)
        self.spheresfm_matcher_combo.setFixedWidth(150)

        self.spheresfm_feature_combo = QComboBox()
        self.spheresfm_feature_combo.setToolTip(i18n.tip("SPHERESFM_FEATURE_PRESET"))
        self.spheresfm_feature_combo.addItem(i18n.t("SPHERESFM_FEATURE_FAST"), _SPHERESFM_FEATURE_FAST)
        self.spheresfm_feature_combo.addItem(i18n.t("SPHERESFM_FEATURE_STANDARD"), _SPHERESFM_FEATURE_STANDARD)
        self.spheresfm_feature_combo.addItem(i18n.t("SPHERESFM_FEATURE_ROBUST"), _SPHERESFM_FEATURE_ROBUST)
        self.spheresfm_feature_combo.setFixedWidth(150)

        spheresfm_pipeline_row = QWidget()
        spheresfm_pipeline_layout = QHBoxLayout(spheresfm_pipeline_row)
        spheresfm_pipeline_layout.setContentsMargins(0, 0, 0, 0)
        spheresfm_pipeline_layout.setSpacing(8)
        spheresfm_pipeline_layout.addWidget(QLabel(i18n.t("COLMAP_MATCHER_COMPACT")))
        spheresfm_pipeline_layout.addWidget(self.spheresfm_matcher_combo)
        spheresfm_pipeline_layout.addWidget(QLabel(i18n.t("SPHERESFM_FEATURE_COMPACT")))
        spheresfm_pipeline_layout.addWidget(self.spheresfm_feature_combo)
        spheresfm_pipeline_layout.addStretch()
        spheresfm_form.addRow(spheresfm_pipeline_row)

        self.spheresfm_pose_browse = BrowseWidget(
            mode="file",
            filter_str="Text (*.txt *.csv);;All (*.*)",
            placeholder="POS.txt",
        )
        self.spheresfm_pose_browse.setToolTip(i18n.tip("SPHERESFM_POSE_FILE"))
        add_tooltip_row(
            spheresfm_form,
            i18n.t("SPHERESFM_POSE_FILE"),
            self.spheresfm_pose_browse,
            i18n.tip("SPHERESFM_POSE_FILE"),
        )

        self.spheresfm_open_gui_btn = QPushButton(i18n.t("SPHERESFM_OPEN_GUI"))
        self.spheresfm_open_gui_btn.setToolTip(i18n.tip("SPHERESFM_OPEN_GUI"))
        self.spheresfm_open_gui_btn.clicked.connect(self._open_spheresfm_result)
        spheresfm_form.addRow("", self.spheresfm_open_gui_btn)

        spheresfm_layout.addLayout(spheresfm_form)
        spheresfm_layout.addStretch()

        # Metashapeインポート設定
        preprocess = QWidget()
        self.metashape_section = preprocess
        preprocess_layout = QVBoxLayout(preprocess)
        preprocess_layout.setContentsMargins(8, 8, 8, 8)
        preprocess_layout.setSpacing(6)
        profile_form = QFormLayout()
        profile_form.setSpacing(6)

        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip(i18n.tip("TARGET_PROFILE"))
        self.profile_combo.addItem(i18n.PROFILE_POSTSHOT, _PROFILE_POSTSHOT)
        self.profile_combo.addItem(i18n.PROFILE_BRUSH, _PROFILE_BRUSH)
        self.profile_combo.addItem(i18n.PROFILE_LICHTFELD, _PROFILE_LICHTFELD)
        self.profile_combo.addItem(i18n.PROFILE_CUSTOM, _PROFILE_CUSTOM)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        add_tooltip_row(profile_form, i18n.TARGET_PROFILE, self.profile_combo, i18n.tip("TARGET_PROFILE"))

        self.output_shape_combo = QComboBox()
        self.output_shape_combo.setToolTip(i18n.tip("OUTPUT_SHAPE"))
        self.output_shape_combo.addItem(i18n.t("OUTPUT_SHAPE_PROJECTED"), _OUTPUT_SHAPE_PROJECTED)
        self.output_shape_combo.addItem(i18n.t("OUTPUT_SHAPE_EQUIRECT_3DGUT"), _OUTPUT_SHAPE_EQUIRECT_3DGUT)
        self.output_shape_combo.currentIndexChanged.connect(self._on_output_shape_changed)
        add_tooltip_row(profile_form, i18n.t("OUTPUT_SHAPE"), self.output_shape_combo, i18n.tip("OUTPUT_SHAPE"))

        self.profile_hint = QLabel("")
        self.profile_hint.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self.profile_hint.setVisible(False)
        profile_form.addRow("", self.profile_hint)

        self.axis_transform_combo = QComboBox()
        self.axis_transform_combo.setToolTip(i18n.tip("AXIS_TRANSFORM"))
        self.axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_POSTSHOT"), _AXIS_POSTSHOT)
        self.axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_BRUSH"), _AXIS_BRUSH)
        self.axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_NONE"), _AXIS_NONE)
        self.axis_transform_combo.setFixedWidth(180)
        self.axis_transform_combo.currentIndexChanged.connect(self._on_profile_option_changed)
        add_tooltip_row(profile_form, i18n.t("AXIS_TRANSFORM"), self.axis_transform_combo, i18n.tip("AXIS_TRANSFORM"))

        self.export_colmap_cb = QCheckBox(i18n.t("EXPORT_COLMAP"))
        self.export_colmap_cb.setToolTip(i18n.t("EXPORT_COLMAP_HINT"))
        profile_form.addRow("", self.export_colmap_cb)

        preprocess_layout.addLayout(profile_form)
        pp_form = QFormLayout()

        self.ms_images_path_label = ElidedPathLabel("-")
        self.ms_images_path_label.setToolTip(i18n.tip("MS_IMAGES"))
        add_tooltip_row(pp_form, i18n.t("MS_IMAGES_LABEL"), self.ms_images_path_label, i18n.tip("MS_IMAGES"))

        self.ms_xml_browse = BrowseWidget(mode="file", filter_str="XML (*.xml);;すべて (*.*)")
        self.ms_xml_browse.setToolTip(i18n.tip("MS_XML"))
        add_tooltip_row(pp_form, i18n.METASHAPE_XML, self.ms_xml_browse, i18n.tip("MS_XML"))

        self.ms_ply_browse = BrowseWidget(mode="file", filter_str="PLY (*.ply);;すべて (*.*)")
        self.ms_ply_browse.setToolTip(i18n.tip("MS_PLY"))
        add_tooltip_row(pp_form, i18n.METASHAPE_PLY, self.ms_ply_browse, i18n.tip("MS_PLY"))

        import_advanced = CollapsibleSection(i18n.t("ADVANCED_SETTINGS"), expanded=False)
        import_adv_form = QFormLayout()
        import_adv_form.setSpacing(6)

        self.metashape_import_options_row = QWidget()
        import_option_row = QHBoxLayout(self.metashape_import_options_row)
        import_option_row.setContentsMargins(0, 0, 0, 0)
        import_option_row.setSpacing(8)
        self.ms_scale_label = QLabel(i18n.t("SCALE_FACTOR_COMPACT"))
        self.ms_scale_label.setToolTip(i18n.tip("SCALE_FACTOR"))
        import_option_row.addWidget(self.ms_scale_label)

        self.ms_scale_edit = QLineEdit("1.0")
        self.ms_scale_edit.setFixedWidth(72)
        self.ms_scale_edit.setToolTip(i18n.tip("SCALE_FACTOR"))
        self.ms_scale_edit.textEdited.connect(self._on_profile_option_changed)
        import_option_row.addWidget(self.ms_scale_edit)

        self.ms_use_ply_cb = QCheckBox(i18n.t("MS_USE_PLY"))
        self.ms_use_ply_cb.setToolTip(i18n.tip("MS_USE_PLY"))
        self.ms_use_ply_cb.toggled.connect(self._on_profile_option_changed)
        import_option_row.addWidget(self.ms_use_ply_cb)

        self.ms_no_fix_rot_cb = QCheckBox(i18n.NO_FIX_ROTATION)
        self.ms_no_fix_rot_cb.setToolTip(i18n.tip("NO_FIX_ROTATION"))
        self.ms_no_fix_rot_cb.toggled.connect(self._on_profile_option_changed)
        import_option_row.addWidget(self.ms_no_fix_rot_cb)
        import_option_row.addStretch()
        import_adv_form.addRow(self.metashape_import_options_row)
        import_advanced.content_layout.addLayout(import_adv_form)

        preprocess_layout.addLayout(pp_form)
        preprocess_layout.addWidget(import_advanced)
        preprocess_layout.addStretch()

        self.view_config = ViewConfigWidget(show_settings=False, show_summary=False)
        self.view_config.views_changed.connect(self._on_views_changed)
        self.view_config.hovered_view_changed.connect(lambda _name: self._render_preview())

        # 視点書き出し設定
        adv_output = QWidget()
        self.advanced_output_section = adv_output
        adv_output_layout = QVBoxLayout(adv_output)
        adv_output_layout.setContentsMargins(8, 8, 8, 8)
        adv_output_layout.setSpacing(6)
        adv_form = QFormLayout()
        adv_form.setSpacing(6)

        self.scale_combo = QComboBox()
        self.scale_combo.setToolTip(i18n.tip("OUTPUT_SCALE"))
        self.scale_combo.addItem("Full", 1.0)
        self.scale_combo.addItem("Normal", _NORMAL_OUTPUT_SCALE)
        self.scale_combo.addItem("Half", 0.5)
        full_scale_index = self.scale_combo.findData(1.0)
        if full_scale_index >= 0:
            self.scale_combo.setCurrentIndex(full_scale_index)
        self.scale_combo.setFixedWidth(90)
        self.output_scale_label = QLabel(i18n.OUTPUT_SCALE + ":")
        self.output_scale_label.setToolTip(i18n.tip("OUTPUT_SCALE"))
        self.view_config.angle_row.addWidget(self.output_scale_label)
        self.view_config.angle_row.addWidget(self.scale_combo)
        self.view_config.angle_row.addStretch()

        self.yaw_per_frame_edit = DragDoubleSpinBox(
            minimum=-180.0,
            maximum=180.0,
            step=1.0,
            decimals=1,
            value=30.0,
            drag_pixels_per_step=6.0,
        )
        self.yaw_per_frame_edit.setFixedWidth(76)
        self.yaw_per_frame_edit.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))
        self.yaw_per_frame_row = QWidget()
        yaw_per_frame_layout = QHBoxLayout(self.yaw_per_frame_row)
        yaw_per_frame_layout.setContentsMargins(0, 0, 0, 0)
        yaw_per_frame_layout.setSpacing(8)
        self.yaw_per_frame_label = QLabel(i18n.t("YAW_OFFSET_PER_FRAME"))
        self.yaw_per_frame_label.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))
        yaw_per_frame_layout.addWidget(self.yaw_per_frame_label)
        yaw_per_frame_layout.addWidget(self.yaw_per_frame_edit)
        yaw_per_frame_layout.addStretch()
        self.view_config.extra_controls_layout.addWidget(self.yaw_per_frame_row)

        adv_form.addRow(self.view_config.settings_widget)

        output_details = CollapsibleSection(i18n.t("OUTPUT_DETAIL"), expanded=False)
        self.output_details_section = output_details

        self.output_format_combo = QComboBox()
        self.output_format_combo.setToolTip(i18n.tip("OUTPUT_FORMAT"))
        self.output_format_combo.addItem(i18n.t("OUTPUT_FORMAT_AUTO"), "auto")
        for fmt in ("jpg", "png", "tiff", "webp"):
            self.output_format_combo.addItem(fmt, fmt)
        self.output_format_combo.setFixedWidth(96)

        self.output_bit_depth_combo = QComboBox()
        self.output_bit_depth_combo.setToolTip(i18n.tip("OUTPUT_BIT_DEPTH"))
        self.output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_8"), "8")
        self.output_bit_depth_combo.addItem(i18n.t("OUTPUT_BIT_DEPTH_SOURCE"), "source")
        self.output_bit_depth_combo.setFixedWidth(86)

        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(8)
        self.output_format_label = QLabel(i18n.t("OUTPUT_FORMAT_COMPACT"))
        self.output_format_label.setToolTip(i18n.tip("OUTPUT_FORMAT"))
        format_layout.addWidget(self.output_format_label)
        format_layout.addWidget(self.output_format_combo)
        self.output_bit_depth_label = QLabel(i18n.t("OUTPUT_BIT_DEPTH_COMPACT"))
        self.output_bit_depth_label.setToolTip(i18n.tip("OUTPUT_BIT_DEPTH"))
        format_layout.addWidget(self.output_bit_depth_label)
        format_layout.addWidget(self.output_bit_depth_combo)
        format_layout.addStretch()
        output_details.content_layout.addWidget(format_row)

        self.invert_masks_cb = QCheckBox(i18n.INVERT_MASKS)
        self.invert_masks_cb.setToolTip(i18n.tip("INVERT_MASKS"))

        self.jpg_quality_edit = QLineEdit("95")
        self.jpg_quality_edit.setToolTip(i18n.tip("JPG_QUALITY"))
        self.jpg_quality_edit.setFixedWidth(64)

        quality_row = QWidget()
        quality_layout = QHBoxLayout(quality_row)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(8)
        quality_layout.addWidget(self.invert_masks_cb)
        quality_layout.addSpacing(8)
        self.jpg_quality_label = QLabel(i18n.t("JPG_QUALITY_COMPACT"))
        self.jpg_quality_label.setToolTip(i18n.tip("JPG_QUALITY"))
        quality_layout.addWidget(self.jpg_quality_label)
        quality_layout.addWidget(self.jpg_quality_edit)
        quality_layout.addStretch()
        output_details.content_layout.addWidget(quality_row)

        adv_output_layout.addLayout(adv_form)
        adv_output_layout.addWidget(output_details)
        adv_output_layout.addStretch()

        self.view_export_tab_index = self.settings_tabs.addTab(
            self.advanced_output_section,
            i18n.t("STEP4_TAB_VIEW_EXPORT"),
        )
        self.metashape_tab_index = self.settings_tabs.addTab(
            self.metashape_section,
            i18n.t("STEP4_TAB_METASHAPE"),
        )
        self.colmap_tab_index = self.settings_tabs.addTab(
            self.colmap_section,
            i18n.t("STEP4_TAB_COLMAP"),
        )
        self.spheresfm_tab_index = self.settings_tabs.addTab(
            self.spheresfm_section,
            i18n.t("STEP4_TAB_SPHERESFM"),
        )
        left_layout.addWidget(self.settings_tabs, stretch=1)

        left_layout.addStretch()

        self.export_summary_bar = QWidget()
        self.export_summary_bar.setObjectName("stickySummaryBar")
        summary_layout = QHBoxLayout(self.export_summary_bar)
        summary_layout.setContentsMargins(0, 6, SETTINGS_PANE_MARGINS[2], 2)
        summary_layout.setSpacing(0)
        self.export_summary_label = QLabel(self.view_config.summary_text())
        self.export_summary_label.setObjectName("stickySummaryLabel")
        self.export_summary_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.export_summary_label.setWordWrap(False)
        self.view_config.summary_changed.connect(self.export_summary_label.setText)
        summary_layout.addStretch()
        summary_layout.addWidget(self.export_summary_label)

        # 右パネル: プレビュー
        preview_pane = QWidget()
        preview_pane.setObjectName("workPane")
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)
        preview_title = QLabel(i18n.t("CUBEMAP_PREVIEW_SECTION"))
        preview_title.setObjectName("paneTitle")
        self.preview = PreviewWidget()
        self.preview.mask_overlay_changed.connect(lambda: self._schedule_render_preview())
        self.preview.current_image_changed.connect(lambda: self._schedule_render_preview())
        preview_header = QHBoxLayout()
        preview_header.setContentsMargins(0, 0, 0, 0)
        preview_header.setSpacing(8)
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.preview.projection_toggle_btn)
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.preview, stretch=1)

        top_scroll.setWidget(top)
        left_pane_layout.addWidget(top_scroll, stretch=1)
        left_pane_layout.addWidget(self.export_summary_bar)
        splitter.addWidget(left_pane)
        splitter.addWidget(preview_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SETTINGS_PANE_WIDTH, 760])
        layout.addWidget(splitter)

        lichtfeld_index = self.profile_combo.findData(_PROFILE_LICHTFELD)
        if lichtfeld_index >= 0:
            self.profile_combo.setCurrentIndex(lichtfeld_index)
        self._on_profile_changed(self.profile_combo.currentIndex())
        self._on_output_shape_changed(self.output_shape_combo.currentIndex())
        self._on_colmap_mapper_changed()
        self._on_colmap_run_toggled(self.run_colmap_cb.isChecked())
        self._set_export_method(_METHOD_METASHAPE)

    # -- シーンディレクトリ --

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if not path:
            self.output_path_label.setToolTip(i18n.tip("OUTPUT_DIR_CUBEMAP"))
            self.output_path_label.set_full_text("-")
            self.ms_images_path_label.setToolTip(i18n.tip("MS_IMAGES"))
            self.ms_images_path_label.set_full_text("-")
            self.ms_xml_browse.set_text("")
            self.ms_ply_browse.set_text("")
            self.preview.set_scene_dir("")
            self._refresh_input_image_count()
            self._update_output_count()
            self._render_preview()
            return
        p = Path(path)
        images_dir = str(self._metashape_images_dir())
        self._update_path_labels()
        self.ms_images_path_label.setToolTip(f"{i18n.tip('MS_IMAGES')}\n{images_dir}")
        self.ms_images_path_label.set_full_text(images_dir)
        self.ms_xml_browse.set_text(str(self._guess_xml(p)))
        self.ms_ply_browse.set_text(self._guess_ply(p))
        self.preview.set_scene_dir(path)
        self._refresh_input_image_count()
        self._update_output_count()
        self._render_preview()

    def primary_action_text(self) -> str:
        return i18n.t("EXPORT")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("RUN_CUBEMAP")

    def on_activated(self) -> None:
        self.preview.refresh_image_list(prefer_current=True)
        self._refresh_input_image_count()
        self._update_path_labels()
        self._update_output_count()
        self._render_preview()

    # -- ユーザー設定 --

    def enable_user_preferences(self) -> None:
        if self._user_preferences_enabled:
            return
        self._user_preferences_enabled = True
        self._load_user_preferences()
        self.colmap_exec_browse.path_changed.connect(lambda _path: self._save_user_preferences())
        self.glomap_exec_browse.path_changed.connect(lambda _path: self._save_user_preferences())
        self.colmap_matcher_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
        self.colmap_mapper_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
        self.spheresfm_exec_browse.path_changed.connect(lambda _path: self._save_user_preferences())
        self.spheresfm_matcher_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
        self.spheresfm_feature_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())

    def _load_user_preferences(self) -> None:
        settings = load_user_settings_section(_USER_SETTINGS_SECTION)
        self._syncing_user_preferences = True
        try:
            colmap_exec = str(settings.get("colmap_executable", "")).strip()
            glomap_exec = str(settings.get("glomap_executable", "")).strip()
            if colmap_exec:
                self.colmap_exec_browse.set_text(colmap_exec)
            if glomap_exec:
                self.glomap_exec_browse.set_text(glomap_exec)
            spheresfm_exec = str(settings.get("spheresfm_executable", "")).strip()
            if spheresfm_exec:
                self.spheresfm_exec_browse.set_text(spheresfm_exec)

            matcher = str(settings.get("matcher", "")).strip()
            mapper = str(settings.get("mapper", "")).strip()
            if matcher:
                self._set_combo_data(self.colmap_matcher_combo, matcher)
            if mapper:
                self._set_combo_data(self.colmap_mapper_combo, mapper)
            spheresfm_matcher = str(settings.get("spheresfm_matcher", "")).strip()
            spheresfm_feature = str(settings.get("spheresfm_feature_preset", "")).strip()
            if spheresfm_matcher:
                self._set_combo_data(self.spheresfm_matcher_combo, spheresfm_matcher)
            if spheresfm_feature:
                self._set_combo_data(self.spheresfm_feature_combo, spheresfm_feature)
        finally:
            self._syncing_user_preferences = False
        self._on_colmap_mapper_changed()

    def _save_user_preferences(self) -> None:
        if self._syncing_user_preferences:
            return
        update_user_settings_section(
            _USER_SETTINGS_SECTION,
            {
                "colmap_executable": self.colmap_exec_browse.text(),
                "glomap_executable": self.glomap_exec_browse.text(),
                "matcher": self.colmap_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
                "mapper": self.colmap_mapper_combo.currentData() or _COLMAP_MAPPER_GLOBAL,
                "spheresfm_executable": self.spheresfm_exec_browse.text(),
                "spheresfm_matcher": self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
                "spheresfm_feature_preset": self.spheresfm_feature_combo.currentData() or _SPHERESFM_FEATURE_STANDARD,
            },
        )

    # -- 書き出し方式 --

    def _export_method(self) -> str:
        return self._export_method_value

    def _is_metashape_method(self) -> bool:
        return self._export_method() == _METHOD_METASHAPE

    def _is_colmap_method(self) -> bool:
        return self._export_method() == _METHOD_COLMAP

    def _is_spheresfm_method(self) -> bool:
        return self._export_method() == _METHOD_SPHERESFM

    def _set_export_method(self, method: str) -> None:
        if method not in {_METHOD_METASHAPE, _METHOD_COLMAP, _METHOD_SPHERESFM}:
            method = _METHOD_METASHAPE
        self._export_method_value = method
        btn = self.export_method_buttons.get(method)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self._on_export_method_changed()

    def _on_export_method_changed(self) -> None:
        metashape = self._is_metashape_method()
        self._sync_settings_tabs()
        if not metashape:
            self.export_colmap_cb.setChecked(False)
        self._sync_output_shape_controls()
        self._update_path_labels()
        self._update_output_count()
        self.primary_action_state_changed.emit()

    def _sync_settings_tabs(self) -> None:
        current = self.settings_tabs.currentIndex()
        was_route_specific = current in {self.metashape_tab_index, self.colmap_tab_index, self.spheresfm_tab_index}
        self.settings_tabs.setTabVisible(self.metashape_tab_index, self._is_metashape_method())
        self.settings_tabs.setTabVisible(self.colmap_tab_index, self._is_colmap_method())
        self.settings_tabs.setTabVisible(self.spheresfm_tab_index, self._is_spheresfm_method())
        view_enabled = not self._uses_direct_equirect_output() and not self._is_spheresfm_method()
        self.settings_tabs.setTabEnabled(self.view_export_tab_index, view_enabled)
        if self._is_spheresfm_method():
            route_index = self.spheresfm_tab_index
        elif self._is_colmap_method():
            route_index = self.colmap_tab_index
        else:
            route_index = self.metashape_tab_index
        if (not view_enabled) and current == self.view_export_tab_index:
            self.settings_tabs.setCurrentIndex(route_index)
        elif was_route_specific or not self.settings_tabs.isTabVisible(self.settings_tabs.currentIndex()):
            self.settings_tabs.setCurrentIndex(route_index)

    def _update_path_labels(self) -> None:
        if not self.scene_dir:
            return
        output = str(self._display_output_dir())
        if self._uses_direct_equirect_output():
            tip_key = "OUTPUT_DIR_LICHTFELD_DIRECT"
        elif self._is_spheresfm_method():
            tip_key = "OUTPUT_DIR_SPHERESFM_PROJECT"
        else:
            tip_key = "OUTPUT_DIR_CUBEMAP" if self._is_metashape_method() else "OUTPUT_DIR_COLMAP_PROJECT"
        self.output_path_label.setToolTip(f"{i18n.tip(tip_key)}\n{output}")
        self.output_path_label.set_full_text(output)

    def _on_colmap_run_toggled(self, checked: bool) -> None:
        self.colmap_exec_browse.setEnabled(checked)
        self.colmap_pipeline_row.setEnabled(checked)
        self._on_colmap_mapper_changed()

    def _on_colmap_mapper_changed(self, *_args) -> None:
        needs_glomap = (
            self.run_colmap_cb.isChecked()
            and self.colmap_mapper_combo.currentData() == _COLMAP_MAPPER_GLOMAP
        )
        self.glomap_exec_row_label.setVisible(needs_glomap)
        self.glomap_exec_browse.setVisible(needs_glomap)
        self.glomap_exec_browse.setEnabled(needs_glomap)

    # -- プロファイル --

    def _profile_id(self) -> str:
        return self.profile_combo.currentData() or _PROFILE_CUSTOM

    def _effective_profile(self) -> str:
        mode = self._axis_transform_mode()
        if mode == _AXIS_NONE:
            return _PROFILE_LICHTFELD
        if mode == _AXIS_BRUSH:
            return _PROFILE_BRUSH
        return _PROFILE_POSTSHOT

    @staticmethod
    def _profile_axis_default(profile: str) -> str:
        if profile == _PROFILE_LICHTFELD:
            return _AXIS_NONE
        if profile == _PROFILE_BRUSH:
            return _AXIS_BRUSH
        return _AXIS_POSTSHOT

    @staticmethod
    def _profile_use_ply_default(profile: str) -> bool:
        return profile == _PROFILE_LICHTFELD

    @staticmethod
    def _profile_scale_default(_profile: str) -> float:
        return 1.0

    @staticmethod
    def _profile_no_fix_rotation_default(_profile: str) -> bool:
        return False

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _sync_profile_defaults(self, profile: str) -> None:
        if profile == _PROFILE_CUSTOM:
            return
        self._syncing_profile_controls = True
        try:
            self._set_combo_data(self.axis_transform_combo, self._profile_axis_default(profile))
            self.ms_use_ply_cb.setChecked(self._profile_use_ply_default(profile))
            self.ms_scale_edit.setText("1.0")
            self.ms_no_fix_rot_cb.setChecked(self._profile_no_fix_rotation_default(profile))
        finally:
            self._syncing_profile_controls = False

    def _axis_transform_mode(self) -> str:
        data = self.axis_transform_combo.currentData()
        return data if data in {_AXIS_POSTSHOT, _AXIS_BRUSH, _AXIS_NONE} else _AXIS_POSTSHOT

    def _uses_lichtfeld_final_correction(self) -> bool:
        return self._is_metashape_method() and self._effective_profile() == _PROFILE_LICHTFELD

    def _on_profile_changed(self, _index: int) -> None:
        p = self._profile_id()
        self._sync_profile_defaults(p)
        self.profile_hint.setText(i18n.t("PROFILE_CUSTOM_HINT") if p == _PROFILE_CUSTOM else "")
        self.profile_hint.setVisible(p == _PROFILE_CUSTOM)
        if self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT and (
            self._axis_transform_mode() != _AXIS_NONE or not self._preprocess_uses_ply()
        ):
            self._set_combo_data(self.output_shape_combo, _OUTPUT_SHAPE_PROJECTED)
        self._sync_ply_browse_enabled()
        self._sync_output_shape_controls()
        self._update_path_labels()
        self._update_output_count()

    def _on_profile_option_changed(self, *_args) -> None:
        if self._syncing_profile_controls:
            return
        current = self._profile_id()
        if current != _PROFILE_CUSTOM:
            axis_changed = self._axis_transform_mode() != self._profile_axis_default(current)
            ply_changed = self.ms_use_ply_cb.isChecked() != self._profile_use_ply_default(current)
            no_fix_changed = (
                self.ms_no_fix_rot_cb.isChecked()
                != self._profile_no_fix_rotation_default(current)
            )
            try:
                scale_changed = not math.isclose(
                    float(self.ms_scale_edit.text().strip()),
                    self._profile_scale_default(current),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            except ValueError:
                scale_changed = self.ms_scale_edit.text().strip() != "1.0"
            if axis_changed or ply_changed or no_fix_changed or scale_changed:
                custom_idx = self.profile_combo.findData(_PROFILE_CUSTOM)
                if custom_idx >= 0:
                    self.profile_combo.setCurrentIndex(custom_idx)
        self._sync_ply_browse_enabled()
        self._sync_output_shape_controls()

    # -- 出力形状 --

    def _output_shape(self) -> str:
        data = self.output_shape_combo.currentData()
        return data if data in {_OUTPUT_SHAPE_PROJECTED, _OUTPUT_SHAPE_EQUIRECT_3DGUT} else _OUTPUT_SHAPE_PROJECTED

    def _uses_direct_equirect_output(self) -> bool:
        return self._is_metashape_method() and self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT

    def _on_output_shape_changed(self, *_args) -> None:
        if self._syncing_output_shape_controls:
            return
        if self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            self._ensure_direct_equirect_defaults()
        self._sync_output_shape_controls()
        self._sync_settings_tabs()
        self._update_path_labels()
        self._update_output_count()
        self._schedule_render_preview()
        self.primary_action_state_changed.emit()

    def _ensure_direct_equirect_defaults(self) -> None:
        self._syncing_profile_controls = True
        try:
            self._set_combo_data(self.axis_transform_combo, _AXIS_NONE)
            self.ms_use_ply_cb.setChecked(True)
        finally:
            self._syncing_profile_controls = False

        if self._profile_id() not in {_PROFILE_LICHTFELD, _PROFILE_CUSTOM}:
            self._set_combo_data(self.profile_combo, _PROFILE_LICHTFELD)

    def _sync_output_shape_controls(self) -> None:
        direct = self._uses_direct_equirect_output()
        spheresfm = self._is_spheresfm_method()
        if direct:
            if self._saved_projected_export_targets is None:
                self._saved_projected_export_targets = (
                    self.export_images_cb.isChecked(),
                    self.export_masks_cb.isChecked(),
                )
            self.export_images_cb.setChecked(True)
            self.export_masks_cb.setChecked(True)
            self.export_colmap_cb.setChecked(False)
        elif spheresfm:
            if self._saved_projected_export_targets is None:
                self._saved_projected_export_targets = (
                    self.export_images_cb.isChecked(),
                    self.export_masks_cb.isChecked(),
                )
            self.export_images_cb.setChecked(False)
            self.export_masks_cb.setChecked(False)
            self.export_colmap_cb.setChecked(False)
        elif self._saved_projected_export_targets is not None:
            images, masks = self._saved_projected_export_targets
            self.export_images_cb.setChecked(images)
            self.export_masks_cb.setChecked(masks)
            self._saved_projected_export_targets = None

        route_uses_view_export = not direct and not spheresfm
        self.export_targets_row.setEnabled(route_uses_view_export)
        self.view_config.settings_widget.setEnabled(route_uses_view_export)
        self.output_details_section.setEnabled(route_uses_view_export)
        self.output_shape_combo.setEnabled(self._is_metashape_method())
        self.axis_transform_combo.setEnabled(not direct and not spheresfm)
        self.ms_use_ply_cb.setEnabled(not direct and not spheresfm)
        self.export_colmap_cb.setEnabled(self._is_metashape_method() and not direct)
        self.settings_tabs.setTabEnabled(self.view_export_tab_index, route_uses_view_export)

    def _preprocess_uses_ply(self) -> bool:
        return self.ms_use_ply_cb.isChecked()

    def _sync_ply_browse_enabled(self) -> None:
        self.ms_ply_browse.setEnabled(self._preprocess_uses_ply())

    def _writes_images(self) -> bool:
        return self.export_images_cb.isChecked()

    def _writes_masks(self) -> bool:
        return self.export_masks_cb.isChecked()

    def _writes_any_view_assets(self) -> bool:
        return self._writes_images() or self._writes_masks()

    # -- ビュー --

    def _on_views_changed(self) -> None:
        self._update_output_count()
        self._schedule_render_preview()

    def _schedule_render_preview(self) -> None:
        if self._preview_render_timer.isActive():
            self._preview_render_pending = True
            self._preview_render_timer.start()
            return
        self._preview_render_pending = False
        self._render_preview()
        self._preview_render_timer.start()

    def _flush_scheduled_render_preview(self) -> None:
        if not self._preview_render_pending:
            return
        self._preview_render_pending = False
        self._render_preview()

    def _render_preview(self) -> None:
        try:
            views = (
                []
                if self._uses_direct_equirect_output() or self._is_spheresfm_method()
                else self.view_config.collect_views(include_disabled=True)
            )
        except Exception:
            views = []
        mask_dir = str(self._mask_dir()) if self.scene_dir else ""
        self.preview.render(views, mask_dir)

    def _count_input_images(self) -> int:
        if not self.scene_dir:
            return 0
        scene = Path(self.scene_dir)
        images = scene / "images"
        roots = [images] if images.is_dir() else [scene]
        exts = {".jpg", ".jpeg", ".png"}
        seen: set[str] = set()
        count = 0
        for root in roots:
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    key = str(p.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        count += 1
        return count

    def _refresh_input_image_count(self) -> None:
        self._input_image_count = len(getattr(self.preview, "preview_images", []) or [])

    def _update_output_count(self) -> None:
        label = i18n.t("OUTPUT_IMAGE_COUNT_LABEL")
        if self._uses_direct_equirect_output() or self._is_spheresfm_method():
            count_text = i18n.t("OUTPUT_IMAGE_COUNT_DIRECT_FORMAT").format(count=self._input_image_count)
            self.view_config.set_output_count_text(f"{label}: {count_text}")
            return
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            self.view_config.set_output_count_text(f"{label}: -")
            return
        enabled = sum(1 for v in views if v["enabled"])
        sources = self._input_image_count
        total = sources * enabled
        warn = ""
        if enabled > _BLOCK_ENABLED_VIEWS:
            warn = " [超過]"
        elif enabled > _WARN_ENABLED_VIEWS:
            warn = " [多い]"
        count_text = i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=total)
        self.view_config.set_output_count_text(f"{label}: {count_text}{warn}")

    # -- コマンド構築 --

    def build_commands(self) -> list[tuple[str, list[str]]]:
        if self._is_spheresfm_method():
            self._validate_spheresfm_export()
            if not self._prepare_spheresfm_dir():
                return []
            return self._build_spheresfm_commands()

        if self._is_colmap_method():
            self._validate_image_only_export()
            if not self._prepare_colmap_rig_dir():
                return []
            steps = [("colmap_rig_export", self._build_cubemap_cmd(image_only=True, colmap_rig=True))]
            if self.run_colmap_cb.isChecked():
                steps.extend(self._build_colmap_sfm_commands())
            return steps

        self._validate_bundle()

        preprocess_cmd = self._build_preprocess_cmd()

        if self._uses_direct_equirect_output():
            return [("metashape", preprocess_cmd)]

        if not self._prepare_output_dir():
            return []

        steps = [("metashape", preprocess_cmd)]
        steps.append(("cubemap", self._build_cubemap_cmd()))
        if self.export_colmap_cb.isChecked():
            steps.append(("colmap", self._build_colmap_cmd()))
        return steps

    def _build_preprocess_cmd(self) -> list[str]:
        script = self.base_dir / "vendor" / "metashape_360_lfs" / "metashape_360_lfs.py"
        if not script.exists():
            raise FileNotFoundError(f"metashape_360_lfs.py が見つかりません: {script}")
        scene = Path(self.scene_dir)
        if not scene.is_dir():
            raise ValueError(f"シーンフォルダが見つかりません: {scene}")

        images = str(self._metashape_images_dir())
        xml = self.ms_xml_browse.text()
        if not images or not Path(images).is_dir():
            raise ValueError(f"Metashape画像フォルダが見つかりません: {images}")
        if not xml or not Path(xml).is_file():
            raise ValueError(f"Metashape XMLが見つかりません: {xml}")

        scale = float(self.ms_scale_edit.text().strip())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("スケール係数は正の有限値である必要があります")

        ply = ""
        if self._preprocess_uses_ply():
            ply = self.ms_ply_browse.text()
            if not ply or not Path(ply).is_file():
                raise ValueError(f"PLYファイルが見つかりません: {ply}")
        return build_metashape_preprocess_cmd(
            MetashapePreprocessCommand(
                python_executable=sys.executable,
                script=script,
                images=Path(images),
                xml=xml,
                output=scene,
                scale=scale,
                use_ply=self._preprocess_uses_ply(),
                ply=ply,
                no_fix_rotation=self.ms_no_fix_rot_cb.isChecked(),
            )
        )

    def _build_cubemap_cmd(self, image_only: bool = False, colmap_rig: bool = False) -> list[str]:
        script = self.base_dir / "cubemap_transforms_json.py"
        if not script.exists():
            raise FileNotFoundError(f"cubemap_transforms_json.py が見つかりません: {script}")

        scene = Path(self.scene_dir)
        if not scene.is_dir():
            raise ValueError(f"シーンフォルダが見つかりません: {scene}")

        output = self._output_dir()

        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for v in views if v["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")

        views_json = self._write_views_config(output, views)

        if colmap_rig:
            yaw_step = 0.0
        else:
            yaw_step = float(self.yaw_per_frame_edit.value())

        out_fmt = self.output_format_combo.currentData() or "auto"
        out_depth = self.output_bit_depth_combo.currentData() or "8"

        try:
            jpgq = int(self.jpg_quality_edit.text().strip())
        except ValueError as exc:
            raise ValueError("JPG/WebP 品質は整数で指定してください") from exc
        if not 1 <= jpgq <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")
        return build_cubemap_conversion_cmd(
            CubemapConversionCommand(
                python_executable=sys.executable,
                script=script,
                scene=scene,
                output=output,
                views_json=views_json,
                scale=float(self.scale_combo.currentData()),
                axis_mode=self._axis_transform_mode(),
                image_only=image_only,
                colmap_rig=colmap_rig,
                invert_masks=self.invert_masks_cb.isChecked(),
                writes_images=self._writes_images(),
                writes_masks=self._writes_masks(),
                yaw_offset_per_frame=yaw_step,
                output_format=out_fmt,
                output_bit_depth=out_depth,
                jpg_quality=jpgq,
            )
        )

    def _build_colmap_cmd(self) -> list[str]:
        script = self.base_dir / "transforms_to_colmap.py"
        if not script.exists():
            raise FileNotFoundError(f"transforms_to_colmap.py が見つかりません: {script}")

        output = self._output_dir()
        colmap_dir = output / "colmap"

        ply_path: Path | None = None
        ply = output / "pointcloud.ply"
        if ply.is_file():
            ply_path = ply
        else:
            # cubemap 出力ディレクトリ内の任意 .ply をフォールバック
            plys = sorted([p for p in output.glob("*.ply") if p.is_file()])
            if plys:
                ply_path = plys[0]
        return build_colmap_export_cmd(
            ColmapExportCommand(
                python_executable=sys.executable,
                script=script,
                output=output,
                colmap_dir=colmap_dir,
                ply=ply_path,
            )
        )

    def _default_colmap_executable(self) -> str:
        return "colmap.exe" if os.name == "nt" else "colmap"

    def _default_glomap_executable(self) -> str:
        return "glomap.exe" if os.name == "nt" else "glomap"

    def _default_spheresfm_executable(self) -> str:
        return "colmap.exe" if os.name == "nt" else "colmap"

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return any(sep in value for sep in ("/", "\\")) or Path(value).is_absolute()

    def _resolve_executable(self, raw: str, default_name: str, message_key: str) -> str:
        value = raw.strip() or default_name
        if self._looks_like_path(value):
            path = Path(value)
            if not path.is_file():
                raise ValueError(i18n.t(message_key).format(path=value))
            return str(path)
        found = shutil.which(value)
        if not found:
            raise ValueError(i18n.t(message_key).format(path=value))
        return found

    def _resolve_colmap_executable(self) -> str:
        return self._resolve_executable(
            self.colmap_exec_browse.text(),
            self._default_colmap_executable(),
            "COLMAP_EXEC_NOT_FOUND",
        )

    def _resolve_spheresfm_executable(self) -> str:
        return self._resolve_executable(
            self.spheresfm_exec_browse.text(),
            self._default_spheresfm_executable(),
            "SPHERESFM_EXEC_NOT_FOUND",
        )

    def _resolve_glomap_executable(self) -> str:
        return self._resolve_executable(
            self.glomap_exec_browse.text(),
            self._default_glomap_executable(),
            "GLOMAP_EXEC_NOT_FOUND",
        )

    def _build_colmap_sfm_commands(self) -> list[tuple[str, list[str]]]:
        colmap = self._resolve_colmap_executable()
        rig_dir = self._colmap_rig_dir()
        images_dir = self._colmap_rig_images_dir()
        masks_dir = self._colmap_rig_masks_dir()
        database = self._colmap_database_path()
        sparse = self._colmap_sparse_dir()

        matcher = self.colmap_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL
        mapper = self.colmap_mapper_combo.currentData() or _COLMAP_MAPPER_INCREMENTAL
        glomap = self._resolve_glomap_executable() if mapper == _COLMAP_MAPPER_GLOMAP else self._default_glomap_executable()
        return build_colmap_sfm_commands(
            ColmapSfmCommand(
                colmap=colmap,
                glomap=glomap,
                rig_dir=rig_dir,
                images_dir=images_dir,
                masks_dir=masks_dir,
                database=database,
                sparse=sparse,
                camera_params=self._colmap_camera_params_arg(),
                writes_images=self._writes_images(),
                writes_masks=self._writes_masks(),
                matcher=matcher,
                mapper=mapper,
            )
        )

    def _build_spheresfm_commands(self) -> list[tuple[str, list[str]]]:
        script = self.base_dir / "scripts" / "prepare_spheresfm_project.py"
        if not script.exists():
            raise FileNotFoundError(f"prepare_spheresfm_project.py が見つかりません: {script}")

        matcher = self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL
        pose_path = self.spheresfm_pose_browse.text().strip()
        if matcher == _SPHERESFM_MATCHER_SPATIAL and not pose_path:
            raise ValueError(i18n.t("SPHERESFM_POSE_REQUIRED"))
        if pose_path and not Path(pose_path).is_file():
            raise ValueError(i18n.t("SPHERESFM_POSE_NOT_FOUND").format(path=pose_path))

        return build_spheresfm_commands(
            SphereSfmCommand(
                python_executable=sys.executable,
                prepare_script=script,
                colmap=self._resolve_spheresfm_executable(),
                images_dir=self._metashape_images_dir(),
                source_masks_dir=self._mask_dir(),
                prepared_masks_dir=self._spheresfm_masks_dir(),
                database=self._spheresfm_database_path(),
                sparse=self._spheresfm_sparse_dir(),
                camera_params=self._spheresfm_camera_params_arg(),
                use_masks=self._spheresfm_uses_masks(),
                matcher=matcher,
                feature_preset=self.spheresfm_feature_combo.currentData() or _SPHERESFM_FEATURE_STANDARD,
                pose_path=pose_path,
            )
        )

    def _open_spheresfm_result(self) -> None:
        if not self.scene_dir:
            return
        model = self._find_spheresfm_sparse_model()
        if model is None:
            QMessageBox.warning(
                self,
                i18n.t("SPHERESFM_OPEN_GUI"),
                i18n.t("SPHERESFM_RESULT_NOT_FOUND").format(path=str(self._spheresfm_sparse_dir())),
            )
            return
        try:
            colmap = self._resolve_spheresfm_executable()
        except ValueError as exc:
            QMessageBox.warning(self, i18n.t("SPHERESFM_OPEN_GUI"), str(exc))
            return
        args = [
            "gui",
            "--database_path",
            str(self._spheresfm_database_path()),
            "--image_path",
            str(self._metashape_images_dir()),
            "--import_path",
            str(model),
        ]
        if not QProcess.startDetached(colmap, args):
            QMessageBox.warning(self, i18n.t("SPHERESFM_OPEN_GUI"), i18n.t("SPHERESFM_OPEN_GUI_FAILED"))

    def _write_views_config(self, output_dir: Path, views: list[dict]) -> Path:
        return write_views_config(output_dir, views)

    @staticmethod
    def _views_config_payload(views: list[dict]) -> dict:
        return views_config_payload(views)

    def _export_settings_path(self) -> Path:
        if self._uses_direct_equirect_output():
            root = self._direct_output_dir()
        elif self._is_spheresfm_method():
            root = self._spheresfm_project_dir()
        else:
            root = self._output_dir()
        return root / _EXPORT_SETTINGS_NAME

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _collect_export_settings(self) -> dict:
        views = self.view_config.collect_views(include_disabled=True)
        scale = float(self.scale_combo.currentData())
        direct = self._uses_direct_equirect_output()
        spheresfm = self._is_spheresfm_method()
        yaw_step = (
            0.0
            if self._export_method() == _METHOD_COLMAP or direct or spheresfm
            else float(self.yaw_per_frame_edit.value())
        )
        jpg_quality = int(self.jpg_quality_edit.text().strip())
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        scene = Path(self.scene_dir)
        output = self._display_output_dir()
        profile = self._profile_id()
        views_config_snapshot = None if direct or spheresfm else self._views_config_payload(views)

        return {
            "app": "stechdrive-3dgs-utils",
            "app_version": APP_VERSION,
            "settings_version": 1,
            "created_at": self._utc_now_iso(),
            "scene_dir": str(scene),
            "output_dir": str(output),
            "export_method": self._export_method(),
            "output_shape": self._output_shape(),
            "target_profile": profile,
            "effective_profile": self._effective_profile(),
            "axis_transform": self._axis_transform_mode(),
            "fov": 90.0,
            "image_size": {
                "label": self.scale_combo.currentText(),
                "scale": scale,
            },
            "view_config": {
                "mode": self.view_config.view_mode(),
                "yaw_offset": self.view_config.yaw_offset(),
                "yaw_slots": self.view_config.yaw_slot_count(),
                "pitch_rows": self.view_config.pitch_values(),
                "pitch_rows_text": self.view_config.pitch_rows_text(),
                "cube6_drop_top": False,
                "cube6_drop_bottom": False,
                "views": [
                    {
                        "name": v["name"],
                        "yaw": float(v["yaw"]),
                        "pitch": float(v["pitch"]),
                        "enabled": bool(v["enabled"]),
                    }
                    for v in views
                ],
            },
            "views_config_path": "" if direct or spheresfm else "views_config.json",
            "views_config_snapshot": views_config_snapshot,
            "conversion": {
                "yaw_offset_per_frame": yaw_step,
                "output_format": self.output_format_combo.currentData() or "auto",
                "output_bit_depth": self.output_bit_depth_combo.currentData() or "8",
                "jpg_quality": jpg_quality,
                "invert_masks": self.invert_masks_cb.isChecked(),
                "write_images": self._writes_images() and not direct and not spheresfm,
                "write_masks": self._writes_masks() and not direct and not spheresfm,
                "no_image": direct or spheresfm or not self._writes_any_view_assets(),
                "uses_source_images": direct or spheresfm,
                "uses_source_masks": (direct or spheresfm) and self._mask_dir().is_dir(),
                "export_colmap": self._is_metashape_method() and self.export_colmap_cb.isChecked(),
            },
            "postprocess": {
                "lichtfeld_final_orientation_correction": self._uses_lichtfeld_final_correction(),
                "lichtfeld_final_orientation_matrix": _LICHTFELD_FINAL_CORRECTION.tolist()
                if self._uses_lichtfeld_final_correction()
                else None,
            },
            "metashape_import": {
                "enabled": self._is_metashape_method(),
                "use_ply": self._preprocess_uses_ply(),
                "images_dir": str(self._metashape_images_dir()),
                "xml": self.ms_xml_browse.text(),
                "ply": self.ms_ply_browse.text()
                if self._is_metashape_method() and self._preprocess_uses_ply()
                else "",
                "scale": float(self.ms_scale_edit.text().strip()),
                "no_fix_rotation": self.ms_no_fix_rot_cb.isChecked(),
            },
            "colmap_rig": {
                "enabled": self._export_method() == _METHOD_COLMAP,
                "dir": str(self._colmap_rig_dir()),
                "project_dir": str(self._colmap_project_dir()),
                "images_dir": str(self._colmap_rig_images_dir()),
                "masks_dir": str(self._colmap_rig_masks_dir()),
                "rig_config": str(self._colmap_rig_dir() / "rig_config.json"),
                "database": str(self._colmap_database_path()),
                "sparse_dir": str(self._colmap_sparse_dir()),
                "sparse_model_dir": str(self._find_colmap_sparse_model() or ""),
                "run_sfm": self.run_colmap_cb.isChecked(),
                "colmap_executable": self.colmap_exec_browse.text(),
                "glomap_executable": self.glomap_exec_browse.text(),
                "matcher": self.colmap_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
                "mapper": self.colmap_mapper_combo.currentData() or _COLMAP_MAPPER_INCREMENTAL,
                "per_frame_yaw_forced_zero": self._export_method() == _METHOD_COLMAP,
            },
            "spheresfm": {
                "enabled": spheresfm,
                "project_dir": str(self._spheresfm_project_dir()),
                "images_dir": str(self._metashape_images_dir()),
                "source_masks_dir": str(self._mask_dir()),
                "prepared_masks_dir": str(self._spheresfm_masks_dir()),
                "database": str(self._spheresfm_database_path()),
                "sparse_dir": str(self._spheresfm_sparse_dir()),
                "sparse_model_dir": str(self._find_spheresfm_sparse_model() or ""),
                "use_masks": self._spheresfm_uses_masks(),
                "colmap_executable": self.spheresfm_exec_browse.text(),
                "matcher": self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
                "feature_preset": self.spheresfm_feature_combo.currentData() or _SPHERESFM_FEATURE_STANDARD,
                "pose_path": self.spheresfm_pose_browse.text(),
                "camera_model": "SPHERE",
                "camera_params": self._spheresfm_camera_params_arg() if spheresfm else "",
            },
            "inputs": {
                "transforms_json": str(scene / "transforms.json"),
                "masks_dir": str(self._mask_dir()),
                "ply_source": str(self._resolve_ply_source() or ""),
            },
            "output_files": {
                "settings": _EXPORT_SETTINGS_NAME,
                "views_config": "" if direct or spheresfm else "views_config.json",
                "transforms_json": "transforms.json",
                "images_dir": "images",
                "masks_dir": "masks",
                "pointcloud": "pointcloud.ply" if direct else "",
                "colmap_rig_dir": "colmap_rig",
                "colmap_rig_config": "colmap_rig/rig_config.json",
                "colmap_project_manifest": f"colmap_rig/{_COLMAP_PROJECT_MANIFEST_NAME}",
                "spheresfm_project_dir": "spheresfm",
                "spheresfm_project_manifest": f"spheresfm/{_SPHERESFM_PROJECT_MANIFEST_NAME}",
            },
        }

    def _write_export_settings(self) -> None:
        path = self._export_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._collect_export_settings()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_colmap_project_manifest(self) -> None:
        project = self._colmap_project_dir()
        sparse_model = self._find_colmap_sparse_model()
        payload = {
            "app": "stechdrive-3dgs-utils",
            "app_version": APP_VERSION,
            "export_type": "colmap_project",
            "created_at": self._utc_now_iso(),
            "project_dir": str(project),
            "images_dir": "images",
            "masks_dir": "masks",
            "sparse_dir": "sparse",
            "sparse_model_dir": str(sparse_model.relative_to(project).as_posix()) if sparse_model else "",
            "ready_for_import": sparse_model is not None,
            "database": "database.db",
            "rig_config": "rig_config.json",
            "run_sfm": self.run_colmap_cb.isChecked(),
            "matcher": self.colmap_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
            "mapper": self.colmap_mapper_combo.currentData() or _COLMAP_MAPPER_INCREMENTAL,
            "camera_model": "PINHOLE",
            "camera_params": self._colmap_camera_params_arg(),
        }
        project.mkdir(parents=True, exist_ok=True)
        (project / _COLMAP_PROJECT_MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_spheresfm_project_manifest(self) -> None:
        project = self._spheresfm_project_dir()
        sparse_model = self._find_spheresfm_sparse_model()
        payload = {
            "app": "stechdrive-3dgs-utils",
            "app_version": APP_VERSION,
            "export_type": "spheresfm_project",
            "created_at": self._utc_now_iso(),
            "project_dir": str(project),
            "images_dir": str(self._metashape_images_dir()),
            "source_masks_dir": str(self._mask_dir()),
            "prepared_masks_dir": "masks_colmap",
            "sparse_dir": "sparse",
            "sparse_model_dir": str(sparse_model.relative_to(project).as_posix()) if sparse_model else "",
            "ready_for_import": sparse_model is not None,
            "database": "database.db",
            "use_masks": self._spheresfm_uses_masks(),
            "matcher": self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
            "feature_preset": self.spheresfm_feature_combo.currentData() or _SPHERESFM_FEATURE_STANDARD,
            "pose_path": self.spheresfm_pose_browse.text(),
            "camera_model": "SPHERE",
            "camera_params": self._spheresfm_camera_params_arg(),
        }
        project.mkdir(parents=True, exist_ok=True)
        (project / _SPHERESFM_PROJECT_MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return Path(self.scene_dir) / "output"

    def _direct_output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return Path(self.scene_dir)

    def _display_output_dir(self) -> Path:
        if self._uses_direct_equirect_output():
            return self._direct_output_dir()
        if self._is_spheresfm_method():
            return self._spheresfm_project_dir()
        return self._output_dir() if self._is_metashape_method() else self._colmap_rig_dir()

    def _mask_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return Path(self.scene_dir) / "masks"

    def _colmap_rig_dir(self) -> Path:
        return self._output_dir() / "colmap_rig"

    def _colmap_project_dir(self) -> Path:
        return self._colmap_rig_dir()

    def _colmap_rig_images_dir(self) -> Path:
        return self._colmap_rig_dir() / "images"

    def _colmap_rig_masks_dir(self) -> Path:
        return self._colmap_rig_dir() / "masks"

    def _colmap_database_path(self) -> Path:
        return self._colmap_rig_dir() / "database.db"

    def _colmap_sparse_dir(self) -> Path:
        return self._colmap_rig_dir() / "sparse"

    def _spheresfm_project_dir(self) -> Path:
        return self._output_dir() / "spheresfm"

    def _spheresfm_masks_dir(self) -> Path:
        return self._spheresfm_project_dir() / "masks_colmap"

    def _spheresfm_database_path(self) -> Path:
        return self._spheresfm_project_dir() / "database.db"

    def _spheresfm_sparse_dir(self) -> Path:
        return self._spheresfm_project_dir() / "sparse"

    def _find_spheresfm_sparse_model(self) -> Path | None:
        sparse = self._spheresfm_sparse_dir()
        if self._has_colmap_sparse_model(sparse):
            return sparse
        if not sparse.is_dir():
            return None

        def sort_key(path: Path) -> tuple[int, int | str]:
            if path.name.isdigit():
                return (0, int(path.name))
            return (1, path.name.lower())

        candidates = [p for p in sparse.iterdir() if p.is_dir() and self._has_colmap_sparse_model(p)]
        if not candidates:
            return None

        def score(path: Path) -> tuple[int, tuple[int, int | str]]:
            images_file = path / "images.txt"
            registered = 0
            if images_file.is_file():
                try:
                    registered = sum(
                        1
                        for line in images_file.read_text(encoding="utf-8", errors="replace").splitlines()
                        if line.strip() and not line.startswith("#")
                    ) // 2
                except OSError:
                    registered = 0
            return (registered, sort_key(path))

        return max(candidates, key=score)

    def _find_colmap_sparse_model(self) -> Path | None:
        sparse = self._colmap_sparse_dir()
        if self._has_colmap_sparse_model(sparse):
            return sparse
        if not sparse.is_dir():
            return None

        def sort_key(path: Path) -> tuple[int, int | str]:
            if path.name.isdigit():
                return (0, int(path.name))
            return (1, path.name.lower())

        for candidate in sorted((p for p in sparse.iterdir() if p.is_dir()), key=sort_key):
            if self._has_colmap_sparse_model(candidate):
                return candidate
        return None

    @staticmethod
    def _has_colmap_sparse_model(path: Path) -> bool:
        if not path.is_dir():
            return False
        return (
            all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))
            or all((path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))
        )

    def _colmap_camera_params_arg(self) -> str:
        width, height = self._planned_colmap_image_size()
        params = pinhole_camera_params(width, height, 90.0)
        return ",".join(f"{value:.12g}" for value in params)

    def _spheresfm_camera_params_arg(self) -> str:
        source = self._first_image_size(self._metashape_images_dir())
        if source is None:
            raise ValueError(f"画像フォルダに対象画像がありません: {self._metashape_images_dir()}")
        width, height = source
        return f"1,{width / 2:.12g},{height / 2:.12g}"

    def _planned_colmap_image_size(self) -> tuple[int, int]:
        if not self._writes_images():
            existing = self._first_image_size(self._colmap_rig_images_dir())
            if existing is not None:
                return existing

        source = self._first_image_size(Path(self.scene_dir) / "images") if self.scene_dir else None
        if source is not None:
            scale = float(self.scale_combo.currentData())
            output_size = max(1, int(round(source[1] * scale)))
            return output_size, output_size

        existing = self._first_image_size(self._colmap_rig_images_dir())
        if existing is not None:
            return existing

        raise ValueError("COLMAP用の画像サイズを判定できません。images/ に画像が必要です。")

    @staticmethod
    def _first_image_size(root: Path) -> tuple[int, int] | None:
        if not root.is_dir():
            return None
        supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
            if not path.is_file() or path.suffix.lower() not in supported:
                continue
            try:
                from PIL import Image

                with Image.open(path) as img:
                    return int(img.width), int(img.height)
            except Exception:
                continue
        return None

    def _metashape_images_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return Path(self.scene_dir) / "images"

    def _prepare_output_dir(self) -> bool:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        output = self._output_dir()

        scene = Path(self.scene_dir).resolve()
        try:
            resolved_output = output.resolve()
        except OSError:
            resolved_output = output.absolute()
        if resolved_output.parent != scene:
            raise ValueError(f"出力フォルダがシーンフォルダ外です: {output}")

        if not self._writes_any_view_assets():
            output.mkdir(parents=True, exist_ok=True)
            return True

        if self._writes_images() and self._writes_masks():
            if output.exists() and any(output.iterdir()):
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_RESET_TITLE"),
                    i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(output)),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                self._clear_output_dir(output)
        else:
            targets = []
            if self._writes_images():
                targets.append(output / "images")
            if self._writes_masks():
                targets.append(output / "masks")
            existing_targets = [p for p in targets if self._path_has_contents(p)]
            if existing_targets:
                target_text = "\n".join(str(p) for p in existing_targets)
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_PARTIAL_RESET_TITLE"),
                    i18n.t("OUTPUT_PARTIAL_RESET_MESSAGE").format(paths=target_text),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                for target in existing_targets:
                    self._clear_path(target)

        output.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_colmap_rig_dir(self) -> bool:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        output = self._output_dir()
        rig_dir = self._colmap_rig_dir()

        try:
            resolved_rig = rig_dir.resolve()
        except OSError:
            resolved_rig = rig_dir.absolute()
        if resolved_rig.parent != output.resolve():
            raise ValueError(f"COLMAP Rig出力フォルダが不正です: {rig_dir}")

        if self._writes_images() and self._writes_masks():
            if rig_dir.exists() and any(rig_dir.iterdir()):
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_RESET_TITLE"),
                    i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(rig_dir)),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                self._clear_path(rig_dir)
        else:
            targets: list[Path] = []
            if self._writes_images():
                targets.append(self._colmap_rig_images_dir())
            if self._writes_masks():
                targets.append(self._colmap_rig_masks_dir())
            if self.run_colmap_cb.isChecked():
                targets.extend([self._colmap_database_path(), self._colmap_sparse_dir()])
            existing_targets = [p for p in targets if self._path_has_contents(p)]
            if existing_targets:
                target_text = "\n".join(str(p) for p in existing_targets)
                result = QMessageBox.question(
                    self,
                    i18n.t("OUTPUT_PARTIAL_RESET_TITLE"),
                    i18n.t("OUTPUT_PARTIAL_RESET_MESSAGE").format(paths=target_text),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if result != QMessageBox.Yes:
                    return False
                for target in existing_targets:
                    self._clear_path(target)

        rig_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_spheresfm_dir(self) -> bool:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        output = self._output_dir()
        project = self._spheresfm_project_dir()

        try:
            resolved_project = project.resolve()
        except OSError:
            resolved_project = project.absolute()
        if resolved_project.parent != output.resolve():
            raise ValueError(f"SphereSfM出力フォルダが不正です: {project}")

        if project.exists() and any(project.iterdir()):
            result = QMessageBox.question(
                self,
                i18n.t("OUTPUT_RESET_TITLE"),
                i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(project)),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return False
            self._clear_path(project)

        project.mkdir(parents=True, exist_ok=True)
        return True

    @staticmethod
    def _path_has_contents(path: Path) -> bool:
        if path.is_dir():
            return any(path.iterdir())
        return path.exists()

    @staticmethod
    def _clear_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    @staticmethod
    def _clear_output_dir(output: Path) -> None:
        for child in output.iterdir():
            CubemapStep._clear_path(child)

    # -- バンドル検証 --

    def _spheresfm_uses_masks(self) -> bool:
        return self.spheresfm_use_masks_cb.isChecked()

    def _validate_spheresfm_export(self) -> None:
        self._validate_image_only_export()
        if self._spheresfm_uses_masks() and not self._mask_dir().is_dir():
            raise ValueError(i18n.t("SPHERESFM_MASKS_NOT_FOUND").format(path=str(self._mask_dir())))

    def _validate_image_only_export(self) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        images = self._metashape_images_dir()
        if not images.is_dir():
            raise ValueError(f"画像フォルダが見つかりません: {images}")
        supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        if not any(p.is_file() and p.suffix.lower() in supported for p in images.rglob("*")):
            raise ValueError(f"画像フォルダに対象画像がありません: {images}")

    def _validate_bundle(self) -> None:
        profile = self._effective_profile()
        source = self._resolve_ply_source()
        if source is not None:
            return
        if profile == _PROFILE_LICHTFELD and self._preprocess_uses_ply():
            return
        if profile == _PROFILE_LICHTFELD:
            raise ValueError("LichtFeldプロファイルにはpointcloud.plyが必要です。Metashapeインポート設定でPLY使用を有効にしてください。")
        raise ValueError(
            "Postshot/BrushプロファイルにはMetashapeからエクスポートしたRAW PLYが必要です。"
            "LichtFeld用のpointcloud.plyは使用できません。"
        )

    def _resolve_ply_source(self) -> Path | None:
        if not self.scene_dir:
            return None
        scene = Path(self.scene_dir)
        if self._axis_transform_mode() == _AXIS_NONE:
            candidates = [scene / "pointcloud.ply"]
            for c in candidates:
                if c.is_file():
                    return c
            return None
        else:
            candidates = [scene / "metashape.ply", scene / "sparse.ply"]

        for c in candidates:
            if c.is_file():
                return c
        plys = sorted(
            [p for p in scene.glob("*.ply") if p.is_file() and p.name.lower() != "pointcloud.ply"],
            key=lambda x: x.name.lower(),
        )
        if plys:
            return plys[0]
        return None

    # -- バンドル後処理 --

    def on_queue_finished(self, success: bool) -> None:
        if success:
            try:
                self._finalize_bundle()
            except Exception:
                pass

    def _finalize_bundle(self) -> None:
        if self._is_spheresfm_method():
            self._write_export_settings()
            self._write_spheresfm_project_manifest()
            return

        if self._is_colmap_method():
            self._write_export_settings()
            self._write_colmap_project_manifest()
            return

        if self._uses_direct_equirect_output():
            if self._uses_lichtfeld_final_correction():
                self._apply_lichtfeld_final_correction(self._direct_output_dir())
            self._write_export_settings()
            return

        output = self._output_dir()
        output.mkdir(parents=True, exist_ok=True)

        source = self._resolve_ply_source()
        if source is not None:
            dest = output / source.name
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)

            transforms = output / "transforms.json"
            if transforms.is_file():
                data = json.loads(transforms.read_text(encoding="utf-8"))
                data["ply_file_path"] = dest.name
                transforms.write_text(json.dumps(data, indent=2), encoding="utf-8")

        if self._uses_lichtfeld_final_correction():
            self._apply_lichtfeld_final_correction(output)

        if self._writes_any_view_assets():
            self._write_export_settings()

    def _apply_lichtfeld_final_correction(self, output: Path) -> None:
        transforms = output / "transforms.json"
        if transforms.is_file():
            self._transform_transforms_json(transforms, _LICHTFELD_FINAL_CORRECTION)

        pointcloud = output / "pointcloud.ply"
        if pointcloud.is_file():
            self._transform_ply_points(pointcloud, _LICHTFELD_FINAL_CORRECTION)

    @staticmethod
    def _transform_transforms_json(path: Path, matrix: np.ndarray) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        if not isinstance(frames, list):
            return
        for frame in frames:
            if not isinstance(frame, dict) or "transform_matrix" not in frame:
                continue
            transform = np.array(frame["transform_matrix"], dtype=np.float64)
            if transform.shape != (4, 4):
                continue
            frame["transform_matrix"] = (matrix @ transform).tolist()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def _transform_ply_points(cls, path: Path, matrix: np.ndarray) -> None:
        if cls._transform_ply_with_open3d(path, matrix):
            return
        cls._transform_ascii_ply(path, matrix)

    @staticmethod
    def _transform_ply_with_open3d(path: Path, matrix: np.ndarray) -> bool:
        try:
            import open3d as o3d  # type: ignore
        except Exception:
            return False
        try:
            pc = o3d.io.read_point_cloud(str(path))
            if pc.is_empty():
                return False
            pc.transform(matrix)
            return bool(o3d.io.write_point_cloud(str(path), pc))
        except Exception:
            return False

    @staticmethod
    def _transform_ascii_ply(path: Path, matrix: np.ndarray) -> None:
        text = path.read_text(encoding="ascii", errors="strict")
        lines = text.splitlines(keepends=True)
        try:
            end_idx = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
        except StopIteration as e:
            raise ValueError(f"PLY header is missing end_header: {path}") from e

        header = lines[: end_idx + 1]
        if not any(line.strip().startswith("format ascii") for line in header):
            raise ValueError(
                f"Binary PLY correction requires open3d, but open3d could not transform: {path}"
            )

        vertex_count = 0
        vertex_props: list[str] = []
        in_vertex = False
        for line in header:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and parts[0] == "property" and len(parts) >= 3:
                vertex_props.append(parts[-1])

        try:
            x_idx = vertex_props.index("x")
            y_idx = vertex_props.index("y")
            z_idx = vertex_props.index("z")
        except ValueError as e:
            raise ValueError(f"PLY vertex element must contain x/y/z properties: {path}") from e

        data_start = end_idx + 1
        if len(lines) < data_start + vertex_count:
            raise ValueError(f"PLY vertex data is truncated: {path}")

        rot = matrix[:3, :3]
        trans = matrix[:3, 3]
        for i in range(vertex_count):
            line_idx = data_start + i
            line = lines[line_idx]
            newline = "\n" if line.endswith("\n") else ""
            tokens = line.split()
            if len(tokens) < len(vertex_props):
                raise ValueError(f"PLY vertex row is truncated at row {i}: {path}")
            point = np.array(
                [float(tokens[x_idx]), float(tokens[y_idx]), float(tokens[z_idx])],
                dtype=np.float64,
            )
            corrected = rot @ point + trans
            tokens[x_idx] = f"{corrected[0]:.9g}"
            tokens[y_idx] = f"{corrected[1]:.9g}"
            tokens[z_idx] = f"{corrected[2]:.9g}"
            lines[line_idx] = " ".join(tokens) + newline

        path.write_text("".join(lines), encoding="ascii")

    # -- プログレス --

    def phase_display_name(self, phase: str) -> str:
        labels = {
            "metashape": "PHASE_METASHAPE_IMPORT",
            "colmap_rig_export": "PHASE_COLMAP_RIG_EXPORT",
            "colmap_feature": "PHASE_COLMAP_FEATURE",
            "colmap_rig_config": "PHASE_COLMAP_RIG_CONFIG",
            "colmap_match": "PHASE_COLMAP_MATCH",
            "colmap_mapper": "PHASE_COLMAP_MAPPER",
            "spheresfm_prepare": "PHASE_SPHERESFM_PREPARE",
            "spheresfm_database": "PHASE_SPHERESFM_DATABASE",
            "spheresfm_feature": "PHASE_SPHERESFM_FEATURE",
            "spheresfm_match": "PHASE_SPHERESFM_MATCH",
            "spheresfm_mapper": "PHASE_SPHERESFM_MAPPER",
        }
        key = labels.get(phase)
        return i18n.t(key) if key else phase

    def on_phase_started(self, phase: str) -> tuple[int, int] | None:
        if phase == "colmap_rig_export":
            self._converted_total = 0
            self._processed = 0
            self._explicit_progress = False
            return None
        if phase == "colmap_feature":
            total = self._count_colmap_rig_images()
            return 0, total if total > 0 else 0
        if phase == "spheresfm_prepare":
            total = self._count_source_images()
            return 0, total if total > 0 else 0
        if phase == "spheresfm_feature":
            total = self._count_source_images()
            return 0, total if total > 0 else 0
        if phase in {"colmap_rig_config", "colmap_match", "colmap_mapper"}:
            self._colmap_ba_iterations = 0
            return 0, 0
        if phase in {"spheresfm_database", "spheresfm_match", "spheresfm_mapper"}:
            self._colmap_ba_iterations = 0
            return 0, 0
        return None

    def on_line(self, line: str) -> tuple[int, int] | None:
        colmap_feature = _COLMAP_FEATURE_RE.search(line)
        if colmap_feature:
            return int(colmap_feature.group(1)), int(colmap_feature.group(2))

        colmap_match_image = _COLMAP_MATCH_IMAGE_RE.search(line)
        if colmap_match_image:
            return int(colmap_match_image.group(1)), int(colmap_match_image.group(2))

        colmap_match_block = _COLMAP_MATCH_BLOCK_RE.search(line)
        if colmap_match_block:
            block_row = int(colmap_match_block.group(1))
            block_rows = int(colmap_match_block.group(2))
            block_col = int(colmap_match_block.group(3))
            block_cols = int(colmap_match_block.group(4))
            total = max(1, block_rows * block_cols)
            done = min(total, max(1, (block_row - 1) * block_cols + block_col))
            return done, total

        colmap_ba_fixed = _COLMAP_GLOBAL_BA_FIXED_RE.search(line)
        if colmap_ba_fixed:
            return self._colmap_global_ba_progress(
                int(colmap_ba_fixed.group(1)),
                int(colmap_ba_fixed.group(2)),
                fixed_rotation=True,
            )

        colmap_ba_done = _COLMAP_GLOBAL_BA_DONE_RE.search(line)
        if colmap_ba_done:
            return self._colmap_global_ba_progress(
                int(colmap_ba_done.group(1)),
                int(colmap_ba_done.group(2)),
                fixed_rotation=False,
            )

        if _COLMAP_RETRIANGULATION_START_RE.search(line):
            return self._colmap_retriangulation_progress(done=False)

        if _COLMAP_RETRIANGULATION_DONE_RE.search(line) or _COLMAP_RECONSTRUCTION_DONE_RE.search(line):
            return self._colmap_retriangulation_progress(done=True)

        progress = _PROGRESS_RE.match(line)
        if progress:
            self._processed = int(progress.group(1))
            self._converted_total = int(progress.group(2))
            self._explicit_progress = True
            return self._processed, self._converted_total

        m = _CONVERT_RE.match(line)
        if m:
            self._converted_total = int(m.group(1))
            self._processed = 0
            self._explicit_progress = False
            return 0, self._converted_total

        if line.startswith("Processing:") and self._converted_total > 0 and not self._explicit_progress:
            self._processed += 1
            return self._processed, self._converted_total

        return None

    def _colmap_global_ba_progress(
        self,
        iteration: int,
        total_iterations: int,
        *,
        fixed_rotation: bool,
    ) -> tuple[int, int]:
        total_iterations = max(1, total_iterations)
        iteration = min(max(1, iteration), total_iterations)
        self._colmap_ba_iterations = max(self._colmap_ba_iterations, total_iterations)
        total_units = total_iterations * 2 + 2
        done_units = (iteration - 1) * 2 + (1 if fixed_rotation else 2)
        return done_units, total_units

    def _colmap_retriangulation_progress(self, *, done: bool) -> tuple[int, int]:
        iterations = max(1, self._colmap_ba_iterations)
        total_units = iterations * 2 + 2
        return (total_units if done else total_units - 1), total_units

    def _count_colmap_rig_images(self) -> int:
        images_dir = self._colmap_rig_images_dir()
        if not images_dir.is_dir():
            return 0
        supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        return sum(1 for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported)

    def _count_source_images(self) -> int:
        if not self.scene_dir:
            return 0
        images_dir = self._metashape_images_dir()
        if not images_dir.is_dir():
            return 0
        supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        return sum(1 for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported)

    # -- ヘルパー --

    @staticmethod
    def _guess_xml(scene_dir: Path) -> Path:
        for name in ["metashape.xml", "cameras.xml"]:
            c = scene_dir / name
            if c.is_file():
                return c
        xmls = sorted([p for p in scene_dir.glob("*.xml") if p.is_file()], key=lambda x: x.name.lower())
        return xmls[0] if xmls else scene_dir / "metashape.xml"

    @staticmethod
    def _guess_ply(scene_dir: Path) -> str:
        for name in ["metashape.ply", "sparse.ply"]:
            c = scene_dir / name
            if c.is_file():
                return str(c)
        plys = sorted(
            [p for p in scene_dir.glob("*.ply") if p.is_file() and p.name.lower() != "pointcloud.ply"],
            key=lambda x: x.name.lower(),
        )
        return str(plys[0]) if plys else ""
