"""Step 4: 視点画像書き出し (Metashape / COLMAP modes)."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.scene_layout import (
    scene_images_dir,
    step4_meta_dir,
)
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.drag_spinbox import DragDoubleSpinBox
from gui.common.form_rows import add_tooltip_row
from gui.cubemap.preview_renderer import PreviewWidget
from gui.feature_flags import apriltag_scale_enabled
from gui.cubemap.view_config import _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS, ViewConfigWidget
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
)
from gui.steps.sfm_route_backends import get_sfm_route_backend
from gui.steps.sfm_route_selector import SfmRouteSelector
from gui.steps.sfm_route_specs import (
    normalize_sfm_route,
)
from gui.steps.step4_command_plan import Step4CommandPlanMixin
from gui.steps.step4_contracts import (
    _AXIS_BRUSH,
    _AXIS_NONE,
    _AXIS_POSTSHOT,
    _COLMAP_MAPPER_GLOBAL,
    _COLMAP_MAPPER_GLOMAP,
    _COLMAP_MAPPER_INCREMENTAL,
    _COLMAP_MATCHER_EXHAUSTIVE,
    _COLMAP_MATCHER_SEQUENTIAL,
    _COLMAP_REPOSITORY_URL,
    _GENERATED_POINTCLOUD_NAME,
    _METHOD_COLMAP,
    _METHOD_METASHAPE,
    _METHOD_SPHERESFM,
    _NORMAL_OUTPUT_SCALE,
    _OUTPUT_SHAPE_EQUIRECT_3DGUT,
    _OUTPUT_SHAPE_PROJECTED,
    _PIPELINE_STAGE_CONVERSION,
    _PIPELINE_STAGE_SFM,
    _PIPELINE_STATUS_WARNING,
    _PROFILE_BRUSH,
    _PROFILE_CUSTOM,
    _PROFILE_LICHTFELD,
    _PROFILE_POSTSHOT,
    _SPHERESFM_MATCHER_SPATIAL,
    _SPHERESFM_QUALITY_FAST,
    _SPHERESFM_QUALITY_QUALITY,
    _SPHERESFM_QUALITY_STANDARD,
    _SPHERESFM_REPOSITORY_URL,
    _SPHERESFM_RUN_CONVERT_ONLY,
    _SPHERESFM_RUN_FULL,
    _SPHERESFM_RUN_SFM_ONLY,
    _USER_SETTINGS_SECTION,
    _normalize_spheresfm_quality_preset,
    is_colmap_gui_unavailable_output,  # noqa: F401 - re-exported for existing tests/imports
    is_spheresfm_rtx50_cuda_error_line,  # noqa: F401 - re-exported for existing tests/imports
)
from gui.steps.step4_manifest import Step4ManifestMixin
from gui.steps.step4_paths import Step4PathMixin
from gui.steps.step4_pipeline import Step4PipelineMixin
from gui.steps.step4_runtime import Step4RuntimeMixin
from gui.steps.step4_settings import (
    load_step4_export_settings,
)
from gui.steps.step4_training import Step4TrainingMixin
from gui.steps.step4_widgets import ElidedPathLabel
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_LICHTFELD as _TRAINING_BACKEND_LICHTFELD,
)
from gui.user_settings import load_user_settings_section, update_user_settings_section

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


def _make_external_link(text: str, url: str, tooltip: str, object_name: str) -> QLabel:
    link = QLabel(f'<a href="{url}">{text}</a>')
    link.setObjectName(object_name)
    link.setOpenExternalLinks(True)
    link.setTextInteractionFlags(Qt.TextBrowserInteraction)
    link.setToolTip(tooltip)
    return link


class CubemapStep(
    Step4RuntimeMixin,
    Step4TrainingMixin,
    Step4PipelineMixin,
    Step4CommandPlanMixin,
    Step4ManifestMixin,
    Step4PathMixin,
    BaseStepWidget,
):
    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(base_dir, parent)
        self._converted_total = 0
        self._processed = 0
        self._explicit_progress = False
        self._colmap_ba_iterations = 0
        self._syncing_profile_controls = False
        self._syncing_output_shape_controls = False
        self._syncing_user_preferences = False
        self._syncing_project_settings = False
        self._user_preferences_enabled = False
        self._export_method_value = _METHOD_METASHAPE
        self._conversion_intent = True
        self._colmap_sfm_intent = False
        self._spheresfm_sfm_intent = True
        self._spheresfm_conversion_intent = True
        self._pipeline_notice_text = ""
        self._saved_projected_export_targets: tuple[bool, bool] | None = None
        self._input_image_count = 0
        self._spheresfm_phase_logs: dict[str, Path] = {}
        self._training_phase_logs: dict[str, Path] = {}
        self._spheresfm_rtx50_cuda_error_seen = False
        self._spheresfm_rtx50_cuda_error_phase: str | None = None
        self._spheresfm_rtx50_cuda_error_shown = False
        self._spheresfm_gui_processes: list[QProcess] = []
        self._active_runner_phase = ""
        self._preview_render_pending = False
        self._training_backend_value = _TRAINING_BACKEND_LICHTFELD
        self._training_dataset_user_edited = False
        self._training_output_user_edited = False
        self._syncing_training_paths = False
        self._lfs_output_name_user_edited = False
        self._syncing_lfs_output_name = False
        self._postshot_project_name_user_edited = False
        self._syncing_postshot_project_name = False
        self._syncing_lfs_auto_fields = False
        self._yaw_per_frame_non_colmap_value = 30.0
        self._metashape_auto_xml_candidates: tuple[Path, ...] = ()
        self._metashape_auto_ply_candidates: tuple[Path, ...] = ()
        self._syncing_metashape_auto_inputs = False
        self._metashape_ply_approved = False
        self._metashape_ply_auto_candidate = False
        self._apriltag_scale_ui_enabled = apriltag_scale_enabled()
        self._apriltag_estimate_process: QProcess | None = None
        self._apriltag_last_scale: float | None = None
        self._colmap_sparse_user_edited = False
        self._spheresfm_sparse_user_edited = False
        self._syncing_sfm_input_paths = False
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

        # 左パネル: 固定ヘッダー/タブ + 固定サマリー
        left_pane = QWidget()
        left_pane.setFixedWidth(SETTINGS_PANE_WIDTH)
        left_pane_layout = QVBoxLayout(left_pane)
        left_pane_layout.setContentsMargins(0, 0, 0, 0)
        left_pane_layout.setSpacing(0)

        top = QWidget()
        top.setObjectName("settingsPane")
        top.setFixedWidth(SETTINGS_PANE_WIDTH)
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(*SETTINGS_PANE_MARGINS)
        top_layout.setSpacing(8)
        left_layout = top_layout  # 既存コードとの互換用エイリアス

        self.export_method_label = QLabel(i18n.t("EXPORT_METHOD_COMPACT"))
        self.export_method_label.setToolTip(i18n.tip("EXPORT_METHOD"))
        self.export_method_label.setVisible(False)
        self.export_method_selector = SfmRouteSelector()
        self.export_method_selector.route_changed.connect(self._set_export_method)
        self.export_method_row = self.export_method_selector
        self.export_method_row.setMaximumWidth(SETTINGS_PANE_WIDTH - SETTINGS_PANE_MARGINS[2] - 18)
        self.export_method_buttons = self.export_method_selector.route_buttons

        self.export_targets_row = QWidget()
        self.export_targets_row.setToolTip(i18n.tip("EXPORT_TARGETS"))
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

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("step4SettingsTabs")
        self.settings_tabs.tabBar().setUsesScrollButtons(False)
        self.settings_tabs.tabBar().setExpanding(False)

        (
            self.sfm_path_summary_row,
            self.sfm_path_summary_kind,
            self.sfm_path_summary_value,
        ) = self._make_tab_path_summary_row()
        (
            self.cubemap_path_summary_row,
            self.cubemap_path_summary_kind,
            self.cubemap_path_summary_value,
        ) = self._make_tab_path_summary_row()

        colmap_section = QWidget()
        self.colmap_section = colmap_section
        colmap_section_layout = QVBoxLayout(colmap_section)
        colmap_section_layout.setContentsMargins(8, 8, 8, 8)
        colmap_section_layout.setSpacing(6)
        colmap_form = QFormLayout()
        colmap_form.setSpacing(6)

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
        self.colmap_repo_link = _make_external_link(
            i18n.t("COLMAP_REPOSITORY_LINK"),
            _COLMAP_REPOSITORY_URL,
            i18n.tip("COLMAP_REPOSITORY_LINK"),
            "colmapRepositoryLink",
        )
        colmap_section_layout.addWidget(self.colmap_repo_link, alignment=Qt.AlignLeft)

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

        self.spheresfm_quality_combo = QComboBox()
        self.spheresfm_quality_combo.setToolTip(i18n.tip("SPHERESFM_QUALITY_PRESET"))
        self.spheresfm_quality_combo.addItem(i18n.t("SPHERESFM_QUALITY_FAST"), _SPHERESFM_QUALITY_FAST)
        self.spheresfm_quality_combo.addItem(i18n.t("SPHERESFM_QUALITY_STANDARD"), _SPHERESFM_QUALITY_STANDARD)
        self.spheresfm_quality_combo.addItem(i18n.t("SPHERESFM_QUALITY_QUALITY"), _SPHERESFM_QUALITY_QUALITY)
        self.spheresfm_quality_combo.setCurrentIndex(1)
        self.spheresfm_quality_combo.setFixedWidth(150)

        self.spheresfm_pipeline_row = QWidget()
        spheresfm_pipeline_layout = QHBoxLayout(self.spheresfm_pipeline_row)
        spheresfm_pipeline_layout.setContentsMargins(0, 0, 0, 0)
        spheresfm_pipeline_layout.setSpacing(8)
        spheresfm_pipeline_layout.addWidget(QLabel(i18n.t("COLMAP_MATCHER_COMPACT")))
        spheresfm_pipeline_layout.addWidget(self.spheresfm_matcher_combo)
        spheresfm_pipeline_layout.addWidget(QLabel(i18n.t("SPHERESFM_QUALITY_COMPACT")))
        spheresfm_pipeline_layout.addWidget(self.spheresfm_quality_combo)
        spheresfm_pipeline_layout.addStretch()
        spheresfm_form.addRow(self.spheresfm_pipeline_row)

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
        self.spheresfm_repo_link = _make_external_link(
            i18n.t("SPHERESFM_REPOSITORY_LINK"),
            _SPHERESFM_REPOSITORY_URL,
            i18n.tip("SPHERESFM_REPOSITORY_LINK"),
            "spheresfmRepositoryLink",
        )
        spheresfm_layout.addWidget(self.spheresfm_repo_link, alignment=Qt.AlignLeft)

        spheresfm_convert_section = QWidget()
        self.spheresfm_convert_section = spheresfm_convert_section
        spheresfm_convert_layout = QVBoxLayout(spheresfm_convert_section)
        spheresfm_convert_layout.setContentsMargins(8, 8, 8, 8)
        spheresfm_convert_layout.setSpacing(6)
        spheresfm_convert_form = QFormLayout()
        spheresfm_convert_form.setSpacing(6)

        self.spheresfm_output_shape_combo = QComboBox()
        self.spheresfm_output_shape_combo.setToolTip(i18n.tip("SPHERESFM_OUTPUT_SHAPE"))
        self.spheresfm_output_shape_combo.addItem(i18n.t("OUTPUT_SHAPE_PROJECTED"), _OUTPUT_SHAPE_PROJECTED)
        self.spheresfm_output_shape_combo.addItem(i18n.t("OUTPUT_SHAPE_EQUIRECT_3DGUT"), _OUTPUT_SHAPE_EQUIRECT_3DGUT)
        self.spheresfm_output_shape_combo.currentIndexChanged.connect(self._on_output_shape_changed)

        self.spheresfm_profile_combo = QComboBox()
        self.spheresfm_profile_combo.setToolTip(i18n.tip("SPHERESFM_TARGET_PROFILE"))
        self.spheresfm_profile_combo.addItem(i18n.PROFILE_POSTSHOT, _PROFILE_POSTSHOT)
        self.spheresfm_profile_combo.addItem(i18n.PROFILE_BRUSH, _PROFILE_BRUSH)
        self.spheresfm_profile_combo.addItem(i18n.PROFILE_LICHTFELD, _PROFILE_LICHTFELD)
        self.spheresfm_profile_combo.addItem(i18n.PROFILE_CUSTOM, _PROFILE_CUSTOM)
        self.spheresfm_profile_combo.currentIndexChanged.connect(self._on_spheresfm_profile_changed)
        add_tooltip_row(
            spheresfm_convert_form,
            i18n.TARGET_PROFILE,
            self.spheresfm_profile_combo,
            i18n.tip("SPHERESFM_TARGET_PROFILE"),
        )
        add_tooltip_row(
            spheresfm_convert_form,
            i18n.t("OUTPUT_SHAPE"),
            self.spheresfm_output_shape_combo,
            i18n.tip("SPHERESFM_OUTPUT_SHAPE"),
        )

        self.spheresfm_axis_transform_combo = QComboBox()
        self.spheresfm_axis_transform_combo.setToolTip(i18n.tip("SPHERESFM_AXIS_TRANSFORM"))
        self.spheresfm_axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_POSTSHOT"), _AXIS_POSTSHOT)
        self.spheresfm_axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_BRUSH"), _AXIS_BRUSH)
        self.spheresfm_axis_transform_combo.addItem(i18n.t("AXIS_TRANSFORM_NONE"), _AXIS_NONE)
        self.spheresfm_axis_transform_combo.setFixedWidth(180)
        self.spheresfm_axis_transform_combo.currentIndexChanged.connect(self._on_spheresfm_profile_option_changed)
        add_tooltip_row(
            spheresfm_convert_form,
            i18n.t("AXIS_TRANSFORM"),
            self.spheresfm_axis_transform_combo,
            i18n.tip("SPHERESFM_AXIS_TRANSFORM"),
        )

        self.spheresfm_profile_hint = QLabel("")
        self.spheresfm_profile_hint.setStyleSheet("color: #8888aa; font-size: 9pt;")
        self.spheresfm_profile_hint.setVisible(False)
        spheresfm_convert_form.addRow("", self.spheresfm_profile_hint)

        spheresfm_convert_layout.addLayout(spheresfm_convert_form)
        spheresfm_convert_layout.addStretch()

        # Metashapeインポート設定
        self.metashape_section = QWidget()
        self.metashape_section.setLayout(QVBoxLayout())
        preprocess = QWidget()
        self.metashape_sfm_input_widget = preprocess
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

        self.metashape_output_section = QWidget()
        self.metashape_output_section.setLayout(profile_form)
        pp_form = QFormLayout()

        self.ms_images_path_label = ElidedPathLabel("-")
        self.ms_images_path_label.setToolTip(i18n.tip("MS_IMAGES"))
        add_tooltip_row(pp_form, i18n.t("MS_IMAGES_LABEL"), self.ms_images_path_label, i18n.tip("MS_IMAGES"))

        self.ms_xml_browse = BrowseWidget(
            mode="file",
            filter_str="XML (*.xml);;すべて (*.*)",
            placeholder=i18n.t("MS_XML_PLACEHOLDER"),
        )
        self.ms_xml_browse.setToolTip(i18n.tip("MS_XML"))
        self.ms_xml_browse.line_edit.setToolTip(i18n.tip("MS_XML"))
        add_tooltip_row(pp_form, i18n.METASHAPE_XML, self.ms_xml_browse, i18n.tip("MS_XML"))

        self.ms_ply_browse = BrowseWidget(
            mode="file",
            filter_str="PLY (*.ply);;すべて (*.*)",
            placeholder=i18n.t("MS_PLY_PLACEHOLDER"),
        )
        self.ms_ply_browse.setToolTip(i18n.tip("MS_PLY"))
        self.ms_ply_browse.line_edit.setToolTip(i18n.tip("MS_PLY"))
        add_tooltip_row(pp_form, i18n.METASHAPE_PLY, self.ms_ply_browse, i18n.tip("MS_PLY"))

        self.ms_xml_browse.path_changed.connect(self._on_metashape_input_path_changed)
        self.ms_ply_browse.path_changed.connect(self._on_metashape_input_path_changed)
        self.ms_ply_browse.line_edit.textEdited.connect(self._on_metashape_ply_text_edited)

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

        self.colmap_sfm_input_widget = QWidget()
        colmap_sfm_input_layout = QVBoxLayout(self.colmap_sfm_input_widget)
        colmap_sfm_input_layout.setContentsMargins(8, 8, 8, 8)
        colmap_sfm_input_layout.setSpacing(6)
        colmap_sfm_input_form = QFormLayout()
        colmap_sfm_input_form.setSpacing(6)
        self.colmap_sparse_browse = BrowseWidget(
            mode="dir",
            placeholder=i18n.t("SFM_SPARSE_MODEL_PLACEHOLDER"),
        )
        self.colmap_sparse_browse.setToolTip(i18n.tip("COLMAP_SPARSE_MODEL"))
        self.colmap_sparse_browse.line_edit.setToolTip(i18n.tip("COLMAP_SPARSE_MODEL"))
        add_tooltip_row(
            colmap_sfm_input_form,
            i18n.t("SFM_SPARSE_MODEL"),
            self.colmap_sparse_browse,
            i18n.tip("COLMAP_SPARSE_MODEL"),
        )
        colmap_sfm_input_layout.addLayout(colmap_sfm_input_form)
        colmap_sfm_input_layout.addStretch()

        self.spheresfm_sfm_input_widget = QWidget()
        spheresfm_sfm_input_layout = QVBoxLayout(self.spheresfm_sfm_input_widget)
        spheresfm_sfm_input_layout.setContentsMargins(8, 8, 8, 8)
        spheresfm_sfm_input_layout.setSpacing(6)
        spheresfm_sfm_input_form = QFormLayout()
        spheresfm_sfm_input_form.setSpacing(6)
        self.spheresfm_sparse_browse = BrowseWidget(
            mode="dir",
            placeholder=i18n.t("SFM_SPARSE_MODEL_PLACEHOLDER"),
        )
        self.spheresfm_sparse_browse.setToolTip(i18n.tip("SPHERESFM_SPARSE_MODEL"))
        self.spheresfm_sparse_browse.line_edit.setToolTip(i18n.tip("SPHERESFM_SPARSE_MODEL"))
        add_tooltip_row(
            spheresfm_sfm_input_form,
            i18n.t("SFM_SPARSE_MODEL"),
            self.spheresfm_sparse_browse,
            i18n.tip("SPHERESFM_SPARSE_MODEL"),
        )
        spheresfm_sfm_input_layout.addLayout(spheresfm_sfm_input_form)
        spheresfm_sfm_input_layout.addStretch()

        self.colmap_sparse_browse.path_changed.connect(self._on_colmap_sparse_path_changed)
        self.spheresfm_sparse_browse.path_changed.connect(self._on_spheresfm_sparse_path_changed)

        self.sfm_input_section = QWidget()
        sfm_input_layout = QVBoxLayout(self.sfm_input_section)
        sfm_input_layout.setContentsMargins(0, 0, 0, 0)
        sfm_input_layout.setSpacing(4)
        self.sfm_input_title = QLabel(i18n.t("SFM_INPUT_SECTION"))
        self.sfm_input_title.setToolTip(i18n.tip("SFM_INPUT_SECTION"))
        sfm_input_layout.addWidget(self.sfm_input_title)
        sfm_input_layout.addWidget(self.metashape_sfm_input_widget)
        sfm_input_layout.addWidget(self.colmap_sfm_input_widget)
        sfm_input_layout.addWidget(self.spheresfm_sfm_input_widget)

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

        output_details = QWidget()
        self.output_details_section = output_details
        output_details_layout = QVBoxLayout(output_details)
        output_details_layout.setContentsMargins(0, 0, 0, 0)
        output_details_layout.setSpacing(8)

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
        output_details_layout.addWidget(format_row)

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
        output_details_layout.addWidget(quality_row)
        output_details_layout.addStretch()

        adv_output_layout.addLayout(adv_form)
        adv_output_layout.addStretch()

        self.training_section = self._build_training_section(exe_filter)

        input_tab = QWidget()
        self.input_tab = input_tab
        input_layout = QVBoxLayout(input_tab)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(6)
        input_layout.addWidget(self.sfm_path_summary_row)
        input_layout.addWidget(self.export_method_row)
        input_layout.addWidget(self.sfm_input_section)
        input_layout.addWidget(self.colmap_section)
        input_layout.addWidget(self.spheresfm_section)
        input_layout.addStretch()

        output_tab = QWidget()
        self.output_tab = output_tab
        output_layout = QVBoxLayout(output_tab)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setSpacing(6)
        output_layout.addWidget(self.cubemap_path_summary_row)
        output_layout.addWidget(self.export_targets_row)
        output_layout.addWidget(self.metashape_output_section)
        output_layout.addWidget(self.spheresfm_convert_section)
        output_layout.addWidget(self.advanced_output_section)
        output_layout.addStretch()

        details_tab = QWidget()
        self.details_tab = details_tab
        details_layout = QVBoxLayout(details_tab)
        details_layout.setContentsMargins(8, 8, 8, 8)
        details_layout.setSpacing(6)
        details_layout.addWidget(self.output_details_section)
        details_layout.addStretch()

        self.apriltag_tab_index: int | None = None
        if self._apriltag_scale_ui_enabled:
            self.apriltag_tab = self._build_apriltag_scale_tab()

        self.input_tab_index = self.settings_tabs.addTab(
            self._make_tab_scroll_area(self.input_tab),
            i18n.t("STEP4_TAB_INPUT"),
        )
        self.output_tab_index = self.settings_tabs.addTab(
            self._make_tab_scroll_area(self.output_tab),
            i18n.t("STEP4_TAB_OUTPUT"),
        )
        if self._apriltag_scale_ui_enabled:
            self.apriltag_tab_index = self.settings_tabs.addTab(
                self._make_tab_scroll_area(self.apriltag_tab),
                i18n.t("STEP4_TAB_APRILTAG_SCALE"),
            )
        self.details_tab_index = self.settings_tabs.addTab(
            self._make_tab_scroll_area(self.details_tab),
            i18n.t("STEP4_TAB_DETAILS"),
        )
        self.metashape_tab_index = self.input_tab_index
        self.colmap_tab_index = self.input_tab_index
        self.spheresfm_tab_index = self.input_tab_index
        self.view_export_tab_index = self.output_tab_index
        self.spheresfm_convert_tab_index = self.output_tab_index
        self.settings_tabs.currentChanged.connect(lambda _index: self.primary_action_state_changed.emit())
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

        left_pane_layout.addWidget(top, stretch=1)
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
        spheresfm_lichtfeld_index = self.spheresfm_profile_combo.findData(_PROFILE_LICHTFELD)
        if spheresfm_lichtfeld_index >= 0:
            self.spheresfm_profile_combo.setCurrentIndex(spheresfm_lichtfeld_index)
        self._on_profile_changed(self.profile_combo.currentIndex())
        self._on_spheresfm_profile_changed(self.spheresfm_profile_combo.currentIndex())
        self._on_output_shape_changed(self.output_shape_combo.currentIndex())
        self._on_colmap_mapper_changed()
        self._sync_colmap_sfm_controls()
        self._set_export_method(_METHOD_METASHAPE)

    def _make_tab_scroll_area(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("step4TabScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(content)
        return scroll

    def _make_tab_path_summary_row(self) -> tuple[QWidget, QLabel, ElidedPathLabel]:
        row = QWidget()
        row.setObjectName("tabPathSummary")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        kind = QLabel("-")
        kind.setObjectName("tabPathSummaryKind")
        kind.setWordWrap(False)
        value = ElidedPathLabel("-")
        value.setObjectName("tabPathSummaryValue")
        layout.addWidget(kind)
        layout.addWidget(value, stretch=1)
        return row, kind, value

    def _make_training_path_summary_row(
        self,
    ) -> tuple[QWidget, QLabel, ElidedPathLabel, QLabel, ElidedPathLabel]:
        row = QWidget()
        row.setObjectName("tabPathSummary")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        dataset_kind = QLabel(i18n.t("STEP4_SUMMARY_INPUT"))
        dataset_kind.setObjectName("tabPathSummaryKind")
        dataset_value = ElidedPathLabel("-")
        dataset_value.setObjectName("tabPathSummaryValue")
        output_kind = QLabel(i18n.t("STEP4_SUMMARY_OUTPUT"))
        output_kind.setObjectName("tabPathSummaryKind")
        output_value = ElidedPathLabel("-")
        output_value.setObjectName("tabPathSummaryValue")
        layout.addWidget(dataset_kind)
        layout.addWidget(dataset_value, stretch=1)
        layout.addWidget(output_kind)
        layout.addWidget(output_value, stretch=1)
        return row, dataset_kind, dataset_value, output_kind, output_value

    def set_scene_dir(self, path: str) -> None:
        super().set_scene_dir(path)
        if not path:
            self.ms_images_path_label.setToolTip(i18n.tip("MS_IMAGES"))
            self.ms_images_path_label.set_full_text("-")
            self.ms_xml_browse.set_text("")
            self.ms_ply_browse.set_text("")
            self._metashape_auto_xml_candidates = ()
            self._metashape_auto_ply_candidates = ()
            self._set_metashape_ply_approved(False)
            self._colmap_sparse_user_edited = False
            self._spheresfm_sparse_user_edited = False
            self._sync_sfm_input_paths(force=True)
            self._update_metashape_input_hint()
            self.preview.set_scene_dir("")
            self._refresh_input_image_count()
            self._training_dataset_user_edited = False
            self._training_output_user_edited = False
            self._lfs_output_name_user_edited = False
            self._postshot_project_name_user_edited = False
            self._update_training_paths(force=True)
            self._update_path_labels()
            self._update_lfs_output_name(force=True)
            self._update_postshot_project_name(force=True)
            self._update_output_count()
            self._render_preview()
            return
        p = Path(path)
        images_dir = str(self._metashape_images_dir())
        self._update_path_labels()
        self.ms_images_path_label.setToolTip(f"{i18n.tip('MS_IMAGES')}\n{images_dir}")
        self.ms_images_path_label.set_full_text(images_dir)
        self._apply_metashape_auto_inputs(p)
        self._colmap_sparse_user_edited = False
        self._spheresfm_sparse_user_edited = False
        self._sync_sfm_input_paths(force=True)
        self._training_dataset_user_edited = False
        self._training_output_user_edited = False
        self._lfs_output_name_user_edited = False
        self._postshot_project_name_user_edited = False
        restored = self._restore_project_settings(p)
        self.preview.set_scene_dir(path, refresh=False)
        self._input_image_count = 0
        self._update_training_paths(force=not restored)
        self._update_lfs_output_name(force=not restored)
        self._update_postshot_project_name(force=not restored)
        self._update_lfs_auto_steps_scaler()
        self._update_output_count()
        self._update_metashape_input_hint()

    def _restore_project_settings(self, scene: Path) -> bool:
        settings = load_step4_export_settings(scene)
        if not settings:
            return False

        self._syncing_project_settings = True
        self._syncing_user_preferences = True
        try:
            self._apply_project_settings(scene, settings)
        finally:
            self._syncing_user_preferences = False
            self._syncing_project_settings = False
        return True

    def _apply_project_settings(self, scene: Path, settings: dict) -> None:
        route = normalize_sfm_route(str(settings.get("export_method", "")))
        external_import = self._settings_origin_kind(settings) == "external_import"
        self._set_export_method(route)
        if external_import:
            self._conversion_intent = False
            self._colmap_sfm_intent = False
            self._spheresfm_sfm_intent = False
            self._spheresfm_conversion_intent = False

        self._restore_conversion_settings(settings)
        self._restore_route_settings(scene, settings)
        self._restore_training_settings(scene, settings)
        if external_import:
            self._arm_external_import_metashape_defaults_if_ready()

        self._sync_output_shape_controls()
        self._sync_yaw_per_frame_control()
        self._sync_settings_tabs()
        self._update_path_labels()

    @staticmethod
    def _settings_origin_kind(settings: dict) -> str:
        origin = settings.get("origin")
        if not isinstance(origin, dict):
            return ""
        return str(origin.get("kind") or "").strip()

    def _arm_external_import_metashape_defaults_if_ready(self) -> None:
        if not self._is_metashape_method() or not self._external_import_metashape_inputs_ready():
            return

        self._conversion_intent = True
        self._colmap_sfm_intent = False
        self._spheresfm_sfm_intent = False
        self._spheresfm_conversion_intent = False
        self.export_images_cb.setChecked(True)
        self.export_masks_cb.setChecked(True)
        self.export_colmap_cb.setChecked(False)
        self._set_combo_data(self.output_shape_combo, _OUTPUT_SHAPE_PROJECTED)
        self.view_config.apply_settings_snapshot({"mode": "cube6", "yaw_offset": 0.0})
        self._set_combo_data(self.profile_combo, _PROFILE_LICHTFELD)
        self._sync_profile_defaults(_PROFILE_LICHTFELD)
        self._set_metashape_ply_approved(True, auto_candidate=False)
        self._sync_ply_browse_enabled()

    def _external_import_metashape_inputs_ready(self) -> bool:
        xml_text = self.ms_xml_browse.text().strip()
        ply_text = self.ms_ply_browse.text().strip()
        if not xml_text or not ply_text:
            return False
        xml = Path(xml_text)
        ply = Path(ply_text)
        if not xml.is_file() or not ply.is_file():
            return False
        return (
            self._metashape_input_output_path_issue(xml) is None
            and self._metashape_input_output_path_issue(ply) is None
        )

    def _restore_conversion_settings(self, settings: dict) -> None:
        image_size = settings.get("image_size") if isinstance(settings.get("image_size"), dict) else {}
        scale = image_size.get("scale")
        if scale is not None:
            try:
                self._set_combo_data(self.scale_combo, float(scale))
            except (TypeError, ValueError):
                pass

        view_config = settings.get("view_config")
        if isinstance(view_config, dict):
            self.view_config.apply_settings_snapshot(view_config)

        conversion = settings.get("conversion") if isinstance(settings.get("conversion"), dict) else {}
        if "write_images" in conversion:
            self.export_images_cb.setChecked(bool(conversion.get("write_images")))
        if "write_masks" in conversion:
            self.export_masks_cb.setChecked(bool(conversion.get("write_masks")))
        if "yaw_offset_per_frame" in conversion and not self._is_colmap_method():
            try:
                self.yaw_per_frame_edit.setValue(float(conversion.get("yaw_offset_per_frame")))
            except (TypeError, ValueError):
                pass
        output_format = str(conversion.get("output_format", "")).strip()
        output_bit_depth = str(conversion.get("output_bit_depth", "")).strip()
        if output_format:
            self._set_combo_data(self.output_format_combo, output_format)
        if output_bit_depth:
            self._set_combo_data(self.output_bit_depth_combo, output_bit_depth)
        if "jpg_quality" in conversion:
            self.jpg_quality_edit.setText(str(conversion.get("jpg_quality")))
        if "invert_masks" in conversion:
            self.invert_masks_cb.setChecked(bool(conversion.get("invert_masks")))

    def _restore_route_settings(self, scene: Path, settings: dict) -> None:
        output_shape = str(settings.get("output_shape", "")).strip()
        if output_shape:
            combo = self.spheresfm_output_shape_combo if self._is_spheresfm_method() else self.output_shape_combo
            self._set_combo_data(combo, output_shape)

        target_profile = str(settings.get("target_profile", "")).strip()
        axis_transform = str(settings.get("axis_transform", "")).strip()
        if self._is_spheresfm_method():
            if target_profile:
                self._set_combo_data(self.spheresfm_profile_combo, target_profile)
            if axis_transform:
                self._set_combo_data(self.spheresfm_axis_transform_combo, axis_transform)
        else:
            if target_profile:
                self._set_combo_data(self.profile_combo, target_profile)
            if axis_transform:
                self._set_combo_data(self.axis_transform_combo, axis_transform)

        metashape = settings.get("metashape_import")
        if isinstance(metashape, dict):
            xml = self._settings_path_text(scene, metashape.get("xml"), require_file=True)
            ply = self._settings_path_text(scene, metashape.get("ply"), require_file=True)
            if xml:
                self.ms_xml_browse.set_text(xml)
            if ply:
                self.ms_ply_browse.set_text(ply)
                self._set_metashape_ply_approved(True)
            if "use_ply" in metashape:
                self.ms_use_ply_cb.setChecked(bool(metashape.get("use_ply")))
            if "scale" in metashape:
                self.ms_scale_edit.setText(str(metashape.get("scale")))
            if "no_fix_rotation" in metashape:
                self.ms_no_fix_rot_cb.setChecked(bool(metashape.get("no_fix_rotation")))

        colmap = settings.get("colmap_rig")
        if isinstance(colmap, dict):
            if "run_sfm" in colmap:
                self._set_colmap_stage_intents(
                    run_sfm=bool(colmap.get("run_sfm")),
                    run_conversion=self._conversion_intent or bool(colmap.get("run_sfm")),
                )
            self._set_combo_data(self.colmap_matcher_combo, str(colmap.get("matcher", "")).strip())
            self._set_combo_data(self.colmap_mapper_combo, str(colmap.get("mapper", "")).strip())
            colmap_exec = self._settings_text(colmap.get("colmap_executable"))
            glomap_exec = self._settings_text(colmap.get("glomap_executable"))
            if colmap_exec:
                self.colmap_exec_browse.set_text(colmap_exec)
            if glomap_exec:
                self.glomap_exec_browse.set_text(glomap_exec)
            sparse = self._settings_path_text(
                scene,
                colmap.get("selected_sparse_model_dir"),
            )
            if sparse:
                self.colmap_sparse_browse.set_text(sparse)
                self._colmap_sparse_user_edited = True

        spheresfm = settings.get("spheresfm")
        if isinstance(spheresfm, dict):
            if "use_masks" in spheresfm:
                self.spheresfm_use_masks_cb.setChecked(bool(spheresfm.get("use_masks")))
            self._set_combo_data(self.spheresfm_matcher_combo, str(spheresfm.get("matcher", "")).strip())
            self._set_combo_data(
                self.spheresfm_quality_combo,
                _normalize_spheresfm_quality_preset(str(spheresfm.get("quality_preset", "")).strip()),
            )
            run_scope = str(spheresfm.get("run_scope", "")).strip()
            if run_scope:
                run_scope = self._normalize_spheresfm_run_scope(run_scope)
                self._set_spheresfm_stage_intents(
                    run_sfm=run_scope != _SPHERESFM_RUN_CONVERT_ONLY,
                    run_conversion=run_scope != _SPHERESFM_RUN_SFM_ONLY,
                )
            pose = self._settings_path_text(scene, spheresfm.get("pose_path"), require_file=True)
            if pose:
                self.spheresfm_pose_browse.set_text(pose)
            spheresfm_exec = self._settings_text(spheresfm.get("colmap_executable"))
            if spheresfm_exec:
                self.spheresfm_exec_browse.set_text(spheresfm_exec)
            sparse = self._settings_path_text(
                scene,
                spheresfm.get("selected_sparse_model_dir"),
            )
            if sparse:
                self.spheresfm_sparse_browse.set_text(sparse)
                self._spheresfm_sparse_user_edited = True

    @staticmethod
    def _settings_text(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _settings_path_text(scene: Path, value: object, *, require_file: bool = False) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        path = Path(text)
        if not path.is_absolute():
            path = scene / path
        if require_file and not path.is_file():
            return ""
        return str(path)

    def primary_action_text(self) -> str:
        return i18n.t("RUN")

    def primary_action_tooltip(self) -> str:
        return i18n.tip("RUN_CUBEMAP")

    def primary_action_enabled(self) -> bool:
        selected = False
        status_by_stage = {
            _PIPELINE_STAGE_SFM: self._pipeline_sfm_status(),
            _PIPELINE_STAGE_CONVERSION: self._pipeline_conversion_status(),
        }
        for stage, (status, _symbol, _detail) in status_by_stage.items():
            if not self._pipeline_stage_runs_in_app(stage):
                continue
            selected = True
            if status == _PIPELINE_STATUS_WARNING:
                return False
        return selected

    def on_activated(self) -> None:
        self._refresh_metashape_auto_inputs_if_empty()
        self._sync_sfm_input_paths()
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
        self.spheresfm_quality_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
        self.spheresfm_output_shape_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
        self.spheresfm_profile_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
        self.spheresfm_axis_transform_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
        self.training_executable_browse.path_changed.connect(lambda _path: self._save_user_preferences())

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
            spheresfm_quality = str(settings.get("spheresfm_quality_preset", "")).strip()
            spheresfm_output_shape = str(settings.get("spheresfm_output_shape", "")).strip()
            spheresfm_profile = str(settings.get("spheresfm_profile", "")).strip()
            spheresfm_axis = str(settings.get("spheresfm_axis_transform", "")).strip()
            if spheresfm_matcher:
                self._set_combo_data(self.spheresfm_matcher_combo, spheresfm_matcher)
            if spheresfm_quality:
                self._set_combo_data(
                    self.spheresfm_quality_combo,
                    _normalize_spheresfm_quality_preset(spheresfm_quality),
                )
            if spheresfm_output_shape:
                self._set_combo_data(self.spheresfm_output_shape_combo, spheresfm_output_shape)
            if spheresfm_profile:
                self._set_combo_data(self.spheresfm_profile_combo, spheresfm_profile)
            if spheresfm_axis:
                self._set_combo_data(self.spheresfm_axis_transform_combo, spheresfm_axis)

            training_backend = str(settings.get("training_backend", "")).strip()
            if training_backend:
                self._set_training_backend(training_backend)
            training_executable = str(settings.get("training_executable", "")).strip()
            training_output = str(settings.get("training_output", "")).strip()
            if training_executable:
                self.training_executable_browse.set_text(training_executable)
            if training_output:
                self.training_output_browse.set_text(training_output)
                self._training_output_user_edited = True
        finally:
            self._syncing_user_preferences = False
        self._on_colmap_mapper_changed()

    def _save_user_preferences(self) -> None:
        if self._syncing_user_preferences or self._syncing_project_settings:
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
                "spheresfm_quality_preset": self._spheresfm_quality_preset(),
                "spheresfm_output_shape": self.spheresfm_output_shape_combo.currentData() or _OUTPUT_SHAPE_PROJECTED,
                "spheresfm_profile": self.spheresfm_profile_combo.currentData() or _PROFILE_LICHTFELD,
                "spheresfm_axis_transform": self.spheresfm_axis_transform_combo.currentData() or _AXIS_NONE,
                "training_backend": self._training_backend(),
                "training_executable": self.training_executable_browse.text(),
                "training_output": self.training_output_browse.text(),
            },
        )

    # -- 書き出し方式 --

    def _export_method(self) -> str:
        return self._export_method_value

    def _sfm_route_backend(self):
        return get_sfm_route_backend(self._export_method())

    def _is_metashape_method(self) -> bool:
        return self._export_method() == _METHOD_METASHAPE

    def _is_colmap_method(self) -> bool:
        return self._export_method() == _METHOD_COLMAP

    def _is_spheresfm_method(self) -> bool:
        return self._export_method() == _METHOD_SPHERESFM

    def _spheresfm_runs_conversion(self) -> bool:
        if not self._is_spheresfm_method():
            return False
        return self._spheresfm_conversion_intent

    def _spheresfm_runs_sfm(self) -> bool:
        if not self._is_spheresfm_method():
            return False
        return self._spheresfm_sfm_intent

    @staticmethod
    def _normalize_spheresfm_run_scope(value: str) -> str:
        if value in {_SPHERESFM_RUN_FULL, _SPHERESFM_RUN_SFM_ONLY, _SPHERESFM_RUN_CONVERT_ONLY}:
            return value
        return _SPHERESFM_RUN_FULL

    def _set_export_method(self, method: str) -> None:
        method = normalize_sfm_route(method)
        self._export_method_value = method
        if hasattr(self, "export_method_selector"):
            self.export_method_selector.set_route(method)
        self._on_export_method_changed()

    def _on_export_method_changed(self) -> None:
        metashape = self._is_metashape_method()
        self._sync_settings_tabs(prefer_route_tab=True)
        if not metashape:
            self.export_colmap_cb.setChecked(False)
        self._sync_output_shape_controls()
        self._sync_yaw_per_frame_control()
        self._sync_colmap_sfm_controls()
        self._sync_sfm_input_paths()
        self._update_path_labels()
        self._update_training_paths()
        self._update_output_count()
        self.primary_action_state_changed.emit()

    def _sync_yaw_per_frame_control(self) -> None:
        if not hasattr(self, "yaw_per_frame_edit"):
            return
        if self._is_colmap_method():
            if self.yaw_per_frame_edit.isEnabled():
                self._yaw_per_frame_non_colmap_value = float(self.yaw_per_frame_edit.value())
            self.yaw_per_frame_edit.setValue(0.0)
            self.yaw_per_frame_edit.setEnabled(False)
            self.yaw_per_frame_edit.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_COLMAP_HINT"))
            self.yaw_per_frame_label.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_COLMAP_HINT"))
            return

        if not self.yaw_per_frame_edit.isEnabled():
            self.yaw_per_frame_edit.setEnabled(True)
            self.yaw_per_frame_edit.setValue(self._yaw_per_frame_non_colmap_value)
        self.yaw_per_frame_edit.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))
        self.yaw_per_frame_label.setToolTip(i18n.t("YAW_OFFSET_PER_FRAME_HINT"))

    def _sync_settings_tabs(self, *, prefer_route_tab: bool = False) -> None:
        current = self.settings_tabs.currentIndex()
        self.metashape_section.setVisible(False)
        self.sfm_input_section.setVisible(True)
        self.metashape_sfm_input_widget.setVisible(self._is_metashape_method())
        self.colmap_sfm_input_widget.setVisible(self._is_colmap_method())
        self.spheresfm_sfm_input_widget.setVisible(self._is_spheresfm_method())
        self.metashape_output_section.setVisible(self._is_metashape_method())
        self.colmap_section.setVisible(self._is_colmap_method())
        self.spheresfm_section.setVisible(self._is_spheresfm_method())
        self.spheresfm_convert_section.setVisible(self._is_spheresfm_method() and self._spheresfm_runs_conversion())
        spheresfm_conversion_off = self._is_spheresfm_method() and not self._spheresfm_runs_conversion()
        conversion_enabled = not spheresfm_conversion_off
        self.settings_tabs.setTabEnabled(self.output_tab_index, conversion_enabled)
        if self.apriltag_tab_index is not None:
            self.settings_tabs.setTabEnabled(self.apriltag_tab_index, self._is_metashape_method())
        route_index = self.input_tab_index
        if spheresfm_conversion_off and current == self.output_tab_index:
            self.settings_tabs.setCurrentIndex(route_index)
        elif self.apriltag_tab_index is not None and current == self.apriltag_tab_index and not self._is_metashape_method():
            self.settings_tabs.setCurrentIndex(route_index)
        elif prefer_route_tab:
            self.settings_tabs.setCurrentIndex(route_index)
        elif not self._settings_tab_available(self.settings_tabs.currentIndex()):
            self.settings_tabs.setCurrentIndex(route_index)

    def _build_apriltag_scale_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(6)

        self.apriltag_enable_cb = QCheckBox(i18n.t("APRILTAG_SCALE_ENABLE"))
        self.apriltag_enable_cb.setToolTip(i18n.tip("APRILTAG_SCALE_ENABLE"))
        self.apriltag_enable_cb.toggled.connect(self._sync_apriltag_controls)
        form.addRow("", self.apriltag_enable_cb)

        self.apriltag_tag_size_edit = QLineEdit("0.160")
        self.apriltag_tag_size_edit.setFixedWidth(86)
        self.apriltag_tag_size_edit.setToolTip(i18n.tip("APRILTAG_TAG_SIZE"))
        add_tooltip_row(form, i18n.t("APRILTAG_TAG_SIZE"), self.apriltag_tag_size_edit, i18n.tip("APRILTAG_TAG_SIZE"))

        self.apriltag_family_combo = QComboBox()
        self.apriltag_family_combo.setFixedWidth(120)
        self.apriltag_family_combo.setToolTip(i18n.tip("APRILTAG_FAMILY"))
        for family in ("tag36h11", "tag25h9", "tag16h5"):
            self.apriltag_family_combo.addItem(family, family)
        add_tooltip_row(form, i18n.t("APRILTAG_FAMILY"), self.apriltag_family_combo, i18n.tip("APRILTAG_FAMILY"))

        self.apriltag_id_edit = QLineEdit("")
        self.apriltag_id_edit.setFixedWidth(86)
        self.apriltag_id_edit.setToolTip(i18n.tip("APRILTAG_TAG_ID"))
        add_tooltip_row(form, i18n.t("APRILTAG_TAG_ID"), self.apriltag_id_edit, i18n.tip("APRILTAG_TAG_ID"))

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.apriltag_estimate_btn = QPushButton(i18n.t("APRILTAG_ESTIMATE"))
        self.apriltag_estimate_btn.setToolTip(i18n.tip("APRILTAG_ESTIMATE"))
        self.apriltag_estimate_btn.clicked.connect(self._run_apriltag_scale_estimate)
        action_layout.addWidget(self.apriltag_estimate_btn)
        self.apriltag_apply_btn = QPushButton(i18n.t("APRILTAG_APPLY_SCALE"))
        self.apriltag_apply_btn.setToolTip(i18n.tip("APRILTAG_APPLY_SCALE"))
        self.apriltag_apply_btn.clicked.connect(self._apply_apriltag_scale)
        action_layout.addWidget(self.apriltag_apply_btn)
        action_layout.addStretch()
        form.addRow("", action_row)

        self.apriltag_result_label = QLabel(i18n.t("APRILTAG_RESULT_EMPTY"))
        self.apriltag_result_label.setWordWrap(True)
        self.apriltag_result_label.setToolTip(i18n.tip("APRILTAG_RESULT"))
        form.addRow(i18n.t("APRILTAG_RESULT"), self.apriltag_result_label)

        self.apriltag_status_label = QLabel(i18n.t("APRILTAG_DEV_STATUS"))
        self.apriltag_status_label.setWordWrap(True)
        self.apriltag_status_label.setStyleSheet("color: #8888aa; font-size: 9pt;")
        form.addRow("", self.apriltag_status_label)

        layout.addLayout(form)
        layout.addStretch()
        self._sync_apriltag_controls()
        return tab

    def _sync_apriltag_controls(self) -> None:
        if not self._apriltag_scale_ui_enabled or not hasattr(self, "apriltag_estimate_btn"):
            return
        running = self._apriltag_estimate_process is not None
        enabled = self.apriltag_enable_cb.isChecked() and not running
        self.apriltag_estimate_btn.setEnabled(enabled)
        self.apriltag_apply_btn.setEnabled(enabled and self._apriltag_last_scale is not None)

    def _build_apriltag_scale_cmd(self, report_path: Path) -> list[str]:
        script = self.base_dir / "scripts" / "estimate_apriltag_scale.py"
        if not script.is_file():
            raise ValueError(f"estimate_apriltag_scale.py not found: {script}")
        transforms = self._output_dir() / "transforms.json"
        if not transforms.is_file():
            raise ValueError(i18n.t("APRILTAG_TRANSFORMS_MISSING").format(path=str(transforms)))

        tag_size_text = self.apriltag_tag_size_edit.text().strip()
        try:
            tag_size = float(tag_size_text)
        except ValueError as exc:
            raise ValueError(i18n.t("APRILTAG_TAG_SIZE_INVALID")) from exc
        if not math.isfinite(tag_size) or tag_size <= 0.0:
            raise ValueError(i18n.t("APRILTAG_TAG_SIZE_INVALID"))

        cmd = [
            sys.executable,
            "-u",
            str(script),
            str(transforms),
            "--tag-size-m",
            tag_size_text,
            "--family",
            str(self.apriltag_family_combo.currentData() or "tag36h11"),
            "--report-json",
            str(report_path),
            "--equirect-temp-dir",
            str(step4_meta_dir(Path(self.scene_dir)) / "apriltag_projection"),
        ]
        tag_id = self.apriltag_id_edit.text().strip()
        if tag_id:
            cmd.extend(["--tag-id", tag_id])
        return cmd

    def _run_apriltag_scale_estimate(self) -> None:
        if self._apriltag_estimate_process is not None:
            return
        if not self.scene_dir:
            self.apriltag_result_label.setText(i18n.t("APRILTAG_SCENE_REQUIRED"))
            return
        report_path = step4_meta_dir(Path(self.scene_dir)) / "apriltag_scale_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = self._build_apriltag_scale_cmd(report_path)
        except Exception as exc:
            self.apriltag_result_label.setText(str(exc))
            return

        self._apriltag_last_scale = None
        self.apriltag_result_label.setText(i18n.t("APRILTAG_RUNNING"))
        self._sync_apriltag_controls()
        process = QProcess(self)
        self._apriltag_estimate_process = process
        process.setProgram(cmd[0])
        process.setArguments(cmd[1:])
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.finished.connect(lambda exit_code, status, path=report_path: self._on_apriltag_scale_finished(exit_code, status, path))
        process.start()
        if not process.waitForStarted(3000):
            detail = process.errorString().strip() or "-"
            self._apriltag_estimate_process = None
            self.apriltag_result_label.setText(i18n.t("APRILTAG_FAILED").format(detail=detail))
            self._sync_apriltag_controls()

    def _on_apriltag_scale_finished(
        self,
        exit_code: int,
        _status: QProcess.ExitStatus,
        report_path: Path,
    ) -> None:
        process = self._apriltag_estimate_process
        detail = self._qprocess_output_text(process) if process is not None else ""
        if process is not None:
            process.deleteLater()
        self._apriltag_estimate_process = None

        if exit_code != 0:
            self.apriltag_result_label.setText(
                i18n.t("APRILTAG_FAILED").format(detail=self._message_detail_tail(detail))
            )
            self._sync_apriltag_controls()
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            estimate = report["estimate"]
            scale = float(estimate["scale"])
        except Exception as exc:
            self.apriltag_result_label.setText(i18n.t("APRILTAG_FAILED").format(detail=str(exc)))
            self._sync_apriltag_controls()
            return

        self._apriltag_last_scale = scale
        self.apriltag_result_label.setText(
            i18n.t("APRILTAG_RESULT_FORMAT").format(
                scale=f"{scale:.9g}",
                observations=int(estimate.get("observation_count", 0)),
                pairs=int(estimate.get("pair_count", 0)),
                inliers=int(estimate.get("inlier_count", 0)),
                rms=float(estimate.get("rms_residual_m", 0.0)),
            )
        )
        self._sync_apriltag_controls()

    def _apply_apriltag_scale(self) -> None:
        if self._apriltag_last_scale is None:
            return
        self.ms_scale_edit.setText(f"{self._apriltag_last_scale:.9g}")
        self._on_profile_option_changed()
        self.apriltag_status_label.setText(i18n.t("APRILTAG_APPLIED"))

    def _settings_tab_available(self, index: int) -> bool:
        return (
            0 <= index < self.settings_tabs.count()
            and self.settings_tabs.isTabVisible(index)
            and self.settings_tabs.isTabEnabled(index)
        )

    def _update_path_labels(self) -> None:
        if not self.scene_dir:
            self._set_single_path_summary(
                self.sfm_path_summary_row,
                self.sfm_path_summary_kind,
                self.sfm_path_summary_value,
                i18n.t("STEP4_SUMMARY_INPUT"),
                "-",
            )
            self._set_single_path_summary(
                self.cubemap_path_summary_row,
                self.cubemap_path_summary_kind,
                self.cubemap_path_summary_value,
                i18n.t("STEP4_SUMMARY_OUTPUT"),
                "-",
            )
            self._set_training_path_summary("-", "-")
            return
        self._update_sfm_path_summary()
        self._update_cubemap_path_summary()
        self._update_training_path_summary()

    def _set_single_path_summary(
        self,
        row: QWidget,
        kind_label: QLabel,
        value_label: ElidedPathLabel,
        kind: str,
        value: str,
        *,
        tooltip: str = "",
    ) -> None:
        kind_label.setText(kind)
        value_label.set_full_text(value or "-")
        summary_tooltip = tooltip or f"{kind}: {value or '-'}"
        row.setToolTip(summary_tooltip)
        kind_label.setToolTip(summary_tooltip)
        value_label.setToolTip(summary_tooltip)

    def _set_training_path_summary(self, dataset: str, output: str, *, tooltip: str = "") -> None:
        self.training_dataset_summary_kind.setText(i18n.t("STEP4_SUMMARY_INPUT"))
        self.training_output_summary_kind.setText(i18n.t("STEP4_SUMMARY_OUTPUT"))
        self.training_dataset_summary_value.set_full_text(dataset or "-")
        self.training_output_summary_value.set_full_text(output or "-")
        summary_tooltip = tooltip or (
            f"{i18n.t('STEP4_SUMMARY_INPUT')}: {dataset or '-'}\n{i18n.t('STEP4_SUMMARY_OUTPUT')}: {output or '-'}"
        )
        for widget in (
            self.training_path_summary_row,
            self.training_dataset_summary_kind,
            self.training_dataset_summary_value,
            self.training_output_summary_kind,
            self.training_output_summary_value,
        ):
            widget.setToolTip(summary_tooltip)

    def _summary_path_text(self, path: Path, *, directory: bool = True) -> str:
        separator = os.sep
        if self.scene_dir:
            try:
                text = path.relative_to(Path(self.scene_dir)).as_posix()
                separator = "/"
            except ValueError:
                text = str(path)
        else:
            text = str(path)
        if directory and text != "-":
            text = text.rstrip("/\\") + separator
        return text

    def _summary_full_path(self, path: Path, *, directory: bool = True) -> str:
        text = str(path)
        if directory:
            text = text.rstrip("/\\") + os.sep
        return text

    def _update_sfm_path_summary(self) -> None:
        if self._is_metashape_method():
            parts: list[str] = []
            tooltip_parts: list[str] = []
            xml_text = self.ms_xml_browse.text().strip()
            if xml_text:
                xml = Path(xml_text)
                parts.append(f"XML {self._summary_path_text(xml, directory=False)}")
                tooltip_parts.append(f"{i18n.METASHAPE_XML}: {xml}")
            else:
                parts.append("XML")
                tooltip_parts.append(f"{i18n.METASHAPE_XML}: -")
            if self._preprocess_uses_ply():
                ply_text = self.ms_ply_browse.text().strip()
                if ply_text:
                    ply = Path(ply_text)
                    parts.append(f"PLY {self._summary_path_text(ply, directory=False)}")
                    tooltip_parts.append(f"{i18n.METASHAPE_PLY}: {ply}")
                else:
                    parts.append("PLY")
                    tooltip_parts.append(f"{i18n.METASHAPE_PLY}: -")
            self._set_single_path_summary(
                self.sfm_path_summary_row,
                self.sfm_path_summary_kind,
                self.sfm_path_summary_value,
                i18n.t("STEP4_SUMMARY_INPUT"),
                " / ".join(parts),
                tooltip="\n".join(tooltip_parts),
            )
            return

        if self._is_colmap_method():
            if self.pipeline_stage_intent(_PIPELINE_STAGE_SFM):
                target = self._colmap_rig_dir()
                kind = i18n.t("STEP4_SUMMARY_OUTPUT")
                tip_key = "OUTPUT_DIR_COLMAP_PROJECT"
            else:
                target = self._find_colmap_sparse_model() or self._colmap_sparse_dir()
                kind = i18n.t("STEP4_SUMMARY_INPUT")
                tip_key = "COLMAP_SPARSE_MODEL"
            self._set_single_path_summary(
                self.sfm_path_summary_row,
                self.sfm_path_summary_kind,
                self.sfm_path_summary_value,
                kind,
                self._summary_path_text(target),
                tooltip=f"{i18n.tip(tip_key)}\n{self._summary_full_path(target)}",
            )
            return

        if self._spheresfm_runs_sfm():
            target = self._spheresfm_project_dir()
            kind = i18n.t("STEP4_SUMMARY_WORK")
            tip_key = "OUTPUT_DIR_SPHERESFM_PROJECT"
        else:
            target = self._find_spheresfm_sparse_model() or self._spheresfm_sparse_dir()
            kind = i18n.t("STEP4_SUMMARY_INPUT")
            tip_key = "SPHERESFM_SPARSE_MODEL"
        self._set_single_path_summary(
            self.sfm_path_summary_row,
            self.sfm_path_summary_kind,
            self.sfm_path_summary_value,
            kind,
            self._summary_path_text(target),
            tooltip=f"{i18n.tip(tip_key)}\n{self._summary_full_path(target)}",
        )

    def _update_cubemap_path_summary(self) -> None:
        target = self._colmap_rig_dir() if self._is_colmap_method() else self._output_dir()
        if self._is_colmap_method():
            tip_key = "OUTPUT_DIR_COLMAP_PROJECT"
        elif self._uses_direct_equirect_output() or (
            self._spheresfm_runs_conversion() and self._uses_spheresfm_3dgut_output()
        ):
            tip_key = "OUTPUT_DIR_LICHTFELD_DIRECT"
        else:
            tip_key = "OUTPUT_DIR_CUBEMAP"
        value = self._summary_path_text(target)
        if not self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION):
            value = i18n.t("STEP4_SUMMARY_DISABLED_PATH").format(path=value)
        self._set_single_path_summary(
            self.cubemap_path_summary_row,
            self.cubemap_path_summary_kind,
            self.cubemap_path_summary_value,
            i18n.t("STEP4_SUMMARY_OUTPUT"),
            value,
            tooltip=f"{i18n.tip(tip_key)}\n{self._summary_full_path(target)}",
        )

    def _update_training_path_summary(self) -> None:
        dataset = self._training_dataset_dir()
        output = self._training_output_dir()
        self._set_training_path_summary(
            self._summary_path_text(dataset),
            self._summary_path_text(output),
            tooltip=(
                f"{i18n.t('STEP4_SUMMARY_INPUT')}: {self._summary_full_path(dataset)}\n"
                f"{i18n.t('STEP4_SUMMARY_OUTPUT')}: {self._summary_full_path(output)}"
            ),
        )

    def _on_metashape_input_path_changed(self, *_args) -> None:
        if self.sender() is self.ms_ply_browse and not self._syncing_metashape_auto_inputs:
            self._set_metashape_ply_approved(bool(self.ms_ply_browse.text().strip()))
        self._update_metashape_input_hint()
        self._update_path_labels()
        self.primary_action_state_changed.emit()

    def _on_metashape_ply_text_edited(self, _text: str) -> None:
        if self._syncing_metashape_auto_inputs:
            return
        self._set_metashape_ply_approved(bool(self.ms_ply_browse.text().strip()))

    def _approve_metashape_ply(self) -> None:
        self._set_metashape_ply_approved(bool(self.ms_ply_browse.text().strip()))
        self._update_metashape_input_hint()
        self.primary_action_state_changed.emit()

    def _set_metashape_ply_approved(self, approved: bool, *, auto_candidate: bool = False) -> None:
        self._metashape_ply_approved = bool(approved)
        self._metashape_ply_auto_candidate = bool(auto_candidate and not approved)
        if not hasattr(self, "ms_ply_approve_btn"):
            return
        has_path = bool(self.ms_ply_browse.text().strip())
        self.ms_ply_approve_btn.setEnabled(has_path and self._preprocess_uses_ply())
        self.ms_ply_approve_btn.setChecked(self._metashape_ply_approved)
        if self._metashape_ply_approved:
            tooltip = i18n.t("MS_PLY_APPROVED")
        elif self._metashape_ply_auto_candidate:
            tooltip = i18n.t("MS_PLY_APPROVE_CANDIDATE")
        else:
            tooltip = i18n.t("MS_PLY_APPROVE")
        self.ms_ply_approve_btn.setToolTip(tooltip)

    def _on_colmap_sparse_path_changed(self, _path: str) -> None:
        if not self._syncing_sfm_input_paths:
            self._colmap_sparse_user_edited = True
        self._update_sfm_input_tooltips()
        self._update_training_paths()
        self._update_path_labels()
        self.primary_action_state_changed.emit()

    def _on_spheresfm_sparse_path_changed(self, _path: str) -> None:
        if not self._syncing_sfm_input_paths:
            self._spheresfm_sparse_user_edited = True
        self._update_sfm_input_tooltips()
        self._update_training_paths()
        self._update_path_labels()
        self.primary_action_state_changed.emit()

    def _sync_sfm_input_paths(self, *, force: bool = False) -> None:
        if not hasattr(self, "colmap_sparse_browse"):
            return
        self._syncing_sfm_input_paths = True
        try:
            if not self.scene_dir:
                self.colmap_sparse_browse.set_text("")
                self.spheresfm_sparse_browse.set_text("")
                return
            if force or (not self._colmap_sparse_user_edited and not self.colmap_sparse_browse.text().strip()):
                model = self._auto_find_colmap_sparse_model()
                self.colmap_sparse_browse.set_text(str(model) if model else "")
            if force or (not self._spheresfm_sparse_user_edited and not self.spheresfm_sparse_browse.text().strip()):
                model = self._auto_find_spheresfm_sparse_model()
                self.spheresfm_sparse_browse.set_text(str(model) if model else "")
        finally:
            self._syncing_sfm_input_paths = False
        self._update_sfm_input_tooltips()
        self._update_path_labels()

    def _update_sfm_input_tooltips(self) -> None:
        if not hasattr(self, "colmap_sparse_browse"):
            return
        colmap_note = self._sparse_model_input_note(self.colmap_sparse_browse.text())
        colmap_tip = self._append_tooltip_note(i18n.tip("COLMAP_SPARSE_MODEL"), colmap_note)
        self.colmap_sparse_browse.setToolTip(colmap_tip)
        self.colmap_sparse_browse.line_edit.setToolTip(colmap_tip)

        spheresfm_note = self._sparse_model_input_note(self.spheresfm_sparse_browse.text())
        spheresfm_tip = self._append_tooltip_note(i18n.tip("SPHERESFM_SPARSE_MODEL"), spheresfm_note)
        self.spheresfm_sparse_browse.setToolTip(spheresfm_tip)
        self.spheresfm_sparse_browse.line_edit.setToolTip(spheresfm_tip)

    def _sparse_model_input_note(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            return i18n.t("SFM_SPARSE_MODEL_AUTO_HINT")
        path = Path(raw)
        if self._has_colmap_sparse_model(path):
            return i18n.t("SFM_SPARSE_MODEL_READY_HINT")
        return i18n.t("SFM_SPARSE_MODEL_INVALID_HINT").format(path=raw)

    def _apply_metashape_auto_inputs(self, scene: Path) -> None:
        xml = self._guess_xml(scene)
        ply = self._guess_ply(scene)
        self._syncing_metashape_auto_inputs = True
        try:
            self.ms_xml_browse.set_text(str(xml) if xml else "")
            self.ms_ply_browse.set_text(str(ply) if ply else "")
        finally:
            self._syncing_metashape_auto_inputs = False
        self._set_metashape_ply_approved(ply is not None, auto_candidate=False)

    def _refresh_metashape_auto_inputs_if_empty(self) -> None:
        if not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        old_xml_candidates = self._metashape_auto_xml_candidates
        old_ply_candidates = self._metashape_auto_ply_candidates
        changed = False
        if not self.ms_xml_browse.text().strip():
            xml = self._guess_xml(scene)
            if xml:
                self._syncing_metashape_auto_inputs = True
                try:
                    self.ms_xml_browse.set_text(str(xml))
                finally:
                    self._syncing_metashape_auto_inputs = False
                changed = True
        if not self.ms_ply_browse.text().strip():
            ply = self._guess_ply(scene)
            if ply:
                self._syncing_metashape_auto_inputs = True
                try:
                    self.ms_ply_browse.set_text(str(ply))
                finally:
                    self._syncing_metashape_auto_inputs = False
                self._set_metashape_ply_approved(True, auto_candidate=False)
                changed = True
        if (
            changed
            or old_xml_candidates != self._metashape_auto_xml_candidates
            or old_ply_candidates != self._metashape_auto_ply_candidates
        ):
            self._update_metashape_input_hint()

    def _update_metashape_input_hint(self) -> None:
        if not hasattr(self, "ms_xml_browse") or not hasattr(self, "ms_ply_browse"):
            return
        if not self.scene_dir:
            self._set_metashape_input_tooltips("", "")
            return

        scene = Path(self.scene_dir)
        xml_note = ""
        ply_note = ""
        xml_text = self.ms_xml_browse.text().strip()
        if not xml_text:
            if self._metashape_auto_xml_candidates:
                xml_note = i18n.t("MS_XML_MANUAL_SELECTION_HINT").format(
                    names=self._format_candidate_names(self._metashape_auto_xml_candidates)
                )
            else:
                xml_note = i18n.t("MS_XML_MISSING_HINT")
        else:
            xml = Path(xml_text)
            issue = self._metashape_input_output_path_issue(xml)
            if issue:
                xml_note = issue
            elif not xml.is_file():
                xml_note = i18n.t("MS_XML_SELECTED_MISSING_HINT").format(path=xml_text)

        if self._preprocess_uses_ply():
            ply_text = self.ms_ply_browse.text().strip()
            if not ply_text:
                if self._metashape_auto_ply_candidates:
                    ply_note = i18n.t("MS_PLY_MANUAL_SELECTION_HINT").format(
                        names=self._format_candidate_names(self._metashape_auto_ply_candidates)
                    )
                elif (scene / _GENERATED_POINTCLOUD_NAME).is_file():
                    ply_note = i18n.t("MS_PLY_ONLY_POINTCLOUD_HINT")
                else:
                    ply_note = i18n.t("MS_PLY_MISSING_HINT")
            else:
                ply = Path(ply_text)
                issue = self._metashape_input_output_path_issue(ply)
                if issue:
                    ply_note = issue
                elif not ply.is_file():
                    ply_note = i18n.t("MS_PLY_SELECTED_MISSING_HINT").format(path=ply_text)

        self._set_metashape_input_tooltips(xml_note, ply_note)
        self._set_metashape_ply_approved(
            self._metashape_ply_approved,
            auto_candidate=self._metashape_ply_auto_candidate,
        )

    def _set_metashape_input_tooltips(self, xml_note: str, ply_note: str) -> None:
        xml_tip = self._append_tooltip_note(i18n.tip("MS_XML"), xml_note)
        ply_tip = self._append_tooltip_note(i18n.tip("MS_PLY"), ply_note)
        self.ms_xml_browse.setToolTip(xml_tip)
        self.ms_xml_browse.line_edit.setToolTip(xml_tip)
        self.ms_ply_browse.setToolTip(ply_tip)
        self.ms_ply_browse.line_edit.setToolTip(ply_tip)

    @staticmethod
    def _append_tooltip_note(base: str, note: str) -> str:
        return f"{base}\n{note}" if note else base

    def _sync_colmap_sfm_controls(self) -> None:
        enabled = self._is_colmap_method() and self._colmap_sfm_intent
        self.colmap_exec_browse.setEnabled(enabled)
        self.colmap_pipeline_row.setEnabled(enabled)
        self._on_colmap_mapper_changed()

    def _on_colmap_mapper_changed(self, *_args) -> None:
        needs_glomap = (
            self._is_colmap_method()
            and self._colmap_sfm_intent
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

    def _set_combo_data(self, combo: QComboBox, value: object) -> None:
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

    def _spheresfm_profile_id(self) -> str:
        return self.spheresfm_profile_combo.currentData() or _PROFILE_LICHTFELD

    def _spheresfm_axis_transform_mode(self) -> str:
        data = self.spheresfm_axis_transform_combo.currentData()
        return data if data in {_AXIS_POSTSHOT, _AXIS_BRUSH, _AXIS_NONE} else _AXIS_NONE

    def _spheresfm_effective_profile(self) -> str:
        if self._uses_spheresfm_3dgut_output():
            return _PROFILE_LICHTFELD
        mode = self._spheresfm_axis_transform_mode()
        if mode == _AXIS_NONE:
            return _PROFILE_LICHTFELD
        if mode == _AXIS_BRUSH:
            return _PROFILE_BRUSH
        return _PROFILE_POSTSHOT

    def _uses_lichtfeld_final_correction(self) -> bool:
        return self._is_metashape_method() and self._effective_profile() == _PROFILE_LICHTFELD

    def _uses_spheresfm_lichtfeld_final_correction(self) -> bool:
        return self._is_spheresfm_method() and self._spheresfm_effective_profile() == _PROFILE_LICHTFELD

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
            no_fix_changed = self.ms_no_fix_rot_cb.isChecked() != self._profile_no_fix_rotation_default(current)
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

    def _on_spheresfm_profile_changed(self, _index: int) -> None:
        p = self._spheresfm_profile_id()
        if p != _PROFILE_CUSTOM:
            self._syncing_profile_controls = True
            try:
                self._set_combo_data(self.spheresfm_axis_transform_combo, self._profile_axis_default(p))
            finally:
                self._syncing_profile_controls = False
        self.spheresfm_profile_hint.setText(i18n.t("PROFILE_CUSTOM_HINT") if p == _PROFILE_CUSTOM else "")
        self.spheresfm_profile_hint.setVisible(p == _PROFILE_CUSTOM)
        self._sync_output_shape_controls()
        self._update_path_labels()
        self._update_output_count()

    def _on_spheresfm_profile_option_changed(self, *_args) -> None:
        if self._syncing_profile_controls:
            return
        current = self._spheresfm_profile_id()
        if current != _PROFILE_CUSTOM:
            axis_changed = self._spheresfm_axis_transform_mode() != self._profile_axis_default(current)
            if axis_changed:
                custom_idx = self.spheresfm_profile_combo.findData(_PROFILE_CUSTOM)
                if custom_idx >= 0:
                    self.spheresfm_profile_combo.setCurrentIndex(custom_idx)
        self._sync_output_shape_controls()

    # -- 出力形状 --

    def _output_shape(self) -> str:
        combo = self.spheresfm_output_shape_combo if self._is_spheresfm_method() else self.output_shape_combo
        data = combo.currentData()
        return data if data in {_OUTPUT_SHAPE_PROJECTED, _OUTPUT_SHAPE_EQUIRECT_3DGUT} else _OUTPUT_SHAPE_PROJECTED

    def _uses_direct_equirect_output(self) -> bool:
        return self._is_metashape_method() and self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT

    def _uses_spheresfm_3dgut_output(self) -> bool:
        return self._is_spheresfm_method() and self._output_shape() == _OUTPUT_SHAPE_EQUIRECT_3DGUT

    def _uses_spheresfm_projected_output(self) -> bool:
        return self._is_spheresfm_method() and self._output_shape() == _OUTPUT_SHAPE_PROJECTED

    def _on_output_shape_changed(self, *_args) -> None:
        if self._syncing_output_shape_controls:
            return
        if self._uses_direct_equirect_output():
            self._ensure_direct_equirect_defaults()
        if self._uses_spheresfm_3dgut_output():
            self._ensure_spheresfm_3dgut_defaults()
        self._sync_output_shape_controls()
        self._sync_settings_tabs()
        self._update_path_labels()
        self._update_training_paths()
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

    def _ensure_spheresfm_3dgut_defaults(self) -> None:
        self._syncing_profile_controls = True
        try:
            self._set_combo_data(self.spheresfm_axis_transform_combo, _AXIS_NONE)
        finally:
            self._syncing_profile_controls = False

        if self._spheresfm_profile_id() not in {_PROFILE_LICHTFELD, _PROFILE_CUSTOM}:
            self._set_combo_data(self.spheresfm_profile_combo, _PROFILE_LICHTFELD)

    def _sync_output_shape_controls(self) -> None:
        direct = self._uses_direct_equirect_output()
        spheresfm = self._is_spheresfm_method()
        spheresfm_runs_conversion = self._spheresfm_runs_conversion()
        spheresfm_runs_sfm = self._spheresfm_runs_sfm()
        spheresfm_3dgut = spheresfm_runs_conversion and self._uses_spheresfm_3dgut_output()
        spheresfm_projected = spheresfm_runs_conversion and self._uses_spheresfm_projected_output()
        if direct or spheresfm_3dgut:
            if self._saved_projected_export_targets is None:
                self._saved_projected_export_targets = (
                    self.export_images_cb.isChecked(),
                    self.export_masks_cb.isChecked(),
                )
            self.export_images_cb.setChecked(True)
            self.export_masks_cb.setChecked(True)
            self.export_colmap_cb.setChecked(False)
        elif self._saved_projected_export_targets is not None:
            images, masks = self._saved_projected_export_targets
            self.export_images_cb.setChecked(images)
            self.export_masks_cb.setChecked(masks)
            self._saved_projected_export_targets = None

        if spheresfm:
            self.export_colmap_cb.setChecked(False)

        route_uses_view_export = not direct and (not spheresfm or spheresfm_projected)
        self.export_targets_row.setEnabled(route_uses_view_export)
        self.view_config.settings_widget.setEnabled(route_uses_view_export)
        self.output_details_section.setEnabled(route_uses_view_export)
        self.output_shape_combo.setEnabled(self._is_metashape_method())
        self.spheresfm_output_shape_combo.setEnabled(spheresfm_runs_conversion)
        self.spheresfm_exec_browse.setEnabled(spheresfm_runs_sfm)
        self.spheresfm_use_masks_cb.setEnabled(spheresfm_runs_sfm)
        self.spheresfm_pipeline_row.setEnabled(spheresfm_runs_sfm)
        self.spheresfm_pose_browse.setEnabled(spheresfm_runs_sfm)
        self.axis_transform_combo.setEnabled(self._is_metashape_method() and not direct)
        self.spheresfm_profile_combo.setEnabled(spheresfm_projected)
        self.spheresfm_axis_transform_combo.setEnabled(spheresfm_projected)
        self.ms_use_ply_cb.setEnabled(self._is_metashape_method() and not direct)
        self.export_colmap_cb.setEnabled(self._is_metashape_method() and not direct)
        self.settings_tabs.setTabEnabled(self.output_tab_index, (not spheresfm) or spheresfm_runs_conversion)

    def _preprocess_uses_ply(self) -> bool:
        return self.ms_use_ply_cb.isChecked()

    def _sync_ply_browse_enabled(self) -> None:
        self.ms_ply_browse.setEnabled(self._preprocess_uses_ply())
        self._set_metashape_ply_approved(
            self._metashape_ply_approved,
            auto_candidate=self._metashape_ply_auto_candidate,
        )
        self._update_metashape_input_hint()

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
                if self._uses_direct_equirect_output()
                or (self._spheresfm_runs_conversion() and self._uses_spheresfm_3dgut_output())
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
        images = scene_images_dir(scene)
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
        if self._uses_direct_equirect_output() or (
            self._spheresfm_runs_conversion() and self._uses_spheresfm_3dgut_output()
        ):
            count_text = i18n.t("OUTPUT_IMAGE_COUNT_DIRECT_FORMAT").format(count=self._input_image_count)
            self.view_config.set_output_count_text(f"{label}: {count_text}")
            self._update_lfs_auto_steps_scaler()
            return
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            self.view_config.set_output_count_text(f"{label}: -")
            self._update_lfs_auto_steps_scaler()
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
        self._update_lfs_auto_steps_scaler()

    # -- コマンド構築 --

    def process_log_dir(self) -> Path | None:
        if not self._is_spheresfm_method():
            return None
        try:
            return step4_meta_dir(Path(self.scene_dir)) / "logs" / "spheresfm"
        except ValueError:
            return None

    def training_process_log_dir(self) -> Path | None:
        if not self.scene_dir:
            return None
        try:
            return step4_meta_dir(Path(self.scene_dir)) / "logs" / "training"
        except ValueError:
            return None

    def _reset_spheresfm_rtx50_diagnostics(self) -> None:
        self._spheresfm_phase_logs.clear()
        self._spheresfm_rtx50_cuda_error_seen = False
        self._spheresfm_rtx50_cuda_error_phase = None
        self._spheresfm_rtx50_cuda_error_shown = False
