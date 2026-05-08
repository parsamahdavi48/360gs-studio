"""Step 4: 視点画像書き出し (Metashape / COLMAP modes)."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QSize, Qt, QTimer
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
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
from gui.common.icons import check_icon
from gui.cubemap.preview_renderer import PreviewWidget
from gui.cubemap.view_config import _BLOCK_ENABLED_VIEWS, _WARN_ENABLED_VIEWS, ViewConfigWidget
from gui.steps.base_step import (
    SETTINGS_PANE_MARGINS,
    SETTINGS_PANE_WIDTH,
    BaseStepWidget,
)
from gui.steps.cubemap_commands import (
    ColmapExportCommand,
    ColmapSfmCommand,
    CubemapConversionCommand,
    MetashapePreprocessCommand,
    SphereSfmCommand,
    SphereSfmTransformsCommand,
    build_colmap_export_cmd,
    build_colmap_sfm_commands,
    build_cubemap_conversion_cmd,
    build_metashape_preprocess_cmd,
    build_spheresfm_commands,
    build_spheresfm_transforms_cmd,
    views_config_payload,
    write_views_config,
)
from gui.steps.sfm_route_backends import get_sfm_route_backend
from gui.steps.sfm_route_selector import SfmRouteSelector
from gui.steps.sfm_route_specs import (
    SFM_ROUTE_COLMAP as _METHOD_COLMAP,
    SFM_ROUTE_METASHAPE as _METHOD_METASHAPE,
    SFM_ROUTE_SPHERESFM as _METHOD_SPHERESFM,
    normalize_sfm_route,
)
from gui.steps.step4_settings import (
    STEP4_SETTINGS_VERSION,
    load_step4_export_settings,
    write_step4_export_settings,
)
from gui.steps.training_backends import (
    CustomTrainingOptions,
    LichtFeldTrainingOptions,
    PostshotTrainingOptions,
    TrainingDataset,
    build_custom_training_cmd,
    build_lichtfeld_training_cmd,
    build_postshot_training_cmd,
    lichtfeld_auto_steps_scaler,
    lichtfeld_defaults,
    lichtfeld_output_name_stem,
)
from gui.steps.training_backend_selector import TrainingBackendSelector
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_CUSTOM as _TRAINING_BACKEND_CUSTOM,
    TRAINING_BACKEND_LICHTFELD as _TRAINING_BACKEND_LICHTFELD,
    TRAINING_BACKEND_POSTSHOT as _TRAINING_BACKEND_POSTSHOT,
    get_training_backend_spec,
    normalize_training_backend,
    training_backend_default_executable,
    training_backend_phase_name,
)
from gui.user_settings import load_user_settings_section, update_user_settings_section
from gui.version import APP_VERSION
from scene_layout import (
    STEP4_EXPORT_SETTINGS_JSON,
    STEP4_META_DIR_NAME,
    STEP4_VIEWS_CONFIG_JSON,
    step4_export_settings_path,
    step4_meta_dir,
    step4_views_config_path,
)
from scene_project import (
    append_step4_dataset_run,
    append_step4_sfm_run,
    append_step4_training_run,
    file_identity,
    scene_relative,
    utc_now_iso,
)

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
_SPHERESFM_CUDA_ARCH_ERROR_MARKERS = (
    "no kernel image is available for execution on the device",
    "cudaerrornokernelimagefordevice",
)
_COLMAP_GUI_UNAVAILABLE_MARKERS = (
    "cannot start colmap gui",
    "built without gui support",
    "qt dependency is missing",
)
_PROFILE_POSTSHOT = "postshot"
_PROFILE_BRUSH = "brush"
_PROFILE_LICHTFELD = "lichtfeld"
_PROFILE_CUSTOM = "custom"
_PIPELINE_STAGE_SFM = "sfm"
_PIPELINE_STAGE_CONVERSION = "conversion"
_PIPELINE_STAGE_TRAINING = "training"
_PIPELINE_STATUS_READY = "ready"
_PIPELINE_STATUS_WARNING = "warning"
_PIPELINE_STATUS_OFF = "off"
_OUTPUT_SHAPE_PROJECTED = "projected"
_OUTPUT_SHAPE_EQUIRECT_3DGUT = "equirect_3dgut"
_COLMAP_MAPPER_INCREMENTAL = "incremental"
_COLMAP_MAPPER_GLOBAL = "global"
_COLMAP_MAPPER_GLOMAP = "glomap"
_COLMAP_MATCHER_SEQUENTIAL = "sequential"
_COLMAP_MATCHER_EXHAUSTIVE = "exhaustive"
_SPHERESFM_MATCHER_SPATIAL = "spatial"
_SPHERESFM_QUALITY_FAST = "fast"
_SPHERESFM_QUALITY_STANDARD = "standard"
_SPHERESFM_QUALITY_QUALITY = "quality"
_SPHERESFM_RUN_FULL = "full"
_SPHERESFM_RUN_SFM_ONLY = "sfm_only"
_SPHERESFM_RUN_CONVERT_ONLY = "convert_only"
_AXIS_POSTSHOT = "postshot"
_AXIS_BRUSH = "brush"
_AXIS_NONE = "none"
_NORMAL_OUTPUT_SCALE = 2.0 / math.pi
_SUPPORTED_TRAINING_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
_GENERATED_POINTCLOUD_NAME = "pointcloud.ply"
_LFS_ADVANCED_INT_KEYS = {
    "refine_every",
    "start_refine",
    "stop_refine",
    "grow_until_iter",
    "reset_every",
    "sh_degree_interval",
    "bilateral_grid_X",
    "bilateral_grid_Y",
    "bilateral_grid_W",
    "init_num_pts",
    "pause_refine_after_reset",
    "sparsify_steps",
    "ppisp_warmup_steps",
    "ppisp_controller_activation_step",
}
_LFS_ADVANCED_LIST_KEYS = {"eval_steps", "save_steps"}
_LFS_STRATEGIES = ("mrnf", "igs+", "mcmc")
_LFS_UI_STEP_TEXT_KEYS = {"iterations"}
_LFS_UI_STEP_ADVANCED_KEYS = {
    "sh_degree_interval",
    "refine_every",
    "start_refine",
    "stop_refine",
    "reset_every",
    "grow_until_iter",
}
_LFS_ADVANCED_FIELD_WIDTHS = {
    "means_lr": 112,
    "means_lr_end": 122,
    "shs_lr": 102,
    "opacity_lr": 102,
    "scaling_lr": 102,
    "scaling_lr_end": 112,
    "rotation_lr": 102,
    "grad_threshold": 116,
    "mask_opacity_penalty_weight": 104,
    "mask_opacity_penalty_power": 104,
    "bilateral_grid_lr": 116,
    "lambda_dssim": 100,
    "init_num_pts": 108,
    "min_opacity": 108,
    "prune_opacity": 108,
    "grow_scale3d": 100,
    "grow_scale2d": 100,
    "prune_scale3d": 100,
    "prune_scale2d": 100,
    "growth_grad_threshold": 116,
    "means_noise_weight": 104,
    "bounds_percentile": 100,
    "sparsify_steps": 108,
    "ppisp_controller_activation_step": 108,
    "max_width": 92,
    "test_every": 86,
    "eval_steps": 136,
    "save_steps": 136,
}
_LFS_ADVANCED_FLOAT_FORMATS = {
    "means_lr": ".6f",
    "means_lr_end": ".8f",
    "shs_lr": ".4f",
    "opacity_lr": ".4f",
    "scaling_lr": ".4f",
    "scaling_lr_end": ".4f",
    "rotation_lr": ".4f",
    "grad_threshold": ".6f",
    "bilateral_grid_lr": ".6f",
    "lambda_dssim": ".3f",
    "opacity_reg": ".4f",
    "scale_reg": ".4f",
    "tv_loss_weight": ".1f",
    "init_opacity": ".3f",
    "init_scaling": ".3f",
    "init_extent": ".1f",
    "min_opacity": ".4f",
    "prune_opacity": ".4f",
    "grow_scale3d": ".4f",
    "grow_scale2d": ".3f",
    "prune_scale3d": ".3f",
    "prune_scale2d": ".3f",
    "growth_grad_threshold": ".5f",
    "grow_fraction": ".3f",
    "opacity_decay": ".4f",
    "scale_decay": ".4f",
    "means_noise_weight": ".1f",
    "bounds_percentile": ".2f",
    "init_rho": ".4f",
    "prune_ratio": ".3f",
    "mask_threshold": ".3f",
    "mask_opacity_penalty_weight": ".3f",
    "mask_opacity_penalty_power": ".3f",
    "ppisp_controller_lr": ".5f",
}
_EXPORT_SETTINGS_NAME = STEP4_EXPORT_SETTINGS_JSON
_COLMAP_PROJECT_MANIFEST_NAME = "stechdrive_colmap_project.json"
_SPHERESFM_PROJECT_MANIFEST_NAME = "stechdrive_spheresfm_project.json"
_COLMAP_REPOSITORY_URL = "https://github.com/colmap/colmap"
_SPHERESFM_REPOSITORY_URL = "https://github.com/json87/spheresfm"
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


def is_spheresfm_rtx50_cuda_error_line(line: str) -> bool:
    """Detect CUDA binary/device-architecture failures seen with non-sm_120 SphereSfM builds."""
    lowered = line.lower()
    if any(marker in lowered for marker in _SPHERESFM_CUDA_ARCH_ERROR_MARKERS):
        return True
    if "invalid device function" not in lowered:
        return False
    return any(marker in lowered for marker in ("cuda", "sift", "pyramidcu", "cuteximage"))


def is_colmap_gui_unavailable_output(text: str) -> bool:
    """Detect COLMAP builds that can run CLI commands but cannot launch the Qt GUI."""
    lowered = text.lower()
    return any(marker in lowered for marker in _COLMAP_GUI_UNAVAILABLE_MARKERS)


def _normalize_spheresfm_quality_preset(value: str) -> str:
    if value == _SPHERESFM_QUALITY_FAST:
        return _SPHERESFM_QUALITY_FAST
    if value in {_SPHERESFM_QUALITY_QUALITY, "robust"}:
        return _SPHERESFM_QUALITY_QUALITY
    return _SPHERESFM_QUALITY_STANDARD


def _make_external_link(text: str, url: str, tooltip: str, object_name: str) -> QLabel:
    link = QLabel(f'<a href="{url}">{text}</a>')
    link.setObjectName(object_name)
    link.setOpenExternalLinks(True)
    link.setTextInteractionFlags(Qt.TextBrowserInteraction)
    link.setToolTip(tooltip)
    return link


class CurrentPageStack(QStackedWidget):
    """Stacked widget whose size hint follows only the visible page."""

    def sizeHint(self) -> QSize:
        widget = self.currentWidget()
        return widget.sizeHint() if widget is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        widget = self.currentWidget()
        return widget.minimumSizeHint() if widget is not None else super().minimumSizeHint()


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
        self._syncing_project_settings = False
        self._syncing_spheresfm_scope_from_intent = False
        self._user_preferences_enabled = False
        self._export_method_value = _METHOD_METASHAPE
        self._conversion_intent = True
        self._spheresfm_sfm_intent_override: bool | None = None
        self._spheresfm_conversion_intent_override: bool | None = None
        self._saved_projected_export_targets: tuple[bool, bool] | None = None
        self._input_image_count = 0
        self._spheresfm_phase_logs: dict[str, Path] = {}
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

        self.spheresfm_run_scope_combo = QComboBox()
        self.spheresfm_run_scope_combo.setToolTip(i18n.tip("SPHERESFM_RUN_SCOPE"))
        self.spheresfm_run_scope_combo.addItem(i18n.t("SPHERESFM_RUN_FULL"), _SPHERESFM_RUN_FULL)
        self.spheresfm_run_scope_combo.addItem(i18n.t("SPHERESFM_RUN_SFM_ONLY"), _SPHERESFM_RUN_SFM_ONLY)
        self.spheresfm_run_scope_combo.addItem(i18n.t("SPHERESFM_RUN_CONVERT_ONLY"), _SPHERESFM_RUN_CONVERT_ONLY)
        self.spheresfm_run_scope_combo.currentIndexChanged.connect(self._on_spheresfm_run_scope_changed)
        add_tooltip_row(
            spheresfm_form,
            i18n.t("SPHERESFM_RUN_SCOPE"),
            self.spheresfm_run_scope_combo,
            i18n.tip("SPHERESFM_RUN_SCOPE"),
        )

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

        self.ms_ply_approve_btn = self.ms_ply_browse.add_icon_button(
            check_icon(),
            i18n.t("MS_PLY_APPROVE"),
            self._approve_metashape_ply,
            accessible_name=i18n.t("MS_PLY_APPROVE"),
        )
        self.ms_ply_approve_btn.setCheckable(True)
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

        self.input_tab_index = self.settings_tabs.addTab(
            self._make_tab_scroll_area(self.input_tab),
            i18n.t("STEP4_TAB_INPUT"),
        )
        self.output_tab_index = self.settings_tabs.addTab(
            self._make_tab_scroll_area(self.output_tab),
            i18n.t("STEP4_TAB_OUTPUT"),
        )
        self.training_tab_index = self.settings_tabs.addTab(
            self.training_section,
            i18n.t("STEP4_TAB_TRAINING"),
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
        self._on_colmap_run_toggled(self.run_colmap_cb.isChecked())
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

    def set_pipeline_stage(self, stage: str) -> None:
        if stage == _PIPELINE_STAGE_TRAINING:
            index = self.training_tab_index
        elif stage == _PIPELINE_STAGE_CONVERSION:
            index = self.output_tab_index
        else:
            index = self.input_tab_index
        if not self._settings_tab_available(index):
            index = self.input_tab_index
        self.settings_tabs.setCurrentIndex(index)

    def pipeline_stage(self) -> str:
        current = self.settings_tabs.currentIndex()
        if current == self.training_tab_index:
            return _PIPELINE_STAGE_TRAINING
        if current == self.output_tab_index:
            return _PIPELINE_STAGE_CONVERSION
        return _PIPELINE_STAGE_SFM

    def pipeline_stage_intent(self, stage: str) -> bool:
        if stage == _PIPELINE_STAGE_SFM:
            return self._sfm_route_backend().sfm_intent(self)
        if stage == _PIPELINE_STAGE_CONVERSION:
            return self._sfm_route_backend().conversion_intent(self)
        if stage == _PIPELINE_STAGE_TRAINING:
            return self.run_training_cb.isChecked()
        return False

    def pipeline_stage_intent_enabled(self, stage: str) -> bool:
        return stage in {_PIPELINE_STAGE_SFM, _PIPELINE_STAGE_CONVERSION, _PIPELINE_STAGE_TRAINING}

    def pipeline_stage_intent_toggle_enabled(self, stage: str) -> bool:
        if stage == _PIPELINE_STAGE_SFM:
            return self._sfm_route_backend().sfm_intent_toggle_enabled(self)
        return self.pipeline_stage_intent_enabled(stage)

    def _pipeline_stage_runs_in_app(self, stage: str) -> bool:
        if stage == _PIPELINE_STAGE_SFM:
            return self._sfm_route_backend().sfm_runs_in_app(self)
        return self.pipeline_stage_intent(stage)

    def toggle_pipeline_stage_intent(self, stage: str) -> None:
        self.set_pipeline_stage_intent(stage, not self.pipeline_stage_intent(stage))

    def set_pipeline_stage_intent(self, stage: str, enabled: bool) -> None:
        enabled = bool(enabled)
        if stage == _PIPELINE_STAGE_SFM:
            self._sfm_route_backend().set_sfm_intent(self, enabled)
        elif stage == _PIPELINE_STAGE_CONVERSION:
            self._sfm_route_backend().set_conversion_intent(self, enabled)
        elif stage == _PIPELINE_STAGE_TRAINING:
            self.run_training_cb.setChecked(enabled)
        self._sync_settings_tabs()
        self._update_path_labels()
        self._update_training_paths()
        self._update_output_count()
        self.primary_action_state_changed.emit()

    def _set_spheresfm_stage_intents(self, *, run_sfm: bool, run_conversion: bool) -> None:
        self._spheresfm_sfm_intent_override = bool(run_sfm)
        self._spheresfm_conversion_intent_override = bool(run_conversion)
        if run_sfm and run_conversion:
            scope = _SPHERESFM_RUN_FULL
        elif run_sfm:
            scope = _SPHERESFM_RUN_SFM_ONLY
        else:
            scope = _SPHERESFM_RUN_CONVERT_ONLY
        self._syncing_spheresfm_scope_from_intent = True
        try:
            self._set_combo_data(self.spheresfm_run_scope_combo, scope)
        finally:
            self._syncing_spheresfm_scope_from_intent = False

    def pipeline_nav_items(self) -> list[dict[str, object]]:
        items: list[tuple[str, str, tuple[str, str, str]]] = [
            (_PIPELINE_STAGE_SFM, i18n.t("STEP4_PIPELINE_SFM"), self._pipeline_sfm_status()),
            (
                _PIPELINE_STAGE_CONVERSION,
                i18n.t("STEP4_PIPELINE_CONVERSION"),
                self._pipeline_conversion_status(),
            ),
            (
                _PIPELINE_STAGE_TRAINING,
                i18n.t("STEP4_PIPELINE_TRAINING"),
                self._pipeline_training_status(),
            ),
        ]
        result: list[dict[str, object]] = []
        for stage, label, (status, symbol, detail) in items:
            status_text = i18n.t(f"STEP4_PIPELINE_STATUS_{status.upper()}")
            intent = self.pipeline_stage_intent(stage)
            intent_enabled = self.pipeline_stage_intent_enabled(stage)
            intent_toggle_enabled = self.pipeline_stage_intent_toggle_enabled(stage)
            intent_key = "STEP4_PIPELINE_INTENT_ON" if intent else "STEP4_PIPELINE_INTENT_OFF"
            if stage == _PIPELINE_STAGE_SFM and self._is_metashape_method():
                intent_tooltip = i18n.t("STEP4_PIPELINE_INTENT_METASHAPE_INPUT")
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            elif intent and not intent_enabled:
                intent_tooltip = i18n.t("STEP4_PIPELINE_INTENT_LOCKED_ON").format(stage=label)
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            elif not intent_enabled:
                intent_tooltip = i18n.t("STEP4_PIPELINE_INTENT_DISABLED").format(stage=label)
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            else:
                intent_tooltip = i18n.t(intent_key).format(stage=label)
                intent_symbol = "●" if intent else "○"
                intent_checked = intent
            result.append(
                {
                    "stage": stage,
                    "label": label,
                    "status": status,
                    "status_symbol": symbol,
                    "status_tooltip": f"{label}: {status_text}\n{detail}",
                    "intent_checked": intent_checked,
                    "intent_enabled": intent_enabled,
                    "intent_toggle_enabled": intent_toggle_enabled,
                    "intent_symbol": intent_symbol,
                    "intent_tooltip": intent_tooltip,
                    "navigate_tooltip": i18n.t("STEP4_PIPELINE_NAVIGATE").format(stage=label),
                }
            )
        return result

    def _pipeline_sfm_status(self) -> tuple[str, str, str]:
        if not self.pipeline_stage_intent(_PIPELINE_STAGE_SFM):
            if self._is_metashape_method():
                return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_OFF"))
            if self._is_colmap_method():
                return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_COLMAP_OFF"))
            return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_SPHERESFM_OFF"))
        if not self.scene_dir:
            return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_SCENE_REQUIRED"))
        if self._is_metashape_method():
            missing = self._metashape_input_missing_detail()
            if missing is not None:
                return (_PIPELINE_STATUS_WARNING, "!", missing)
            return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_READY"))
        if self._is_colmap_method():
            if self.run_colmap_cb.isChecked():
                if not self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION) and not self._colmap_rig_images_dir().is_dir():
                    return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_COLMAP_NEEDS_RIG"))
                return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_COLMAP_RUNS"))
        return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_SPHERESFM_RUNS"))

    def _pipeline_conversion_status(self) -> tuple[str, str, str]:
        if not self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION):
            return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_CONVERSION_OFF"))
        if not self.scene_dir:
            return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_SCENE_REQUIRED"))
        if self._is_spheresfm_method() and self._spheresfm_runs_sfm():
            return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_CONVERSION_AFTER_SFM"))
        if self._is_spheresfm_method() and self._find_spheresfm_sparse_model() is None:
            return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_SPHERESFM_NEEDS_SPARSE"))
        if self._is_metashape_method():
            missing = self._metashape_input_missing_detail()
            if missing is not None:
                return (_PIPELINE_STATUS_WARNING, "!", missing)
        return (_PIPELINE_STATUS_READY, "✓", i18n.t("STEP4_PIPELINE_DETAIL_CONVERSION_RUNS"))

    def _metashape_input_missing_detail(self) -> str | None:
        if not self.scene_dir:
            return i18n.t("STEP4_PIPELINE_DETAIL_SCENE_REQUIRED")
        self._refresh_metashape_auto_inputs_if_empty()
        self._update_metashape_input_hint()
        xml_text = self.ms_xml_browse.text().strip()
        if not xml_text:
            return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_XML")
        xml = Path(xml_text)
        if self._metashape_input_output_path_issue(xml):
            return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_INPUT_IN_OUTPUT")
        if not xml.is_file():
            return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_XML")
        if self._preprocess_uses_ply():
            ply_text = self.ms_ply_browse.text().strip()
            if not ply_text:
                return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_PLY")
            ply = Path(ply_text)
            if self._metashape_input_output_path_issue(ply):
                return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_INPUT_IN_OUTPUT")
            if not ply.is_file():
                return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_PLY")
            if not self._metashape_ply_approved:
                return i18n.t("STEP4_PIPELINE_DETAIL_METASHAPE_NEEDS_PLY_APPROVAL")
        return None

    def _pipeline_training_status(self) -> tuple[str, str, str]:
        if not self.run_training_cb.isChecked():
            return (_PIPELINE_STATUS_OFF, "-", i18n.t("STEP4_PIPELINE_DETAIL_TRAINING_OFF"))
        if not self.scene_dir:
            return (_PIPELINE_STATUS_WARNING, "!", i18n.t("STEP4_PIPELINE_DETAIL_SCENE_REQUIRED"))
        issue = self._training_dataset_issue()
        if issue is not None:
            return (_PIPELINE_STATUS_WARNING, "!", issue)
        if self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION):
            backend = self._training_backend_display_name(self._training_backend()) if hasattr(self, "training_backend_buttons") else ""
            return (
                _PIPELINE_STATUS_READY,
                "✓",
                i18n.t("STEP4_PIPELINE_DETAIL_TRAINING_RUNS").format(backend=backend),
            )
        backend = self._training_backend_display_name(self._training_backend()) if hasattr(self, "training_backend_buttons") else ""
        return (
            _PIPELINE_STATUS_READY,
            "✓",
            i18n.t("STEP4_PIPELINE_DETAIL_TRAINING_RUNS").format(backend=backend),
        )

    def _training_required_output_shape(self) -> str:
        backend = self._training_backend()
        if backend == _TRAINING_BACKEND_LICHTFELD:
            return _OUTPUT_SHAPE_EQUIRECT_3DGUT if self.lfs_gut_cb.isChecked() else _OUTPUT_SHAPE_PROJECTED
        if backend == _TRAINING_BACKEND_POSTSHOT:
            return _OUTPUT_SHAPE_PROJECTED
        return ""

    def _training_dataset_issue(self) -> str | None:
        required_shape = self._training_required_output_shape()
        if self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION):
            return self._planned_training_dataset_issue(required_shape)
        return self._existing_training_dataset_issue(required_shape)

    def _planned_training_dataset_issue(self, required_shape: str) -> str | None:
        if not required_shape:
            return None
        planned_shape = self._output_shape()
        if required_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            if planned_shape != _OUTPUT_SHAPE_EQUIRECT_3DGUT:
                return i18n.t("TRAINING_DATASET_NEEDS_3DGUT_OUTPUT")
            return None
        if planned_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            return i18n.t("TRAINING_DATASET_NEEDS_PROJECTED_OUTPUT")
        return None

    def _existing_training_dataset_issue(self, required_shape: str) -> str | None:
        if not self._training_dataset_available():
            return i18n.t("TRAINING_REQUIRES_DATASET_OUTPUT")
        if not required_shape:
            return None
        dataset = self._training_dataset()
        actual_shape = self._training_dataset_export_shape(dataset.dataset_root)
        if required_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            if actual_shape == _OUTPUT_SHAPE_PROJECTED:
                return i18n.t("TRAINING_DATASET_EXISTING_NOT_3DGUT")
            if not (dataset.dataset_root / "pointcloud.ply").is_file():
                return i18n.t("TRAINING_DATASET_3DGUT_NEEDS_PLY")
            return None
        if actual_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            return i18n.t("TRAINING_DATASET_EXISTING_NOT_PROJECTED")
        return None

    def _training_dataset_export_shape(self, dataset_root: Path) -> str:
        if not self.scene_dir:
            return ""
        try:
            if dataset_root.resolve() != self._output_dir().resolve():
                return ""
        except OSError:
            return ""
        settings = load_step4_export_settings(Path(self.scene_dir))
        if not settings:
            return ""
        shape = str(settings.get("output_shape", "")).strip()
        return shape if shape in {_OUTPUT_SHAPE_PROJECTED, _OUTPUT_SHAPE_EQUIRECT_3DGUT} else ""

    def _build_training_section(self, exe_filter: str) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.run_training_cb = QCheckBox(i18n.t("RUN_TRAINING_AFTER_EXPORT"))
        self.run_training_cb.setToolTip(i18n.tip("RUN_TRAINING_AFTER_EXPORT"))
        self.run_training_cb.toggled.connect(self._on_training_settings_changed)
        self.training_headless_cb = QCheckBox(i18n.t("TRAINING_HEADLESS"))
        self.training_headless_cb.setToolTip(i18n.tip("TRAINING_HEADLESS"))
        self.training_headless_cb.toggled.connect(self._on_training_settings_changed)

        self.training_backend_selector = TrainingBackendSelector()
        self.training_backend_selector.backend_changed.connect(self._set_training_backend)
        self.training_backend_row = self.training_backend_selector
        self.training_backend_label = self.training_backend_selector.label
        self.training_backend_buttons = self.training_backend_selector.backend_buttons
        self.training_backend_other_button = self.training_backend_selector.other_button
        self.training_backend_other_menu_button = self.training_backend_selector.other_menu_button
        self.training_backend_other_actions = self.training_backend_selector.other_backend_actions
        layout.addWidget(self.training_backend_row)

        self.training_run_options_row = QWidget()
        run_options_layout = QHBoxLayout(self.training_run_options_row)
        run_options_layout.setContentsMargins(0, 0, 0, 0)
        run_options_layout.setSpacing(12)
        run_options_layout.addWidget(self.training_headless_cb)
        run_options_layout.addStretch()
        layout.addWidget(self.training_run_options_row)

        self.training_settings_content = QWidget()
        training_settings_layout = QVBoxLayout(self.training_settings_content)
        training_settings_layout.setContentsMargins(0, 0, 0, 0)
        training_settings_layout.setSpacing(6)

        self.training_settings_scroll = QScrollArea()
        self.training_settings_scroll.setObjectName("trainingSettingsScroll")
        self.training_settings_scroll.setWidgetResizable(True)
        self.training_settings_scroll.setFrameShape(QScrollArea.NoFrame)
        self.training_settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.training_settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.training_settings_scroll.setWidget(self.training_settings_content)

        form = QFormLayout()
        form.setSpacing(6)

        self.training_executable_browse = BrowseWidget(
            mode="file",
            filter_str=exe_filter,
            placeholder=self._default_training_executable(_TRAINING_BACKEND_LICHTFELD),
        )
        self.training_executable_browse.setToolTip(i18n.tip("TRAINING_EXECUTABLE"))
        add_tooltip_row(
            form,
            i18n.t("TRAINING_EXECUTABLE"),
            self.training_executable_browse,
            i18n.tip("TRAINING_EXECUTABLE"),
        )

        self.training_dataset_browse = BrowseWidget(mode="dir")
        self.training_dataset_browse.setToolTip(i18n.tip("TRAINING_DATASET"))
        self._syncing_training_paths = False
        self.training_dataset_browse.path_changed.connect(self._on_training_dataset_edited)
        add_tooltip_row(
            form,
            i18n.t("TRAINING_DATASET"),
            self.training_dataset_browse,
            i18n.tip("TRAINING_DATASET"),
        )

        self.training_output_browse = BrowseWidget(mode="dir")
        self.training_output_browse.setToolTip(i18n.tip("TRAINING_OUTPUT"))
        self.training_output_browse.path_changed.connect(self._on_training_output_edited)
        add_tooltip_row(
            form,
            i18n.t("TRAINING_OUTPUT"),
            self.training_output_browse,
            i18n.tip("TRAINING_OUTPUT"),
        )

        training_settings_layout.addLayout(form)

        self.training_options_stack = CurrentPageStack()
        self.lichtfeld_training_options = self._build_lichtfeld_training_options()
        self.postshot_training_options = self._build_postshot_training_options()
        self.custom_training_options = self._build_custom_training_options()
        self.training_option_widgets = {
            _TRAINING_BACKEND_LICHTFELD: self.lichtfeld_training_options,
            _TRAINING_BACKEND_POSTSHOT: self.postshot_training_options,
            _TRAINING_BACKEND_CUSTOM: self.custom_training_options,
        }
        self.training_options_stack_indices: dict[str, int] = {}
        for backend, widget in sorted(
            self.training_option_widgets.items(),
            key=lambda item: get_training_backend_spec(item[0]).stack_order,
        ):
            index = self.training_options_stack.addWidget(widget)
            self.training_options_stack_indices[backend] = (
                index if isinstance(index, int) else self.training_options_stack.indexOf(widget)
            )
        self.training_options_stack.currentChanged.connect(lambda _index: self._refresh_training_settings_layout())
        training_settings_layout.addWidget(self.training_options_stack)
        training_settings_layout.addStretch()
        layout.addWidget(self.training_settings_scroll, stretch=1)

        self._training_dataset_user_edited = False
        self._set_training_backend(_TRAINING_BACKEND_LICHTFELD)
        return section

    def _refresh_training_settings_layout(self) -> None:
        self.training_options_stack.updateGeometry()
        self.training_settings_content.updateGeometry()
        self.training_settings_content.adjustSize()

    def _build_lichtfeld_training_options(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(6)
        self.lfs_options_form = form
        self.lfs_advanced_edits: dict[str, QLineEdit] = {}
        self.lfs_advanced_checks: dict[str, QCheckBox] = {}
        self.lfs_advanced_sections: dict[str, CollapsibleSection] = {}
        self.lfs_advanced_rows: dict[str, tuple[QFormLayout, int]] = {}

        self.lfs_strategy_combo = QComboBox()
        self.lfs_strategy_combo.addItem("MRNF", "mrnf")
        self.lfs_strategy_combo.addItem("IGS+", "igs+")
        self.lfs_strategy_combo.addItem("MCMC", "mcmc")
        add_tooltip_row(form, i18n.t("LFS_STRATEGY"), self.lfs_strategy_combo, i18n.tip("LFS_STRATEGY"))

        self.lfs_iterations_edit = QLineEdit("30000")
        self.lfs_iterations_edit.setFixedWidth(96)
        self.lfs_iterations_edit.setToolTip(i18n.tip("LFS_ITERATIONS"))
        add_tooltip_row(form, i18n.t("LFS_ITERATIONS"), self.lfs_iterations_edit, i18n.tip("LFS_ITERATIONS"))

        self.lfs_max_gaussians_edit = QLineEdit("5000000")
        self.lfs_max_gaussians_edit.setFixedWidth(112)
        self.lfs_max_gaussians_edit.setToolTip(i18n.tip("LFS_MAX_GAUSSIANS"))
        add_tooltip_row(
            form,
            i18n.t("LFS_MAX_GAUSSIANS"),
            self.lfs_max_gaussians_edit,
            i18n.tip("LFS_MAX_GAUSSIANS"),
        )

        self.lfs_output_name_edit = QLineEdit()
        self.lfs_output_name_edit.setToolTip(i18n.tip("LFS_OUTPUT_PLY_NAME"))
        self.lfs_output_name_edit.textEdited.connect(self._on_lfs_output_name_edited)
        add_tooltip_row(
            form,
            i18n.t("LFS_OUTPUT_PLY_NAME"),
            self.lfs_output_name_edit,
            i18n.tip("LFS_OUTPUT_PLY_NAME"),
        )

        self.lfs_sh_degree_combo = QComboBox()
        for degree in range(4):
            self.lfs_sh_degree_combo.addItem(str(degree), degree)
        self.lfs_sh_degree_combo.setCurrentIndex(3)
        add_tooltip_row(form, i18n.t("LFS_SH_DEGREE"), self.lfs_sh_degree_combo, i18n.tip("LFS_SH_DEGREE"))

        self.lfs_tile_mode_combo = QComboBox()
        self.lfs_tile_mode_combo.addItem("1 (Full)", 1)
        self.lfs_tile_mode_combo.addItem("2", 2)
        self.lfs_tile_mode_combo.addItem("4", 4)
        self.lfs_tile_mode_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_TILE_MODE"), self.lfs_tile_mode_combo, i18n.tip("LFS_TILE_MODE"))

        self.lfs_steps_scaler_edit = QLineEdit("1.0")
        self.lfs_steps_scaler_edit.setFixedWidth(72)
        self.lfs_steps_scaler_edit.setToolTip(i18n.tip("LFS_STEPS_SCALER"))
        self.lfs_auto_steps_scaler_cb = QCheckBox(i18n.t("AUTO"))
        self.lfs_auto_steps_scaler_cb.setChecked(True)
        self.lfs_auto_steps_scaler_cb.setToolTip(i18n.tip("LFS_STEPS_SCALER_AUTO"))
        self.lfs_auto_steps_scaler_cb.toggled.connect(self._on_lfs_auto_steps_scaler_changed)
        self.lfs_iterations_edit.textEdited.connect(self._on_lfs_iterations_edited)
        self.lfs_steps_scaler_edit.setEnabled(False)
        scaler_row = QWidget()
        scaler_layout = QHBoxLayout(scaler_row)
        scaler_layout.setContentsMargins(0, 0, 0, 0)
        scaler_layout.setSpacing(6)
        scaler_layout.addWidget(self.lfs_steps_scaler_edit)
        scaler_layout.addWidget(self.lfs_auto_steps_scaler_cb)
        scaler_layout.addStretch()
        add_tooltip_row(
            form,
            i18n.t("LFS_STEPS_SCALER"),
            scaler_row,
            i18n.tip("LFS_STEPS_SCALER"),
        )

        self.lfs_bilateral_grid_cb = QCheckBox()
        self.lfs_bilateral_grid_cb.setToolTip(i18n.tip("LFS_BILATERAL_GRID"))
        add_tooltip_row(form, i18n.t("LFS_BILATERAL_GRID"), self.lfs_bilateral_grid_cb, i18n.tip("LFS_BILATERAL_GRID"))

        self.lfs_mask_mode_combo = QComboBox()
        self.lfs_mask_mode_combo.addItem("None", "none")
        self.lfs_mask_mode_combo.addItem("Segment", "segment")
        self.lfs_mask_mode_combo.addItem("Ignore", "ignore")
        self.lfs_mask_mode_combo.addItem("Alpha Consistent", "alpha_consistent")
        add_tooltip_row(form, i18n.t("LFS_MASK_MODE"), self.lfs_mask_mode_combo, i18n.tip("LFS_MASK_MODE"))

        self.lfs_mask_invert_row = form.rowCount()
        self.lfs_invert_masks_cb = QCheckBox()
        add_tooltip_row(form, i18n.t("LFS_INVERT_MASKS"), self.lfs_invert_masks_cb, i18n.tip("LFS_INVERT_MASKS"))
        self.lfs_mask_threshold_edit = QLineEdit("0.500")
        self.lfs_mask_threshold_edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get("mask_threshold", 86))
        self.lfs_mask_threshold_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_MASK_THRESHOLD"), self.lfs_mask_threshold_edit)
        self.lfs_use_alpha_as_mask_cb = QCheckBox()
        self.lfs_use_alpha_as_mask_cb.setChecked(True)
        self.lfs_use_alpha_as_mask_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_USE_ALPHA_AS_MASK"), self.lfs_use_alpha_as_mask_cb)
        self.lfs_mask_opacity_penalty_weight_edit = QLineEdit("1.000")
        self.lfs_mask_opacity_penalty_weight_edit.setFixedWidth(
            _LFS_ADVANCED_FIELD_WIDTHS.get("mask_opacity_penalty_weight", 86)
        )
        self.lfs_mask_opacity_penalty_weight_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_MASK_OPACITY_PENALTY_WEIGHT"),
            self.lfs_mask_opacity_penalty_weight_edit,
        )
        self.lfs_mask_opacity_penalty_power_edit = QLineEdit("2.000")
        self.lfs_mask_opacity_penalty_power_edit.setFixedWidth(
            _LFS_ADVANCED_FIELD_WIDTHS.get("mask_opacity_penalty_power", 86)
        )
        self.lfs_mask_opacity_penalty_power_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_MASK_OPACITY_PENALTY_POWER"),
            self.lfs_mask_opacity_penalty_power_edit,
        )

        checks = QWidget()
        checks_layout = QGridLayout(checks)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        checks_layout.setHorizontalSpacing(10)
        checks_layout.setVerticalSpacing(3)
        self.lfs_sparsity_cb = QCheckBox(i18n.t("LFS_SPARSITY"))
        self.lfs_gut_cb = QCheckBox(i18n.t("LFS_GUT"))
        self.lfs_undistort_cb = QCheckBox(i18n.t("LFS_UNDISTORT"))
        self.lfs_mip_filter_cb = QCheckBox(i18n.t("LFS_MIP_FILTER"))
        self.lfs_ppisp_cb = QCheckBox(i18n.t("LFS_PPISP"))
        for index, cb in enumerate(
            (
                self.lfs_sparsity_cb,
                self.lfs_gut_cb,
                self.lfs_undistort_cb,
                self.lfs_mip_filter_cb,
                self.lfs_ppisp_cb,
            )
        ):
            checks_layout.addWidget(cb, index // 2, index % 2)
        form.addRow("", checks)

        self.lfs_ppisp_freeze_from_sidecar_cb = QCheckBox()
        self.lfs_ppisp_freeze_from_sidecar_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_PPISP_FREEZE_FROM_SIDECAR"), self.lfs_ppisp_freeze_from_sidecar_cb)
        self.lfs_ppisp_sidecar_browse = BrowseWidget(mode="file")
        self.lfs_ppisp_sidecar_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_PPISP_SIDECAR_PATH"), self.lfs_ppisp_sidecar_browse)
        self.lfs_ppisp_use_controller_cb = QCheckBox()
        self.lfs_ppisp_use_controller_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_PPISP_USE_CONTROLLER"), self.lfs_ppisp_use_controller_cb)
        self.lfs_ppisp_controller_activation_step_edit = QLineEdit("-1")
        self.lfs_ppisp_controller_activation_step_edit.setFixedWidth(
            _LFS_ADVANCED_FIELD_WIDTHS.get("ppisp_controller_activation_step", 86)
        )
        self.lfs_ppisp_controller_activation_step_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_CONTROLLER_ACTIVATION_STEP"),
            self.lfs_ppisp_controller_activation_step_edit,
        )
        self.lfs_ppisp_controller_lr_edit = QLineEdit("0.0020")
        self.lfs_ppisp_controller_lr_edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get("ppisp_controller_lr", 86))
        self.lfs_ppisp_controller_lr_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_PPISP_CONTROLLER_LR"), self.lfs_ppisp_controller_lr_edit)
        self.lfs_ppisp_freeze_gaussians_on_distill_cb = QCheckBox()
        self.lfs_ppisp_freeze_gaussians_on_distill_cb.setChecked(True)
        self.lfs_ppisp_freeze_gaussians_on_distill_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_FREEZE_GAUSSIANS_ON_DISTILL"),
            self.lfs_ppisp_freeze_gaussians_on_distill_cb,
        )

        self.lfs_bg_mode_combo = QComboBox()
        self.lfs_bg_mode_combo.addItem("Color", "solid_color")
        self.lfs_bg_mode_combo.addItem("Modulation", "modulation")
        self.lfs_bg_mode_combo.addItem("Image", "image")
        self.lfs_bg_mode_combo.addItem("Random", "random")
        add_tooltip_row(form, i18n.t("LFS_BG_MODE"), self.lfs_bg_mode_combo, i18n.tip("LFS_BG_MODE"))

        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(4)
        self.lfs_bg_r_edit = QLineEdit("0")
        self.lfs_bg_g_edit = QLineEdit("0")
        self.lfs_bg_b_edit = QLineEdit("0")
        for label, edit in (("R:", self.lfs_bg_r_edit), ("G:", self.lfs_bg_g_edit), ("B:", self.lfs_bg_b_edit)):
            color_layout.addWidget(QLabel(label))
            edit.setFixedWidth(42)
            edit.textChanged.connect(lambda _text: self._update_lfs_color_swatch())
            color_layout.addWidget(edit)
        self.lfs_bg_color_swatch = QLabel()
        self.lfs_bg_color_swatch.setFixedSize(22, 18)
        self.lfs_bg_color_swatch.setFrameShape(QLabel.Box)
        color_layout.addWidget(self.lfs_bg_color_swatch)
        color_layout.addStretch()
        self.lfs_bg_color_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_BG_COLOR"), color_row, i18n.tip("LFS_BG_COLOR"))

        self.lfs_bg_image_browse = BrowseWidget(mode="file")
        self.lfs_bg_image_browse.setToolTip(i18n.tip("LFS_BG_IMAGE"))
        self.lfs_bg_image_row = form.rowCount()
        add_tooltip_row(form, i18n.t("LFS_BG_IMAGE"), self.lfs_bg_image_browse, i18n.tip("LFS_BG_IMAGE"))

        advanced = CollapsibleSection(i18n.t("LFS_ADVANCED_PARAMETERS"), expanded=False)
        adv_layout = advanced.content_layout

        def add_section(title_key: str) -> QFormLayout:
            section = CollapsibleSection(i18n.t(title_key), expanded=False)
            section.setObjectName("lfsAdvancedSubsection")
            self.lfs_advanced_sections[title_key] = section
            adv_layout.addWidget(section)
            section_form = QFormLayout()
            section_form.setSpacing(5)
            section.content_layout.addLayout(section_form)
            return section_form

        def add_edit(section_form: QFormLayout, key: str, default: str, width: int = 86) -> QLineEdit:
            edit = QLineEdit(default)
            edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get(key, width))
            self.lfs_advanced_edits[key] = edit
            self.lfs_advanced_rows[key] = (section_form, section_form.rowCount())
            add_tooltip_row(section_form, i18n.t(f"LFS_{key.upper()}"), edit)
            return edit

        def add_check(section_form: QFormLayout, key: str, checked: bool = False) -> QCheckBox:
            cb = QCheckBox()
            cb.setChecked(checked)
            self.lfs_advanced_checks[key] = cb
            self.lfs_advanced_rows[key] = (section_form, section_form.rowCount())
            add_tooltip_row(section_form, i18n.t(f"LFS_{key.upper()}"), cb)
            return cb

        dataset_form = add_section("LFS_SECTION_DATASET")
        self.lfs_dataset_resize_factor_combo = QComboBox()
        self.lfs_dataset_resize_factor_combo.addItem("Auto", "auto")
        for factor in (1, 2, 4, 8):
            self.lfs_dataset_resize_factor_combo.addItem(str(factor), str(factor))
        add_tooltip_row(dataset_form, i18n.t("LFS_RESIZE_FACTOR"), self.lfs_dataset_resize_factor_combo)
        self.lfs_dataset_max_width_edit = QLineEdit("3840")
        self.lfs_dataset_max_width_edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get("max_width", 86))
        add_tooltip_row(dataset_form, i18n.t("LFS_MAX_WIDTH"), self.lfs_dataset_max_width_edit)
        self.lfs_dataset_cpu_cache_cb = QCheckBox()
        self.lfs_dataset_cpu_cache_cb.setChecked(True)
        add_tooltip_row(dataset_form, i18n.t("LFS_CPU_CACHE"), self.lfs_dataset_cpu_cache_cb)
        self.lfs_dataset_fs_cache_cb = QCheckBox()
        self.lfs_dataset_fs_cache_cb.setChecked(True)
        add_tooltip_row(dataset_form, i18n.t("LFS_FS_CACHE"), self.lfs_dataset_fs_cache_cb)
        add_check(dataset_form, "enable_eval", False)
        self.lfs_dataset_test_every_edit = QLineEdit("8")
        self.lfs_dataset_test_every_edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get("test_every", 86))
        self.lfs_dataset_test_every_row = dataset_form.rowCount()
        self.lfs_advanced_rows["test_every"] = (dataset_form, self.lfs_dataset_test_every_row)
        add_tooltip_row(dataset_form, i18n.t("LFS_TEST_EVERY"), self.lfs_dataset_test_every_edit)
        add_check(dataset_form, "enable_save_eval_images", True)

        opt_form = add_section("LFS_SECTION_OPTIMIZATION")
        for key, default in (
            ("means_lr", "0.000020"),
            ("means_lr_end", "0.0000002"),
            ("shs_lr", "0.0020"),
            ("opacity_lr", "0.0120"),
            ("scaling_lr", "0.0070"),
            ("scaling_lr_end", "0.0050"),
            ("rotation_lr", "0.0020"),
        ):
            add_edit(opt_form, key, default)
        for key, default in (
            ("refine_every", "200"),
            ("start_refine", "0"),
            ("stop_refine", "28500"),
            ("grow_until_iter", "15000"),
            ("grad_threshold", "0.003000"),
            ("reset_every", "3000"),
            ("sh_degree_interval", "1000"),
        ):
            add_edit(opt_form, key, default)

        bilateral_form = add_section("LFS_SECTION_BILATERAL")
        for key, default in (
            ("bilateral_grid_X", "16"),
            ("bilateral_grid_Y", "16"),
            ("bilateral_grid_W", "8"),
            ("bilateral_grid_lr", "0.002000"),
        ):
            add_edit(bilateral_form, key, default)

        loss_form = add_section("LFS_SECTION_LOSSES")
        for key, default in (
            ("lambda_dssim", "0.200"),
            ("opacity_reg", "0.0000"),
            ("scale_reg", "0.0000"),
            ("tv_loss_weight", "10.0"),
        ):
            add_edit(loss_form, key, default)

        init_form = add_section("LFS_SECTION_INITIALIZATION")
        add_edit(init_form, "init_opacity", "0.500")
        add_edit(init_form, "init_scaling", "0.100")
        add_check(init_form, "random", False)
        add_edit(init_form, "init_num_pts", "100000")
        add_edit(init_form, "init_extent", "3.0")

        pruning_form = add_section("LFS_SECTION_PRUNING_GROWING")
        for key, default in (
            ("min_opacity", "0.003922"),
            ("prune_opacity", "0.0050"),
            ("grow_scale3d", "0.0100"),
            ("grow_scale2d", "0.050"),
            ("prune_scale3d", "0.100"),
            ("prune_scale2d", "0.150"),
            ("pause_refine_after_reset", "0"),
        ):
            add_edit(pruning_form, key, default)
        add_check(pruning_form, "revised_opacity", True)

        mrnf_form = add_section("LFS_SECTION_MRNF")
        for key, default in (
            ("growth_grad_threshold", "0.00300"),
            ("grow_fraction", "0.070"),
            ("opacity_decay", "0.0040"),
            ("scale_decay", "0.0020"),
            ("means_noise_weight", "50.0"),
            ("bounds_percentile", "0.80"),
        ):
            add_edit(mrnf_form, key, default)
        add_check(mrnf_form, "use_error_map", True)
        add_check(mrnf_form, "use_edge_map", True)

        sparsity_form = add_section("LFS_SECTION_SPARSITY")
        add_edit(sparsity_form, "sparsify_steps", "15000")
        add_edit(sparsity_form, "init_rho", "0.0005")
        add_edit(sparsity_form, "prune_ratio", "0.600")

        save_form = add_section("LFS_SECTION_SAVE_EVAL")
        add_edit(save_form, "eval_steps", "7000,30000", width=120)
        add_edit(save_form, "save_steps", "7000,30000", width=120)

        form.addRow(advanced)
        self._syncing_lfs_strategy_state = False
        self._lfs_active_strategy = "mrnf"
        self._lfs_strategy_states = {
            strategy: self._default_lfs_ui_state(strategy) for strategy in _LFS_STRATEGIES
        }
        self._apply_lfs_ui_state(self._lfs_strategy_states[self._lfs_active_strategy])
        self._update_lfs_color_swatch()
        self.lfs_strategy_combo.currentIndexChanged.connect(self._on_lfs_strategy_combo_changed)
        self.lfs_steps_scaler_edit.editingFinished.connect(self._on_lfs_steps_scaler_editing_finished)
        self.lfs_bilateral_grid_cb.toggled.connect(lambda _checked: self._update_lfs_conditional_visibility())
        self.lfs_mask_mode_combo.currentIndexChanged.connect(lambda _idx: self._update_lfs_conditional_visibility())
        self.lfs_sparsity_cb.toggled.connect(lambda _checked: self._update_lfs_conditional_visibility())
        self.lfs_gut_cb.toggled.connect(self._on_lfs_gut_changed)
        self.lfs_ppisp_cb.toggled.connect(lambda _checked: self._update_lfs_conditional_visibility())
        self.lfs_ppisp_freeze_from_sidecar_cb.toggled.connect(
            lambda _checked: self._update_lfs_conditional_visibility()
        )
        self.lfs_ppisp_use_controller_cb.toggled.connect(lambda _checked: self._update_lfs_conditional_visibility())
        self.lfs_bg_mode_combo.currentIndexChanged.connect(lambda _idx: self._update_lfs_conditional_visibility())
        self.lfs_advanced_checks["random"].toggled.connect(lambda _checked: self._update_lfs_conditional_visibility())
        self.lfs_advanced_checks["enable_eval"].toggled.connect(
            lambda _checked: self._update_lfs_conditional_visibility()
        )
        self._update_lfs_conditional_visibility()
        return widget

    def _build_postshot_training_options(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(6)
        self.postshot_options_form = form
        self.postshot_advanced_sections: dict[str, CollapsibleSection] = {}
        self.postshot_advanced_rows: dict[str, tuple[QFormLayout, int]] = {}

        self.postshot_project_name_edit = QLineEdit("postshot.psht")
        self.postshot_project_name_edit.setToolTip(i18n.tip("POSTSHOT_PROJECT_NAME"))
        self.postshot_project_name_edit.textEdited.connect(self._on_postshot_project_name_edited)
        add_tooltip_row(
            form,
            i18n.t("POSTSHOT_PROJECT_NAME"),
            self.postshot_project_name_edit,
            i18n.tip("POSTSHOT_PROJECT_NAME"),
        )

        self.postshot_profile_combo = QComboBox()
        self.postshot_profile_combo.addItem("Splat3", "Splat3")
        self.postshot_profile_combo.addItem("Splat MCMC", "Splat MCMC")
        self.postshot_profile_combo.addItem("Splat ADC", "Splat ADC")
        add_tooltip_row(form, i18n.t("POSTSHOT_PROFILE"), self.postshot_profile_combo, i18n.tip("POSTSHOT_PROFILE"))

        self.postshot_ksteps_edit = QLineEdit("60")
        self.postshot_ksteps_edit.setFixedWidth(72)
        self.postshot_ksteps_edit.setToolTip(i18n.tip("POSTSHOT_KSTEPS"))
        self.postshot_ksteps_auto_cb = QCheckBox(i18n.t("AUTO"))
        self.postshot_ksteps_auto_cb.setChecked(True)
        self.postshot_ksteps_auto_cb.setToolTip(i18n.tip("POSTSHOT_KSTEPS_AUTO"))
        self.postshot_ksteps_edit.setEnabled(False)
        ksteps_row = QWidget()
        ksteps_layout = QHBoxLayout(ksteps_row)
        ksteps_layout.setContentsMargins(0, 0, 0, 0)
        ksteps_layout.setSpacing(6)
        ksteps_layout.addWidget(self.postshot_ksteps_edit)
        ksteps_layout.addWidget(self.postshot_ksteps_auto_cb)
        ksteps_layout.addStretch()
        add_tooltip_row(form, i18n.t("POSTSHOT_KSTEPS"), ksteps_row, i18n.tip("POSTSHOT_KSTEPS"))

        self.postshot_max_image_size_edit = QLineEdit("3840")
        self.postshot_max_image_size_edit.setFixedWidth(72)
        self.postshot_max_image_size_edit.setToolTip(i18n.tip("POSTSHOT_MAX_IMAGE_SIZE"))
        add_tooltip_row(
            form,
            i18n.t("POSTSHOT_MAX_IMAGE_SIZE"),
            self.postshot_max_image_size_edit,
            i18n.tip("POSTSHOT_MAX_IMAGE_SIZE"),
        )
        self.postshot_import_masks_cb = QCheckBox(i18n.t("POSTSHOT_IMPORT_MASKS"))
        self.postshot_import_masks_cb.setToolTip(i18n.tip("POSTSHOT_IMPORT_MASKS"))
        form.addRow("", self.postshot_import_masks_cb)

        self.postshot_mask_mode_combo = QComboBox()
        self.postshot_mask_mode_combo.addItem(i18n.t("POSTSHOT_MASK_BACKGROUND"), "background")
        self.postshot_mask_mode_combo.addItem(i18n.t("POSTSHOT_MASK_OCCLUDERS"), "occluders")
        self.postshot_mask_mode_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("POSTSHOT_MASK_MODE"),
            self.postshot_mask_mode_combo,
            i18n.tip("POSTSHOT_MASK_MODE"),
        )

        self.postshot_image_select_combo = QComboBox()
        self.postshot_image_select_combo.addItem(i18n.t("POSTSHOT_IMAGE_SELECT_ALL"), "all")
        self.postshot_image_select_combo.addItem(i18n.t("POSTSHOT_IMAGE_SELECT_BEST"), "best")
        add_tooltip_row(
            form,
            i18n.t("POSTSHOT_IMAGE_SELECT"),
            self.postshot_image_select_combo,
            i18n.tip("POSTSHOT_IMAGE_SELECT"),
        )

        self.postshot_num_train_images_edit = QLineEdit("0")
        self.postshot_num_train_images_edit.setFixedWidth(72)
        add_tooltip_row(
            form,
            i18n.t("POSTSHOT_NUM_TRAIN_IMAGES"),
            self.postshot_num_train_images_edit,
            i18n.tip("POSTSHOT_NUM_TRAIN_IMAGES"),
        )

        self.postshot_camera_poses_combo = QComboBox()
        self.postshot_camera_poses_combo.addItem(i18n.t("POSTSHOT_CAMERA_POSES_IMPORT"), "import")
        self.postshot_camera_poses_combo.addItem(i18n.t("POSTSHOT_CAMERA_POSES_ESTIMATE"), "estimate")
        add_tooltip_row(
            form,
            i18n.t("POSTSHOT_CAMERA_POSES"),
            self.postshot_camera_poses_combo,
            i18n.tip("POSTSHOT_CAMERA_POSES"),
        )

        self.postshot_pose_quality_combo = QComboBox()
        for value, label in (
            (1, i18n.t("POSTSHOT_POSE_QUALITY_FAST")),
            (2, "2"),
            (3, i18n.t("POSTSHOT_POSE_QUALITY_DEFAULT")),
            (4, i18n.t("POSTSHOT_POSE_QUALITY_BEST")),
        ):
            self.postshot_pose_quality_combo.addItem(label, value)
        self.postshot_pose_quality_combo.setCurrentIndex(2)
        add_tooltip_row(
            form,
            i18n.t("POSTSHOT_POSE_QUALITY"),
            self.postshot_pose_quality_combo,
            i18n.tip("POSTSHOT_POSE_QUALITY"),
        )

        advanced = CollapsibleSection(i18n.t("POSTSHOT_ADVANCED_PARAMETERS"), expanded=False)
        adv_layout = advanced.content_layout

        def add_section(title_key: str) -> QFormLayout:
            section = CollapsibleSection(i18n.t(title_key), expanded=False)
            section.setObjectName("postshotAdvancedSubsection")
            self.postshot_advanced_sections[title_key] = section
            adv_layout.addWidget(section)
            section_form = QFormLayout()
            section_form.setSpacing(5)
            section.content_layout.addLayout(section_form)
            return section_form

        def add_row(section_form: QFormLayout, key: str, field: QWidget, tooltip: str | None = None) -> None:
            self.postshot_advanced_rows[key] = (section_form, section_form.rowCount())
            add_tooltip_row(section_form, i18n.t(f"POSTSHOT_{key.upper()}"), field, tooltip)

        camera_form = add_section("POSTSHOT_SECTION_CAMERA")
        self.postshot_gpu_index_edit = QLineEdit("")
        self.postshot_gpu_index_edit.setFixedWidth(72)
        add_row(camera_form, "gpu_index", self.postshot_gpu_index_edit, i18n.tip("POSTSHOT_GPU_INDEX"))
        self.postshot_no_recenter_points_cb = QCheckBox()
        add_row(camera_form, "no_recenter_points", self.postshot_no_recenter_points_cb, i18n.tip("POSTSHOT_NO_RECENTER_POINTS"))

        model_form = add_section("POSTSHOT_SECTION_MODEL")
        self.postshot_splat_density_edit = QLineEdit("1.0")
        self.postshot_splat_density_edit.setFixedWidth(72)
        add_row(model_form, "splat_density", self.postshot_splat_density_edit, i18n.tip("POSTSHOT_SPLAT_DENSITY"))
        self.postshot_max_num_splats_edit = QLineEdit("3000")
        self.postshot_max_num_splats_edit.setFixedWidth(72)
        add_row(model_form, "max_num_splats", self.postshot_max_num_splats_edit, i18n.tip("POSTSHOT_MAX_NUM_SPLATS"))
        self.postshot_anti_aliasing_combo = QComboBox()
        self.postshot_anti_aliasing_combo.addItem(i18n.t("POSTSHOT_DEFAULT"), "default")
        self.postshot_anti_aliasing_combo.addItem(i18n.t("ON"), "on")
        self.postshot_anti_aliasing_combo.addItem(i18n.t("OFF"), "off")
        add_row(model_form, "anti_aliasing", self.postshot_anti_aliasing_combo, i18n.tip("POSTSHOT_ANTI_ALIASING"))
        self.postshot_max_sh_degree_combo = QComboBox()
        for degree in range(4):
            self.postshot_max_sh_degree_combo.addItem(str(degree), degree)
        self.postshot_max_sh_degree_combo.setCurrentIndex(3)
        add_row(model_form, "max_sh_degree", self.postshot_max_sh_degree_combo, i18n.tip("POSTSHOT_MAX_SH_DEGREE"))
        self.postshot_create_sky_model_cb = QCheckBox()
        add_row(model_form, "create_sky_model", self.postshot_create_sky_model_cb, i18n.tip("POSTSHOT_CREATE_SKY_MODEL"))
        self.postshot_store_training_context_cb = QCheckBox()
        add_row(
            model_form,
            "store_training_context",
            self.postshot_store_training_context_cb,
            i18n.tip("POSTSHOT_STORE_TRAINING_CONTEXT"),
        )
        self.postshot_show_train_error_cb = QCheckBox()
        add_row(model_form, "show_train_error", self.postshot_show_train_error_cb, i18n.tip("POSTSHOT_SHOW_TRAIN_ERROR"))

        region_form = add_section("POSTSHOT_SECTION_REGION")
        self.postshot_crop_box_combo = QComboBox()
        self.postshot_crop_box_combo.addItem(i18n.t("POSTSHOT_BOX_NONE"), "none")
        self.postshot_crop_box_combo.addItem(i18n.t("POSTSHOT_BOX_DEFAULT"), "default")
        self.postshot_crop_box_combo.addItem(i18n.t("POSTSHOT_BOX_CUSTOM"), "custom")
        add_row(region_form, "crop_box", self.postshot_crop_box_combo, i18n.tip("POSTSHOT_CROP_BOX"))
        self.postshot_crop_box_min_edit = QLineEdit("")
        add_row(region_form, "crop_box_min", self.postshot_crop_box_min_edit, i18n.tip("POSTSHOT_BOX_MIN"))
        self.postshot_crop_box_max_edit = QLineEdit("")
        add_row(region_form, "crop_box_max", self.postshot_crop_box_max_edit, i18n.tip("POSTSHOT_BOX_MAX"))
        self.postshot_roi_box_combo = QComboBox()
        self.postshot_roi_box_combo.addItem(i18n.t("POSTSHOT_BOX_NONE"), "none")
        self.postshot_roi_box_combo.addItem(i18n.t("POSTSHOT_BOX_DEFAULT"), "default")
        self.postshot_roi_box_combo.addItem(i18n.t("POSTSHOT_BOX_CUSTOM"), "custom")
        add_row(region_form, "roi_box", self.postshot_roi_box_combo, i18n.tip("POSTSHOT_ROI_BOX"))
        self.postshot_roi_box_min_edit = QLineEdit("")
        add_row(region_form, "roi_box_min", self.postshot_roi_box_min_edit, i18n.tip("POSTSHOT_BOX_MIN"))
        self.postshot_roi_box_max_edit = QLineEdit("")
        add_row(region_form, "roi_box_max", self.postshot_roi_box_max_edit, i18n.tip("POSTSHOT_BOX_MAX"))

        output_form = add_section("POSTSHOT_SECTION_OUTPUT")
        self.postshot_export_splat_edit = QLineEdit("")
        add_row(output_form, "export_splat", self.postshot_export_splat_edit, i18n.tip("POSTSHOT_EXPORT_SPLAT"))

        form.addRow(advanced)
        self.postshot_ksteps_auto_cb.toggled.connect(lambda _checked: self._update_postshot_conditional_visibility())
        self.postshot_import_masks_cb.toggled.connect(lambda _checked: self._update_postshot_conditional_visibility())
        self.postshot_profile_combo.currentIndexChanged.connect(lambda _idx: self._update_postshot_conditional_visibility())
        self.postshot_image_select_combo.currentIndexChanged.connect(lambda _idx: self._update_postshot_conditional_visibility())
        self.postshot_camera_poses_combo.currentIndexChanged.connect(lambda _idx: self._update_postshot_conditional_visibility())
        self.postshot_crop_box_combo.currentIndexChanged.connect(lambda _idx: self._update_postshot_conditional_visibility())
        self.postshot_roi_box_combo.currentIndexChanged.connect(lambda _idx: self._update_postshot_conditional_visibility())
        self._update_postshot_conditional_visibility()
        return widget

    def _build_custom_training_options(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(6)
        self.custom_training_args_edit = QLineEdit("{dataset} {output}")
        self.custom_training_args_edit.setToolTip(i18n.tip("CUSTOM_TRAINING_ARGS"))
        add_tooltip_row(
            form,
            i18n.t("CUSTOM_TRAINING_ARGS"),
            self.custom_training_args_edit,
            i18n.tip("CUSTOM_TRAINING_ARGS"),
        )
        return widget

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
        self.preview.set_scene_dir(path)
        self._refresh_input_image_count()
        self._update_training_paths(force=not restored)
        self._update_lfs_output_name(force=not restored)
        self._update_postshot_project_name(force=not restored)
        self._update_lfs_auto_steps_scaler()
        self._update_output_count()
        self._update_metashape_input_hint()
        self._render_preview()

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
        self._set_export_method(route)

        self._restore_conversion_settings(settings)
        self._restore_route_settings(scene, settings)
        self._restore_training_settings(scene, settings)

        self._sync_output_shape_controls()
        self._sync_yaw_per_frame_control()
        self._sync_settings_tabs()
        self._update_path_labels()

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
                self._set_metashape_ply_approved(bool(metashape.get("ply_approved", True)))
            if "use_ply" in metashape:
                self.ms_use_ply_cb.setChecked(bool(metashape.get("use_ply")))
            if "scale" in metashape:
                self.ms_scale_edit.setText(str(metashape.get("scale")))
            if "no_fix_rotation" in metashape:
                self.ms_no_fix_rot_cb.setChecked(bool(metashape.get("no_fix_rotation")))

        colmap = settings.get("colmap_rig")
        if isinstance(colmap, dict):
            self.run_colmap_cb.setChecked(bool(colmap.get("run_sfm", self.run_colmap_cb.isChecked())))
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
            self._set_combo_data(
                self.spheresfm_run_scope_combo,
                self._normalize_spheresfm_run_scope(str(spheresfm.get("run_scope", "")).strip()),
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

    def _restore_training_settings(self, scene: Path, settings: dict) -> None:
        training = settings.get("training")
        if not isinstance(training, dict):
            return
        self.run_training_cb.setChecked(bool(training.get("enabled", False)))
        backend = str(training.get("backend", "")).strip()
        if backend:
            self._set_training_backend(backend)
        executable = self._settings_text(training.get("executable"))
        if executable:
            self.training_executable_browse.set_text(executable)
        dataset_root = self._settings_path_text(scene, training.get("dataset_root"))
        if dataset_root:
            self.training_dataset_browse.set_text(dataset_root)
            self._training_dataset_user_edited = Path(dataset_root) != self._default_training_dataset_dir()
        output_dir = self._settings_path_text(scene, training.get("output_dir"))
        if output_dir:
            self.training_output_browse.set_text(output_dir)
            self._training_output_user_edited = Path(output_dir) != self._default_training_output_dir()

        self._restore_lfs_settings(training.get("lichtfeld"))
        self._restore_postshot_settings(training.get("postshot"))
        custom = training.get("custom") if isinstance(training.get("custom"), dict) else {}
        if "arguments_template" in custom:
            self.custom_training_args_edit.setText(str(custom.get("arguments_template", "")))

    def _restore_lfs_settings(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        advanced = payload.get("advanced") if isinstance(payload.get("advanced"), dict) else {}
        state = {
            "strategy": payload.get("strategy", "mrnf"),
            "iterations": payload.get("iterations", "30,000"),
            "max_gaussians": payload.get("max_gaussians", "1,000,000"),
            "sh_degree": payload.get("sh_degree", 3),
            "tile_mode": payload.get("tile_mode", 1),
            "steps_scaler": payload.get("steps_scaler", "1.00"),
            "bilateral_grid": payload.get("bilateral_grid", False),
            "mask_mode": payload.get("mask_mode", "none"),
            "invert_masks": payload.get("invert_masks", False),
            "mask_threshold": payload.get("mask_threshold", self.lfs_mask_threshold_edit.text()),
            "use_alpha_as_mask": payload.get("use_alpha_as_mask", self.lfs_use_alpha_as_mask_cb.isChecked()),
            "mask_opacity_penalty_weight": payload.get(
                "mask_opacity_penalty_weight",
                self.lfs_mask_opacity_penalty_weight_edit.text(),
            ),
            "mask_opacity_penalty_power": payload.get(
                "mask_opacity_penalty_power",
                self.lfs_mask_opacity_penalty_power_edit.text(),
            ),
            "sparsity": payload.get("sparsity", False),
            "gut": payload.get("gut", False),
            "undistort": payload.get("undistort", False),
            "mip_filter": payload.get("mip_filter", False),
            "ppisp": payload.get("ppisp", False),
            "ppisp_freeze_from_sidecar": payload.get(
                "ppisp_freeze_from_sidecar",
                self.lfs_ppisp_freeze_from_sidecar_cb.isChecked(),
            ),
            "ppisp_sidecar_path": advanced.get("ppisp_sidecar_path", ""),
            "ppisp_use_controller": payload.get("ppisp_use_controller", self.lfs_ppisp_use_controller_cb.isChecked()),
            "ppisp_controller_activation_step": payload.get(
                "ppisp_controller_activation_step",
                self.lfs_ppisp_controller_activation_step_edit.text(),
            ),
            "ppisp_controller_lr": payload.get("ppisp_controller_lr", self.lfs_ppisp_controller_lr_edit.text()),
            "ppisp_freeze_gaussians_on_distill": payload.get(
                "ppisp_freeze_gaussians_on_distill",
                self.lfs_ppisp_freeze_gaussians_on_distill_cb.isChecked(),
            ),
            "background_mode": payload.get("background_mode", "solid_color"),
            "background_color": payload.get("background_color", [0, 0, 0]),
            "background_image": payload.get("background_image", ""),
            "advanced_numbers": advanced.get("numbers", {}),
            "advanced_checks": advanced.get("checks", {}),
        }
        strategy = str(state.get("strategy", "mrnf"))
        self._lfs_active_strategy = strategy if strategy in _LFS_STRATEGIES else "mrnf"
        self._lfs_strategy_states[self._lfs_active_strategy] = state
        self._apply_lfs_ui_state(state)
        if "auto_steps_scaler" in payload:
            self.lfs_auto_steps_scaler_cb.setChecked(bool(payload.get("auto_steps_scaler")))
        if "headless" in payload:
            self.training_headless_cb.setChecked(bool(payload.get("headless")))
        if "output_name" in payload:
            self.lfs_output_name_edit.setText(str(payload.get("output_name", "")))
            self._lfs_output_name_user_edited = bool(self.lfs_output_name_edit.text().strip())

    def _restore_postshot_settings(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        if "project_name" in payload:
            self.postshot_project_name_edit.setText(str(payload.get("project_name", "")))
            self._postshot_project_name_user_edited = bool(self.postshot_project_name_edit.text().strip())
        self._set_combo_data(self.postshot_profile_combo, str(payload.get("profile", "")).strip())
        if "ksteps" in payload:
            self.postshot_ksteps_edit.setText(str(payload.get("ksteps", "")))
        if "auto_ksteps" in payload:
            self.postshot_ksteps_auto_cb.setChecked(bool(payload.get("auto_ksteps")))
        if "max_image_size" in payload:
            self.postshot_max_image_size_edit.setText(str(payload.get("max_image_size", "")))
        self._set_combo_data(self.postshot_camera_poses_combo, str(payload.get("camera_poses", "")).strip())
        if "import_masks" in payload:
            self.postshot_import_masks_cb.setChecked(bool(payload.get("import_masks")))
        self._set_combo_data(self.postshot_mask_mode_combo, str(payload.get("mask_mode", "")).strip())
        self._set_combo_data(self.postshot_image_select_combo, str(payload.get("image_select", "")).strip())
        if "num_train_images" in payload:
            self.postshot_num_train_images_edit.setText(str(payload.get("num_train_images", "")))
        self._set_combo_data(self.postshot_pose_quality_combo, payload.get("pose_quality"))
        self._update_postshot_conditional_visibility()

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
            _PIPELINE_STAGE_TRAINING: self._pipeline_training_status(),
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
        self._update_lfs_auto_steps_scaler()
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
        self.spheresfm_run_scope_combo.currentIndexChanged.connect(lambda _idx: self._save_user_preferences())
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
            spheresfm_run_scope = str(settings.get("spheresfm_run_scope", "")).strip()
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
            if spheresfm_run_scope:
                self._set_combo_data(self.spheresfm_run_scope_combo, self._normalize_spheresfm_run_scope(spheresfm_run_scope))
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
                "spheresfm_run_scope": self._spheresfm_run_scope(),
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

    def _training_backend(self) -> str:
        return self._training_backend_value

    def _training_backend_display_name(self, backend: str) -> str:
        if hasattr(self, "training_backend_selector"):
            return self.training_backend_selector.display_name(backend)
        return i18n.t(get_training_backend_spec(backend).short_label_key)

    def _set_training_backend(self, backend: str) -> None:
        backend = normalize_training_backend(backend)
        spec = get_training_backend_spec(backend)
        self._training_backend_value = backend
        if hasattr(self, "training_backend_selector"):
            self.training_backend_selector.set_backend(backend)
        self.training_backend_label.setToolTip(i18n.tip(spec.tooltip_key))
        stack_index = self.training_options_stack_indices[backend]
        self.training_options_stack.setCurrentIndex(stack_index)
        self.training_executable_browse.line_edit.setPlaceholderText(self._default_training_executable(backend))
        self.training_headless_cb.setVisible(spec.supports_headless)
        self._refresh_training_settings_layout()
        self._update_training_paths()
        if backend == _TRAINING_BACKEND_POSTSHOT:
            self._update_postshot_project_name()
        if backend == _TRAINING_BACKEND_LICHTFELD:
            self._update_lfs_auto_steps_scaler()
        if getattr(self, "_user_preferences_enabled", False):
            self._save_user_preferences()

    def _on_training_dataset_edited(self, _path: str) -> None:
        if self._syncing_training_paths:
            return
        self._training_dataset_user_edited = True
        self._update_lfs_auto_steps_scaler()

    def _on_training_output_edited(self, _path: str) -> None:
        if self._syncing_training_paths:
            return
        self._training_output_user_edited = True
        self._save_user_preferences()

    def _on_lfs_output_name_edited(self, _text: str) -> None:
        if self._syncing_lfs_output_name:
            return
        self._lfs_output_name_user_edited = True
        self._on_training_settings_changed()

    def _on_postshot_project_name_edited(self, _text: str) -> None:
        if self._syncing_postshot_project_name:
            return
        self._postshot_project_name_user_edited = True
        self._on_training_settings_changed()

    def _on_training_settings_changed(self, *_args) -> None:
        self._update_path_labels()
        self.primary_action_state_changed.emit()

    def _on_lfs_iterations_edited(self, _text: str) -> None:
        if self._syncing_lfs_auto_fields or getattr(self, "_syncing_lfs_strategy_state", False):
            return
        self._on_training_settings_changed()

    def _on_lfs_auto_steps_scaler_changed(self, checked: bool) -> None:
        self.lfs_steps_scaler_edit.setEnabled(not checked)
        if checked:
            self._update_lfs_auto_steps_scaler()
        else:
            self._save_lfs_active_state()
        self._on_training_settings_changed()

    def _on_lfs_gut_changed(self, _checked: bool) -> None:
        self._update_lfs_conditional_visibility()
        self._on_training_settings_changed()

    @staticmethod
    def _round_lfs_ui_step(value: float) -> int:
        return max(0, int(math.floor(value + 0.5)))

    @staticmethod
    def _format_lfs_int(value: object) -> str:
        return f"{int(value):,}"

    @staticmethod
    def _format_lfs_steps_list(values: object) -> str:
        if not isinstance(values, (list, tuple)):
            return ""
        return ",".join(str(int(value)) for value in values)

    @staticmethod
    def _parse_lfs_int_text_value(text: object, default: int = 0) -> int:
        try:
            return int(str(text).strip().replace(",", ""))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_lfs_float_text_value(text: object, default: float = 1.0) -> float:
        try:
            value = float(str(text).strip().replace(",", ""))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    @staticmethod
    def _parse_lfs_steps_text_value(text: object) -> list[int]:
        values: list[int] = []
        for part in str(text).split(","):
            item = part.strip()
            if not item:
                continue
            try:
                value = int(item)
            except ValueError:
                continue
            if value > 0:
                values.append(value)
        return values

    @staticmethod
    def _lfs_ppisp_activation_display(iterations: int, steps_scaler: float, configured_step: int) -> int:
        if configured_step >= 0:
            return configured_step
        tail_iters = int(math.floor(5000.0 * max(steps_scaler, 1.0) + 0.5))
        return max(0, int(iterations) - tail_iters)

    def _format_lfs_float(self, key: str, value: object) -> str:
        fmt = _LFS_ADVANCED_FLOAT_FORMATS.get(key, ".3f")
        try:
            return format(float(value), fmt)
        except (TypeError, ValueError):
            return format(0.0, fmt)

    def _format_lfs_advanced_value(self, key: str, value: object) -> str:
        if key in _LFS_ADVANCED_LIST_KEYS:
            return self._format_lfs_steps_list(value)
        if key in _LFS_ADVANCED_INT_KEYS:
            return self._format_lfs_int(value)
        return self._format_lfs_float(key, value)

    def _lfs_current_strategy(self) -> str:
        strategy = self.lfs_strategy_combo.currentData() if hasattr(self, "lfs_strategy_combo") else None
        return str(strategy) if strategy in _LFS_STRATEGIES else "mrnf"

    def _default_lfs_ui_state(self, strategy: str) -> dict[str, object]:
        strategy = strategy if strategy in _LFS_STRATEGIES else "mrnf"
        defaults = lichtfeld_defaults(strategy)
        bg_color = defaults.get("bg_color", [0.0, 0.0, 0.0])
        color_values = [
            max(0, min(255, int(math.floor(float(component) * 255.0 + 0.5))))
            for component in list(bg_color)[:3]
        ]
        while len(color_values) < 3:
            color_values.append(0)
        activation_step = self._lfs_ppisp_activation_display(
            int(defaults.get("iterations", 30000)),
            float(defaults.get("steps_scaler", 1.0)),
            int(defaults.get("ppisp_controller_activation_step", -1)),
        )
        return {
            "strategy": strategy,
            "iterations": self._format_lfs_int(defaults.get("iterations", 30000)),
            "max_gaussians": self._format_lfs_int(defaults.get("max_cap", 1_000_000)),
            "sh_degree": int(defaults.get("sh_degree", 3)),
            "tile_mode": int(defaults.get("tile_mode", 1)),
            "steps_scaler": self._format_lfs_float("steps_scaler", defaults.get("steps_scaler", 1.0)),
            "bilateral_grid": bool(defaults.get("use_bilateral_grid", False)),
            "mask_mode": str(defaults.get("mask_mode", "none")),
            "invert_masks": bool(defaults.get("invert_masks", False)),
            "mask_threshold": self._format_lfs_float("mask_threshold", defaults.get("mask_threshold", 0.5)),
            "use_alpha_as_mask": bool(defaults.get("use_alpha_as_mask", True)),
            "mask_opacity_penalty_weight": self._format_lfs_float(
                "mask_opacity_penalty_weight",
                defaults.get("mask_opacity_penalty_weight", 1.0),
            ),
            "mask_opacity_penalty_power": self._format_lfs_float(
                "mask_opacity_penalty_power",
                defaults.get("mask_opacity_penalty_power", 2.0),
            ),
            "sparsity": bool(defaults.get("enable_sparsity", False)),
            "gut": bool(defaults.get("gut", False)),
            "undistort": bool(defaults.get("undistort", False)),
            "mip_filter": bool(defaults.get("mip_filter", False)),
            "ppisp": bool(defaults.get("use_ppisp", False)),
            "ppisp_freeze_from_sidecar": bool(defaults.get("ppisp_freeze_from_sidecar", False)),
            "ppisp_sidecar_path": str(defaults.get("ppisp_sidecar_path", "")),
            "ppisp_use_controller": bool(defaults.get("ppisp_use_controller", False)),
            "ppisp_controller_activation_step": self._format_lfs_int(activation_step),
            "ppisp_controller_lr": self._format_lfs_float(
                "ppisp_controller_lr",
                defaults.get("ppisp_controller_lr", 0.002),
            ),
            "ppisp_freeze_gaussians_on_distill": bool(
                defaults.get("ppisp_freeze_gaussians_on_distill", True)
            ),
            "background_mode": str(defaults.get("bg_mode", "solid_color")),
            "background_color": color_values,
            "background_image": str(defaults.get("bg_image_path", "")),
            "advanced_numbers": {
                key: self._format_lfs_advanced_value(key, defaults[key])
                for key in self.lfs_advanced_edits
                if key in defaults
            },
            "advanced_checks": {
                key: bool(defaults.get(key, False))
                for key in self.lfs_advanced_checks
            },
        }

    @staticmethod
    def _copy_lfs_ui_state(state: dict[str, object]) -> dict[str, object]:
        copied = dict(state)
        copied["background_color"] = list(copied.get("background_color", [0, 0, 0]))
        copied["advanced_numbers"] = dict(copied.get("advanced_numbers", {}))
        copied["advanced_checks"] = dict(copied.get("advanced_checks", {}))
        return copied

    def _capture_lfs_ui_state(self, *, strategy: str | None = None) -> dict[str, object]:
        return {
            "strategy": strategy or self._lfs_current_strategy(),
            "iterations": self.lfs_iterations_edit.text().strip(),
            "max_gaussians": self.lfs_max_gaussians_edit.text().strip(),
            "sh_degree": int(self.lfs_sh_degree_combo.currentData()),
            "tile_mode": int(self.lfs_tile_mode_combo.currentData()),
            "steps_scaler": self.lfs_steps_scaler_edit.text().strip(),
            "bilateral_grid": self.lfs_bilateral_grid_cb.isChecked(),
            "mask_mode": self.lfs_mask_mode_combo.currentData() or "none",
            "invert_masks": self.lfs_invert_masks_cb.isChecked(),
            "mask_threshold": self.lfs_mask_threshold_edit.text().strip(),
            "use_alpha_as_mask": self.lfs_use_alpha_as_mask_cb.isChecked(),
            "mask_opacity_penalty_weight": self.lfs_mask_opacity_penalty_weight_edit.text().strip(),
            "mask_opacity_penalty_power": self.lfs_mask_opacity_penalty_power_edit.text().strip(),
            "sparsity": self.lfs_sparsity_cb.isChecked(),
            "gut": self.lfs_gut_cb.isChecked(),
            "undistort": self.lfs_undistort_cb.isChecked(),
            "mip_filter": self.lfs_mip_filter_cb.isChecked(),
            "ppisp": self.lfs_ppisp_cb.isChecked(),
            "ppisp_freeze_from_sidecar": self.lfs_ppisp_freeze_from_sidecar_cb.isChecked(),
            "ppisp_sidecar_path": self.lfs_ppisp_sidecar_browse.text().strip(),
            "ppisp_use_controller": self.lfs_ppisp_use_controller_cb.isChecked(),
            "ppisp_controller_activation_step": self.lfs_ppisp_controller_activation_step_edit.text().strip(),
            "ppisp_controller_lr": self.lfs_ppisp_controller_lr_edit.text().strip(),
            "ppisp_freeze_gaussians_on_distill": self.lfs_ppisp_freeze_gaussians_on_distill_cb.isChecked(),
            "background_mode": self.lfs_bg_mode_combo.currentData() or "solid_color",
            "background_color": [
                self.lfs_bg_r_edit.text().strip(),
                self.lfs_bg_g_edit.text().strip(),
                self.lfs_bg_b_edit.text().strip(),
            ],
            "background_image": self.lfs_bg_image_browse.text().strip(),
            "advanced_numbers": {key: edit.text().strip() for key, edit in self.lfs_advanced_edits.items()},
            "advanced_checks": {key: cb.isChecked() for key, cb in self.lfs_advanced_checks.items()},
        }

    def _apply_lfs_ui_state(self, state: dict[str, object]) -> None:
        state = self._copy_lfs_ui_state(state)
        self._syncing_lfs_strategy_state = True
        try:
            self._set_combo_data(self.lfs_strategy_combo, str(state.get("strategy", "mrnf")))
            self.lfs_iterations_edit.setText(str(state.get("iterations", "30,000")))
            self.lfs_max_gaussians_edit.setText(str(state.get("max_gaussians", "1,000,000")))
            self._set_combo_data(self.lfs_sh_degree_combo, int(state.get("sh_degree", 3)))
            self._set_combo_data(self.lfs_tile_mode_combo, int(state.get("tile_mode", 1)))
            self.lfs_steps_scaler_edit.setText(str(state.get("steps_scaler", "1.00")))
            self.lfs_bilateral_grid_cb.setChecked(bool(state.get("bilateral_grid", False)))
            self._set_combo_data(self.lfs_mask_mode_combo, str(state.get("mask_mode", "none")))
            self.lfs_invert_masks_cb.setChecked(bool(state.get("invert_masks", False)))
            self.lfs_mask_threshold_edit.setText(str(state.get("mask_threshold", "0.500")))
            self.lfs_use_alpha_as_mask_cb.setChecked(bool(state.get("use_alpha_as_mask", True)))
            self.lfs_mask_opacity_penalty_weight_edit.setText(
                str(state.get("mask_opacity_penalty_weight", "1.000"))
            )
            self.lfs_mask_opacity_penalty_power_edit.setText(
                str(state.get("mask_opacity_penalty_power", "2.000"))
            )
            self.lfs_sparsity_cb.setChecked(bool(state.get("sparsity", False)))
            self.lfs_gut_cb.setChecked(bool(state.get("gut", False)))
            self.lfs_undistort_cb.setChecked(bool(state.get("undistort", False)))
            self.lfs_mip_filter_cb.setChecked(bool(state.get("mip_filter", False)))
            self.lfs_ppisp_cb.setChecked(bool(state.get("ppisp", False)))
            self.lfs_ppisp_freeze_from_sidecar_cb.setChecked(
                bool(state.get("ppisp_freeze_from_sidecar", False))
            )
            self.lfs_ppisp_sidecar_browse.set_text(str(state.get("ppisp_sidecar_path", "")))
            self.lfs_ppisp_use_controller_cb.setChecked(bool(state.get("ppisp_use_controller", False)))
            self.lfs_ppisp_controller_activation_step_edit.setText(
                str(state.get("ppisp_controller_activation_step", "25,000"))
            )
            self.lfs_ppisp_controller_lr_edit.setText(str(state.get("ppisp_controller_lr", "0.00200")))
            self.lfs_ppisp_freeze_gaussians_on_distill_cb.setChecked(
                bool(state.get("ppisp_freeze_gaussians_on_distill", True))
            )
            self._set_combo_data(self.lfs_bg_mode_combo, str(state.get("background_mode", "solid_color")))
            color = list(state.get("background_color", [0, 0, 0]))
            while len(color) < 3:
                color.append(0)
            self.lfs_bg_r_edit.setText(str(color[0]))
            self.lfs_bg_g_edit.setText(str(color[1]))
            self.lfs_bg_b_edit.setText(str(color[2]))
            self.lfs_bg_image_browse.set_text(str(state.get("background_image", "")))
            numbers = dict(state.get("advanced_numbers", {}))
            for key, edit in self.lfs_advanced_edits.items():
                if key in numbers:
                    edit.setText(str(numbers[key]))
            checks = dict(state.get("advanced_checks", {}))
            for key, checkbox in self.lfs_advanced_checks.items():
                if key in checks:
                    checkbox.setChecked(bool(checks[key]))
        finally:
            self._syncing_lfs_strategy_state = False
        self._update_lfs_color_swatch()
        self._update_lfs_conditional_visibility()

    def _lfs_resolved_ppisp_activation_from_state(self, state: dict[str, object]) -> int:
        return self._lfs_ppisp_activation_display(
            self._parse_lfs_int_text_value(state.get("iterations"), 30000),
            self._parse_lfs_float_text_value(state.get("steps_scaler"), 1.0),
            -1,
        )

    def _scale_lfs_ui_state_steps(self, state: dict[str, object], new_scaler: float) -> None:
        if new_scaler <= 0.0 or not math.isfinite(new_scaler):
            return
        old_scaler = self._parse_lfs_float_text_value(state.get("steps_scaler"), 1.0)
        ratio = new_scaler / old_scaler if old_scaler > 0.0 else new_scaler
        old_activation_default = self._lfs_resolved_ppisp_activation_from_state(state)
        state["steps_scaler"] = self._format_lfs_float("steps_scaler", new_scaler)
        if abs(ratio - 1.0) < 0.001:
            return

        for key in _LFS_UI_STEP_TEXT_KEYS:
            value = self._parse_lfs_int_text_value(state.get(key), 0)
            state[key] = self._format_lfs_int(self._round_lfs_ui_step(value * ratio))

        numbers = dict(state.get("advanced_numbers", {}))
        for key in _LFS_UI_STEP_ADVANCED_KEYS:
            if key in numbers:
                value = self._parse_lfs_int_text_value(numbers[key], 0)
                numbers[key] = self._format_lfs_int(self._round_lfs_ui_step(value * ratio))
        for key in _LFS_ADVANCED_LIST_KEYS:
            if key in numbers:
                scaled = sorted(
                    {
                        self._round_lfs_ui_step(value * ratio)
                        for value in self._parse_lfs_steps_text_value(numbers[key])
                    }
                )
                numbers[key] = self._format_lfs_steps_list([value for value in scaled if value > 0])
        state["advanced_numbers"] = numbers

        current_activation = self._parse_lfs_int_text_value(state.get("ppisp_controller_activation_step"), -1)
        if current_activation <= 0 or current_activation == old_activation_default:
            state["ppisp_controller_activation_step"] = self._format_lfs_int(
                self._lfs_resolved_ppisp_activation_from_state(state)
            )

    def _save_lfs_active_state(self, *, strategy: str | None = None) -> None:
        if not hasattr(self, "lfs_strategy_combo") or not hasattr(self, "_lfs_strategy_states"):
            return
        active = strategy or getattr(self, "_lfs_active_strategy", self._lfs_current_strategy())
        if active not in _LFS_STRATEGIES:
            active = "mrnf"
        self._lfs_strategy_states[active] = self._capture_lfs_ui_state(strategy=active)

    def _commit_lfs_steps_scaler_if_changed(self) -> None:
        if self._syncing_lfs_strategy_state:
            return
        strategy = getattr(self, "_lfs_active_strategy", self._lfs_current_strategy())
        state = self._capture_lfs_ui_state(strategy=strategy)
        stored = self._lfs_strategy_states.get(strategy, state)
        old_scaler_text = str(stored.get("steps_scaler", state.get("steps_scaler", "1.00")))
        new_scaler = self._parse_lfs_float_text_value(state.get("steps_scaler"), -1.0)
        if new_scaler <= 0.0:
            return
        state["steps_scaler"] = old_scaler_text
        self._scale_lfs_ui_state_steps(state, new_scaler)
        self._lfs_strategy_states[strategy] = state
        self._apply_lfs_ui_state(state)

    def _on_lfs_steps_scaler_editing_finished(self) -> None:
        if self.lfs_auto_steps_scaler_cb.isChecked():
            return
        self._commit_lfs_steps_scaler_if_changed()
        self._on_training_settings_changed()

    def _on_lfs_strategy_combo_changed(self, _index: int) -> None:
        if self._syncing_lfs_strategy_state:
            return
        new_strategy = self._lfs_current_strategy()
        old_strategy = getattr(self, "_lfs_active_strategy", "mrnf")
        if new_strategy == old_strategy:
            self._update_lfs_conditional_visibility()
            return
        self._save_lfs_active_state(strategy=old_strategy)
        self._lfs_active_strategy = new_strategy
        state = self._lfs_strategy_states.get(new_strategy)
        if state is None:
            state = self._default_lfs_ui_state(new_strategy)
            self._lfs_strategy_states[new_strategy] = state
        self._apply_lfs_ui_state(state)
        self._on_training_settings_changed()

    @staticmethod
    def _set_form_row_visible(form: QFormLayout | None, row: int, visible: bool) -> None:
        if form is not None and hasattr(form, "setRowVisible"):
            form.setRowVisible(row, visible)

    def _set_lfs_form_row_visible(self, row: int, visible: bool) -> None:
        self._set_form_row_visible(getattr(self, "lfs_options_form", None), row, visible)

    def _set_postshot_form_row_visible(self, row: int, visible: bool) -> None:
        self._set_form_row_visible(getattr(self, "postshot_options_form", None), row, visible)

    def _set_lfs_advanced_row_visible(self, key: str, visible: bool) -> None:
        row = getattr(self, "lfs_advanced_rows", {}).get(key)
        if row is not None:
            form, row_index = row
            self._set_form_row_visible(form, row_index, visible)

    def _set_postshot_advanced_row_visible(self, key: str, visible: bool) -> None:
        row = getattr(self, "postshot_advanced_rows", {}).get(key)
        if row is not None:
            form, row_index = row
            self._set_form_row_visible(form, row_index, visible)

    def _update_postshot_conditional_visibility(self) -> None:
        if not hasattr(self, "postshot_profile_combo"):
            return
        self.postshot_ksteps_edit.setEnabled(not self.postshot_ksteps_auto_cb.isChecked())
        self._set_postshot_form_row_visible(self.postshot_mask_mode_row, self.postshot_import_masks_cb.isChecked())
        self.postshot_num_train_images_edit.setEnabled(
            (self.postshot_image_select_combo.currentData() or "all") == "best"
        )
        self.postshot_pose_quality_combo.setEnabled(
            (self.postshot_camera_poses_combo.currentData() or "import") == "estimate"
        )

        profile = self.postshot_profile_combo.currentData() or "Splat3"
        self._set_postshot_advanced_row_visible("splat_density", profile == "Splat ADC")
        self._set_postshot_advanced_row_visible("max_num_splats", profile == "Splat MCMC")

        crop_custom = (self.postshot_crop_box_combo.currentData() or "none") == "custom"
        self._set_postshot_advanced_row_visible("crop_box_min", crop_custom)
        self._set_postshot_advanced_row_visible("crop_box_max", crop_custom)
        roi_custom = (self.postshot_roi_box_combo.currentData() or "none") == "custom"
        self._set_postshot_advanced_row_visible("roi_box_min", roi_custom)
        self._set_postshot_advanced_row_visible("roi_box_max", roi_custom)

    def _update_lfs_background_visibility(self) -> None:
        if not hasattr(self, "lfs_bg_mode_combo"):
            return
        mode = self.lfs_bg_mode_combo.currentData()
        self._set_lfs_form_row_visible(self.lfs_bg_color_row, mode in {"solid_color", "modulation"})
        self._set_lfs_form_row_visible(self.lfs_bg_image_row, mode == "image")
        self._on_training_settings_changed()

    def _update_lfs_conditional_visibility(self) -> None:
        if not hasattr(self, "lfs_mask_mode_combo"):
            return

        mask_mode = self.lfs_mask_mode_combo.currentData() or "none"
        has_mask = mask_mode != "none"
        segment_mask = mask_mode == "segment"
        self._set_lfs_form_row_visible(self.lfs_mask_invert_row, has_mask)
        self._set_lfs_form_row_visible(self.lfs_mask_threshold_row, has_mask)
        self._set_lfs_form_row_visible(self.lfs_use_alpha_as_mask_row, has_mask)
        self._set_lfs_form_row_visible(self.lfs_mask_opacity_penalty_weight_row, segment_mask)
        self._set_lfs_form_row_visible(self.lfs_mask_opacity_penalty_power_row, segment_mask)

        ppisp = self.lfs_ppisp_cb.isChecked()
        ppisp_sidecar = ppisp and self.lfs_ppisp_freeze_from_sidecar_cb.isChecked()
        ppisp_controller = ppisp and self.lfs_ppisp_use_controller_cb.isChecked()
        self._set_lfs_form_row_visible(self.lfs_ppisp_freeze_from_sidecar_row, ppisp)
        self._set_lfs_form_row_visible(self.lfs_ppisp_sidecar_row, ppisp_sidecar)
        self._set_lfs_form_row_visible(self.lfs_ppisp_use_controller_row, ppisp)
        self._set_lfs_form_row_visible(self.lfs_ppisp_controller_activation_step_row, ppisp_controller)
        self._set_lfs_form_row_visible(self.lfs_ppisp_controller_lr_row, ppisp_controller)
        self._set_lfs_form_row_visible(self.lfs_ppisp_freeze_gaussians_on_distill_row, ppisp_controller)

        strategy = self.lfs_strategy_combo.currentData() or "mrnf"
        is_igs = strategy == "igs+"
        is_mrnf = strategy in {"mrnf", "mnrf", "lfs"}
        if is_igs and self.lfs_gut_cb.isChecked():
            self.lfs_gut_cb.setChecked(False)
        self.lfs_gut_cb.setEnabled(not is_igs)
        self._set_lfs_form_row_visible(self.lfs_tile_mode_row, self.lfs_gut_cb.isChecked())

        self.lfs_advanced_sections["LFS_SECTION_BILATERAL"].setVisible(self.lfs_bilateral_grid_cb.isChecked())
        self.lfs_advanced_sections["LFS_SECTION_PRUNING_GROWING"].setVisible(is_igs)
        self.lfs_advanced_sections["LFS_SECTION_MRNF"].setVisible(is_mrnf)
        self.lfs_advanced_sections["LFS_SECTION_SPARSITY"].setVisible(self.lfs_sparsity_cb.isChecked())
        self._set_lfs_advanced_row_visible("grow_until_iter", is_mrnf)
        self._set_lfs_advanced_row_visible("prune_opacity", is_igs)
        self._set_lfs_advanced_row_visible("revised_opacity", is_igs)
        random_init = self.lfs_advanced_checks["random"].isChecked()
        self._set_lfs_advanced_row_visible("init_num_pts", random_init)
        self._set_lfs_advanced_row_visible("init_extent", random_init)
        self._set_lfs_advanced_row_visible("test_every", self.lfs_advanced_checks["enable_eval"].isChecked())

        self._update_lfs_background_visibility()

    def _lfs_color_component(self, edit: QLineEdit) -> int:
        try:
            value = int(edit.text().strip())
        except ValueError:
            return 0
        return min(255, max(0, value))

    def _update_lfs_color_swatch(self) -> None:
        if not hasattr(self, "lfs_bg_color_swatch"):
            return
        r = self._lfs_color_component(self.lfs_bg_r_edit)
        g = self._lfs_color_component(self.lfs_bg_g_edit)
        b = self._lfs_color_component(self.lfs_bg_b_edit)
        self.lfs_bg_color_swatch.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #4b5563;"
        )

    def _default_training_executable(self, backend: str | None = None) -> str:
        return training_backend_default_executable(
            backend or self._training_backend(),
            windows=os.name == "nt",
        )

    def _resolve_training_executable(self) -> str:
        return self._resolve_executable(
            self.training_executable_browse.text(),
            self._default_training_executable(),
            "TRAINING_EXEC_NOT_FOUND",
        )

    def _training_output_dir(self) -> Path:
        raw = self.training_output_browse.text().strip()
        if raw:
            return Path(raw)
        return self._default_training_output_dir()

    def _default_training_output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return Path(self.scene_dir) / "output"

    def _training_config_path(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return step4_meta_dir(Path(self.scene_dir)) / "training" / "lichtfeld_config.json"

    def _default_training_dataset_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return self._display_output_dir()

    def _default_lfs_output_name(self) -> str:
        return Path(self.scene_dir).name if self.scene_dir else ""

    def _default_postshot_project_name(self) -> str:
        if not self.scene_dir:
            return "postshot.psht"
        return f"{Path(self.scene_dir).name}.psht"

    def _update_lfs_output_name(self, *, force: bool = False) -> None:
        if not hasattr(self, "lfs_output_name_edit"):
            return
        default_name = self._default_lfs_output_name()
        if force or not self._lfs_output_name_user_edited or not self.lfs_output_name_edit.text().strip():
            self._syncing_lfs_output_name = True
            try:
                self.lfs_output_name_edit.setText(default_name)
            finally:
                self._syncing_lfs_output_name = False

    def _update_postshot_project_name(self, *, force: bool = False) -> None:
        if not hasattr(self, "postshot_project_name_edit"):
            return
        default_name = self._default_postshot_project_name()
        if force or not self._postshot_project_name_user_edited or not self.postshot_project_name_edit.text().strip():
            self._syncing_postshot_project_name = True
            try:
                self.postshot_project_name_edit.setText(default_name)
            finally:
                self._syncing_postshot_project_name = False

    def _update_training_paths(self, *, force: bool = False) -> None:
        if not hasattr(self, "training_dataset_browse"):
            return
        if not self.scene_dir:
            self._syncing_training_paths = True
            try:
                self.training_dataset_browse.set_text("")
                self.training_output_browse.set_text("")
            finally:
                self._syncing_training_paths = False
            return

        self._syncing_training_paths = True
        try:
            if force or not self._training_dataset_user_edited or not self.training_dataset_browse.text():
                self.training_dataset_browse.set_text(str(self._default_training_dataset_dir()))
            if force or not self._training_output_user_edited or not self.training_output_browse.text():
                self.training_output_browse.set_text(str(self._default_training_output_dir()))
        finally:
            self._syncing_training_paths = False

    def _training_sparse_model_dir(self) -> Path | None:
        if self._is_colmap_method():
            return self._find_colmap_sparse_model() or (self._colmap_sparse_dir() / "0")
        if self._is_spheresfm_method():
            return self._find_spheresfm_sparse_model() or (self._spheresfm_sparse_dir() / "0")
        output_colmap = self._display_output_dir() / "colmap"
        return output_colmap if output_colmap.is_dir() else None

    def _training_dataset(self) -> TrainingDataset:
        dataset_root = Path(self.training_dataset_browse.text().strip()) if self.training_dataset_browse.text() else self._default_training_dataset_dir()
        if self._is_colmap_method():
            images_dir = self._colmap_rig_images_dir()
            masks_dir = self._colmap_rig_masks_dir()
        else:
            images_dir = dataset_root / "images"
            masks_dir = dataset_root / "masks"
        return TrainingDataset(
            dataset_root=dataset_root,
            images_dir=images_dir,
            masks_dir=masks_dir,
            colmap_sparse_dir=self._training_sparse_model_dir(),
            transforms_json=dataset_root / "transforms.json",
            pointcloud_ply=self._resolve_ply_source() or dataset_root / "pointcloud.ply",
            output_shape=self._output_shape(),
        )

    def _training_dataset_available(self) -> bool:
        if not self.scene_dir or not hasattr(self, "training_dataset_browse"):
            return False
        try:
            dataset = self._training_dataset()
        except ValueError:
            return False
        if dataset.images_dir is not None and self._count_images_in_dir(dataset.images_dir) > 0:
            return True
        return bool(dataset.transforms_json and dataset.transforms_json.is_file())

    def _selected_projection_view_count(self) -> int:
        if self._uses_direct_equirect_output() or self._uses_spheresfm_3dgut_output():
            return 1
        try:
            views = self.view_config.collect_views(include_disabled=True)
        except Exception:
            return 0
        return sum(1 for view in views if view["enabled"])

    @staticmethod
    def _count_images_in_dir(images_dir: Path | None) -> int:
        if images_dir is None or not images_dir.is_dir():
            return 0
        return sum(
            1
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_TRAINING_IMAGE_EXTS
        )

    def _training_image_count(self, dataset: TrainingDataset | None = None) -> int:
        if not self.scene_dir:
            return 0
        dataset = dataset or self._training_dataset()
        if self._uses_direct_equirect_output() or self._uses_spheresfm_3dgut_output():
            count = self._count_images_in_dir(dataset.images_dir)
            return count if count > 0 else self._count_source_images()

        if not self._training_dataset_user_edited:
            source_count = self._count_source_images()
            view_count = self._selected_projection_view_count()
            estimated = source_count * view_count
            if estimated > 0:
                return estimated
        return self._count_images_in_dir(dataset.images_dir)

    def _update_lfs_auto_steps_scaler(self) -> None:
        if not hasattr(self, "lfs_auto_steps_scaler_cb") or not self.lfs_auto_steps_scaler_cb.isChecked():
            return
        image_count = self._training_image_count() if self.scene_dir else 0
        scaler = lichtfeld_auto_steps_scaler(image_count)
        self._syncing_lfs_auto_fields = True
        try:
            self._save_lfs_active_state()
            for strategy in _LFS_STRATEGIES:
                state = self._copy_lfs_ui_state(
                    self._lfs_strategy_states.get(strategy, self._default_lfs_ui_state(strategy))
                )
                self._scale_lfs_ui_state_steps(state, scaler)
                self._lfs_strategy_states[strategy] = state
            active = getattr(self, "_lfs_active_strategy", self._lfs_current_strategy())
            self._apply_lfs_ui_state(self._lfs_strategy_states[active])
        finally:
            self._syncing_lfs_auto_fields = False

    def _spheresfm_runs_conversion(self) -> bool:
        if not self._is_spheresfm_method():
            return False
        if self._spheresfm_conversion_intent_override is not None:
            return self._spheresfm_conversion_intent_override
        return self._spheresfm_run_scope() != _SPHERESFM_RUN_SFM_ONLY

    def _spheresfm_runs_sfm(self) -> bool:
        if not self._is_spheresfm_method():
            return False
        if self._spheresfm_sfm_intent_override is not None:
            return self._spheresfm_sfm_intent_override
        return self._spheresfm_run_scope() != _SPHERESFM_RUN_CONVERT_ONLY

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
        route_index = self.input_tab_index
        if spheresfm_conversion_off and current == self.output_tab_index:
            self.settings_tabs.setCurrentIndex(route_index)
        elif prefer_route_tab:
            self.settings_tabs.setCurrentIndex(route_index)
        elif not self._settings_tab_available(self.settings_tabs.currentIndex()):
            self.settings_tabs.setCurrentIndex(route_index)

    def _settings_tab_available(self, index: int) -> bool:
        return (
            0 <= index < self.settings_tabs.count()
            and self.settings_tabs.isTabVisible(index)
            and self.settings_tabs.isTabEnabled(index)
        )

    def _on_spheresfm_run_scope_changed(self, *_args) -> None:
        if not self._syncing_spheresfm_scope_from_intent:
            self._spheresfm_sfm_intent_override = None
            self._spheresfm_conversion_intent_override = None
        self._sync_output_shape_controls()
        self._sync_settings_tabs()
        self._update_training_paths()
        self._update_output_count()
        self.primary_action_state_changed.emit()

    def _update_path_labels(self) -> None:
        if not self.scene_dir:
            return
        output = str(self._display_output_dir())
        if self._uses_direct_equirect_output() or (
            self._spheresfm_runs_conversion() and self._uses_spheresfm_3dgut_output()
        ):
            tip_key = "OUTPUT_DIR_LICHTFELD_DIRECT"
        elif self._is_spheresfm_method():
            tip_key = "OUTPUT_DIR_SPHERESFM_PROJECT"
        else:
            tip_key = "OUTPUT_DIR_CUBEMAP" if self._is_metashape_method() else "OUTPUT_DIR_COLMAP_PROJECT"
        self.output_path_label.setToolTip(f"{i18n.tip(tip_key)}\n{output}")
        self.output_path_label.set_full_text(output)

    def _on_metashape_input_path_changed(self, *_args) -> None:
        if self.sender() is self.ms_ply_browse and not self._syncing_metashape_auto_inputs:
            self._set_metashape_ply_approved(bool(self.ms_ply_browse.text().strip()))
        self._update_metashape_input_hint()
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
        self.primary_action_state_changed.emit()

    def _on_spheresfm_sparse_path_changed(self, _path: str) -> None:
        if not self._syncing_sfm_input_paths:
            self._spheresfm_sparse_user_edited = True
        self._update_sfm_input_tooltips()
        self._update_training_paths()
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
        self._set_metashape_ply_approved(False, auto_candidate=ply is not None)

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
                self._set_metashape_ply_approved(False, auto_candidate=True)
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
                elif not self._metashape_ply_approved:
                    ply_note = i18n.t("MS_PLY_APPROVAL_HINT")

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

    def _on_colmap_run_toggled(self, checked: bool) -> None:
        self.colmap_exec_browse.setEnabled(checked)
        self.colmap_pipeline_row.setEnabled(checked)
        self._on_colmap_mapper_changed()
        self.primary_action_state_changed.emit()

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

    def _reset_spheresfm_rtx50_diagnostics(self) -> None:
        self._spheresfm_phase_logs.clear()
        self._spheresfm_rtx50_cuda_error_seen = False
        self._spheresfm_rtx50_cuda_error_phase = None
        self._spheresfm_rtx50_cuda_error_shown = False

    def build_commands(self) -> list[tuple[str, list[str]]]:
        if self._is_spheresfm_method():
            self._reset_spheresfm_rtx50_diagnostics()
            run_sfm = self._spheresfm_runs_sfm()
            run_conversion = self._spheresfm_runs_conversion()
            if run_sfm or run_conversion:
                self._validate_spheresfm_export()
            if self.run_training_cb.isChecked() and not run_conversion:
                issue = self._training_dataset_issue()
                if issue is not None:
                    raise ValueError(issue)
            self._guard_existing_training_output_targets()
            if run_conversion and not run_sfm:
                self._require_spheresfm_sparse_model()
                self._validate_spheresfm_conversion_export()
                if not self._prepare_spheresfm_run_outputs(include_project=False, include_conversion=True):
                    return []
                steps = self._build_spheresfm_conversion_commands()
                steps.extend(self._build_training_commands())
                return steps

            steps: list[tuple[str, list[str]]] = []
            if run_sfm:
                if not self._prepare_spheresfm_run_outputs(
                    include_project=True,
                    include_conversion=run_conversion,
                ):
                    return []
                steps.extend(self._build_spheresfm_sfm_commands())
            if run_conversion:
                self._validate_spheresfm_conversion_export()
                steps.extend(self._build_spheresfm_conversion_commands())
            steps.extend(self._build_training_commands())
            return steps

        if self._is_colmap_method():
            run_conversion = self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION)
            run_sfm = self.pipeline_stage_intent(_PIPELINE_STAGE_SFM)
            if run_conversion or run_sfm:
                self._validate_image_only_export()
            self._guard_existing_training_output_targets()
            if run_conversion and not self._prepare_colmap_rig_dir():
                return []
            steps: list[tuple[str, list[str]]] = []
            if run_conversion:
                steps.append(("colmap_rig_export", self._build_cubemap_cmd(image_only=True, colmap_rig=True)))
            if run_sfm:
                if not run_conversion and not self._colmap_rig_images_dir().is_dir():
                    raise ValueError(i18n.t("STEP4_PIPELINE_DETAIL_COLMAP_NEEDS_RIG"))
                steps.extend(self._build_colmap_sfm_commands())
            steps.extend(self._build_training_commands())
            return steps

        run_conversion = self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION)
        if run_conversion:
            self._validate_bundle()
            preprocess_cmd = self._build_preprocess_cmd()
        self._guard_existing_training_output_targets()

        if not run_conversion:
            return self._build_training_commands()

        if self._uses_direct_equirect_output():
            if not self._prepare_3dgut_output_dir():
                return []
            steps = [("metashape", preprocess_cmd)]
            steps.extend(self._build_training_commands())
            return steps

        if not self._prepare_output_dir():
            return []

        steps = [("metashape", preprocess_cmd)]
        steps.append(("cubemap", self._build_cubemap_cmd()))
        if self.export_colmap_cb.isChecked():
            steps.append(("colmap", self._build_colmap_cmd()))
        steps.extend(self._build_training_commands())
        return steps

    def _build_preprocess_cmd(self) -> list[str]:
        self._refresh_metashape_auto_inputs_if_empty()
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
        if self._metashape_input_output_path_issue(Path(xml)):
            raise ValueError(i18n.t("METASHAPE_INPUT_IN_OUTPUT_ERROR").format(path=xml))

        scale = float(self.ms_scale_edit.text().strip())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("スケール係数は正の有限値である必要があります")

        ply = ""
        if self._preprocess_uses_ply():
            ply = self.ms_ply_browse.text()
            if not ply or not Path(ply).is_file():
                raise ValueError(f"PLYファイルが見つかりません: {ply}")
            if self._metashape_input_output_path_issue(Path(ply)):
                raise ValueError(i18n.t("METASHAPE_INPUT_IN_OUTPUT_ERROR").format(path=ply))
            if not self._metashape_ply_approved:
                raise ValueError(i18n.t("METASHAPE_PLY_APPROVAL_ERROR").format(path=ply))
        return build_metashape_preprocess_cmd(
            MetashapePreprocessCommand(
                python_executable=sys.executable,
                script=script,
                images=Path(images),
                xml=xml,
                output=self._display_output_dir() if self._uses_direct_equirect_output() else scene,
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

        views_json = self._write_views_config(step4_meta_dir(scene), views)

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

    def _parse_positive_int(self, edit: QLineEdit, label: str) -> int:
        try:
            value = int(edit.text().strip().replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"{label} は整数で指定してください") from exc
        if value <= 0:
            raise ValueError(f"{label} は1以上で指定してください")
        return value

    def _parse_nonnegative_int(self, edit: QLineEdit, label: str) -> int:
        try:
            value = int(edit.text().strip().replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"{label} は整数で指定してください") from exc
        if value < 0:
            raise ValueError(f"{label} は0以上で指定してください")
        return value

    def _parse_int(self, edit: QLineEdit, label: str) -> int:
        try:
            return int(edit.text().strip().replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"{label} は整数で指定してください") from exc

    def _parse_float(self, edit: QLineEdit, label: str) -> float:
        try:
            value = float(edit.text().strip())
        except ValueError as exc:
            raise ValueError(f"{label} は数値で指定してください") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} は有限値で指定してください")
        return value

    def _parse_optional_gpu_index(self, edit: QLineEdit, label: str) -> int | None:
        raw = edit.text().strip()
        if not raw:
            return None
        value = self._parse_nonnegative_int(edit, label)
        if value > 255:
            raise ValueError(f"{label} は0から255で指定してください")
        return value

    def _parse_postshot_box_coords(self, edit: QLineEdit, label: str) -> tuple[float, float, float]:
        raw = edit.text().replace(",", " ").strip()
        parts = [part for part in raw.split() if part]
        if len(parts) != 3:
            raise ValueError(f"{label} はX Y Zの3つの数値で指定してください")
        values: list[float] = []
        for part in parts:
            try:
                value = float(part)
            except ValueError as exc:
                raise ValueError(f"{label} はX Y Zの3つの数値で指定してください") from exc
            if not math.isfinite(value):
                raise ValueError(f"{label} は有限値で指定してください")
            values.append(value)
        return (values[0], values[1], values[2])

    def _postshot_box_options(
        self,
        combo: QComboBox,
        min_edit: QLineEdit,
        max_edit: QLineEdit,
        min_label: str,
        max_label: str,
    ) -> tuple[bool, tuple[float, float, float] | None, tuple[float, float, float] | None]:
        mode = combo.currentData() or "none"
        if mode == "default":
            return True, None, None
        if mode == "custom":
            return (
                False,
                self._parse_postshot_box_coords(min_edit, min_label),
                self._parse_postshot_box_coords(max_edit, max_label),
            )
        return False, None, None

    def _parse_lfs_steps_list(self, edit: QLineEdit, label: str) -> list[int]:
        raw = edit.text().strip()
        if not raw:
            return []
        values: list[int] = []
        for part in raw.split(","):
            text = part.strip()
            if not text:
                continue
            try:
                value = int(text.replace(",", ""))
            except ValueError as exc:
                raise ValueError(f"{label} はカンマ区切りの整数で指定してください") from exc
            if value <= 0:
                raise ValueError(f"{label} は1以上の値で指定してください")
            values.append(value)
        return values

    def _parse_lfs_color(self) -> tuple[float, float, float]:
        values = []
        for label, edit in (
            ("R", self.lfs_bg_r_edit),
            ("G", self.lfs_bg_g_edit),
            ("B", self.lfs_bg_b_edit),
        ):
            try:
                value = int(edit.text().strip())
            except ValueError as exc:
                raise ValueError(f"{i18n.t('LFS_BG_COLOR')} {label} は整数で指定してください") from exc
            if value < 0 or value > 255:
                raise ValueError(f"{i18n.t('LFS_BG_COLOR')} {label} は0から255で指定してください")
            values.append(value / 255.0)
        return (values[0], values[1], values[2])

    def _collect_lfs_config_overrides(self) -> dict[str, object]:
        self._save_lfs_active_state()
        overrides: dict[str, object] = {}
        current_scaler = self._parse_float(self.lfs_steps_scaler_edit, i18n.t("LFS_STEPS_SCALER"))
        default_state = self._default_lfs_ui_state(self._lfs_current_strategy())
        self._scale_lfs_ui_state_steps(default_state, current_scaler)

        def add_bool(config_key: str, current: bool, state_key: str | None = None) -> None:
            default = bool(default_state[state_key or config_key])
            if bool(current) != default:
                overrides[config_key] = bool(current)

        def add_float(config_key: str, edit: QLineEdit, label: str, state_key: str | None = None) -> None:
            current = self._parse_float(edit, label)
            default = self._parse_lfs_float_text_value(default_state[state_key or config_key], 0.0)
            if not math.isclose(current, default, rel_tol=1e-9, abs_tol=1e-12):
                overrides[config_key] = current

        def add_int(config_key: str, edit: QLineEdit, label: str, state_key: str | None = None) -> None:
            current = self._parse_int(edit, label)
            default = self._parse_lfs_int_text_value(default_state[state_key or config_key], 0)
            if current != default:
                overrides[config_key] = current

        add_bool("invert_masks", self.lfs_invert_masks_cb.isChecked())
        add_float("mask_threshold", self.lfs_mask_threshold_edit, i18n.t("LFS_MASK_THRESHOLD"))
        add_bool("use_alpha_as_mask", self.lfs_use_alpha_as_mask_cb.isChecked())
        add_float(
            "mask_opacity_penalty_weight",
            self.lfs_mask_opacity_penalty_weight_edit,
            i18n.t("LFS_MASK_OPACITY_PENALTY_WEIGHT"),
        )
        add_float(
            "mask_opacity_penalty_power",
            self.lfs_mask_opacity_penalty_power_edit,
            i18n.t("LFS_MASK_OPACITY_PENALTY_POWER"),
        )
        add_bool("ppisp_freeze_from_sidecar", self.lfs_ppisp_freeze_from_sidecar_cb.isChecked())
        add_bool("ppisp_use_controller", self.lfs_ppisp_use_controller_cb.isChecked())
        add_int(
            "ppisp_controller_activation_step",
            self.lfs_ppisp_controller_activation_step_edit,
            i18n.t("LFS_PPISP_CONTROLLER_ACTIVATION_STEP"),
        )
        add_float("ppisp_controller_lr", self.lfs_ppisp_controller_lr_edit, i18n.t("LFS_PPISP_CONTROLLER_LR"))
        add_bool(
            "ppisp_freeze_gaussians_on_distill",
            self.lfs_ppisp_freeze_gaussians_on_distill_cb.isChecked(),
        )

        default_numbers = dict(default_state.get("advanced_numbers", {}))
        for key, edit in self.lfs_advanced_edits.items():
            label = i18n.t(f"LFS_{key.upper()}")
            if key in _LFS_ADVANCED_LIST_KEYS:
                current = self._parse_lfs_steps_list(edit, label)
                default = self._parse_lfs_steps_text_value(default_numbers.get(key, ""))
                if current != default:
                    overrides[key] = current
            elif key in _LFS_ADVANCED_INT_KEYS:
                current = self._parse_nonnegative_int(edit, label)
                default = self._parse_lfs_int_text_value(default_numbers.get(key, ""), 0)
                if current != default:
                    overrides[key] = current
            else:
                current = self._parse_float(edit, label)
                default = self._parse_lfs_float_text_value(default_numbers.get(key, ""), 0.0)
                if not math.isclose(current, default, rel_tol=1e-9, abs_tol=1e-12):
                    overrides[key] = current
        if self.lfs_advanced_checks["enable_eval"].isChecked() and "save_steps" in overrides and "eval_steps" not in overrides:
            overrides["eval_steps"] = list(overrides["save_steps"])

        default_checks = dict(default_state.get("advanced_checks", {}))
        for key, checkbox in self.lfs_advanced_checks.items():
            current = checkbox.isChecked()
            if current != bool(default_checks.get(key, False)):
                overrides[key] = current
        sidecar = self.lfs_ppisp_sidecar_browse.text().strip()
        if sidecar != str(default_state.get("ppisp_sidecar_path", "")):
            overrides["ppisp_sidecar_path"] = sidecar
        return overrides

    @staticmethod
    def _filename_only(value: str, label: str) -> str:
        name = value.strip()
        if any(sep in name for sep in ("/", "\\")):
            raise ValueError(i18n.t("TRAINING_OUTPUT_NAME_PATH_ERROR").format(label=label))
        return name

    @staticmethod
    def _guard_training_output_target(path: Path) -> None:
        if path.exists():
            raise ValueError(i18n.t("TRAINING_OUTPUT_EXISTS").format(path=str(path)))

    def _guard_existing_training_output_targets(self) -> None:
        if not self.run_training_cb.isChecked():
            return
        output_dir = self._training_output_dir()
        backend = self._training_backend()
        if backend == _TRAINING_BACKEND_POSTSHOT:
            project_name = self._filename_only(
                self.postshot_project_name_edit.text().strip() or self._default_postshot_project_name(),
                i18n.t("POSTSHOT_PROJECT_NAME"),
            )
            self._guard_training_output_target(output_dir / project_name)
            return
        if backend == _TRAINING_BACKEND_CUSTOM:
            return

        lfs_output_name = self._filename_only(
            self.lfs_output_name_edit.text().strip(),
            i18n.t("LFS_OUTPUT_PLY_NAME"),
        )
        lfs_output_stem = lichtfeld_output_name_stem(lfs_output_name)
        if not lfs_output_stem:
            return
        self._guard_training_output_target(output_dir / f"{lfs_output_stem}.ply")
        if self.lfs_ppisp_cb.isChecked():
            self._guard_training_output_target(output_dir / f"{lfs_output_stem}.ppisp")

    def _build_training_commands(self) -> list[tuple[str, list[str]]]:
        if not self.run_training_cb.isChecked():
            return []
        issue = self._training_dataset_issue()
        if issue is not None:
            raise ValueError(issue)

        dataset = self._training_dataset()
        output_dir = self._training_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        executable = self._resolve_training_executable()
        backend = self._training_backend()

        if backend == _TRAINING_BACKEND_POSTSHOT:
            project_name = self._filename_only(
                self.postshot_project_name_edit.text().strip() or self._default_postshot_project_name(),
                i18n.t("POSTSHOT_PROJECT_NAME"),
            )
            self._guard_training_output_target(output_dir / project_name)
            export_splat_text = self.postshot_export_splat_edit.text().strip()
            export_splat_path = None
            if export_splat_text:
                raw_export_splat_path = Path(export_splat_text)
                export_splat_path = (
                    raw_export_splat_path
                    if raw_export_splat_path.is_absolute()
                    else output_dir / raw_export_splat_path
                )
            if export_splat_path is not None:
                self._guard_training_output_target(export_splat_path)
            crop_default, crop_min, crop_max = self._postshot_box_options(
                self.postshot_crop_box_combo,
                self.postshot_crop_box_min_edit,
                self.postshot_crop_box_max_edit,
                i18n.t("POSTSHOT_CROP_BOX_MIN"),
                i18n.t("POSTSHOT_CROP_BOX_MAX"),
            )
            roi_default, roi_min, roi_max = self._postshot_box_options(
                self.postshot_roi_box_combo,
                self.postshot_roi_box_min_edit,
                self.postshot_roi_box_max_edit,
                i18n.t("POSTSHOT_ROI_BOX_MIN"),
                i18n.t("POSTSHOT_ROI_BOX_MAX"),
            )
            anti_aliasing_mode = self.postshot_anti_aliasing_combo.currentData() or "default"
            anti_aliasing = None
            if anti_aliasing_mode == "on":
                anti_aliasing = True
            elif anti_aliasing_mode == "off":
                anti_aliasing = False
            cmd = build_postshot_training_cmd(
                PostshotTrainingOptions(
                    executable=executable,
                    dataset=dataset,
                    output_dir=output_dir,
                    project_name=project_name,
                    ksteps=None
                    if self.postshot_ksteps_auto_cb.isChecked()
                    else self._parse_positive_int(self.postshot_ksteps_edit, i18n.t("POSTSHOT_KSTEPS")),
                    max_image_size=self._parse_nonnegative_int(
                        self.postshot_max_image_size_edit,
                        i18n.t("POSTSHOT_MAX_IMAGE_SIZE"),
                    ),
                    use_imported_poses=(self.postshot_camera_poses_combo.currentData() or "import") == "import",
                    profile=self.postshot_profile_combo.currentData() or "Splat3",
                    import_masks=self.postshot_import_masks_cb.isChecked(),
                    mask_mode=self.postshot_mask_mode_combo.currentData() or "background",
                    image_select=self.postshot_image_select_combo.currentData() or "all",
                    num_train_images=self._parse_nonnegative_int(
                        self.postshot_num_train_images_edit,
                        i18n.t("POSTSHOT_NUM_TRAIN_IMAGES"),
                    ),
                    pose_quality=int(self.postshot_pose_quality_combo.currentData()),
                    gpu_index=self._parse_optional_gpu_index(
                        self.postshot_gpu_index_edit,
                        i18n.t("POSTSHOT_GPU_INDEX"),
                    ),
                    splat_density=self._parse_float(
                        self.postshot_splat_density_edit,
                        i18n.t("POSTSHOT_SPLAT_DENSITY"),
                    ),
                    max_num_splats=self._parse_positive_int(
                        self.postshot_max_num_splats_edit,
                        i18n.t("POSTSHOT_MAX_NUM_SPLATS"),
                    ),
                    anti_aliasing=anti_aliasing,
                    max_sh_degree=int(self.postshot_max_sh_degree_combo.currentData()),
                    create_sky_model=self.postshot_create_sky_model_cb.isChecked(),
                    store_training_context=self.postshot_store_training_context_cb.isChecked(),
                    show_train_error=self.postshot_show_train_error_cb.isChecked(),
                    no_recenter_points=self.postshot_no_recenter_points_cb.isChecked(),
                    crop_box_default=crop_default,
                    crop_box_min=crop_min,
                    crop_box_max=crop_max,
                    roi_box_default=roi_default,
                    roi_box_min=roi_min,
                    roi_box_max=roi_max,
                    export_splat_path=export_splat_path,
                )
            )
            return [(training_backend_phase_name(backend), cmd)]

        if backend == _TRAINING_BACKEND_CUSTOM:
            cmd = build_custom_training_cmd(
                CustomTrainingOptions(
                    executable=executable,
                    dataset=dataset,
                    output_dir=output_dir,
                    arguments_template=self.custom_training_args_edit.text(),
                )
            )
            return [(training_backend_phase_name(backend), cmd)]

        if self.lfs_auto_steps_scaler_cb.isChecked():
            self._save_lfs_active_state()
        else:
            self._commit_lfs_steps_scaler_if_changed()
        lfs_output_name = self._filename_only(
            self.lfs_output_name_edit.text().strip(),
            i18n.t("LFS_OUTPUT_PLY_NAME"),
        )
        lfs_output_stem = lichtfeld_output_name_stem(lfs_output_name)
        if lfs_output_stem:
            self._guard_training_output_target(output_dir / f"{lfs_output_stem}.ply")
            if self.lfs_ppisp_cb.isChecked():
                self._guard_training_output_target(output_dir / f"{lfs_output_stem}.ppisp")

        lfs_dataset_resize_factor = self.lfs_dataset_resize_factor_combo.currentData()
        lfs_dataset_max_width = self._parse_positive_int(self.lfs_dataset_max_width_edit, i18n.t("LFS_MAX_WIDTH"))
        lfs_dataset_test_every = (
            self._parse_positive_int(self.lfs_dataset_test_every_edit, i18n.t("LFS_TEST_EVERY"))
            if self.lfs_advanced_checks["enable_eval"].isChecked()
            else None
        )
        cmd = build_lichtfeld_training_cmd(
            LichtFeldTrainingOptions(
                executable=executable,
                dataset=dataset,
                output_dir=output_dir,
                config_path=self._training_config_path(),
                output_name=lfs_output_name,
                strategy=self.lfs_strategy_combo.currentData() or "mrnf",
                iterations=self._parse_positive_int(self.lfs_iterations_edit, i18n.t("LFS_ITERATIONS")),
                max_gaussians=self._parse_positive_int(self.lfs_max_gaussians_edit, i18n.t("LFS_MAX_GAUSSIANS")),
                sh_degree=int(self.lfs_sh_degree_combo.currentData()),
                tile_mode=int(self.lfs_tile_mode_combo.currentData()),
                steps_scaler=self._parse_float(self.lfs_steps_scaler_edit, i18n.t("LFS_STEPS_SCALER")),
                image_count=self._training_image_count(dataset),
                auto_steps_scaler=self.lfs_auto_steps_scaler_cb.isChecked(),
                bilateral_grid=self.lfs_bilateral_grid_cb.isChecked(),
                mask_mode=self.lfs_mask_mode_combo.currentData() or "none",
                sparsity=self.lfs_sparsity_cb.isChecked(),
                gut=self.lfs_gut_cb.isChecked(),
                undistort=self.lfs_undistort_cb.isChecked(),
                mip_filter=self.lfs_mip_filter_cb.isChecked(),
                ppisp=self.lfs_ppisp_cb.isChecked(),
                background_mode=self.lfs_bg_mode_combo.currentData() or "solid_color",
                background_color=self._parse_lfs_color(),
                background_image_path=self.lfs_bg_image_browse.text().strip(),
                dataset_resize_factor=None if lfs_dataset_resize_factor == "auto" else str(lfs_dataset_resize_factor),
                dataset_max_width=None if lfs_dataset_max_width == 3840 else lfs_dataset_max_width,
                dataset_use_cpu_cache=self.lfs_dataset_cpu_cache_cb.isChecked(),
                dataset_use_fs_cache=self.lfs_dataset_fs_cache_cb.isChecked(),
                dataset_test_every=lfs_dataset_test_every,
                config_overrides=self._collect_lfs_config_overrides(),
                headless=self.training_headless_cb.isChecked(),
            )
        )
        return [(training_backend_phase_name(backend), cmd)]

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

    def _build_spheresfm_sfm_commands(self) -> list[tuple[str, list[str]]]:
        preflight_script = self.base_dir / "scripts" / "spheresfm_gpu_preflight.py"
        if not preflight_script.exists():
            raise FileNotFoundError(f"spheresfm_gpu_preflight.py が見つかりません: {preflight_script}")

        prepare_script = self.base_dir / "scripts" / "prepare_spheresfm_project.py"
        if not prepare_script.exists():
            raise FileNotFoundError(f"prepare_spheresfm_project.py が見つかりません: {prepare_script}")

        matcher = self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL
        pose_path = self.spheresfm_pose_browse.text().strip()
        if matcher == _SPHERESFM_MATCHER_SPATIAL and not pose_path:
            raise ValueError(i18n.t("SPHERESFM_POSE_REQUIRED"))
        if pose_path and not Path(pose_path).is_file():
            raise ValueError(i18n.t("SPHERESFM_POSE_NOT_FOUND").format(path=pose_path))

        steps = build_spheresfm_commands(
            SphereSfmCommand(
                python_executable=sys.executable,
                preflight_script=preflight_script,
                prepare_script=prepare_script,
                colmap=self._resolve_spheresfm_executable(),
                images_dir=self._metashape_images_dir(),
                source_masks_dir=self._mask_dir(),
                prepared_masks_dir=self._spheresfm_masks_dir(),
                preflight_dir=self._spheresfm_preflight_dir(),
                database=self._spheresfm_database_path(),
                sparse=self._spheresfm_sparse_dir(),
                camera_params=self._spheresfm_camera_params_arg(),
                use_masks=self._spheresfm_uses_masks(),
                matcher=matcher,
                quality_preset=self._spheresfm_quality_preset(),
                pose_path=pose_path,
            )
        )
        return steps

    def _build_spheresfm_conversion_commands(self) -> list[tuple[str, list[str]]]:
        transforms_script = self.base_dir / "scripts" / "spheresfm_to_transforms.py"
        if not transforms_script.exists():
            raise FileNotFoundError(f"spheresfm_to_transforms.py が見つかりません: {transforms_script}")

        steps: list[tuple[str, list[str]]] = []
        transforms_output = self._spheresfm_3dgut_dir() if self._uses_spheresfm_3dgut_output() else self._spheresfm_equirect_dir()
        image_path_mode = "images-prefix" if self._uses_spheresfm_3dgut_output() else "relative"
        steps.append(
            (
                "spheresfm_transforms",
                build_spheresfm_transforms_cmd(
                    SphereSfmTransformsCommand(
                        python_executable=sys.executable,
                        script=transforms_script,
                        sparse=self._spheresfm_sparse_model_for_conversion(),
                        output=transforms_output,
                        images_dir=self._metashape_images_dir(),
                        image_path_mode=image_path_mode,
                    )
                ),
            )
        )
        if self._uses_spheresfm_projected_output():
            steps.append(("spheresfm_cubemap", self._build_spheresfm_cubemap_cmd()))
        return steps

    def _build_spheresfm_cubemap_cmd(self) -> list[str]:
        script = self.base_dir / "cubemap_transforms_json.py"
        if not script.exists():
            raise FileNotFoundError(f"cubemap_transforms_json.py が見つかりません: {script}")

        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for v in views if v["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")

        output = self._spheresfm_cubemap_dir()
        views_json = self._write_views_config(step4_meta_dir(Path(self.scene_dir)), views)
        out_fmt = self.output_format_combo.currentData() or "auto"
        out_depth = self.output_bit_depth_combo.currentData() or "8"

        try:
            jpgq = int(self.jpg_quality_edit.text().strip())
        except ValueError as exc:
            raise ValueError("JPG/WebP 品質は整数で指定してください") from exc
        if not 1 <= jpgq <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")

        mask_dir = self._mask_dir() if self._mask_dir().is_dir() else None
        return build_cubemap_conversion_cmd(
            CubemapConversionCommand(
                python_executable=sys.executable,
                script=script,
                scene=self._spheresfm_equirect_dir(),
                output=output,
                views_json=views_json,
                scale=float(self.scale_combo.currentData()),
                axis_mode=self._spheresfm_axis_transform_mode(),
                image_only=False,
                colmap_rig=False,
                invert_masks=self.invert_masks_cb.isChecked(),
                writes_images=self._writes_images(),
                writes_masks=self._writes_masks(),
                yaw_offset_per_frame=float(self.yaw_per_frame_edit.value()),
                output_format=out_fmt,
                output_bit_depth=out_depth,
                jpg_quality=jpgq,
                image_dir=self._metashape_images_dir(),
                mask_dir=mask_dir,
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
        process = QProcess(self)
        process.setProgram(colmap)
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.start()
        if not process.waitForStarted(3000):
            detail = process.errorString().strip() or "-"
            QMessageBox.warning(
                self,
                i18n.t("SPHERESFM_OPEN_GUI"),
                i18n.t("SPHERESFM_OPEN_GUI_FAILED_DETAIL").format(
                    exe=colmap,
                    model=str(model),
                    detail=detail,
                ),
            )
            return
        if process.waitForFinished(1500):
            detail = self._qprocess_output_text(process) or process.errorString().strip() or "-"
            if is_colmap_gui_unavailable_output(detail):
                QMessageBox.warning(
                    self,
                    i18n.t("SPHERESFM_OPEN_GUI"),
                    i18n.t("SPHERESFM_OPEN_GUI_UNAVAILABLE").format(
                        exe=colmap,
                        model=str(model),
                        detail=self._message_detail_tail(detail),
                    ),
                )
                return
            if process.exitStatus() != QProcess.NormalExit or process.exitCode() != 0:
                QMessageBox.warning(
                    self,
                    i18n.t("SPHERESFM_OPEN_GUI"),
                    i18n.t("SPHERESFM_OPEN_GUI_FAILED_DETAIL").format(
                        exe=colmap,
                        model=str(model),
                        detail=self._message_detail_tail(detail),
                    ),
                )
            return

        self._spheresfm_gui_processes.append(process)
        process.finished.connect(
            lambda _exit_code, _exit_status, proc=process, exe=colmap, model_path=str(model): (
                self._on_spheresfm_gui_process_finished(proc, exe, model_path)
            )
        )
        if process.state() == QProcess.NotRunning:
            self._on_spheresfm_gui_process_finished(process, colmap, str(model))

    @staticmethod
    def _qprocess_output_text(process: QProcess) -> str:
        raw = bytes(process.readAllStandardOutput()) + bytes(process.readAllStandardError())
        return raw.decode("utf-8", errors="replace").strip()

    @staticmethod
    def _message_detail_tail(detail: str, limit: int = 1800) -> str:
        text = detail.strip()
        if len(text) <= limit:
            return text or "-"
        return "...\n" + text[-limit:]

    def _on_spheresfm_gui_process_finished(self, process: QProcess, colmap: str, model: str) -> None:
        if not self._forget_spheresfm_gui_process(process):
            return
        detail = self._qprocess_output_text(process) or process.errorString().strip() or "-"
        if is_colmap_gui_unavailable_output(detail):
            QMessageBox.warning(
                self,
                i18n.t("SPHERESFM_OPEN_GUI"),
                i18n.t("SPHERESFM_OPEN_GUI_UNAVAILABLE").format(
                    exe=colmap,
                    model=model,
                    detail=self._message_detail_tail(detail),
                ),
            )

    def _forget_spheresfm_gui_process(self, process: QProcess) -> bool:
        if not any(p is process for p in self._spheresfm_gui_processes):
            return False
        self._spheresfm_gui_processes = [p for p in self._spheresfm_gui_processes if p is not process]
        process.deleteLater()
        return True

    def _write_views_config(self, output_dir: Path, views: list[dict]) -> Path:
        return write_views_config(output_dir, views)

    @staticmethod
    def _views_config_payload(views: list[dict]) -> dict:
        return views_config_payload(views)

    def _export_settings_path(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return step4_export_settings_path(Path(self.scene_dir))

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _collect_export_settings(self) -> dict:
        views = self.view_config.collect_views(include_disabled=True)
        scale = float(self.scale_combo.currentData())
        direct = self._uses_direct_equirect_output()
        spheresfm = self._is_spheresfm_method()
        spheresfm_runs_conversion = self._spheresfm_runs_conversion()
        spheresfm_3dgut = spheresfm_runs_conversion and self._uses_spheresfm_3dgut_output()
        spheresfm_projected = spheresfm_runs_conversion and self._uses_spheresfm_projected_output()
        direct_source_output = direct or spheresfm_3dgut
        yaw_step = (
            0.0
            if self._export_method() == _METHOD_COLMAP or direct_source_output
            else float(self.yaw_per_frame_edit.value())
        )
        jpg_quality = int(self.jpg_quality_edit.text().strip())
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        scene = Path(self.scene_dir)
        output = self._display_output_dir()
        profile = self._spheresfm_profile_id() if spheresfm else self._profile_id()
        effective_profile = self._spheresfm_effective_profile() if spheresfm else self._effective_profile()
        axis_transform = self._spheresfm_axis_transform_mode() if spheresfm else self._axis_transform_mode()
        route_uses_view_export = not direct_source_output and (not spheresfm or spheresfm_projected)
        views_config_snapshot = self._views_config_payload(views) if route_uses_view_export else None
        views_config_path = ""
        if route_uses_view_export:
            views_config_path = f"{STEP4_META_DIR_NAME}/{STEP4_VIEWS_CONFIG_JSON}"
        writes_view_images = route_uses_view_export and self._writes_images()
        writes_view_masks = route_uses_view_export and self._writes_masks()
        portable_dataset_kind = "3dgut" if direct_source_output else "projection_views"

        return {
            "app": "stechdrive-3dgs-utils",
            "app_version": APP_VERSION,
            "settings_version": STEP4_SETTINGS_VERSION,
            "created_at": self._utc_now_iso(),
            "scene_dir": str(scene),
            "output_dir": str(output),
            "portable_output": {
                "root": "output",
                "dataset_kind": portable_dataset_kind,
                "active": True,
            },
            "export_method": self._export_method(),
            "output_shape": self._output_shape(),
            "target_profile": profile,
            "effective_profile": effective_profile,
            "axis_transform": axis_transform,
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
            "views_config_path": views_config_path,
            "views_config_snapshot": views_config_snapshot,
            "conversion": {
                "yaw_offset_per_frame": yaw_step,
                "output_format": self.output_format_combo.currentData() or "auto",
                "output_bit_depth": self.output_bit_depth_combo.currentData() or "8",
                "jpg_quality": jpg_quality,
                "invert_masks": self.invert_masks_cb.isChecked(),
                "write_images": writes_view_images,
                "write_masks": writes_view_masks,
                "no_image": direct_source_output or not route_uses_view_export or not self._writes_any_view_assets(),
                "uses_source_images": direct_source_output,
                "uses_source_masks": direct_source_output and self._mask_dir().is_dir(),
                "export_colmap": self._is_metashape_method() and self.export_colmap_cb.isChecked(),
            },
            "postprocess": {
                "lichtfeld_final_orientation_correction": (
                    self._uses_lichtfeld_final_correction()
                    or self._uses_spheresfm_lichtfeld_final_correction()
                ),
                "lichtfeld_final_orientation_matrix": _LICHTFELD_FINAL_CORRECTION.tolist()
                if self._uses_lichtfeld_final_correction()
                or self._uses_spheresfm_lichtfeld_final_correction()
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
                "ply_approved": self._metashape_ply_approved,
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
                "selected_sparse_model_dir": (
                    self.colmap_sparse_browse.text() if self._colmap_sparse_user_edited else ""
                ),
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
                "selected_sparse_model_dir": (
                    self.spheresfm_sparse_browse.text() if self._spheresfm_sparse_user_edited else ""
                ),
                "use_masks": self._spheresfm_uses_masks(),
                "colmap_executable": self.spheresfm_exec_browse.text(),
                "matcher": self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
                "quality_preset": self._spheresfm_quality_preset(),
                "run_scope": self._spheresfm_run_scope(),
                "pose_path": self.spheresfm_pose_browse.text(),
                "camera_model": "SPHERE",
                "camera_params": self._spheresfm_camera_params_arg() if spheresfm else "",
                "output_shape": self._output_shape() if spheresfm else "",
                "target_profile": self._spheresfm_profile_id() if spheresfm else "",
                "effective_profile": self._spheresfm_effective_profile() if spheresfm else "",
                "axis_transform": self._spheresfm_axis_transform_mode() if spheresfm else "",
                "equirect_dir": str(self._spheresfm_equirect_dir()) if spheresfm else "",
                "cubemap_dir": str(self._spheresfm_cubemap_dir()) if spheresfm else "",
                "gut_dir": str(self._spheresfm_3dgut_dir()) if spheresfm else "",
            },
            "training": self._collect_training_settings(),
            "inputs": {
                "transforms_json": str(output / "transforms.json")
                if direct_source_output
                else str(scene / "transforms.json"),
                "masks_dir": str(self._mask_dir()),
                "ply_source": str(self._resolve_ply_source() or ""),
            },
            "output_files": {
                "settings": f"{STEP4_META_DIR_NAME}/{_EXPORT_SETTINGS_NAME}",
                "views_config": views_config_path,
                "transforms_json": "" if spheresfm and not spheresfm_runs_conversion else "transforms.json",
                "images_dir": "images",
                "masks_dir": "masks",
                "pointcloud": "pointcloud.ply" if direct or spheresfm_3dgut or spheresfm_projected else "",
                "colmap_rig_dir": "colmap_rig",
                "colmap_rig_config": "colmap_rig/rig_config.json",
                "colmap_project_manifest": f"{STEP4_META_DIR_NAME}/sfm/{_COLMAP_PROJECT_MANIFEST_NAME}",
                "spheresfm_project_dir": "spheresfm",
                "spheresfm_project_manifest": f"{STEP4_META_DIR_NAME}/sfm/{_SPHERESFM_PROJECT_MANIFEST_NAME}",
            },
        }

    def _collect_training_settings(self) -> dict:
        if hasattr(self, "lfs_strategy_combo"):
            self._save_lfs_active_state()
        dataset = self._training_dataset() if self.scene_dir else TrainingDataset(dataset_root=Path(""))
        return {
            "enabled": self.run_training_cb.isChecked(),
            "backend": self._training_backend(),
            "executable": self.training_executable_browse.text(),
            "dataset_root": str(dataset.dataset_root),
            "images_dir": str(dataset.images_dir or ""),
            "masks_dir": str(dataset.masks_dir or ""),
            "colmap_sparse_dir": str(dataset.colmap_sparse_dir or ""),
            "output_dir": str(self._training_output_dir()) if self.scene_dir else "",
            "lichtfeld_config": str(self._training_config_path()) if self.scene_dir else "",
            "lichtfeld": {
                "strategy": self.lfs_strategy_combo.currentData() or "mrnf",
                "iterations": self.lfs_iterations_edit.text().strip(),
                "max_gaussians": self.lfs_max_gaussians_edit.text().strip(),
                "output_name": self.lfs_output_name_edit.text().strip(),
                "sh_degree": self.lfs_sh_degree_combo.currentData(),
                "tile_mode": self.lfs_tile_mode_combo.currentData(),
                "steps_scaler": self.lfs_steps_scaler_edit.text().strip(),
                "auto_steps_scaler": self.lfs_auto_steps_scaler_cb.isChecked(),
                "image_count": self._training_image_count(dataset) if self.scene_dir else 0,
                "bilateral_grid": self.lfs_bilateral_grid_cb.isChecked(),
                "mask_mode": self.lfs_mask_mode_combo.currentData() or "none",
                "invert_masks": self.lfs_invert_masks_cb.isChecked(),
                "mask_threshold": self.lfs_mask_threshold_edit.text().strip(),
                "use_alpha_as_mask": self.lfs_use_alpha_as_mask_cb.isChecked(),
                "mask_opacity_penalty_weight": self.lfs_mask_opacity_penalty_weight_edit.text().strip(),
                "mask_opacity_penalty_power": self.lfs_mask_opacity_penalty_power_edit.text().strip(),
                "sparsity": self.lfs_sparsity_cb.isChecked(),
                "gut": self.lfs_gut_cb.isChecked(),
                "undistort": self.lfs_undistort_cb.isChecked(),
                "mip_filter": self.lfs_mip_filter_cb.isChecked(),
                "ppisp": self.lfs_ppisp_cb.isChecked(),
                "ppisp_freeze_from_sidecar": self.lfs_ppisp_freeze_from_sidecar_cb.isChecked(),
                "ppisp_use_controller": self.lfs_ppisp_use_controller_cb.isChecked(),
                "ppisp_controller_activation_step": self.lfs_ppisp_controller_activation_step_edit.text().strip(),
                "ppisp_controller_lr": self.lfs_ppisp_controller_lr_edit.text().strip(),
                "ppisp_freeze_gaussians_on_distill": self.lfs_ppisp_freeze_gaussians_on_distill_cb.isChecked(),
                "background_mode": self.lfs_bg_mode_combo.currentData() or "solid_color",
                "background_color": [
                    self.lfs_bg_r_edit.text().strip(),
                    self.lfs_bg_g_edit.text().strip(),
                    self.lfs_bg_b_edit.text().strip(),
                ],
                "background_image": self.lfs_bg_image_browse.text().strip(),
                "advanced": {
                    "numbers": {key: edit.text().strip() for key, edit in self.lfs_advanced_edits.items()},
                    "checks": {key: cb.isChecked() for key, cb in self.lfs_advanced_checks.items()},
                    "ppisp_sidecar_path": self.lfs_ppisp_sidecar_browse.text().strip(),
                },
                "headless": self.training_headless_cb.isChecked(),
            },
            "postshot": {
                "project_name": self.postshot_project_name_edit.text().strip(),
                "profile": self.postshot_profile_combo.currentData() or "Splat3",
                "ksteps": self.postshot_ksteps_edit.text().strip(),
                "auto_ksteps": self.postshot_ksteps_auto_cb.isChecked(),
                "max_image_size": self.postshot_max_image_size_edit.text().strip(),
                "camera_poses": self.postshot_camera_poses_combo.currentData() or "import",
                "import_masks": self.postshot_import_masks_cb.isChecked(),
                "mask_mode": self.postshot_mask_mode_combo.currentData() or "background",
                "image_select": self.postshot_image_select_combo.currentData() or "all",
                "num_train_images": self.postshot_num_train_images_edit.text().strip(),
                "pose_quality": self.postshot_pose_quality_combo.currentData(),
                "gpu_index": self.postshot_gpu_index_edit.text().strip(),
                "splat_density": self.postshot_splat_density_edit.text().strip(),
                "max_num_splats": self.postshot_max_num_splats_edit.text().strip(),
                "anti_aliasing": self.postshot_anti_aliasing_combo.currentData() or "default",
                "max_sh_degree": self.postshot_max_sh_degree_combo.currentData(),
                "create_sky_model": self.postshot_create_sky_model_cb.isChecked(),
                "store_training_context": self.postshot_store_training_context_cb.isChecked(),
                "show_train_error": self.postshot_show_train_error_cb.isChecked(),
                "no_recenter_points": self.postshot_no_recenter_points_cb.isChecked(),
                "crop_box": self.postshot_crop_box_combo.currentData() or "none",
                "crop_box_min": self.postshot_crop_box_min_edit.text().strip(),
                "crop_box_max": self.postshot_crop_box_max_edit.text().strip(),
                "roi_box": self.postshot_roi_box_combo.currentData() or "none",
                "roi_box_min": self.postshot_roi_box_min_edit.text().strip(),
                "roi_box_max": self.postshot_roi_box_max_edit.text().strip(),
                "export_splat": self.postshot_export_splat_edit.text().strip(),
            },
            "custom": {
                "arguments_template": self.custom_training_args_edit.text(),
            },
        }

    def _write_export_settings(self) -> None:
        payload = self._collect_export_settings()
        write_step4_export_settings(Path(self.scene_dir), payload)

    @staticmethod
    def _step4_run_id(prefix: str) -> str:
        return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '')}"

    def _current_export_settings_snapshot(self) -> dict:
        if not self.scene_dir:
            return {}
        return load_step4_export_settings(Path(self.scene_dir))

    def _step4_artifact_snapshot(self, root: Path) -> dict:
        scene = Path(self.scene_dir)
        return {
            "root": scene_relative(scene, root),
            "transforms_json": file_identity(root / "transforms.json"),
            "pointcloud": file_identity(root / "pointcloud.ply"),
            "images_dir": file_identity(root / "images"),
            "masks_dir": file_identity(root / "masks"),
            "colmap_sparse_dir": file_identity(root / "sparse"),
        }

    def _current_dataset_root_for_manifest(self) -> Path:
        if self._is_spheresfm_method():
            if self._uses_spheresfm_3dgut_output():
                return self._spheresfm_3dgut_dir()
            if self._spheresfm_runs_conversion():
                return self._spheresfm_cubemap_dir()
            return self._spheresfm_project_dir()
        if self._is_colmap_method():
            return self._colmap_project_dir()
        if self._uses_direct_equirect_output():
            return self._direct_output_dir()
        return self._output_dir()

    def _record_step4_sfm_run(self, mode: str) -> None:
        if not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        route = self._export_method()
        if route == _METHOD_COLMAP:
            project_dir = self._colmap_project_dir()
            sparse_model = self._find_colmap_sparse_model()
        elif route == _METHOD_SPHERESFM:
            project_dir = self._spheresfm_project_dir()
            sparse_model = self._find_spheresfm_sparse_model()
        else:
            project_dir = scene
            sparse_model = self._resolve_ply_source()
        append_step4_sfm_run(
            scene,
            {
                "id": self._step4_run_id("sfm"),
                "created_at": self._utc_now_iso(),
                "route": route,
                "mode": mode,
                "project_dir": scene_relative(scene, project_dir),
                "sparse_model_dir": scene_relative(scene, sparse_model) if sparse_model else "",
                "ready_for_conversion": sparse_model is not None,
                "settings": self._current_export_settings_snapshot(),
            },
        )

    def _record_step4_dataset_run(self) -> None:
        if not self.scene_dir:
            return
        scene = Path(self.scene_dir)
        root = self._current_dataset_root_for_manifest()
        append_step4_dataset_run(
            scene,
            {
                "id": self._step4_run_id("dataset"),
                "created_at": self._utc_now_iso(),
                "route": self._export_method(),
                "output_shape": self._output_shape(),
                "target_profile": self._spheresfm_profile_id()
                if self._is_spheresfm_method()
                else self._profile_id(),
                "dataset_root": scene_relative(scene, root),
                "artifacts": self._step4_artifact_snapshot(root),
                "settings": self._current_export_settings_snapshot(),
            },
        )

    def _record_step4_training_run(self) -> None:
        if not self.scene_dir or not self.run_training_cb.isChecked():
            return
        scene = Path(self.scene_dir)
        dataset = self._training_dataset()
        append_step4_training_run(
            scene,
            {
                "id": self._step4_run_id("training"),
                "created_at": self._utc_now_iso(),
                "backend": self._training_backend(),
                "dataset_root": scene_relative(scene, dataset.dataset_root),
                "output_dir": scene_relative(scene, self._training_output_dir()),
                "settings": self._collect_training_settings(),
            },
        )

    def _record_step4_runs(self, *, sfm_mode: str | None, dataset: bool) -> None:
        if sfm_mode:
            self._record_step4_sfm_run(sfm_mode)
        if dataset:
            self._record_step4_dataset_run()
        self._record_step4_training_run()

    def _write_colmap_project_manifest(self) -> None:
        project = self._colmap_project_dir()
        sparse_model = self._find_colmap_sparse_model()
        manifest_path = step4_meta_dir(Path(self.scene_dir)) / "sfm" / _COLMAP_PROJECT_MANIFEST_NAME
        payload = {
            "app": "stechdrive-3dgs-utils",
            "app_version": APP_VERSION,
            "export_type": "colmap_project",
            "created_at": self._utc_now_iso(),
            "project_dir": str(project),
            "images_dir": "images",
            "masks_dir": "masks",
            "sparse_dir": "sparse",
            "sparse_model_dir": self._path_text_relative_to(sparse_model, project) if sparse_model else "",
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
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_spheresfm_project_manifest(self) -> None:
        project = self._spheresfm_project_dir()
        sparse_model = self._find_spheresfm_sparse_model()
        manifest_path = step4_meta_dir(Path(self.scene_dir)) / "sfm" / _SPHERESFM_PROJECT_MANIFEST_NAME
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
            "sparse_model_dir": self._path_text_relative_to(sparse_model, project) if sparse_model else "",
            "ready_for_import": sparse_model is not None,
            "database": "database.db",
            "use_masks": self._spheresfm_uses_masks(),
            "matcher": self.spheresfm_matcher_combo.currentData() or _COLMAP_MATCHER_SEQUENTIAL,
            "quality_preset": self._spheresfm_quality_preset(),
            "run_scope": self._spheresfm_run_scope(),
            "pose_path": self.spheresfm_pose_browse.text(),
            "camera_model": "SPHERE",
            "camera_params": self._spheresfm_camera_params_arg(),
            "output_shape": self._output_shape(),
            "target_profile": self._spheresfm_profile_id(),
            "effective_profile": self._spheresfm_effective_profile(),
            "axis_transform": self._spheresfm_axis_transform_mode(),
            "equirect_dir": "equirect",
            "cubemap_dir": str(self._spheresfm_cubemap_dir()),
            "gut_dir": str(self._spheresfm_3dgut_dir()),
        }
        project.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _path_text_relative_to(path: Path, root: Path) -> str:
        try:
            return str(path.relative_to(root).as_posix())
        except ValueError:
            return str(path)

    def _output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return Path(self.scene_dir) / "output"

    def _direct_output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return self._output_dir()

    def _display_output_dir(self) -> Path:
        if self._uses_direct_equirect_output():
            return self._direct_output_dir()
        if self._is_spheresfm_method() and self._spheresfm_runs_conversion():
            return self._spheresfm_3dgut_dir() if self._uses_spheresfm_3dgut_output() else self._spheresfm_cubemap_dir()
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

    def _spheresfm_preflight_dir(self) -> Path:
        return self._spheresfm_project_dir() / "preflight"

    def _spheresfm_database_path(self) -> Path:
        return self._spheresfm_project_dir() / "database.db"

    def _spheresfm_sparse_dir(self) -> Path:
        return self._spheresfm_project_dir() / "sparse"

    def _spheresfm_equirect_dir(self) -> Path:
        return self._spheresfm_project_dir() / "equirect"

    def _spheresfm_cubemap_dir(self) -> Path:
        return self._output_dir()

    def _spheresfm_3dgut_dir(self) -> Path:
        return self._output_dir()

    def _selected_colmap_sparse_model(self) -> Path | None:
        if not hasattr(self, "colmap_sparse_browse"):
            return None
        text = self.colmap_sparse_browse.text().strip()
        if not text:
            return None
        path = Path(text)
        return path if self._has_colmap_sparse_model(path) else None

    def _selected_spheresfm_sparse_model(self) -> Path | None:
        if not hasattr(self, "spheresfm_sparse_browse"):
            return None
        text = self.spheresfm_sparse_browse.text().strip()
        if not text:
            return None
        path = Path(text)
        return path if self._has_colmap_sparse_model(path) else None

    def _auto_find_spheresfm_sparse_model(self) -> Path | None:
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

    def _find_spheresfm_sparse_model(self) -> Path | None:
        selected_text = self.spheresfm_sparse_browse.text().strip() if hasattr(self, "spheresfm_sparse_browse") else ""
        selected = self._selected_spheresfm_sparse_model()
        if selected is not None:
            return selected
        if selected_text and self._spheresfm_sparse_user_edited:
            return None
        return self._auto_find_spheresfm_sparse_model()

    def _spheresfm_sparse_model_for_conversion(self) -> Path:
        if self._spheresfm_runs_sfm():
            return self._spheresfm_sparse_dir()
        model = self._find_spheresfm_sparse_model()
        return model if model is not None else self._spheresfm_sparse_dir()

    def _auto_find_colmap_sparse_model(self) -> Path | None:
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

    def _find_colmap_sparse_model(self) -> Path | None:
        selected_text = self.colmap_sparse_browse.text().strip() if hasattr(self, "colmap_sparse_browse") else ""
        selected = self._selected_colmap_sparse_model()
        if selected is not None:
            return selected
        if selected_text and self._colmap_sparse_user_edited:
            return None
        return self._auto_find_colmap_sparse_model()

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

    def _validate_scene_output_dir(self, output: Path) -> None:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        scene = Path(self.scene_dir).resolve()
        try:
            resolved_output = output.resolve()
        except OSError:
            resolved_output = output.absolute()
        if resolved_output.parent != scene:
            raise ValueError(f"出力フォルダがシーンフォルダ外です: {output}")

    def _3dgut_output_reset_targets(self) -> list[Path]:
        output = self._output_dir()
        targets = [
            output / "images",
            output / "masks",
            output / "transforms.json",
            output / "pointcloud.ply",
        ]
        return targets

    def _prepare_3dgut_output_dir(self) -> bool:
        output = self._output_dir()
        self._validate_scene_output_dir(output)
        existing_targets = self._dedupe_nested_paths(
            [path for path in self._3dgut_output_reset_targets() if self._path_has_contents(path)]
        )
        if existing_targets:
            target_text = "\n".join(str(path) for path in existing_targets)
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
        self._link_3dgut_assets(output)
        return True

    def _link_3dgut_assets(self, output: Path) -> None:
        self._link_or_copy_tree(self._metashape_images_dir(), output / "images")
        masks = self._mask_dir()
        if masks.is_dir():
            self._link_or_copy_tree(masks, output / "masks")

    @staticmethod
    def _link_or_copy_tree(source_root: Path, dest_root: Path) -> None:
        if not source_root.is_dir():
            return
        for source in sorted(source_root.rglob("*"), key=lambda path: str(path).lower()):
            relative = source.relative_to(source_root)
            dest = dest_root / relative
            if source.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if not source.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            try:
                os.link(source, dest)
            except OSError:
                shutil.copy2(source, dest)

    def _prepare_output_dir(self) -> bool:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        output = self._output_dir()

        self._validate_scene_output_dir(output)

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

    def _prepare_spheresfm_run_outputs(self, *, include_project: bool, include_conversion: bool) -> bool:
        self._validate_spheresfm_project_dir()
        targets: list[Path] = []
        if include_project:
            targets.append(self._spheresfm_project_dir())
        if include_conversion:
            targets.extend(self._spheresfm_conversion_reset_targets())

        existing_targets = self._dedupe_nested_paths([p for p in targets if self._path_has_contents(p)])
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

        self._spheresfm_project_dir().mkdir(parents=True, exist_ok=True)
        if include_conversion:
            if self._uses_spheresfm_3dgut_output():
                self._spheresfm_3dgut_dir().mkdir(parents=True, exist_ok=True)
                self._link_3dgut_assets(self._spheresfm_3dgut_dir())
            else:
                self._spheresfm_cubemap_dir().mkdir(parents=True, exist_ok=True)
        return True

    def _spheresfm_conversion_reset_targets(self) -> list[Path]:
        if self._uses_spheresfm_3dgut_output():
            root = self._spheresfm_3dgut_dir()
            return [
                root / "images",
                root / "masks",
                root / "transforms.json",
                root / "pointcloud.ply",
            ]

        output = self._spheresfm_cubemap_dir()
        targets = [
            self._spheresfm_equirect_dir(),
            output / "transforms.json",
            output / "pointcloud.ply",
            step4_views_config_path(Path(self.scene_dir)),
            step4_export_settings_path(Path(self.scene_dir)),
        ]
        if self._writes_images():
            targets.append(output / "images")
        if self._writes_masks():
            targets.append(output / "masks")
        return targets

    @staticmethod
    def _dedupe_nested_paths(paths: list[Path]) -> list[Path]:
        kept: list[Path] = []
        for path in sorted(paths, key=lambda p: len(p.parts)):
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path.absolute()
            nested = False
            for parent in kept:
                try:
                    parent_resolved = parent.resolve()
                except OSError:
                    parent_resolved = parent.absolute()
                if resolved == parent_resolved:
                    nested = True
                    break
                try:
                    resolved.relative_to(parent_resolved)
                    nested = True
                    break
                except ValueError:
                    pass
            if not nested:
                kept.append(path)
        return kept

    def _validate_spheresfm_project_dir(self) -> None:
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

    def _spheresfm_quality_preset(self) -> str:
        return _normalize_spheresfm_quality_preset(str(self.spheresfm_quality_combo.currentData() or ""))

    def _spheresfm_run_scope(self) -> str:
        return self._normalize_spheresfm_run_scope(str(self.spheresfm_run_scope_combo.currentData() or ""))

    def _validate_spheresfm_export(self) -> None:
        self._validate_image_only_export()
        if (
            self._spheresfm_runs_sfm()
            and self._spheresfm_uses_masks()
            and not self._mask_dir().is_dir()
        ):
            raise ValueError(i18n.t("SPHERESFM_MASKS_NOT_FOUND").format(path=str(self._mask_dir())))

    def _require_spheresfm_sparse_model(self) -> Path:
        model = self._find_spheresfm_sparse_model()
        if model is None:
            raise ValueError(i18n.t("SPHERESFM_CONVERT_ONLY_NO_SPARSE").format(path=str(self._spheresfm_sparse_dir())))
        return model

    def _validate_spheresfm_conversion_export(self) -> None:
        transforms_script = self.base_dir / "scripts" / "spheresfm_to_transforms.py"
        if not transforms_script.exists():
            raise FileNotFoundError(f"spheresfm_to_transforms.py が見つかりません: {transforms_script}")
        if not self._uses_spheresfm_projected_output():
            return
        cubemap_script = self.base_dir / "cubemap_transforms_json.py"
        if not cubemap_script.exists():
            raise FileNotFoundError(f"cubemap_transforms_json.py が見つかりません: {cubemap_script}")
        views = self.view_config.collect_views(include_disabled=True)
        enabled = sum(1 for v in views if v["enabled"])
        if enabled <= 0:
            raise ValueError("少なくとも1つのビューを有効にしてください")
        if enabled > _BLOCK_ENABLED_VIEWS:
            raise ValueError(f"ビュー数が多すぎます ({enabled})。{_BLOCK_ENABLED_VIEWS} 以下にしてください。")
        try:
            jpgq = int(self.jpg_quality_edit.text().strip())
        except ValueError as exc:
            raise ValueError("JPG/WebP 品質は整数で指定してください") from exc
        if not 1 <= jpgq <= 100:
            raise ValueError("JPG/WebP 品質は 1-100 の範囲で指定してください")

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
            candidates = [scene / _GENERATED_POINTCLOUD_NAME]
            for c in candidates:
                if c.is_file():
                    return c
            return None
        ply_text = self.ms_ply_browse.text().strip() if hasattr(self, "ms_ply_browse") else ""
        if ply_text:
            ply = Path(ply_text)
            if ply.is_file() and self._metashape_ply_approved:
                return ply
        return None

    # -- バンドル後処理 --

    def on_phase_log_started(self, phase: str, path: str) -> None:
        if phase.startswith("spheresfm_"):
            self._spheresfm_phase_logs[phase] = Path(path)

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        if (
            phase.startswith("spheresfm_")
            and exit_code != 0
            and not canceled
            and self._spheresfm_rtx50_cuda_error_seen
            and self._spheresfm_rtx50_cuda_error_phase == phase
            and not self._spheresfm_rtx50_cuda_error_shown
        ):
            self._show_spheresfm_rtx50_cuda_error(phase)

    def _show_spheresfm_rtx50_cuda_error(self, phase: str) -> None:
        self._spheresfm_rtx50_cuda_error_shown = True
        log_path = self._spheresfm_phase_logs.get(phase)
        log_text = str(log_path) if log_path is not None else "-"
        QMessageBox.warning(
            self,
            i18n.t("SPHERESFM_RTX50_CUDA_ERROR_TITLE"),
            i18n.t("SPHERESFM_RTX50_CUDA_ERROR_BODY").format(log_path=log_text),
        )

    def on_queue_finished(self, success: bool) -> None:
        if success:
            try:
                self._finalize_bundle()
            except Exception:
                pass

    def _finalize_bundle(self) -> None:
        if self._is_spheresfm_method():
            if self._uses_spheresfm_projected_output():
                source_ply = self._spheresfm_equirect_dir() / "pointcloud.ply"
                dest_ply = self._spheresfm_cubemap_dir() / "pointcloud.ply"
                if source_ply.is_file():
                    dest_ply.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_ply, dest_ply)
                    transforms = self._spheresfm_cubemap_dir() / "transforms.json"
                    if transforms.is_file():
                        data = json.loads(transforms.read_text(encoding="utf-8"))
                        data["ply_file_path"] = dest_ply.name
                        transforms.write_text(json.dumps(data, indent=2), encoding="utf-8")
                if self._uses_spheresfm_lichtfeld_final_correction():
                    self._apply_lichtfeld_final_correction(self._spheresfm_cubemap_dir())
            elif self._uses_spheresfm_3dgut_output() and self._uses_spheresfm_lichtfeld_final_correction():
                self._apply_lichtfeld_final_correction(self._spheresfm_3dgut_dir())
            self._write_export_settings()
            self._write_spheresfm_project_manifest()
            self._record_step4_runs(
                sfm_mode="spheresfm" if self._spheresfm_runs_sfm() else None,
                dataset=self._spheresfm_runs_conversion(),
            )
            return

        if self._is_colmap_method():
            self._write_export_settings()
            self._write_colmap_project_manifest()
            self._record_step4_runs(
                sfm_mode="colmap" if self.pipeline_stage_intent(_PIPELINE_STAGE_SFM) else None,
                dataset=self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION),
            )
            return

        if self._uses_direct_equirect_output():
            if self._uses_lichtfeld_final_correction():
                self._apply_lichtfeld_final_correction(self._direct_output_dir())
            self._write_export_settings()
            self._record_step4_runs(
                sfm_mode="metashape_import" if self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION) else None,
                dataset=self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION),
            )
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
        self._record_step4_runs(
            sfm_mode="metashape_import" if self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION) else None,
            dataset=self.pipeline_stage_intent(_PIPELINE_STAGE_CONVERSION),
        )

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
            "spheresfm_preflight": "PHASE_SPHERESFM_PREFLIGHT",
            "spheresfm_prepare": "PHASE_SPHERESFM_PREPARE",
            "spheresfm_database": "PHASE_SPHERESFM_DATABASE",
            "spheresfm_feature": "PHASE_SPHERESFM_FEATURE",
            "spheresfm_match": "PHASE_SPHERESFM_MATCH",
            "spheresfm_mapper": "PHASE_SPHERESFM_MAPPER",
            "spheresfm_transforms": "PHASE_SPHERESFM_TRANSFORMS",
            "spheresfm_cubemap": "PHASE_SPHERESFM_CUBEMAP",
            "training_lichtfeld": "PHASE_TRAINING_LICHTFELD",
            "training_postshot": "PHASE_TRAINING_POSTSHOT",
            "training_custom": "PHASE_TRAINING_CUSTOM",
        }
        key = labels.get(phase)
        return i18n.t(key) if key else phase

    def on_phase_started(self, phase: str) -> tuple[int, int] | None:
        self._active_runner_phase = phase
        if phase == "colmap_rig_export":
            self._converted_total = 0
            self._processed = 0
            self._explicit_progress = False
            return None
        if phase == "spheresfm_cubemap":
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
        if phase == "spheresfm_preflight":
            return 0, 1
        if phase == "spheresfm_feature":
            total = self._count_source_images()
            return 0, total if total > 0 else 0
        if phase in {"colmap_rig_config", "colmap_match", "colmap_mapper"}:
            self._colmap_ba_iterations = 0
            return 0, 0
        if phase in {"spheresfm_database", "spheresfm_match", "spheresfm_mapper", "spheresfm_transforms"}:
            self._colmap_ba_iterations = 0
            return 0, 0
        return None

    def on_line(self, line: str) -> tuple[int, int] | None:
        if self._is_spheresfm_method() and is_spheresfm_rtx50_cuda_error_line(line):
            self._spheresfm_rtx50_cuda_error_seen = True
            if self._active_runner_phase.startswith("spheresfm_"):
                self._spheresfm_rtx50_cuda_error_phase = self._active_runner_phase

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
        return self._count_images_in_dir(images_dir)

    def _count_source_images(self) -> int:
        if not self.scene_dir:
            return 0
        images_dir = self._metashape_images_dir()
        return self._count_images_in_dir(images_dir)

    # -- ヘルパー --

    @staticmethod
    def _format_candidate_names(paths: tuple[Path, ...] | list[Path]) -> str:
        names = [p.name for p in paths]
        if len(names) > 4:
            names = names[:4] + [f"+{len(names) - 4}"]
        return ", ".join(names)

    @staticmethod
    def _path_is_same_or_descendant(path: Path, root: Path) -> bool:
        try:
            resolved_path = path.resolve()
            resolved_root = root.resolve()
        except OSError:
            resolved_path = path.absolute()
            resolved_root = root.absolute()
        return resolved_path == resolved_root or resolved_root in resolved_path.parents

    def _metashape_input_output_path_issue(self, path: Path) -> str | None:
        if not self.scene_dir:
            return None
        try:
            output = self._display_output_dir()
        except ValueError:
            return None
        if self._path_is_same_or_descendant(path, output):
            return i18n.t("MS_INPUT_IN_OUTPUT_HINT").format(path=str(path), output=str(output))
        return None

    @staticmethod
    def _scene_xml_candidates(scene_dir: Path) -> tuple[Path, ...]:
        return tuple(sorted([p for p in scene_dir.glob("*.xml") if p.is_file()], key=lambda x: x.name.lower()))

    @staticmethod
    def _scene_ply_candidates(scene_dir: Path) -> tuple[Path, ...]:
        return tuple(
            sorted(
                [
                    p
                    for p in scene_dir.glob("*.ply")
                    if p.is_file() and p.name.lower() != _GENERATED_POINTCLOUD_NAME
                ],
                key=lambda x: x.name.lower(),
            )
        )

    def _guess_xml(self, scene_dir: Path) -> Path | None:
        scored = [
            score
            for candidate in self._scene_xml_candidates(scene_dir)
            if (score := self._metashape_xml_candidate_score(candidate, scene_dir)) is not None
        ]
        self._metashape_auto_xml_candidates = tuple(score[0] for score in scored)
        if not scored:
            return None
        if len(scored) == 1:
            return scored[0][0]
        scored.sort(key=lambda item: (item[1], item[2], item[3], item[0].name.lower()), reverse=True)
        best = scored[0]
        second = scored[1]
        if best[1] > second[1]:
            return best[0]
        return None

    def _guess_ply(self, scene_dir: Path) -> Path | None:
        candidates = self._scene_ply_candidates(scene_dir)
        self._metashape_auto_ply_candidates = candidates
        return candidates[0] if len(candidates) == 1 else None

    def _metashape_xml_candidate_score(self, path: Path, scene_dir: Path) -> tuple[Path, int, int, int] | None:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            return None
        if self._xml_tag_name(root.tag) != "document":
            return None
        image_names, image_stems = self._metashape_image_name_sets(scene_dir)
        total_cameras = 0
        transformed_cameras = 0
        image_matches = 0
        chunks = [node for node in root.iter() if self._xml_tag_name(node.tag) == "chunk"]
        for chunk in chunks:
            sensors = self._metashape_sensor_ids(chunk)
            if not sensors:
                continue
            cameras_parent = self._xml_child(chunk, "cameras")
            if cameras_parent is None:
                continue
            for camera in self._xml_children(cameras_parent, "camera"):
                if str(camera.get("sensor_id") or "").strip() not in sensors:
                    continue
                label = str(camera.get("label") or "").strip()
                if not label:
                    continue
                total_cameras += 1
                transform = self._xml_child(camera, "transform")
                if self._xml_transform_has_16_numbers(transform):
                    transformed_cameras += 1
                if self._metashape_label_matches_image(label, image_names, image_stems):
                    image_matches += 1
        if total_cameras <= 0 or transformed_cameras <= 0:
            return None
        return (path, image_matches, transformed_cameras, total_cameras)

    @staticmethod
    def _xml_tag_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    @classmethod
    def _xml_child(cls, element: ET.Element, name: str) -> ET.Element | None:
        for child in element:
            if cls._xml_tag_name(child.tag) == name:
                return child
        return None

    @classmethod
    def _xml_children(cls, element: ET.Element, name: str) -> list[ET.Element]:
        return [child for child in element if cls._xml_tag_name(child.tag) == name]

    @classmethod
    def _metashape_sensor_ids(cls, chunk: ET.Element) -> set[str]:
        sensors_parent = cls._xml_child(chunk, "sensors")
        if sensors_parent is None:
            return set()
        sensor_ids: set[str] = set()
        for sensor in cls._xml_children(sensors_parent, "sensor"):
            sensor_id = str(sensor.get("id") or "").strip()
            if not sensor_id:
                continue
            if str(sensor.get("type") or "").strip() or cls._xml_child(sensor, "calibration") is not None:
                sensor_ids.add(sensor_id)
        return sensor_ids

    @staticmethod
    def _xml_transform_has_16_numbers(transform: ET.Element | None) -> bool:
        if transform is None or not transform.text:
            return False
        parts = transform.text.split()
        if len(parts) != 16:
            return False
        try:
            values = [float(part) for part in parts]
        except ValueError:
            return False
        return all(math.isfinite(value) for value in values)

    @staticmethod
    def _metashape_image_name_sets(scene_dir: Path) -> tuple[set[str], set[str]]:
        images_dir = scene_dir / "images"
        if not images_dir.is_dir():
            return set(), set()
        names: set[str] = set()
        stems: set[str] = set()
        for image in images_dir.iterdir():
            if not image.is_file() or image.suffix.lower() not in _SUPPORTED_TRAINING_IMAGE_EXTS:
                continue
            names.add(image.name)
            stems.add(image.stem)
        return names, stems

    @staticmethod
    def _metashape_label_matches_image(label: str, image_names: set[str], image_stems: set[str]) -> bool:
        if not image_names and not image_stems:
            return False
        name = label.replace("\\", "/").rsplit("/", 1)[-1]
        stem = Path(name).stem
        return name in image_names or stem in image_stems or label in image_names or label in image_stems
