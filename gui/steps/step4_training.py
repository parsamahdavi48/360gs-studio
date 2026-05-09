"""Step 4 training backend UI, settings, and launch command helpers."""

from __future__ import annotations

import math
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.scene_layout import project_path, scene_images_dir, scene_output_dir, step4_meta_dir
from core.scene_project import load_json
from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.form_rows import add_tooltip_row
from gui.steps.step4_contracts import (
    _LFS_ADVANCED_FIELD_WIDTHS,
    _LFS_ADVANCED_FLOAT_FORMATS,
    _LFS_ADVANCED_INT_KEYS,
    _LFS_ADVANCED_LIST_KEYS,
    _LFS_STRATEGIES,
    _LFS_UI_STEP_ADVANCED_KEYS,
    _LFS_UI_STEP_TEXT_KEYS,
    _OUTPUT_SHAPE_EQUIRECT_3DGUT,
    _OUTPUT_SHAPE_PROJECTED,
    _SUPPORTED_TRAINING_IMAGE_EXTS,
)
from gui.steps.step4_settings import load_step4_export_settings
from gui.steps.step4_widgets import CurrentPageStack
from gui.steps.training_backend_selector import TrainingBackendSelector
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_CUSTOM as _TRAINING_BACKEND_CUSTOM,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_LICHTFELD as _TRAINING_BACKEND_LICHTFELD,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_POSTSHOT as _TRAINING_BACKEND_POSTSHOT,
)
from gui.steps.training_backend_specs import (
    get_training_backend_spec,
    normalize_training_backend,
    training_backend_default_executable,
    training_backend_phase_name,
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


class Step4TrainingMixin:
    def _training_required_output_shape(self) -> str:
        backend = self._training_backend()
        if backend == _TRAINING_BACKEND_LICHTFELD:
            return _OUTPUT_SHAPE_EQUIRECT_3DGUT if self.lfs_gut_cb.isChecked() else _OUTPUT_SHAPE_PROJECTED
        if backend == _TRAINING_BACKEND_POSTSHOT:
            return _OUTPUT_SHAPE_PROJECTED
        return ""

    def _training_dataset_issue(self) -> str | None:
        required_shape = self._training_required_output_shape()
        return self._existing_training_dataset_issue(required_shape)

    def _existing_training_dataset_issue(self, required_shape: str) -> str | None:
        if not self._training_dataset_available():
            return i18n.t("TRAINING_REQUIRES_DATASET_OUTPUT")
        dataset = self._training_dataset()
        if not required_shape:
            return None
        actual_shape = self._training_dataset_export_shape(dataset.dataset_root)
        if required_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            if actual_shape == _OUTPUT_SHAPE_PROJECTED:
                return i18n.t("TRAINING_DATASET_EXISTING_NOT_3DGUT")
            if not (dataset.dataset_root / "pointcloud.ply").is_file():
                return i18n.t("TRAINING_DATASET_3DGUT_NEEDS_PLY")
            return None
        if actual_shape == _OUTPUT_SHAPE_EQUIRECT_3DGUT:
            return i18n.t("TRAINING_DATASET_EXISTING_NOT_PROJECTED")
        if (
            self._training_backend() == _TRAINING_BACKEND_POSTSHOT
            and (self.postshot_camera_poses_combo.currentData() or "import") == "import"
            and dataset.colmap_sparse_dir is None
            and not dataset.transforms_json.is_file()
        ):
            return i18n.t("POSTSHOT_IMPORT_POSES_NOT_FOUND")
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
        self.training_section = section
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        (
            self.training_path_summary_row,
            self.training_dataset_summary_kind,
            self.training_dataset_summary_value,
            self.training_output_summary_kind,
            self.training_output_summary_value,
        ) = self._make_training_path_summary_row()
        layout.addWidget(self.training_path_summary_row)

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

        self.training_common_fields_widget = QWidget()
        form = QFormLayout(self.training_common_fields_widget)
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

        training_settings_layout.addWidget(self.training_common_fields_widget)

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

    def apply_training_wide_layout(self) -> QWidget:
        if getattr(self, "_training_wide_layout_applied", False):
            return self.training_section

        section_layout = self.training_section.layout()
        if section_layout is None:
            return self.training_section

        for widget in (
            self.training_path_summary_row,
            self.training_backend_row,
            self.training_run_options_row,
            self.training_settings_scroll,
        ):
            section_layout.removeWidget(widget)

        old_settings_layout = self.training_settings_content.layout()
        if old_settings_layout is not None:
            old_settings_layout.removeWidget(self.training_common_fields_widget)
            old_settings_layout.removeWidget(self.training_options_stack)
        self.training_settings_scroll.takeWidget()

        self.training_options_scroll_content = QWidget()
        options_layout = QVBoxLayout(self.training_options_scroll_content)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(0)
        options_layout.addWidget(self.training_options_stack)
        options_layout.addStretch()
        self.training_settings_scroll.setWidget(self.training_options_scroll_content)
        self.training_settings_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.training_options_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.training_wide_controls_panel = QWidget()
        self.training_wide_controls_panel.setObjectName("trainingStepControls")
        self.training_wide_controls_panel.setMinimumWidth(360)
        self.training_wide_controls_panel.setMaximumWidth(460)
        controls_layout = QVBoxLayout(self.training_wide_controls_panel)
        controls_layout.setContentsMargins(0, 0, 12, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self.training_backend_row)
        controls_layout.addWidget(self.training_run_options_row)
        controls_layout.addWidget(self.training_common_fields_widget)
        controls_layout.addStretch()

        self.training_wide_body = QWidget()
        body_layout = QHBoxLayout(self.training_wide_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)
        body_layout.addWidget(self.training_wide_controls_panel)
        body_layout.addWidget(self.training_settings_scroll, stretch=1)

        section_layout.addWidget(self.training_path_summary_row)
        section_layout.addWidget(self.training_wide_body, stretch=1)
        self._training_wide_layout_applied = True
        self._refresh_training_settings_layout()
        return self.training_section

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
        add_tooltip_row(
            form,
            i18n.t("LFS_MASK_THRESHOLD"),
            self.lfs_mask_threshold_edit,
            i18n.tip("LFS_MASK_THRESHOLD"),
        )
        self.lfs_use_alpha_as_mask_cb = QCheckBox()
        self.lfs_use_alpha_as_mask_cb.setChecked(True)
        self.lfs_use_alpha_as_mask_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_USE_ALPHA_AS_MASK"),
            self.lfs_use_alpha_as_mask_cb,
            i18n.tip("LFS_USE_ALPHA_AS_MASK"),
        )
        self.lfs_mask_opacity_penalty_weight_edit = QLineEdit("1.000")
        self.lfs_mask_opacity_penalty_weight_edit.setFixedWidth(
            _LFS_ADVANCED_FIELD_WIDTHS.get("mask_opacity_penalty_weight", 86)
        )
        self.lfs_mask_opacity_penalty_weight_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_MASK_OPACITY_PENALTY_WEIGHT"),
            self.lfs_mask_opacity_penalty_weight_edit,
            i18n.tip("LFS_MASK_OPACITY_PENALTY_WEIGHT"),
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
            i18n.tip("LFS_MASK_OPACITY_PENALTY_POWER"),
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
        for index, (cb, tip_key) in enumerate(
            (
                (self.lfs_sparsity_cb, "LFS_SPARSITY"),
                (self.lfs_gut_cb, "LFS_GUT"),
                (self.lfs_undistort_cb, "LFS_UNDISTORT"),
                (self.lfs_mip_filter_cb, "LFS_MIP_FILTER"),
                (self.lfs_ppisp_cb, "LFS_PPISP"),
            )
        ):
            cb.setToolTip(i18n.tip(tip_key))
            checks_layout.addWidget(cb, index // 2, index % 2)
        form.addRow("", checks)

        self.lfs_ppisp_freeze_from_sidecar_cb = QCheckBox()
        self.lfs_ppisp_freeze_from_sidecar_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_FREEZE_FROM_SIDECAR"),
            self.lfs_ppisp_freeze_from_sidecar_cb,
            i18n.tip("LFS_PPISP_FREEZE_FROM_SIDECAR"),
        )
        self.lfs_ppisp_sidecar_browse = BrowseWidget(mode="file")
        self.lfs_ppisp_sidecar_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_SIDECAR_PATH"),
            self.lfs_ppisp_sidecar_browse,
            i18n.tip("LFS_PPISP_SIDECAR_PATH"),
        )
        self.lfs_ppisp_use_controller_cb = QCheckBox()
        self.lfs_ppisp_use_controller_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_USE_CONTROLLER"),
            self.lfs_ppisp_use_controller_cb,
            i18n.tip("LFS_PPISP_USE_CONTROLLER"),
        )
        self.lfs_ppisp_controller_activation_step_edit = QLineEdit("-1")
        self.lfs_ppisp_controller_activation_step_edit.setFixedWidth(
            _LFS_ADVANCED_FIELD_WIDTHS.get("ppisp_controller_activation_step", 86)
        )
        self.lfs_ppisp_controller_activation_step_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_CONTROLLER_ACTIVATION_STEP"),
            self.lfs_ppisp_controller_activation_step_edit,
            i18n.tip("LFS_PPISP_CONTROLLER_ACTIVATION_STEP"),
        )
        self.lfs_ppisp_controller_lr_edit = QLineEdit("0.0020")
        self.lfs_ppisp_controller_lr_edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get("ppisp_controller_lr", 86))
        self.lfs_ppisp_controller_lr_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_CONTROLLER_LR"),
            self.lfs_ppisp_controller_lr_edit,
            i18n.tip("LFS_PPISP_CONTROLLER_LR"),
        )
        self.lfs_ppisp_freeze_gaussians_on_distill_cb = QCheckBox()
        self.lfs_ppisp_freeze_gaussians_on_distill_cb.setChecked(True)
        self.lfs_ppisp_freeze_gaussians_on_distill_row = form.rowCount()
        add_tooltip_row(
            form,
            i18n.t("LFS_PPISP_FREEZE_GAUSSIANS_ON_DISTILL"),
            self.lfs_ppisp_freeze_gaussians_on_distill_cb,
            i18n.tip("LFS_PPISP_FREEZE_GAUSSIANS_ON_DISTILL"),
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
        advanced.setToolTip(i18n.tip("LFS_ADVANCED_PARAMETERS"))
        advanced.toggle_button.setToolTip(i18n.tip("LFS_ADVANCED_PARAMETERS"))
        adv_layout = advanced.content_layout

        def add_section(title_key: str) -> QFormLayout:
            section = CollapsibleSection(i18n.t(title_key), expanded=False)
            section.setObjectName("lfsAdvancedSubsection")
            section.setToolTip(i18n.tip(title_key))
            section.toggle_button.setToolTip(i18n.tip(title_key))
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
            tip_key = f"LFS_{key.upper()}"
            add_tooltip_row(section_form, i18n.t(tip_key), edit, i18n.tip(tip_key))
            return edit

        def add_check(section_form: QFormLayout, key: str, checked: bool = False) -> QCheckBox:
            cb = QCheckBox()
            cb.setChecked(checked)
            self.lfs_advanced_checks[key] = cb
            self.lfs_advanced_rows[key] = (section_form, section_form.rowCount())
            tip_key = f"LFS_{key.upper()}"
            add_tooltip_row(section_form, i18n.t(tip_key), cb, i18n.tip(tip_key))
            return cb

        dataset_form = add_section("LFS_SECTION_DATASET")
        self.lfs_dataset_resize_factor_combo = QComboBox()
        self.lfs_dataset_resize_factor_combo.addItem("Auto", "auto")
        for factor in (1, 2, 4, 8):
            self.lfs_dataset_resize_factor_combo.addItem(str(factor), str(factor))
        add_tooltip_row(
            dataset_form,
            i18n.t("LFS_RESIZE_FACTOR"),
            self.lfs_dataset_resize_factor_combo,
            i18n.tip("LFS_RESIZE_FACTOR"),
        )
        self.lfs_dataset_max_width_edit = QLineEdit("3840")
        self.lfs_dataset_max_width_edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get("max_width", 86))
        add_tooltip_row(
            dataset_form,
            i18n.t("LFS_MAX_WIDTH"),
            self.lfs_dataset_max_width_edit,
            i18n.tip("LFS_MAX_WIDTH"),
        )
        self.lfs_dataset_cpu_cache_cb = QCheckBox()
        self.lfs_dataset_cpu_cache_cb.setChecked(True)
        add_tooltip_row(
            dataset_form,
            i18n.t("LFS_CPU_CACHE"),
            self.lfs_dataset_cpu_cache_cb,
            i18n.tip("LFS_CPU_CACHE"),
        )
        self.lfs_dataset_fs_cache_cb = QCheckBox()
        self.lfs_dataset_fs_cache_cb.setChecked(True)
        add_tooltip_row(
            dataset_form,
            i18n.t("LFS_FS_CACHE"),
            self.lfs_dataset_fs_cache_cb,
            i18n.tip("LFS_FS_CACHE"),
        )
        add_check(dataset_form, "enable_eval", False)
        self.lfs_dataset_test_every_edit = QLineEdit("8")
        self.lfs_dataset_test_every_edit.setFixedWidth(_LFS_ADVANCED_FIELD_WIDTHS.get("test_every", 86))
        self.lfs_dataset_test_every_row = dataset_form.rowCount()
        self.lfs_advanced_rows["test_every"] = (dataset_form, self.lfs_dataset_test_every_row)
        add_tooltip_row(
            dataset_form,
            i18n.t("LFS_TEST_EVERY"),
            self.lfs_dataset_test_every_edit,
            i18n.tip("LFS_TEST_EVERY"),
        )
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
        self._lfs_strategy_states = {strategy: self._default_lfs_ui_state(strategy) for strategy in _LFS_STRATEGIES}
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
        add_row(
            camera_form,
            "no_recenter_points",
            self.postshot_no_recenter_points_cb,
            i18n.tip("POSTSHOT_NO_RECENTER_POINTS"),
        )

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
        add_row(
            model_form, "create_sky_model", self.postshot_create_sky_model_cb, i18n.tip("POSTSHOT_CREATE_SKY_MODEL")
        )
        self.postshot_store_training_context_cb = QCheckBox()
        add_row(
            model_form,
            "store_training_context",
            self.postshot_store_training_context_cb,
            i18n.tip("POSTSHOT_STORE_TRAINING_CONTEXT"),
        )
        self.postshot_show_train_error_cb = QCheckBox()
        add_row(
            model_form, "show_train_error", self.postshot_show_train_error_cb, i18n.tip("POSTSHOT_SHOW_TRAIN_ERROR")
        )

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
        self.postshot_profile_combo.currentIndexChanged.connect(
            lambda _idx: self._update_postshot_conditional_visibility()
        )
        self.postshot_image_select_combo.currentIndexChanged.connect(
            lambda _idx: self._update_postshot_conditional_visibility()
        )
        self.postshot_camera_poses_combo.currentIndexChanged.connect(
            lambda _idx: self._update_postshot_conditional_visibility()
        )
        self.postshot_crop_box_combo.currentIndexChanged.connect(
            lambda _idx: self._update_postshot_conditional_visibility()
        )
        self.postshot_roi_box_combo.currentIndexChanged.connect(
            lambda _idx: self._update_postshot_conditional_visibility()
        )
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
        self._update_path_labels()
        self._update_lfs_auto_steps_scaler()

    def _on_training_output_edited(self, _path: str) -> None:
        if self._syncing_training_paths:
            return
        self._training_output_user_edited = True
        self._update_path_labels()
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
            max(0, min(255, int(math.floor(float(component) * 255.0 + 0.5)))) for component in list(bg_color)[:3]
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
            "ppisp_freeze_gaussians_on_distill": bool(defaults.get("ppisp_freeze_gaussians_on_distill", True)),
            "background_mode": str(defaults.get("bg_mode", "solid_color")),
            "background_color": color_values,
            "background_image": str(defaults.get("bg_image_path", "")),
            "advanced_numbers": {
                key: self._format_lfs_advanced_value(key, defaults[key])
                for key in self.lfs_advanced_edits
                if key in defaults
            },
            "advanced_checks": {key: bool(defaults.get(key, False)) for key in self.lfs_advanced_checks},
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
            self.lfs_mask_opacity_penalty_weight_edit.setText(str(state.get("mask_opacity_penalty_weight", "1.000")))
            self.lfs_mask_opacity_penalty_power_edit.setText(str(state.get("mask_opacity_penalty_power", "2.000")))
            self.lfs_sparsity_cb.setChecked(bool(state.get("sparsity", False)))
            self.lfs_gut_cb.setChecked(bool(state.get("gut", False)))
            self.lfs_undistort_cb.setChecked(bool(state.get("undistort", False)))
            self.lfs_mip_filter_cb.setChecked(bool(state.get("mip_filter", False)))
            self.lfs_ppisp_cb.setChecked(bool(state.get("ppisp", False)))
            self.lfs_ppisp_freeze_from_sidecar_cb.setChecked(bool(state.get("ppisp_freeze_from_sidecar", False)))
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
                    {self._round_lfs_ui_step(value * ratio) for value in self._parse_lfs_steps_text_value(numbers[key])}
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
        self.lfs_bg_color_swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #4b5563;")

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

    def _training_dataset_dir(self) -> Path:
        raw = self.training_dataset_browse.text().strip()
        if raw:
            return Path(raw)
        return self._default_training_dataset_dir()

    def _default_training_output_dir(self) -> Path:
        if not self.scene_dir:
            raise ValueError(i18n.t("SCENE_REQUIRED_ACTION_HINT"))
        return scene_output_dir(Path(self.scene_dir))

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
            return self._find_colmap_sparse_model()
        if self._is_spheresfm_method():
            return self._find_spheresfm_sparse_model()
        output_colmap = self._display_output_dir() / "colmap"
        return output_colmap if output_colmap.is_dir() else None

    def _training_dataset(self) -> TrainingDataset:
        dataset_root = self._training_dataset_dir()
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

    def _count_images_in_dir(self, images_dir: Path | None) -> int:
        if images_dir is None or not images_dir.is_dir():
            return 0
        metadata_count = self._metadata_image_count_for_dir(images_dir)
        if metadata_count is not None:
            return metadata_count
        return sum(
            1
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_TRAINING_IMAGE_EXTS
        )

    def _training_image_count(self, dataset: TrainingDataset | None = None) -> int:
        if not self.scene_dir:
            return 0
        dataset = dataset or self._training_dataset()
        return self._count_images_in_dir(dataset.images_dir)

    def _metadata_image_count_for_dir(self, images_dir: Path) -> int | None:
        if not self.scene_dir:
            return None
        scene = Path(self.scene_dir)
        try:
            target = images_dir.resolve()
            output_images = (scene_output_dir(scene) / "images").resolve()
            source_images = scene_images_dir(scene).resolve()
        except OSError:
            return None

        project = load_json(project_path(scene), {})
        assets = project.get("assets") if isinstance(project.get("assets"), dict) else {}
        key = ""
        if target == output_images:
            key = "output_image_count"
        elif target == source_images:
            key = "source_image_count"
        if not key:
            return None
        try:
            count = int(assets.get(key))
        except (TypeError, ValueError):
            return None
        return count if count > 0 else None

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
        if (
            self.lfs_advanced_checks["enable_eval"].isChecked()
            and "save_steps" in overrides
            and "eval_steps" not in overrides
        ):
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
                    raw_export_splat_path if raw_export_splat_path.is_absolute() else output_dir / raw_export_splat_path
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

    def prepare_training_step(self) -> None:
        if hasattr(self, "run_training_cb"):
            self.run_training_cb.setChecked(True)
        self._update_training_paths()
        self._update_path_labels()
        self._update_lfs_auto_steps_scaler()

    def training_primary_action_enabled(self) -> bool:
        if not self.scene_dir or not hasattr(self, "run_training_cb"):
            return False
        return self._training_dataset_issue() is None

    def build_training_launch_commands(self) -> list[tuple[str, list[str]]]:
        self._training_phase_logs.clear()
        self.prepare_training_step()
        return self._build_training_commands()

    def on_training_queue_finished(self, success: bool) -> None:
        if success:
            self._record_step4_training_run()
