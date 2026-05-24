"""Step 4 training widget construction helpers."""

from __future__ import annotations

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

from gui import i18n
from gui.common.browse_widget import BrowseWidget
from gui.common.collapsible_section import CollapsibleSection
from gui.common.form_rows import add_tooltip_row
from gui.steps.step4_contracts import _LFS_ADVANCED_FIELD_WIDTHS, _LFS_STRATEGIES
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
)


class Step4TrainingUiMixin:
    def _build_training_section(self, exe_filter: str) -> QWidget:
        section = QWidget()
        self.training_section = section
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._training_executable_by_backend: dict[str, str] = {}
        self._syncing_training_executable = False

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
        self.training_executable_browse.path_changed.connect(self._on_training_executable_changed)
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

