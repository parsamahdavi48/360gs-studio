"""Step 4 LichtFeld training UI state helpers."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QFormLayout, QLineEdit

from gui.steps.step4_contracts import (
    _LFS_ADVANCED_FLOAT_FORMATS,
    _LFS_ADVANCED_INT_KEYS,
    _LFS_ADVANCED_LIST_KEYS,
    _LFS_STRATEGIES,
    _LFS_UI_STEP_ADVANCED_KEYS,
    _LFS_UI_STEP_TEXT_KEYS,
)
from gui.steps.training_backends import lichtfeld_defaults


class Step4TrainingLfsStateMixin:
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
            "steps_scaler": self._format_lfs_float("steps_scaler", defaults.get("steps_scaler", 1.0)),
            "bilateral_grid": bool(defaults.get("use_bilateral_grid", False)),
            "mask_mode": str(defaults.get("mask_mode", "none")),
            "depth_loss": bool(defaults.get("use_depth_loss", False)),
            "depth_loss_mode": str(defaults.get("depth_loss_mode", "adaptive-warped-l1")),
            "depth_loss_weight": self._format_lfs_float("depth_loss_weight", defaults.get("depth_loss_weight", 2.0)),
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
            "steps_scaler": self.lfs_steps_scaler_edit.text().strip(),
            "bilateral_grid": self.lfs_bilateral_grid_cb.isChecked(),
            "mask_mode": self.lfs_mask_mode_combo.currentData() or "none",
            "depth_loss": self.lfs_depth_loss_cb.isChecked(),
            "depth_loss_mode": self.lfs_depth_loss_mode_combo.currentData() or "adaptive-warped-l1",
            "depth_loss_weight": self.lfs_depth_loss_weight_edit.text().strip(),
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
            self.lfs_steps_scaler_edit.setText(str(state.get("steps_scaler", "1.00")))
            self.lfs_bilateral_grid_cb.setChecked(bool(state.get("bilateral_grid", False)))
            self._set_combo_data(self.lfs_mask_mode_combo, str(state.get("mask_mode", "none")))
            self.lfs_depth_loss_cb.setChecked(bool(state.get("depth_loss", False)))
            self._set_combo_data(self.lfs_depth_loss_mode_combo, str(state.get("depth_loss_mode", "adaptive-warped-l1")))
            self.lfs_depth_loss_weight_edit.setText(str(state.get("depth_loss_weight", "2.000")))
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
        segment_mask = mask_mode in {"segment", "segment_and_ignore"}
        self._set_lfs_form_row_visible(self.lfs_mask_invert_row, has_mask)
        self._set_lfs_form_row_visible(self.lfs_mask_threshold_row, has_mask)
        self._set_lfs_form_row_visible(self.lfs_use_alpha_as_mask_row, has_mask)
        self._set_lfs_form_row_visible(self.lfs_mask_opacity_penalty_weight_row, segment_mask)
        self._set_lfs_form_row_visible(self.lfs_mask_opacity_penalty_power_row, segment_mask)
        depth_loss = self.lfs_depth_loss_cb.isChecked()
        self._set_lfs_form_row_visible(self.lfs_depth_loss_mode_row, depth_loss)
        self._set_lfs_form_row_visible(self.lfs_depth_loss_weight_row, depth_loss)

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
