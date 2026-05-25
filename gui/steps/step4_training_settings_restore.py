"""Step 4 training settings restore helpers."""

from __future__ import annotations

from pathlib import Path

from gui.steps.step4_contracts import _LFS_STRATEGIES
from gui.steps.training_backend_specs import training_backend_visible_in_selector


class Step4TrainingSettingsRestoreMixin:
    # -- シーンディレクトリ --

    def _restore_training_settings(self, scene: Path, settings: dict) -> None:
        training = settings.get("training")
        if not isinstance(training, dict):
            return
        self.run_training_cb.setChecked(bool(training.get("enabled", False)))
        backend = str(training.get("backend", "")).strip()
        backend_is_visible = training_backend_visible_in_selector(backend) if backend else False
        self._restore_training_executables(training.get("executables"))
        if backend:
            self._set_training_backend(backend)
        executable = self._settings_text(training.get("executable"))
        if executable and (not backend or backend_is_visible):
            self._training_executable_by_backend[self._training_backend()] = executable
            self._apply_training_executable_for_backend(self._training_backend())
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
