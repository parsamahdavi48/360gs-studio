"""Step 4 training command validation and launch construction."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QLineEdit

from gui import i18n
from gui.common.runner_types import ExternalCommandQueue
from gui.steps.step4_contracts import _LFS_ADVANCED_INT_KEYS, _LFS_ADVANCED_LIST_KEYS
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_BRUSH as _TRAINING_BACKEND_BRUSH,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_GSPLAT as _TRAINING_BACKEND_GSPLAT,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_LICHTFELD as _TRAINING_BACKEND_LICHTFELD,
)
from gui.steps.training_backend_specs import (
    TRAINING_BACKEND_POSTSHOT as _TRAINING_BACKEND_POSTSHOT,
)
from gui.steps.training_backend_specs import (
    training_backend_phase_name,
)
from gui.steps.training_backends import (
    BrushTrainingOptions,
    GsplatTrainingOptions,
    LichtFeldTrainingOptions,
    PostshotTrainingOptions,
    brush_export_filename,
    build_brush_training_cmd,
    build_gsplat_training_cmd,
    build_lichtfeld_training_cmd,
    build_postshot_training_cmd,
    lichtfeld_output_name_stem,
)


class Step4TrainingCommandsMixin:
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

    def _parse_optional_positive_int(self, edit: QLineEdit, label: str) -> int | None:
        raw = edit.text().strip()
        if not raw:
            return None
        return self._parse_positive_int(edit, label)

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
        if backend == _TRAINING_BACKEND_BRUSH:
            iterations = self._parse_positive_int(self.brush_iterations_edit, i18n.t("BRUSH_ITERATIONS"))
            export_name = self._filename_only(
                self.brush_export_name_edit.text().strip(),
                i18n.t("BRUSH_EXPORT_NAME"),
            )
            self._guard_training_output_target(output_dir / brush_export_filename(export_name, iterations))
            return
        if backend == _TRAINING_BACKEND_GSPLAT:
            result_name = self._filename_only(
                self.gsplat_result_name_edit.text().strip() or self._default_gsplat_result_name(),
                i18n.t("GSPLAT_RESULT_NAME"),
            )
            self._guard_training_output_target(output_dir / result_name)
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

    def _build_training_commands(self) -> ExternalCommandQueue:
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

        if backend == _TRAINING_BACKEND_BRUSH:
            iterations = self._parse_positive_int(self.brush_iterations_edit, i18n.t("BRUSH_ITERATIONS"))
            export_name = self._filename_only(
                self.brush_export_name_edit.text().strip(),
                i18n.t("BRUSH_EXPORT_NAME"),
            )
            self._guard_training_output_target(output_dir / brush_export_filename(export_name, iterations))
            cmd = build_brush_training_cmd(
                BrushTrainingOptions(
                    executable=executable,
                    dataset=dataset,
                    output_dir=output_dir,
                    export_name=export_name,
                    total_train_iters=iterations,
                    export_every=self._parse_positive_int(
                        self.brush_export_every_edit,
                        i18n.t("BRUSH_EXPORT_EVERY"),
                    ),
                    max_resolution=self._parse_positive_int(
                        self.brush_max_resolution_edit,
                        i18n.t("BRUSH_MAX_RESOLUTION"),
                    ),
                    with_viewer=self.brush_with_viewer_cb.isChecked(),
                    sh_degree=int(self.brush_sh_degree_combo.currentData()),
                    render_mode=self.brush_render_mode_combo.currentData() or "auto",
                    refine_every=self._parse_positive_int(
                        self.brush_refine_every_edit,
                        i18n.t("BRUSH_REFINE_EVERY"),
                    ),
                    max_splats=self._parse_positive_int(
                        self.brush_max_splats_edit,
                        i18n.t("BRUSH_MAX_SPLATS"),
                    ),
                    eval_split_every=self._parse_optional_positive_int(
                        self.brush_eval_split_every_edit,
                        i18n.t("BRUSH_EVAL_SPLIT_EVERY"),
                    ),
                    alpha_mode=self.brush_alpha_mode_combo.currentData() or "auto",
                    subsample_frames=self._parse_optional_positive_int(
                        self.brush_subsample_frames_edit,
                        i18n.t("BRUSH_SUBSAMPLE_FRAMES"),
                    ),
                    subsample_points=self._parse_optional_positive_int(
                        self.brush_subsample_points_edit,
                        i18n.t("BRUSH_SUBSAMPLE_POINTS"),
                    ),
                )
            )
            return [(training_backend_phase_name(backend), cmd)]

        if backend == _TRAINING_BACKEND_GSPLAT:
            script_path = Path(self.gsplat_script_browse.text().strip() or str(self._default_gsplat_script_path()))
            if not script_path.is_file():
                raise ValueError(i18n.t("GSPLAT_SCRIPT_NOT_FOUND").format(path=str(script_path)))
            result_name = self._filename_only(
                self.gsplat_result_name_edit.text().strip() or self._default_gsplat_result_name(),
                i18n.t("GSPLAT_RESULT_NAME"),
            )
            result_dir = output_dir / result_name
            self._guard_training_output_target(result_dir)
            cmd = build_gsplat_training_cmd(
                GsplatTrainingOptions(
                    executable=executable,
                    script_path=script_path,
                    dataset=dataset,
                    result_dir=result_dir,
                    strategy=self.gsplat_strategy_combo.currentData() or "default",
                    max_steps=self._parse_positive_int(self.gsplat_max_steps_edit, i18n.t("GSPLAT_MAX_STEPS")),
                    data_factor=self._parse_positive_int(
                        self.gsplat_data_factor_edit,
                        i18n.t("GSPLAT_DATA_FACTOR"),
                    ),
                    test_every=self._parse_positive_int(
                        self.gsplat_test_every_edit,
                        i18n.t("GSPLAT_TEST_EVERY"),
                    ),
                    save_ply=self.gsplat_save_ply_cb.isChecked(),
                    disable_viewer=self.gsplat_disable_viewer_cb.isChecked(),
                    with_3dgut=self.gsplat_3dgut_cb.isChecked(),
                )
            )
            return [(training_backend_phase_name(backend), cmd)]

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

        if backend != _TRAINING_BACKEND_LICHTFELD:
            raise ValueError(f"Unsupported training backend: {backend}")

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

    def build_training_launch_commands(self) -> ExternalCommandQueue:
        self._training_phase_logs.clear()
        self.prepare_training_step()
        return self._build_training_commands()

    def on_training_queue_finished(self, success: bool) -> None:
        if success:
            self._record_step4_training_run()
