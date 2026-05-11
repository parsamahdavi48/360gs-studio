"""Step 4 settings and run-manifest persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from core.scene_layout import STEP4_META_DIR_NAME, STEP4_VIEWS_CONFIG_JSON, step4_export_settings_path, step4_meta_dir
from core.scene_project import (
    append_step4_dataset_run,
    append_step4_sfm_run,
    append_step4_training_run,
    file_identity,
    scene_relative,
    utc_now_iso,
)
from gui import i18n
from gui.steps.cubemap_commands import views_config_payload, write_views_config
from gui.steps.step4_contracts import (
    _COLMAP_MAPPER_INCREMENTAL,
    _COLMAP_MATCHER_SEQUENTIAL,
    _COLMAP_PROJECT_MANIFEST_NAME,
    _EXPORT_SETTINGS_NAME,
    _LICHTFELD_FINAL_CORRECTION,
    _METHOD_COLMAP,
    _METHOD_SPHERESFM,
    _SPHERESFM_PROJECT_MANIFEST_NAME,
)
from gui.steps.step4_settings import STEP4_SETTINGS_VERSION, load_step4_export_settings, write_step4_export_settings
from gui.steps.training_backends import TrainingDataset
from gui.version import APP_VERSION


class Step4ManifestMixin:
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
        if direct_source_output:
            input_transforms_json = output / "transforms.json"
        elif self._is_metashape_method():
            input_transforms_json = self._metashape_import_work_dir() / "transforms.json"
        else:
            input_transforms_json = scene / "transforms.json"

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
                    self._uses_lichtfeld_final_correction() or self._uses_spheresfm_lichtfeld_final_correction()
                ),
                "lichtfeld_final_orientation_matrix": _LICHTFELD_FINAL_CORRECTION.tolist()
                if self._uses_lichtfeld_final_correction() or self._uses_spheresfm_lichtfeld_final_correction()
                else None,
            },
            "metashape_import": {
                "enabled": self._is_metashape_method(),
                "use_ply": self._preprocess_uses_ply(),
                "images_dir": str(self._metashape_images_dir()),
                "xml": self.ms_xml_browse.text(),
                "ply": self.ms_ply_browse.text() if self._is_metashape_method() and self._preprocess_uses_ply() else "",
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
                "run_sfm": self._colmap_sfm_intent,
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
                "transforms_json": str(input_transforms_json),
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
                "target_profile": self._spheresfm_profile_id() if self._is_spheresfm_method() else self._profile_id(),
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
                "logs": self._training_log_snapshot(scene),
                "settings": self._collect_training_settings(),
            },
        )

    def _training_log_snapshot(self, scene: Path) -> dict[str, object]:
        log_dir = self.training_process_log_dir()
        return {
            "log_dir": scene_relative(scene, log_dir) if log_dir is not None else "",
            "phase_logs": {
                phase: scene_relative(scene, path) for phase, path in sorted(self._training_phase_logs.items())
            },
        }

    def _record_step4_runs(self, *, sfm_mode: str | None, dataset: bool) -> None:
        if sfm_mode:
            self._record_step4_sfm_run(sfm_mode)
        if dataset:
            self._record_step4_dataset_run()

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
            "run_sfm": self._colmap_sfm_intent,
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
