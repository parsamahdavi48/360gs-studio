from __future__ import annotations

from tests.helpers.step4 import (
    STEP4_SETTINGS_VERSION,
    CollapsibleSection,
    CubemapStep,
    Path,
    QMessageBox,
    TrainingStep,
    _app,
    _is_descendant,
    _ready_step,
    _write_ascii_ply,
    _write_output_dataset,
    _write_test_image,
    i18n,
    json,
    math,
    os,
    project_path,
    pytest,
    register_dataset_artifact,
    step4_export_settings_path,
    step4_meta_dir,
    step4_training_runs_path,
)


def test_training_launch_requires_existing_dataset_shape(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.lfs_gut_cb.setChecked(True)

    assert step.training_primary_action_enabled() is False

    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)

    assert step.training_primary_action_enabled() is False

    _write_output_dataset(tmp_path, output_shape="equirect_3dgut")

    assert step.training_primary_action_enabled() is True


def test_training_status_checks_existing_output_shape_when_cube_is_skipped(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_output_dataset(tmp_path, output_shape="projected", legacy_root=True)
    step.set_pipeline_stage_intent("conversion", False)
    step.lfs_gut_cb.setChecked(True)

    assert step.training_primary_action_enabled() is False
    with pytest.raises(ValueError, match="GUT"):
        step.build_training_launch_commands()

    _write_output_dataset(tmp_path, output_shape="equirect_3dgut")
    step._update_training_paths(force=True)
    assert step.training_primary_action_enabled() is True

    gut_pointcloud = tmp_path / "output" / "metashape_3dgut" / "pointcloud.ply"
    gut_pointcloud.unlink()
    assert step.training_primary_action_enabled() is False
    with pytest.raises(ValueError, match="pointcloud.ply"):
        step.build_training_launch_commands()
    _write_ascii_ply(gut_pointcloud, [(0.0, 0.0, 0.0)])

    step.lfs_gut_cb.setChecked(False)
    assert step.training_primary_action_enabled() is False


def test_step4_scene_settings_restore_export_and_training_choices(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    postshot_profile = step.profile_combo.findData("postshot")
    assert postshot_profile >= 0
    step.profile_combo.setCurrentIndex(postshot_profile)
    step.export_images_cb.setChecked(False)
    step.export_masks_cb.setChecked(True)
    step._set_combo_data(step.scale_combo, 0.5)
    step._set_combo_data(step.output_format_combo, "png")
    step._set_combo_data(step.output_bit_depth_combo, "source")
    step.jpg_quality_edit.setText("88")
    step.invert_masks_cb.setChecked(True)
    custom_mode = step.view_config.view_mode_combo.findData("custom_views")
    assert custom_mode >= 0
    step.view_config.view_mode_combo.setCurrentIndex(custom_mode)
    step.view_config.set_yaw_slot_count(5)
    step.view_config.set_pitch_row_count(2)
    step.view_config.yaw_offset_edit.setValue(12.0)
    step.view_config.pitch_rows[0]["checks"][0].setChecked(False)
    step.run_training_cb.setChecked(True)
    lfs_executable = str(tmp_path / "LichtFeld-Studio.exe")
    postshot_executable = str(tmp_path / "postshot-cli.exe")
    step.training_executable_browse.set_text(lfs_executable)
    step._set_training_backend("postshot")
    step.training_executable_browse.set_text(postshot_executable)
    step.postshot_project_name_edit.setText("compare.psht")
    step._write_export_settings()

    restored = CubemapStep(Path.cwd())
    restored.set_scene_dir(str(tmp_path))

    assert restored.profile_combo.currentData() == "postshot"
    assert restored.axis_transform_combo.currentData() == "postshot"
    assert restored.export_images_cb.isChecked() is True
    assert restored.export_masks_cb.isChecked() is True
    assert restored.scale_combo.currentData() == 0.5
    assert restored.output_format_combo.currentData() == "png"
    assert restored.output_bit_depth_combo.currentData() == "source"
    assert restored.jpg_quality_edit.text() == "88"
    assert restored.invert_masks_cb.isChecked() is True
    assert restored.view_config.view_mode() == "custom_views"
    assert restored.view_config.yaw_slot_count() == 5
    assert restored.view_config.pitch_row_count() == 2
    assert restored.view_config.yaw_offset() == pytest.approx(12.0)
    assert restored.view_config.collect_views(include_disabled=True)[0]["enabled"] is False
    assert restored.run_training_cb.isChecked() is True
    assert restored._training_backend() == "postshot"
    assert restored.training_executable_browse.text() == postshot_executable
    restored._set_training_backend("lichtfeld")
    assert restored.training_executable_browse.text() == lfs_executable
    assert restored.postshot_project_name_edit.text() == "compare.psht"


def test_postshot_does_not_use_lichtfeld_pointcloud(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    idx = step.profile_combo.findData("postshot")
    assert idx >= 0
    step.profile_combo.setCurrentIndex(idx)

    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match="pointcloud"):
        step.build_commands()


def test_postshot_accepts_raw_ply_with_custom_name(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    raw_ply = tmp_path / "raw_scan.ply"
    raw_ply.write_text("ply\n", encoding="utf-8")
    step.ms_ply_browse.set_text(str(raw_ply))
    idx = step.profile_combo.findData("postshot")
    assert idx >= 0
    step.profile_combo.setCurrentIndex(idx)

    assert step._resolve_ply_source() == raw_ply


def test_step5_uses_training_log_dir_independent_of_step4_route(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    training = TrainingStep(Path.cwd(), step)
    training.set_scene_dir(str(tmp_path))

    expected_training_logs = step4_meta_dir(tmp_path) / "logs" / "training"
    assert training.process_log_dir() == expected_training_logs
    assert step.process_log_dir() is None

    step._set_export_method("spheresfm")

    assert step.process_log_dir() == step4_meta_dir(tmp_path) / "logs" / "spheresfm"
    assert training.process_log_dir() == expected_training_logs


def test_step5_records_training_phase_logs(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_output_dataset(tmp_path, output_shape="projected")
    training = TrainingStep(Path.cwd(), step)
    training.set_scene_dir(str(tmp_path))
    training.on_activated()

    log_path = step4_meta_dir(tmp_path) / "logs" / "training" / "20260509_training_lichtfeld.log"
    training.on_phase_log_started("training_lichtfeld", str(log_path))
    training.on_queue_finished(True)

    runs = json.loads(step4_training_runs_path(tmp_path).read_text(encoding="utf-8"))["runs"]
    record = runs[-1]

    assert record["logs"] == {
        "log_dir": "_stechdrive/step4/logs/training",
        "phase_logs": {
            "training_lichtfeld": "_stechdrive/step4/logs/training/20260509_training_lichtfeld.log",
        },
    }


def test_step5_launch_builds_lichtfeld_command_and_writes_config(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_output_dataset(tmp_path, output_shape="projected")
    fake_lfs = tmp_path / "LichtFeld-Studio.exe"
    fake_lfs.write_text("", encoding="utf-8")
    training = TrainingStep(Path.cwd(), step)
    training.set_scene_dir(str(tmp_path))
    assert training.primary_action_text() == i18n.t("TRAINING_SELECT_APP")
    assert training.primary_action_enabled() is False
    training.show_backend("lichtfeld")

    assert step.lfs_output_name_edit.text() == tmp_path.name
    step.training_executable_browse.set_text(str(fake_lfs))
    step.lfs_auto_steps_scaler_cb.setChecked(False)
    step.lfs_steps_scaler_edit.setText("1.56")
    step._on_lfs_steps_scaler_editing_finished()
    step.lfs_iterations_edit.setText("46,700")
    step.lfs_max_gaussians_edit.setText("5,000,000")
    step.lfs_bilateral_grid_cb.setChecked(True)
    mask_mode_idx = step.lfs_mask_mode_combo.findData("ignore")
    assert mask_mode_idx >= 0
    step.lfs_mask_mode_combo.setCurrentIndex(mask_mode_idx)
    step.lfs_invert_masks_cb.setChecked(True)
    step.lfs_mask_threshold_edit.setText("0.250")
    step.lfs_use_alpha_as_mask_cb.setChecked(False)
    step.lfs_ppisp_cb.setChecked(True)
    step.lfs_ppisp_freeze_from_sidecar_cb.setChecked(True)
    step.lfs_ppisp_sidecar_browse.set_text(str(tmp_path / "frozen.ppisp"))
    step.lfs_ppisp_use_controller_cb.setChecked(True)
    step.lfs_ppisp_controller_activation_step_edit.setText("12000")
    step.lfs_ppisp_controller_lr_edit.setText("0.0015")
    step.lfs_ppisp_freeze_gaussians_on_distill_cb.setChecked(False)
    bg_mode_idx = step.lfs_bg_mode_combo.findData("modulation")
    assert bg_mode_idx >= 0
    step.lfs_bg_mode_combo.setCurrentIndex(bg_mode_idx)
    step.lfs_bg_r_edit.setText("12")
    step.lfs_bg_g_edit.setText("34")
    step.lfs_bg_b_edit.setText("56")
    resize_idx = step.lfs_dataset_resize_factor_combo.findData("2")
    assert resize_idx >= 0
    step.lfs_dataset_resize_factor_combo.setCurrentIndex(resize_idx)
    step.lfs_dataset_max_width_edit.setText("2048")
    step.lfs_dataset_cpu_cache_cb.setChecked(False)
    step.lfs_dataset_fs_cache_cb.setChecked(False)
    step.lfs_dataset_test_every_edit.setText("12")
    step.lfs_advanced_edits["means_lr"].setText("0.000123")
    step.lfs_advanced_checks["enable_eval"].setChecked(True)
    step.lfs_advanced_edits["save_steps"].setText("5000,30000")
    step.training_headless_cb.setChecked(True)

    conversion_scene = tmp_path / "convert_only"
    conversion_step = _ready_step(conversion_scene, metashape_inputs=True)
    _write_test_image(conversion_scene / "images" / "frame_0001.jpg")
    conversion_step.run_training_cb.setChecked(True)
    assert all(not phase.startswith("training_") for phase, _cmd in conversion_step.build_commands())
    assert training.primary_action_text() == i18n.t("LAUNCH")
    assert training.primary_action_enabled() is True

    commands = training.build_commands()

    assert [phase for phase, _cmd in commands] == ["training_lichtfeld"]
    cmd = commands[0][1]
    config_path = step._training_config_path()
    assert cmd[0] == str(fake_lfs)
    assert cmd[cmd.index("--data-path") + 1] == str(tmp_path / "output" / "metashape_cubemap")
    assert cmd[cmd.index("--output-path") + 1] == str(tmp_path / "output")
    assert cmd[cmd.index("--output-name") + 1] == tmp_path.name
    assert cmd[cmd.index("--config") + 1] == str(config_path)
    assert "--train" in cmd
    assert "--no-splash" in cmd
    assert "--headless" in cmd
    assert cmd[cmd.index("--resize_factor") + 1] == "2"
    assert cmd[cmd.index("--max-width") + 1] == "2048"
    assert "--no-cpu-cache" in cmd
    assert "--no-fs-cache" in cmd
    assert cmd[cmd.index("--test-every") + 1] == "12"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["strategy"] == "mrnf"
    assert config["iterations"] == 29936
    assert config["max_cap"] == 5_000_000
    assert config["sh_degree"] == 3
    assert config["tile_mode"] == 1
    assert config["steps_scaler"] == pytest.approx(1.56)
    assert config["use_bilateral_grid"] is True
    assert config["mask_mode"] == "ignore"
    assert config["invert_masks"] is True
    assert config["mask_threshold"] == pytest.approx(0.25)
    assert config["use_alpha_as_mask"] is False
    assert config["use_ppisp"] is True
    assert config["ppisp_freeze_from_sidecar"] is True
    assert config["ppisp_sidecar_path"] == str(tmp_path / "frozen.ppisp")
    assert config["ppisp_use_controller"] is True
    assert config["ppisp_controller_activation_step"] == 12000
    assert config["ppisp_controller_lr"] == pytest.approx(0.0015)
    assert config["ppisp_freeze_gaussians_on_distill"] is False
    assert config["bg_mode"] == "modulation"
    assert config["bg_color"] == pytest.approx([12 / 255, 34 / 255, 56 / 255])
    assert config["means_lr"] == pytest.approx(0.000123)
    assert config["enable_eval"] is True
    assert config["eval_steps"] == [3205, 19231]
    assert config["save_steps"] == [3205, 19231]
    assert config["headless"] is True


def test_training_headless_option_stays_in_run_options_row(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.resize(1280, 920)
    step.show()
    _app().processEvents()

    assert not _is_descendant(step.run_training_cb, step.training_run_options_row)
    assert _is_descendant(step.training_headless_cb, step.training_run_options_row)
    assert _is_descendant(step.training_backend_buttons["lichtfeld"], step.training_backend_row)
    assert _is_descendant(step.training_backend_selector.other_picker, step.training_backend_row)
    assert step.training_backend_selector.other_picker.isHidden()
    assert not _is_descendant(step.training_backend_buttons["lichtfeld"], step.training_run_options_row)
    assert not step.training_headless_cb.isHidden()

    step._set_training_backend("postshot")
    assert step.training_headless_cb.isHidden()
    assert not step.training_backend_other_button.isChecked()

    step._set_training_backend("custom")
    assert step._training_backend() == "lichtfeld"
    assert step.training_backend_buttons["lichtfeld"].isChecked()


def test_lichtfeld_training_refuses_existing_output_ply(tmp_path: Path, monkeypatch) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    _write_output_dataset(tmp_path, output_shape="equirect_3dgut")
    fake_lfs = tmp_path / "LichtFeld-Studio.exe"
    fake_lfs.write_text("", encoding="utf-8")
    existing = tmp_path / "output" / f"{tmp_path.name}.ply"
    existing.write_text("existing", encoding="utf-8")
    old_dataset_file = tmp_path / "output" / "metashape_3dgut" / "images" / "old.jpg"
    old_dataset_file.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    step.run_training_cb.setChecked(True)
    step.lfs_gut_cb.setChecked(True)
    step.training_executable_browse.set_text(str(fake_lfs))

    with pytest.raises(ValueError, match=existing.name):
        step.build_training_launch_commands()
    assert old_dataset_file.is_file()


def test_postshot_training_defaults_to_scene_project_and_refuses_collision(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    fake_postshot = tmp_path / "postshot-cli.exe"
    fake_postshot.write_text("", encoding="utf-8")
    existing = tmp_path / "output" / f"{tmp_path.name}.psht"
    existing.parent.mkdir(exist_ok=True)
    existing.write_text("existing", encoding="utf-8")
    colmap_images = tmp_path / "output" / "colmap_rig" / "images"
    colmap_images.mkdir(parents=True)
    _write_test_image(colmap_images / "frame_0001.jpg")
    colmap_sparse = tmp_path / "output" / "colmap_rig" / "sparse" / "0"
    colmap_sparse.mkdir(parents=True)
    (colmap_sparse / "cameras.txt").write_text("", encoding="utf-8")
    (colmap_sparse / "images.txt").write_text("", encoding="utf-8")
    (colmap_sparse / "points3D.txt").write_text("", encoding="utf-8")

    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))
    step._set_training_backend("postshot")
    step.training_executable_browse.set_text(str(fake_postshot))
    step.run_training_cb.setChecked(True)

    assert step.training_output_browse.text() == str(tmp_path / "output")
    assert step.postshot_project_name_edit.text() == f"{tmp_path.name}.psht"
    with pytest.raises(ValueError, match=existing.name):
        step.build_training_launch_commands()


def test_postshot_cli_options_are_grouped_and_conditional(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step._set_training_backend("postshot")

    assert step.postshot_profile_combo.currentData() == "Splat3"
    assert step.postshot_ksteps_auto_cb.isChecked()
    assert not step.postshot_ksteps_edit.isEnabled()
    assert step.postshot_max_image_size_edit.text() == "3840"
    assert step.postshot_import_masks_cb.isChecked() is False
    assert step.postshot_mask_mode_combo.isHidden()
    assert step.postshot_image_select_combo.currentData() == "all"
    assert not step.postshot_num_train_images_edit.isEnabled()
    assert step.postshot_camera_poses_combo.currentData() == "import"
    assert not step.postshot_pose_quality_combo.isEnabled()

    expected_sections = {
        "POSTSHOT_SECTION_CAMERA",
        "POSTSHOT_SECTION_MODEL",
        "POSTSHOT_SECTION_REGION",
        "POSTSHOT_SECTION_OUTPUT",
    }
    assert set(step.postshot_advanced_sections) == expected_sections
    assert all(isinstance(section, CollapsibleSection) for section in step.postshot_advanced_sections.values())
    assert all(not section.toggle_button.isChecked() for section in step.postshot_advanced_sections.values())
    assert step.postshot_splat_density_edit.isHidden()
    assert step.postshot_max_num_splats_edit.isHidden()
    assert step.postshot_crop_box_min_edit.isHidden()
    assert step.postshot_roi_box_min_edit.isHidden()

    step.postshot_import_masks_cb.setChecked(True)
    assert not step.postshot_mask_mode_combo.isHidden()

    adc_idx = step.postshot_profile_combo.findData("Splat ADC")
    assert adc_idx >= 0
    step.postshot_profile_combo.setCurrentIndex(adc_idx)
    assert not step.postshot_splat_density_edit.isHidden()
    assert step.postshot_max_num_splats_edit.isHidden()

    mcmc_idx = step.postshot_profile_combo.findData("Splat MCMC")
    assert mcmc_idx >= 0
    step.postshot_profile_combo.setCurrentIndex(mcmc_idx)
    assert step.postshot_splat_density_edit.isHidden()
    assert not step.postshot_max_num_splats_edit.isHidden()

    best_idx = step.postshot_image_select_combo.findData("best")
    assert best_idx >= 0
    step.postshot_image_select_combo.setCurrentIndex(best_idx)
    assert step.postshot_num_train_images_edit.isEnabled()

    estimate_idx = step.postshot_camera_poses_combo.findData("estimate")
    assert estimate_idx >= 0
    step.postshot_camera_poses_combo.setCurrentIndex(estimate_idx)
    assert step.postshot_pose_quality_combo.isEnabled()

    custom_idx = step.postshot_crop_box_combo.findData("custom")
    assert custom_idx >= 0
    step.postshot_crop_box_combo.setCurrentIndex(custom_idx)
    assert not step.postshot_crop_box_min_edit.isHidden()
    assert not step.postshot_crop_box_max_edit.isHidden()


def test_postshot_training_imports_transforms_and_raw_ply_for_metashape(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_output_dataset(tmp_path, output_shape="projected")
    fake_postshot = tmp_path / "postshot-cli.exe"
    fake_postshot.write_text("", encoding="utf-8")
    postshot_idx = step.profile_combo.findData("postshot")
    assert postshot_idx >= 0
    step.profile_combo.setCurrentIndex(postshot_idx)

    step._set_training_backend("postshot")
    step.training_executable_browse.set_text(str(fake_postshot))
    step.run_training_cb.setChecked(True)

    phase, cmd = step.build_training_launch_commands()[0]

    assert phase == "training_postshot"
    import_index = cmd.index("--import")
    assert cmd[import_index + 1 : import_index + 4] == [
        str(tmp_path / "output" / "metashape_cubemap" / "images"),
        str(tmp_path / "output" / "metashape_cubemap" / "transforms.json"),
        str(tmp_path / "metashape.ply"),
    ]


def test_training_tab_auto_scales_lichtfeld_from_projected_image_count(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    dataset_root = tmp_path / "output" / "metashape_cubemap"
    output_images = dataset_root / "images"
    output_images.mkdir(parents=True)
    expected_image_count = 468
    for idx in range(expected_image_count):
        _write_test_image(output_images / f"view_{idx:04d}.jpg")
    (dataset_root / "transforms.json").write_text("{}", encoding="utf-8")
    settings_path = step4_export_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "settings_version": STEP4_SETTINGS_VERSION,
                "output_shape": "projected",
                "output_dir": str(dataset_root),
            }
        ),
        encoding="utf-8",
    )
    expected_scaler = expected_image_count / 300
    fake_lfs = tmp_path / "LichtFeld-Studio.exe"
    fake_lfs.write_text("", encoding="utf-8")

    step.run_training_cb.setChecked(True)
    step.training_executable_browse.set_text(str(fake_lfs))
    step._update_lfs_auto_steps_scaler()

    assert step.lfs_auto_steps_scaler_cb.isChecked()
    assert not step.lfs_steps_scaler_edit.isEnabled()
    assert float(step.lfs_steps_scaler_edit.text()) == pytest.approx(expected_scaler, abs=0.005)
    assert int(step.lfs_iterations_edit.text().replace(",", "")) == math.floor(30000 * expected_scaler + 0.5)
    assert int(step.lfs_advanced_edits["stop_refine"].text().replace(",", "")) == math.floor(
        28500 * expected_scaler + 0.5
    )

    mcmc_idx = step.lfs_strategy_combo.findData("mcmc")
    assert mcmc_idx >= 0
    step.lfs_strategy_combo.setCurrentIndex(mcmc_idx)
    assert int(step.lfs_iterations_edit.text().replace(",", "")) == math.floor(30000 * expected_scaler + 0.5)
    assert int(step.lfs_advanced_edits["stop_refine"].text().replace(",", "")) == math.floor(
        25000 * expected_scaler + 0.5
    )

    commands = step.build_training_launch_commands()

    assert commands[0][0] == "training_lichtfeld"
    config = json.loads(step._training_config_path().read_text(encoding="utf-8"))
    assert config["strategy"] == "mcmc"
    assert config["steps_scaler"] == pytest.approx(expected_scaler)
    assert config["iterations"] == 30000
    assert config["stop_refine"] == 25000
    assert config["max_cap"] == 1_000_000


def test_training_default_dataset_prefers_registered_artifact(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    settings_output = tmp_path / "output" / "old_dataset"
    artifact_output = tmp_path / "output" / "registered_dataset"
    settings_output.mkdir(parents=True)
    artifact_output.mkdir(parents=True)
    (artifact_output / "transforms.json").write_text("{}", encoding="utf-8")
    settings_path = step4_export_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "settings_version": STEP4_SETTINGS_VERSION,
                "output_shape": "projected",
                "output_dir": str(settings_output),
            }
        ),
        encoding="utf-8",
    )
    register_dataset_artifact(tmp_path, artifact_id="dataset_registered", root=artifact_output)

    step.set_pipeline_stage_intent("conversion", False)

    assert step._default_training_dataset_dir() == artifact_output


def test_training_image_count_uses_import_project_metadata(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    output_images = tmp_path / "output" / "metashape_cubemap" / "images"
    output_images.mkdir(parents=True)
    expected = 12_345
    project = project_path(tmp_path)
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_text(json.dumps({"assets": {"output_image_count": expected}}), encoding="utf-8")

    assert step._training_image_count() == expected


def test_training_executable_placeholders_are_file_names_only() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    expected_lichtfeld = "LichtFeld-Studio.exe" if os.name == "nt" else "LichtFeld-Studio"
    expected_postshot = "postshot-cli.exe" if os.name == "nt" else "postshot-cli"
    lfs_path = str(Path("D:/Tools/LichtFeld-Studio.exe"))
    postshot_path = str(Path("D:/Tools/postshot-cli.exe"))

    assert step.training_executable_browse.line_edit.placeholderText() == expected_lichtfeld
    assert not Path(step._default_training_executable("lichtfeld")).is_absolute()
    step.training_executable_browse.set_text(lfs_path)

    step._set_training_backend("postshot")
    assert step.training_executable_browse.line_edit.placeholderText() == expected_postshot
    assert not Path(step._default_training_executable("postshot")).is_absolute()
    assert step.training_executable_browse.text() == ""
    step.training_executable_browse.set_text(postshot_path)

    step._set_training_backend("lichtfeld")
    assert step.training_executable_browse.line_edit.placeholderText() == expected_lichtfeld
    assert step.training_executable_browse.text() == lfs_path

    step._set_training_backend("postshot")
    assert step.training_executable_browse.line_edit.placeholderText() == expected_postshot
    assert step.training_executable_browse.text() == postshot_path

    step._set_training_backend("custom")
    assert step._training_backend() == "lichtfeld"
    assert step.training_executable_browse.text() == lfs_path
    assert step.training_executable_browse.line_edit.placeholderText() == expected_lichtfeld


def test_hidden_custom_training_legacy_executable_is_not_restored_as_lichtfeld(tmp_path: Path) -> None:
    settings_path = step4_export_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    custom_executable = str(tmp_path / "custom-trainer.exe")
    settings_path.write_text(
        json.dumps(
            {
                "settings_version": STEP4_SETTINGS_VERSION,
                "training": {
                    "enabled": True,
                    "backend": "custom",
                    "executable": custom_executable,
                },
            }
        ),
        encoding="utf-8",
    )

    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step._training_backend() == "lichtfeld"
    assert step.training_executable_browse.text() == ""
