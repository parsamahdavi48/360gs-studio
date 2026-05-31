from __future__ import annotations

from tests.helpers.step4 import (
    STEP4_SETTINGS_VERSION,
    AppJob,
    CollapsibleSection,
    CubemapStep,
    Path,
    QMessageBox,
    _app,
    _ready_lichtfeld_training_step,
    _ready_step,
    _workflow_job,
    _write_ascii_ply,
    _write_metashape_xml,
    _write_mixed_metashape_xml,
    _write_test_image,
    i18n,
    json,
    lichtfeld_defaults,
    math,
    np,
    orientation_correction,
    os,
    pytest,
    read_ply_points,
    step4_cubemap,
    step4_export_settings_path,
    step4_meta_dir,
    step4_views_config_path,
)


def test_metashape_projected_uses_dataset_job_writer(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    (tmp_path / "masks").mkdir()
    metashape_work = step4_meta_dir(tmp_path) / "work" / "metashape_import"
    metashape_work.mkdir(parents=True)
    stale = metashape_work / "old.txt"
    stale.write_text("old", encoding="utf-8")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_nerf"]
    cmd = commands[0][1]
    assert isinstance(cmd, AppJob)
    job = cmd.payload
    assert job["kind"] == "metashape_nerf_dataset"
    assert job["images_dir"] == str(tmp_path / "images")
    assert job["masks_dir"] == str(tmp_path / "masks")
    assert job["xml_path"] == str(tmp_path / "metashape.xml")
    assert job["ply_path"] == str(tmp_path / "metashape.ply")
    assert job["output_dir"] == str(tmp_path / "output" / "metashape_cubemap")
    assert job["write_images"] is True
    assert job["write_masks"] is True
    assert stale.exists()
    assert not (tmp_path / "transforms.json").exists()


def test_dataset_mask_generate_mode_runs_after_dataset_writer(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.enable_dataset_mask_settings()
    assert step._dataset_mask_step is not None
    step._dataset_mask_step.set_mask_mode("generate_training")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_nerf", "yolo", "dataset_mask_paths"]
    dataset_job = _workflow_job(commands[0][1])
    assert dataset_job["kind"] == "metashape_nerf_dataset"
    assert dataset_job["write_images"] is True
    assert dataset_job["write_masks"] is False
    attach_job = _workflow_job(commands[-1][1])
    assert attach_job["kind"] == "attach_dataset_masks"
    assert attach_job["dataset_root"] == str(tmp_path / "output" / "metashape_cubemap")
    assert attach_job["masks_dir"] == str(tmp_path / "output" / "metashape_cubemap" / "masks")
    assert attach_job["clear"] is False


def test_dataset_mask_reuse_mode_attaches_existing_masks_without_converting_sfm_masks(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.enable_dataset_mask_settings()
    assert step._dataset_mask_step is not None
    step._dataset_mask_step.set_mask_mode("reuse_existing")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_nerf", "dataset_mask_paths"]
    dataset_job = _workflow_job(commands[0][1])
    assert dataset_job["write_masks"] is False
    attach_job = _workflow_job(commands[-1][1])
    assert attach_job["kind"] == "attach_dataset_masks"
    assert attach_job["clear"] is False


def test_mixed_metashape_projected_uses_nerf_job_writer(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    postshot_index = step.profile_combo.findData("postshot")
    assert postshot_index >= 0
    step.profile_combo.setCurrentIndex(postshot_index)
    _write_test_image(tmp_path / "images" / "pano.jpg", size=(64, 32))
    _write_test_image(tmp_path / "images" / "frame.jpg", size=(40, 30))
    _write_mixed_metashape_xml(tmp_path / "metashape.xml")
    step.ms_xml_browse.set_text(str(tmp_path / "metashape.xml"))
    step.ms_ply_browse.set_text(str(tmp_path / "metashape.ply"))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_nerf"]
    cmd = commands[0][1]
    assert isinstance(cmd, AppJob)
    job = cmd.payload
    assert job["kind"] == "metashape_nerf_dataset"
    assert job["xml_path"] == str(tmp_path / "metashape.xml")
    assert job["ply_path"] == str(tmp_path / "metashape.ply")
    assert job["output_dir"] == str(tmp_path / "output" / "metashape_cubemap")


def test_mixed_metashape_projected_blocks_lichtfeld_nerf_multicamera(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_test_image(tmp_path / "images" / "pano.jpg", size=(64, 32))
    _write_test_image(tmp_path / "images" / "frame.jpg", size=(40, 30))
    _write_mixed_metashape_xml(tmp_path / "metashape.xml")
    step.ms_xml_browse.set_text(str(tmp_path / "metashape.xml"))

    with pytest.raises(ValueError, match="LichtFeld"):
        step.build_commands()


def test_mixed_metashape_direct_erp_output_is_blocked(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_test_image(tmp_path / "images" / "pano.jpg", size=(64, 32))
    _write_test_image(tmp_path / "images" / "frame.jpg", size=(40, 30))
    _write_mixed_metashape_xml(tmp_path / "metashape.xml")
    step.ms_xml_browse.set_text(str(tmp_path / "metashape.xml"))
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)

    with pytest.raises(ValueError, match="PINHOLE"):
        step.build_commands()


def test_step4_pipeline_intent_controls_execution_plan(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)

    assert step.primary_action_text() == i18n.t("RUN")
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent_enabled("sfm") is True
    assert step.pipeline_stage_intent_toggle_enabled("sfm") is False
    assert step.pipeline_stage_intent("conversion") is True
    sfm_item = next(item for item in step.pipeline_nav_items() if item["stage"] == "sfm")
    assert sfm_item["intent_checked"] is True
    assert sfm_item["intent_enabled"] is True
    assert sfm_item["intent_toggle_enabled"] is False
    assert sfm_item["intent_symbol"] == "●"
    assert sfm_item["status"] == "ready"
    assert sfm_item["status_symbol"] == "✓"

    step.set_pipeline_stage_intent("sfm", False)
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent("conversion") is True
    assert step.primary_action_enabled() is True
    sfm_item = next(item for item in step.pipeline_nav_items() if item["stage"] == "sfm")
    assert sfm_item["intent_checked"] is True
    assert sfm_item["status"] == "ready"

    step.ms_use_ply_cb.setChecked(True)
    ply = Path(step.ms_ply_browse.text())
    ply.unlink()
    sfm_item = next(item for item in step.pipeline_nav_items() if item["stage"] == "sfm")
    assert sfm_item["status"] == "warning"
    assert "PLY" in sfm_item["status_tooltip"]
    _write_ascii_ply(ply, [(1.0, 2.0, 3.0)])

    step.set_pipeline_stage_intent("conversion", False)
    assert step.pipeline_stage_intent("conversion") is False
    assert step.pipeline_stage_intent("sfm") is False
    assert step.take_pipeline_notice() == i18n.t("STEP4_PIPELINE_NOTICE_METASHAPE_DISABLED_INPUT")
    assert step.primary_action_enabled() is False
    sfm_item = next(item for item in step.pipeline_nav_items() if item["stage"] == "sfm")
    assert sfm_item["status"] == "off"
    assert "Metashape" in sfm_item["status_tooltip"]
    assert step.build_commands() == []
    step.set_pipeline_stage_intent("conversion", True)
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent("conversion") is True
    assert step.take_pipeline_notice() == i18n.t("STEP4_PIPELINE_NOTICE_METASHAPE_ENABLED_INPUT")
    step.set_pipeline_stage_intent("conversion", False)
    assert step.take_pipeline_notice() == i18n.t("STEP4_PIPELINE_NOTICE_METASHAPE_DISABLED_INPUT")

    step._set_export_method("colmap")
    assert step.pipeline_stage_intent("sfm") is False
    step.set_pipeline_stage_intent("sfm", True)
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent("conversion") is True
    assert step.take_pipeline_notice() == i18n.t("STEP4_PIPELINE_NOTICE_COLMAP_ENABLED_CUBE")
    step.set_pipeline_stage_intent("conversion", False)
    assert step.pipeline_stage_intent("conversion") is False
    assert step.pipeline_stage_intent("sfm") is False
    assert step.take_pipeline_notice() == i18n.t("STEP4_PIPELINE_NOTICE_COLMAP_DISABLED_SFM")

    step._set_export_method("spheresfm")
    step.set_pipeline_stage_intent("sfm", True)
    step.set_pipeline_stage_intent("conversion", True)
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent("conversion") is True
    step.set_pipeline_stage_intent("conversion", False)
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent("conversion") is False
    step.set_pipeline_stage_intent("sfm", False)
    assert step.pipeline_stage_intent("sfm") is False
    assert step.pipeline_stage_intent("conversion") is False


def test_metashape_import_uses_scene_images_and_lf_ply(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    (tmp_path / "images").mkdir()
    _write_metashape_xml(tmp_path / "metashape.xml")
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    step.ms_xml_browse.set_text(str(tmp_path / "metashape.xml"))
    step.ms_ply_browse.set_text(str(tmp_path / "metashape.ply"))

    cmd = step._build_preprocess_cmd()

    job = _workflow_job(cmd)
    assert job["kind"] == "metashape_preprocess"
    assert job["images_dir"] == str(tmp_path / "images")
    assert job["xml_path"] == str(tmp_path / "metashape.xml")
    assert job["ply_path"] == str(tmp_path / "metashape.ply")
    assert job["lichtfeld_camera_y180"] is True


def test_profile_presets_sync_manual_axis_controls(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    postshot_idx = step.profile_combo.findData("postshot")
    brush_idx = step.profile_combo.findData("brush")
    lichtfeld_idx = step.profile_combo.findData("lichtfeld")
    assert postshot_idx >= 0
    assert brush_idx >= 0
    assert lichtfeld_idx >= 0

    step.profile_combo.setCurrentIndex(postshot_idx)
    assert step.axis_transform_combo.currentData() == "postshot"
    assert not step.ms_use_ply_cb.isChecked()
    assert step.ms_scale_edit.text() == "1.0"
    assert not step.ms_no_fix_rot_cb.isChecked()

    step.profile_combo.setCurrentIndex(brush_idx)
    assert step.axis_transform_combo.currentData() == "brush"
    assert not step.ms_use_ply_cb.isChecked()
    assert step.ms_scale_edit.text() == "1.0"
    assert not step.ms_no_fix_rot_cb.isChecked()

    step.profile_combo.setCurrentIndex(lichtfeld_idx)
    assert step.axis_transform_combo.currentData() == "none"
    assert step.ms_use_ply_cb.isChecked()
    assert step.ms_scale_edit.text() == "1.0"
    assert not step.ms_no_fix_rot_cb.isChecked()


def test_manual_axis_change_switches_to_custom_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    postshot_idx = step.profile_combo.findData("postshot")
    assert postshot_idx >= 0
    step.profile_combo.setCurrentIndex(postshot_idx)

    brush_axis = step.axis_transform_combo.findData("brush")
    assert brush_axis >= 0
    step.axis_transform_combo.setCurrentIndex(brush_axis)

    assert step.profile_combo.currentData() == "custom"
    cmd = step._build_cubemap_cmd()
    job = _workflow_job(cmd)
    assert job["axis_mode"] == "brush"
    assert job["final_orientation"] == "none"


def test_manual_metashape_ply_toggle_switches_to_custom_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    (tmp_path / "images").mkdir()
    _write_metashape_xml(tmp_path / "metashape.xml")
    (tmp_path / "metashape.ply").write_text("ply\n", encoding="utf-8")
    step.ms_xml_browse.set_text(str(tmp_path / "metashape.xml"))
    step.ms_ply_browse.set_text(str(tmp_path / "metashape.ply"))
    postshot_idx = step.profile_combo.findData("postshot")
    assert postshot_idx >= 0
    step.profile_combo.setCurrentIndex(postshot_idx)

    step.ms_use_ply_cb.setChecked(True)

    assert step.profile_combo.currentData() == "custom"
    cmd = step._build_preprocess_cmd()
    assert _workflow_job(cmd)["ply_path"] == str(tmp_path / "metashape.ply")


def test_manual_metashape_import_detail_change_switches_to_custom_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    lichtfeld_idx = step.profile_combo.findData("lichtfeld")
    postshot_idx = step.profile_combo.findData("postshot")
    assert lichtfeld_idx >= 0
    assert postshot_idx >= 0

    step.profile_combo.setCurrentIndex(lichtfeld_idx)
    step.ms_no_fix_rot_cb.setChecked(True)
    assert step.profile_combo.currentData() == "custom"

    step.profile_combo.setCurrentIndex(postshot_idx)
    step.ms_scale_edit.setText("2.0")
    step._on_profile_option_changed()
    assert step.profile_combo.currentData() == "custom"


def test_lichtfeld_import_requires_raw_ply_when_ply_enabled(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "images").mkdir()
    _write_metashape_xml(tmp_path / "metashape.xml")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match="PLY"):
        step.build_commands()


def test_cubemap_step_keeps_mask_inversion_as_advanced_option(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.invert_masks_cb.setChecked(True)

    cmd = step._build_cubemap_cmd()

    assert _workflow_job(cmd)["invert_masks"] is True


def test_cubemap_step_can_skip_image_and_mask_conversion(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.export_images_cb.setChecked(False)
    step.export_masks_cb.setChecked(False)

    cmd = step._build_cubemap_cmd()

    job = _workflow_job(cmd)
    assert job["write_images"] is False
    assert job["write_masks"] is False


def test_lichtfeld_3dgut_direct_mode_runs_metashape_only_and_disables_view_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir(parents=True)
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    _write_test_image(tmp_path / "images" / "frame_0001.jpg")
    step.on_activated()
    step.export_images_cb.setChecked(False)

    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    assert step._uses_direct_equirect_output()
    assert step._effective_profile() == "lichtfeld"
    assert step.axis_transform_combo.currentData() == "none"
    assert step.ms_use_ply_cb.isChecked()
    assert step.cubemap_path_summary_value.full_text() == "output/metashape_3dgut/"
    assert step.settings_tabs.isTabEnabled(step.output_tab_index)
    assert not step.view_config.settings_widget.isEnabled()
    assert not step.export_targets_row.isEnabled()
    assert step.export_images_cb.isChecked()
    assert step.export_masks_cb.isChecked()
    assert not step.output_details_section.isEnabled()
    assert not step.export_colmap_cb.isEnabled()
    assert "元画像" in step.view_config.summary_text()

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape"]
    job = _workflow_job(commands[0][1])
    assert job["use_ply"] is True
    assert job["output_dir"] == str(tmp_path / "output" / "metashape_3dgut")
    assert os.path.samefile(
        tmp_path / "images" / "frame_0001.jpg",
        tmp_path / "output" / "metashape_3dgut" / "images" / "frame_0001.jpg",
    )
    assert old_file.is_file()
    assert not step4_views_config_path(tmp_path).exists()


def test_metashape_erp_output_is_disabled_outside_lichtfeld_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    postshot_idx = step.profile_combo.findData("postshot")
    lichtfeld_idx = step.profile_combo.findData("lichtfeld")
    assert direct_idx >= 0
    assert postshot_idx >= 0
    assert lichtfeld_idx >= 0

    assert step.output_shape_combo.isItemEnabled(direct_idx)
    step.profile_combo.setCurrentIndex(postshot_idx)

    assert step.output_shape_combo.currentData() == "projected"
    assert step.profile_combo.currentData() == "postshot"
    assert not step.output_shape_combo.isItemEnabled(direct_idx)
    assert step.output_shape_combo.itemToolTip(direct_idx) == i18n.tip("OUTPUT_SHAPE_EQUIRECT_3DGUT_DISABLED")

    step.output_shape_combo.setCurrentIndex(direct_idx)
    assert step.output_shape_combo.currentData() == "projected"
    assert step.profile_combo.currentData() == "postshot"

    step.profile_combo.setCurrentIndex(lichtfeld_idx)
    assert step.output_shape_combo.isItemEnabled(direct_idx)
    assert step.output_shape_combo.itemToolTip(direct_idx) == i18n.tip("OUTPUT_SHAPE_EQUIRECT_3DGUT")
    step.output_shape_combo.setCurrentIndex(direct_idx)

    assert step.output_shape_combo.currentData() == "equirect_3dgut"
    assert step.profile_combo.currentData() == "lichtfeld"


def test_lichtfeld_3dgut_asset_links_fallback_to_copy(tmp_path: Path, monkeypatch) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    image = tmp_path / "images" / "frame_0001.jpg"
    _write_test_image(image)
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    monkeypatch.setattr(step4_cubemap.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("no link")))

    commands = step.build_commands()

    linked = tmp_path / "output" / "metashape_3dgut" / "images" / "frame_0001.jpg"
    assert [phase for phase, _cmd in commands] == ["metashape"]
    assert linked.is_file()
    assert not os.path.samefile(image, linked)


def test_lichtfeld_3dgut_direct_mode_restores_projection_export_targets(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    step.export_images_cb.setChecked(False)
    step.export_masks_cb.setChecked(True)

    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    projected_idx = step.output_shape_combo.findData("projected")
    assert direct_idx >= 0
    assert projected_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    step.output_shape_combo.setCurrentIndex(projected_idx)

    assert step.export_images_cb.isChecked() is False
    assert step.export_masks_cb.isChecked() is True
    assert step.settings_tabs.isTabEnabled(step.view_export_tab_index)
    assert step.cubemap_path_summary_value.full_text() == "output/metashape_cubemap/"


def test_lichtfeld_3dgut_saved_settings_restore_default_projection_export_targets(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)

    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    step._write_export_settings()

    restored = CubemapStep(Path.cwd())
    restored.set_scene_dir(str(tmp_path))
    projected_idx = restored.output_shape_combo.findData("projected")
    assert projected_idx >= 0
    restored.output_shape_combo.setCurrentIndex(projected_idx)

    assert restored.export_images_cb.isChecked() is True
    assert restored.export_masks_cb.isChecked() is True


def test_switching_profile_away_from_lichtfeld_exits_3dgut_direct_mode(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    postshot_idx = step.profile_combo.findData("postshot")
    assert direct_idx >= 0
    assert postshot_idx >= 0

    step.output_shape_combo.setCurrentIndex(direct_idx)
    step.profile_combo.setCurrentIndex(postshot_idx)

    assert step.output_shape_combo.currentData() == "projected"
    assert not step.output_shape_combo.isItemEnabled(direct_idx)
    assert step._uses_direct_equirect_output() is False
    assert step.settings_tabs.isTabEnabled(step.view_export_tab_index)


def test_lichtfeld_strategy_defaults_match_written_configs(tmp_path: Path) -> None:
    step = _ready_lichtfeld_training_step(tmp_path)

    for strategy in ("mrnf", "igs+", "mcmc"):
        strategy_idx = step.lfs_strategy_combo.findData(strategy)
        assert strategy_idx >= 0
        step.lfs_strategy_combo.setCurrentIndex(strategy_idx)

        commands = step._build_training_commands()

        assert commands[-1][0] == "training_lichtfeld"
        config = json.loads(step._training_config_path().read_text(encoding="utf-8"))
        defaults = lichtfeld_defaults(strategy)
        for key, expected in defaults.items():
            assert key in config
            if isinstance(expected, float):
                assert config[key] == pytest.approx(expected)
            elif isinstance(expected, list) and any(isinstance(value, float) for value in expected):
                assert config[key] == pytest.approx(expected)
            else:
                assert config[key] == expected


def test_lichtfeld_strategy_switch_preserves_each_strategy_state(tmp_path: Path) -> None:
    step = _ready_lichtfeld_training_step(tmp_path)

    step.lfs_max_gaussians_edit.setText("6,000,000")
    step.lfs_advanced_edits["means_lr"].setText("0.000123")

    mcmc_idx = step.lfs_strategy_combo.findData("mcmc")
    assert mcmc_idx >= 0
    step.lfs_strategy_combo.setCurrentIndex(mcmc_idx)

    assert step.lfs_max_gaussians_edit.text() == "1,000,000"
    assert step.lfs_advanced_edits["means_lr"].text() == "0.000016"
    assert step.lfs_advanced_edits["opacity_lr"].text() == "0.0250"
    assert step.lfs_advanced_edits["opacity_reg"].text() == "0.0100"
    assert step.lfs_advanced_checks["revised_opacity"].isChecked() is False

    igs_idx = step.lfs_strategy_combo.findData("igs+")
    assert igs_idx >= 0
    step.lfs_strategy_combo.setCurrentIndex(igs_idx)

    assert step.lfs_max_gaussians_edit.text() == "4,000,000"
    assert step.lfs_advanced_edits["shs_lr"].text() == "0.0050"
    assert step.lfs_advanced_edits["scaling_lr"].text() == "0.0200"
    assert step.lfs_advanced_edits["stop_refine"].text() == "15,000"
    assert step.lfs_advanced_edits["opacity_reg"].text() == "0.0000"
    assert step.lfs_advanced_edits["init_opacity"].text() == "0.100"
    assert step.lfs_advanced_edits["tv_loss_weight"].text() == "5.0"
    assert step.lfs_advanced_checks["revised_opacity"].isChecked() is True

    mrnf_idx = step.lfs_strategy_combo.findData("mrnf")
    assert mrnf_idx >= 0
    step.lfs_strategy_combo.setCurrentIndex(mrnf_idx)

    assert step.lfs_max_gaussians_edit.text() == "6,000,000"
    assert step.lfs_advanced_edits["means_lr"].text() == "0.000123"


def test_lichtfeld_advanced_parameters_are_nested_collapsible_sections(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)

    expected_sections = {
        "LFS_SECTION_DATASET",
        "LFS_SECTION_OPTIMIZATION",
        "LFS_SECTION_BILATERAL",
        "LFS_SECTION_LOSSES",
        "LFS_SECTION_INITIALIZATION",
        "LFS_SECTION_PRUNING_GROWING",
        "LFS_SECTION_MRNF",
        "LFS_SECTION_SPARSITY",
        "LFS_SECTION_SAVE_EVAL",
    }

    assert set(step.lfs_advanced_sections) == expected_sections
    assert all(isinstance(section, CollapsibleSection) for section in step.lfs_advanced_sections.values())
    assert all(not section.toggle_button.isChecked() for section in step.lfs_advanced_sections.values())
    assert step.lfs_advanced_edits["means_lr_end"].width() >= 122
    assert step.lfs_advanced_edits["grad_threshold"].width() >= 116
    assert step.lfs_advanced_edits["save_steps"].width() >= 136
    assert step.lfs_advanced_edits["save_steps"].width() < 180
    assert step.lfs_dataset_test_every_edit.isHidden()
    assert step.lfs_advanced_sections["LFS_SECTION_BILATERAL"].isHidden()
    assert step.lfs_advanced_sections["LFS_SECTION_PRUNING_GROWING"].isHidden()
    assert not step.lfs_advanced_sections["LFS_SECTION_MRNF"].isHidden()
    assert step.lfs_advanced_sections["LFS_SECTION_SPARSITY"].isHidden()
    assert step.lfs_advanced_edits["init_num_pts"].isHidden()
    assert step.lfs_tile_mode_combo.isHidden()

    step.lfs_bilateral_grid_cb.setChecked(True)
    step.lfs_sparsity_cb.setChecked(True)
    step.lfs_gut_cb.setChecked(True)
    step.lfs_advanced_checks["random"].setChecked(True)
    step.lfs_advanced_checks["enable_eval"].setChecked(True)

    assert not step.lfs_advanced_sections["LFS_SECTION_BILATERAL"].isHidden()
    assert not step.lfs_advanced_sections["LFS_SECTION_SPARSITY"].isHidden()
    assert not step.lfs_advanced_edits["init_num_pts"].isHidden()
    assert not step.lfs_dataset_test_every_edit.isHidden()
    assert not step.lfs_tile_mode_combo.isHidden()

    igs_idx = step.lfs_strategy_combo.findData("igs+")
    assert igs_idx >= 0
    step.lfs_strategy_combo.setCurrentIndex(igs_idx)

    assert step.lfs_advanced_sections["LFS_SECTION_MRNF"].isHidden()
    assert not step.lfs_advanced_sections["LFS_SECTION_PRUNING_GROWING"].isHidden()
    assert step.lfs_advanced_edits["grow_until_iter"].isHidden()
    assert not step.lfs_advanced_edits["prune_opacity"].isHidden()
    assert step.lfs_gut_cb.isChecked() is False
    assert step.lfs_gut_cb.isEnabled() is False
    assert step.lfs_tile_mode_combo.isHidden()


def test_lichtfeld_basic_conditional_parameters_follow_source_visibility(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)

    assert step.lfs_invert_masks_cb.isHidden()
    assert step.lfs_mask_threshold_edit.isHidden()
    assert step.lfs_use_alpha_as_mask_cb.isHidden()
    assert step.lfs_mask_opacity_penalty_weight_edit.isHidden()
    assert step.lfs_mask_opacity_penalty_power_edit.isHidden()
    assert step.lfs_ppisp_freeze_from_sidecar_cb.isHidden()
    assert step.lfs_ppisp_sidecar_browse.isHidden()
    assert step.lfs_ppisp_use_controller_cb.isHidden()
    assert step.lfs_ppisp_controller_activation_step_edit.isHidden()

    ignore_idx = step.lfs_mask_mode_combo.findData("ignore")
    assert ignore_idx >= 0
    step.lfs_mask_mode_combo.setCurrentIndex(ignore_idx)

    assert not step.lfs_invert_masks_cb.isHidden()
    assert not step.lfs_mask_threshold_edit.isHidden()
    assert not step.lfs_use_alpha_as_mask_cb.isHidden()
    assert step.lfs_mask_opacity_penalty_weight_edit.isHidden()
    assert step.lfs_mask_opacity_penalty_power_edit.isHidden()

    segment_idx = step.lfs_mask_mode_combo.findData("segment")
    assert segment_idx >= 0
    step.lfs_mask_mode_combo.setCurrentIndex(segment_idx)

    assert not step.lfs_mask_opacity_penalty_weight_edit.isHidden()
    assert not step.lfs_mask_opacity_penalty_power_edit.isHidden()

    step.lfs_ppisp_cb.setChecked(True)

    assert not step.lfs_ppisp_freeze_from_sidecar_cb.isHidden()
    assert step.lfs_ppisp_sidecar_browse.isHidden()
    assert not step.lfs_ppisp_use_controller_cb.isHidden()
    assert step.lfs_ppisp_controller_activation_step_edit.isHidden()

    step.lfs_ppisp_freeze_from_sidecar_cb.setChecked(True)
    step.lfs_ppisp_use_controller_cb.setChecked(True)

    assert not step.lfs_ppisp_sidecar_browse.isHidden()
    assert not step.lfs_ppisp_controller_activation_step_edit.isHidden()
    assert not step.lfs_ppisp_controller_lr_edit.isHidden()
    assert not step.lfs_ppisp_freeze_gaussians_on_distill_cb.isHidden()


def test_cubemap_preview_uses_scene_mask_folder(tmp_path: Path, monkeypatch) -> None:
    step = _ready_step(tmp_path)
    captured: dict[str, str] = {}

    def fake_render(_views, mask_dir: str) -> None:
        captured["mask_dir"] = mask_dir

    monkeypatch.setattr(step.preview, "render", fake_render)

    step._render_preview()

    assert captured["mask_dir"] == str(tmp_path / "masks")


def test_cubemap_build_cancel_keeps_existing_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output" / "metashape_cubemap"
    output.mkdir(parents=True)
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    commands = step.build_commands()

    assert commands == []
    assert old_file.is_file()
    assert not step4_views_config_path(tmp_path).exists()


def test_cubemap_build_resets_existing_output_when_confirmed(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output" / "metashape_cubemap"
    nested = output / "nested"
    nested.mkdir(parents=True)
    old_file = output / "old.txt"
    nested_file = nested / "old_nested.txt"
    old_file.write_text("old", encoding="utf-8")
    nested_file.write_text("old", encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_nerf"]
    assert _workflow_job(commands[0][1])["kind"] == "metashape_nerf_dataset"
    assert not old_file.exists()
    assert not nested.exists()


def test_metashape_dataset_writer_can_preserve_existing_assets_for_pose_only_update(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output" / "metashape_cubemap"
    output.mkdir(parents=True)
    old_image_dir = output / "images"
    old_image_dir.mkdir()
    old_file = old_image_dir / "old_render.png"
    old_file.write_text("old", encoding="utf-8")
    old_mask_dir = output / "masks"
    old_mask_dir.mkdir()
    old_mask = old_mask_dir / "old_mask.png"
    old_mask.write_text("mask", encoding="utf-8")
    old_settings = step4_export_settings_path(tmp_path)
    old_settings.parent.mkdir(parents=True, exist_ok=True)
    old_settings.write_text('{"old": true}\n', encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    assert step.export_targets_row.isEnabled()
    step.export_images_cb.setChecked(False)
    step.export_masks_cb.setChecked(False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_nerf"]
    job = _workflow_job(commands[0][1])
    assert job["kind"] == "metashape_nerf_dataset"
    assert job["write_images"] is False
    assert job["write_masks"] is False
    assert old_file.exists()
    assert old_mask.exists()
    step._finalize_bundle()
    settings = json.loads(old_settings.read_text(encoding="utf-8"))
    assert settings["conversion"]["no_image"] is True
    assert settings["conversion"]["write_images"] is False
    assert settings["conversion"]["write_masks"] is False


def test_cubemap_build_validates_before_resetting_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output" / "metashape_cubemap"
    output.mkdir(parents=True)
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    (tmp_path / "images").mkdir()
    step = _ready_step(tmp_path)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    with pytest.raises(ValueError, match="XML"):
        step.build_commands()

    assert old_file.is_file()


def test_cubemap_finalize_writes_export_settings(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["metashape_nerf"]
    step._finalize_bundle()

    settings_path = step4_export_settings_path(tmp_path)
    assert settings_path.is_file()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["app"] == "stechdrive-3dgs-utils"
    assert settings["settings_version"] == STEP4_SETTINGS_VERSION
    assert settings["portable_output"] == {
        "root": "output/metashape_cubemap",
        "dataset_kind": "projection_views",
        "active": True,
    }
    assert settings["export_method"] == "metashape"
    assert settings["target_profile"] == "lichtfeld"
    assert settings["effective_profile"] == "lichtfeld"
    assert settings["axis_transform"] == "none"
    assert settings["fov"] == 90.0
    assert settings["image_size"]["scale"] == pytest.approx(2.0 / math.pi)
    assert settings["conversion"]["yaw_offset_per_frame"] == 30.0
    assert settings["conversion"]["output_format"] == "auto"
    assert settings["conversion"]["output_bit_depth"] == "8"
    assert settings["conversion"]["no_image"] is False
    assert settings["conversion"]["write_images"] is True
    assert settings["conversion"]["write_masks"] is True
    assert settings["view_config"]["cube6_drop_top"] is False
    assert settings["view_config"]["cube6_drop_bottom"] is False
    assert settings["metashape_import"]["use_ply"] is True
    assert settings["output_files"]["settings"] == "_stechdrive/step4/export_settings.json"
    assert settings["output_files"]["transforms_json"] == "transforms_lichtfeld.json"
    assert settings["output_files"]["pointcloud"] == "pointcloud_lichtfeld.ply"
    assert settings["postprocess"]["final_orientation"] == "lichtfeld"
    assert settings["postprocess"]["final_orientation_stage"] == "cubemap_cli"
    assert settings["postprocess"]["lichtfeld_final_orientation_stage"] == "cubemap_cli"
    assert settings["view_config"]["views"]
    assert settings["views_config_path"] == "_stechdrive/step4/views_config.json"
    assert settings["views_config_snapshot"] == json.loads(
        step4_views_config_path(tmp_path).read_text(encoding="utf-8")
    )


def test_lichtfeld_3dgut_finalize_writes_scene_dataset_settings_and_correction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)
    direct_idx = step.output_shape_combo.findData("equirect_3dgut")
    assert direct_idx >= 0
    step.output_shape_combo.setCurrentIndex(direct_idx)
    monkeypatch.setattr(orientation_correction, "transform_ply_with_open3d", lambda _path, _matrix: False)
    output = tmp_path / "output" / "metashape_3dgut"
    output.mkdir(parents=True)
    _write_ascii_ply(output / "pointcloud.ply", [(1.0, 2.0, 3.0)])
    (output / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "EQUIRECTANGULAR",
                "frames": [{"file_path": "images/frame_0001.jpg", "transform_matrix": np.eye(4).tolist()}],
            }
        ),
        encoding="utf-8",
    )

    step._finalize_bundle()

    settings = json.loads(step4_export_settings_path(tmp_path).read_text(encoding="utf-8"))
    assert settings["export_method"] == "metashape"
    assert settings["output_shape"] == "equirect_3dgut"
    assert settings["output_dir"] == str(output)
    assert settings["portable_output"]["dataset_kind"] == "3dgut"
    assert settings["views_config_path"] == ""
    assert settings["views_config_snapshot"] is None
    assert settings["conversion"]["no_image"] is True
    assert settings["conversion"]["write_images"] is False
    assert settings["conversion"]["write_masks"] is False
    assert settings["conversion"]["uses_source_images"] is True
    assert settings["output_files"]["pointcloud"] == "pointcloud.ply"
    assert settings["postprocess"]["final_orientation"] == "lichtfeld"
    assert settings["postprocess"]["final_orientation_stage"] == "direct_finalize"

    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert data["postprocess"]["final_orientation_stage"] == "direct_finalize"
    corrected = np.array(data["frames"][0]["transform_matrix"])
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(corrected, expected)
    points, _colors = read_ply_points(output / "pointcloud.ply")
    assert np.allclose(points[0], [3.0, -2.0, 1.0])


def test_lichtfeld_projected_finalize_keeps_cli_final_orientation_outputs(tmp_path: Path) -> None:
    _write_ascii_ply(tmp_path / "pointcloud.ply", [(9.0, 9.0, 9.0)])
    metashape_work = step4_meta_dir(tmp_path) / "work" / "metashape_import"
    metashape_work.mkdir(parents=True)
    _write_ascii_ply(metashape_work / "pointcloud.ply", [(1.0, 2.0, 3.0)])
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    output = tmp_path / "output" / "metashape_cubemap"
    output.mkdir(parents=True)
    _write_ascii_ply(output / "pointcloud.ply", [(3.0, -2.0, 1.0)])
    (output / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "ply_file_path": "pointcloud.ply",
                "frames": [
                    {
                        "file_path": "images/frame_front.jpg",
                        "transform_matrix": [
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
                "postprocess": {
                    "final_orientation": "lichtfeld",
                    "final_orientation_stage": "cubemap_cli",
                    "lichtfeld_final_orientation_correction": True,
                    "lichtfeld_final_orientation_stage": "cubemap_cli",
                },
            }
        ),
        encoding="utf-8",
    )

    step._finalize_bundle()

    data = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    corrected = np.array(data["frames"][0]["transform_matrix"])
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(corrected, expected)
    assert data["ply_file_path"] == "pointcloud.ply"
    assert data["postprocess"]["final_orientation_stage"] == "cubemap_cli"

    points, _colors = read_ply_points(output / "pointcloud.ply")
    assert np.allclose(points[0], [3.0, -2.0, 1.0])

    settings = json.loads(step4_export_settings_path(tmp_path).read_text(encoding="utf-8"))
    assert settings["postprocess"]["lichtfeld_final_orientation_correction"] is True
    assert settings["postprocess"]["final_orientation"] == "lichtfeld"
    assert settings["postprocess"]["final_orientation_stage"] == "cubemap_cli"
