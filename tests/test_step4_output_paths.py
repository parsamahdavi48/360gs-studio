from __future__ import annotations

from tests.helpers.step4 import (
    SFM_ROUTE_IDS,
    SFM_ROUTE_METASHAPE,
    CollapsibleSection,
    CubemapStep,
    Path,
    _app,
    _is_descendant,
    _ready_step,
    _workflow_job,
    _write_ascii_ply,
    _write_metashape_xml,
    _write_output_dataset,
    _write_test_image,
    i18n,
    json,
    math,
    pytest,
    step4_export_settings_path,
    step4_meta_dir,
)


def test_metashape_inputs_start_empty_when_exports_are_missing(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    assert step.ms_xml_browse.text() == ""
    assert step.ms_ply_browse.text() == ""
    assert step.ms_xml_browse.line_edit.placeholderText() == i18n.t("MS_XML_PLACEHOLDER")
    assert step.ms_ply_browse.line_edit.placeholderText() == i18n.t("MS_PLY_PLACEHOLDER")
    assert not hasattr(step, "metashape_input_hint")
    assert "pointcloud.ply" in step.ms_ply_browse.line_edit.toolTip()
    assert step.primary_action_enabled() is False


def test_metashape_inputs_auto_detect_standard_names(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "images").mkdir()
    _write_metashape_xml(tmp_path / "cameras.xml")
    _write_ascii_ply(tmp_path / "sparse.ply", [(1.0, 2.0, 3.0)])

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert Path(step.ms_xml_browse.text()) == tmp_path / "cameras.xml"
    assert Path(step.ms_ply_browse.text()) == tmp_path / "sparse.ply"
    assert step._metashape_ply_approved
    assert not hasattr(step, "metashape_input_hint")


def test_metashape_inputs_auto_detect_single_ply_without_approval(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "images").mkdir()
    _write_metashape_xml(tmp_path / "exported_pose.xml")
    _write_ascii_ply(tmp_path / "raw_scan.ply", [(1.0, 2.0, 3.0)])

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert Path(step.ms_xml_browse.text()) == tmp_path / "exported_pose.xml"
    assert Path(step.ms_ply_browse.text()) == tmp_path / "raw_scan.ply"
    assert step._metashape_ply_approved
    assert "承認" not in step.ms_ply_browse.line_edit.toolTip()
    assert step.primary_action_enabled() is True


def test_metashape_inputs_leave_multiple_scene_ply_for_manual_selection(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "images").mkdir()
    _write_metashape_xml(tmp_path / "exported_pose.xml")
    _write_ascii_ply(tmp_path / "scan_a.ply", [(1.0, 2.0, 3.0)])
    _write_ascii_ply(tmp_path / "scan_b.ply", [(1.0, 2.0, 3.0)])

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert Path(step.ms_xml_browse.text()) == tmp_path / "exported_pose.xml"
    assert step.ms_ply_browse.text() == ""
    assert not step._metashape_ply_approved
    assert "scan_a.ply" in step.ms_ply_browse.line_edit.toolTip()
    assert "scan_b.ply" in step.ms_ply_browse.line_edit.toolTip()
    assert step.primary_action_enabled() is False


def test_metashape_inputs_reject_output_dir_sources(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "images").mkdir()
    output = tmp_path / "output"
    output.mkdir(parents=True)
    (output / "metashape.xml").write_text("<root />", encoding="utf-8")
    _write_ascii_ply(output / "metashape.ply", [(1.0, 2.0, 3.0)])

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.ms_xml_browse.set_text(str(output / "metashape.xml"))
    step.ms_ply_browse.set_text(str(output / "metashape.ply"))

    assert step.primary_action_enabled() is False
    assert "output" in step.ms_xml_browse.line_edit.toolTip()
    assert "output" in step.ms_ply_browse.line_edit.toolTip()
    with pytest.raises(ValueError, match="output"):
        step._build_preprocess_cmd()


def test_cubemap_step_uses_tab_path_summaries(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    assert not hasattr(step, "output_browse")
    assert not hasattr(step, "json_name_edit")
    assert not hasattr(step, "mask_browse")
    assert not hasattr(step, "mask_from_alpha_cb")
    assert not hasattr(step, "no_transform_cb")
    assert not hasattr(step, "duplicate_cb")
    assert not hasattr(step, "ms_images_browse")
    assert not hasattr(step, "preprocess_cb")
    assert not hasattr(step, "no_image_cb")
    assert not hasattr(step.view_config, "cube6_drop_top")
    assert not hasattr(step.view_config, "cube6_drop_bottom")
    assert not hasattr(step.view_config, "yaw_slots_combo")
    assert not hasattr(step.view_config, "pitch_rows_combo")
    assert not hasattr(step.view_config, "custom_controls_widget")
    assert hasattr(step, "ms_use_ply_cb")
    assert hasattr(step, "axis_transform_combo")
    assert hasattr(step, "invert_masks_cb")
    assert not hasattr(step, "export_method_combo")
    assert not hasattr(step.view_config, "pitch_edit")
    assert not hasattr(step.view_config, "apply_btn")
    assert not hasattr(step, "training_backend_combo")
    assert set(step.export_method_buttons) == set(SFM_ROUTE_IDS)
    assert step.export_method_selector.current_route() == SFM_ROUTE_METASHAPE
    assert set(step.training_backend_buttons) == {"lichtfeld", "postshot", "brush", "gsplat"}
    assert set(step.training_backend_selector.primary_backend_buttons) == {"lichtfeld", "postshot", "brush", "gsplat"}
    assert set(step.training_backend_selector.other_backend_actions) == set()
    assert step.training_backend_selector.other_picker.isHidden()
    assert not hasattr(step, "training_backend_other_row")
    assert step.export_images_cb.isChecked()
    assert step.export_masks_cb.isChecked()
    assert step.output_shape_combo.currentData() == "projected"
    assert step.input_tab_index == 0
    assert step.output_tab_index == 1
    assert step.view_export_tab_index == step.output_tab_index
    assert step.apriltag_tab_index is None
    assert step.details_tab_index == 2
    assert step.metashape_tab_index == step.input_tab_index
    assert step.colmap_tab_index == step.input_tab_index
    assert step.spheresfm_tab_index == step.input_tab_index
    assert step.spheresfm_convert_tab_index == step.output_tab_index
    assert [step.settings_tabs.tabText(i) for i in range(step.settings_tabs.count())] == [
        i18n.t("STEP4_TAB_INPUT"),
        i18n.t("STEP4_TAB_OUTPUT"),
        i18n.t("STEP4_TAB_DETAILS"),
    ]
    assert step.export_method_label.isHidden()
    assert _is_descendant(step.export_method_row, step.input_tab)
    assert _is_descendant(step.sfm_input_section, step.input_tab)
    assert _is_descendant(step.metashape_sfm_input_widget, step.sfm_input_section)
    assert _is_descendant(step.metashape_output_section, step.output_tab)
    assert _is_descendant(step.export_targets_row, step.output_tab)
    assert step.settings_tabs.tabText(step.output_tab_index) == i18n.t("STEP4_TAB_OUTPUT")
    assert step.metashape_section.isHidden()
    assert not step.metashape_sfm_input_widget.isHidden()
    assert step.colmap_sfm_input_widget.isHidden()
    assert step.spheresfm_sfm_input_widget.isHidden()
    assert not step.metashape_output_section.isHidden()
    assert step.colmap_section.isHidden()
    assert step.spheresfm_section.isHidden()
    assert step.spheresfm_convert_section.isHidden()
    assert _is_descendant(step.view_config.settings_widget, step.advanced_output_section)
    assert _is_descendant(step.advanced_output_section, step.output_tab)
    assert _is_descendant(step.view_config.grid_section, step.advanced_output_section)
    assert _is_descendant(step.view_config.all_toggle_btn, step.view_config.grid_controls_widget)
    assert not _is_descendant(step.view_config.all_toggle_btn, step.view_config.grid_widget)
    assert step.view_config.all_toggle_btn.isCheckable()
    assert not hasattr(step.view_config, "all_on_btn")
    assert not hasattr(step.view_config, "all_off_btn")
    assert _is_descendant(step.view_config.pitch_add_btn, step.view_config.pitch_controls_widget)
    assert _is_descendant(step.view_config.pitch_count_label, step.view_config.pitch_controls_widget)
    assert not _is_descendant(step.view_config.pitch_add_btn, step.view_config.grid_controls_widget)
    assert _is_descendant(step.output_details_section, step.details_tab)
    assert not isinstance(step.output_details_section, CollapsibleSection)
    assert _is_descendant(step.output_format_combo, step.output_details_section)
    assert _is_descendant(step.output_bit_depth_combo, step.output_details_section)
    assert _is_descendant(step.invert_masks_cb, step.output_details_section)
    assert not _is_descendant(step.output_details_section, step.advanced_output_section)
    assert _is_descendant(step.sfm_path_summary_row, step.input_tab)
    assert _is_descendant(step.cubemap_path_summary_row, step.output_tab)
    assert _is_descendant(step.training_path_summary_row, step.training_section)
    assert not _is_descendant(step.export_summary_label, step.advanced_output_section)
    assert not _is_descendant(step.export_summary_label, step.view_config.settings_widget)
    assert step.export_summary_label.text() == step.view_config.summary_text()
    cube6_views = step.view_config.collect_views(include_disabled=True)
    cube6_enabled = [v for v in cube6_views if v["enabled"]]
    assert step.view_config.view_mode() == "cube6"
    assert step.view_config.yaw_slot_count() == 4
    assert step.view_config.pitch_values() == [-90.0, 0.0, 90.0]
    assert len(cube6_views) == 12
    assert {v["name"] for v in cube6_enabled} == {"px", "nx", "pz", "nz", "py", "ny"}
    assert {v["slot"] for v in cube6_enabled if v["name"] in {"py", "ny"}} == {3}
    assert sum(1 for v in cube6_views if not v["enabled"]) == 6
    assert step._export_method() == "metashape"
    assert step.export_method_buttons["metashape"].isChecked()
    assert not hasattr(step, "output_path_label")
    assert not step.cubemap_path_summary_value.wordWrap()
    assert step.cubemap_path_summary_kind.text() == i18n.t("STEP4_SUMMARY_OUTPUT")
    assert step.cubemap_path_summary_value.full_text() == "output/metashape_cubemap/"
    assert step.ms_images_path_label.full_text() == str(tmp_path / "images")
    assert step.scale_combo.itemText(0) == "Full"
    assert step.scale_combo.itemText(1) == "Normal"
    assert step.scale_combo.itemText(2) == "Half"
    assert float(step.scale_combo.itemData(1)) == pytest.approx(2.0 / math.pi)
    assert float(step.scale_combo.currentData()) == pytest.approx(2.0 / math.pi)

    cmd = step._build_cubemap_cmd()

    metashape_work = step4_meta_dir(tmp_path) / "work" / "metashape_import"
    job = _workflow_job(cmd)
    assert job["kind"] == "cubemap_conversion"
    assert job["input_dir"] == str(metashape_work)
    assert job["output_dir"] == str(tmp_path / "output" / "metashape_cubemap")
    assert job["image_dir"] == str(tmp_path)
    assert job["mask_dir"] == ""
    assert job["mask_from_alpha"] is False
    assert job["write_images"] is True
    assert job["write_masks"] is True
    assert job["allow_duplicate"] is False
    assert job["axis_mode"] == "none"
    assert job["final_orientation"] == "lichtfeld"
    assert step.axis_transform_combo.currentData() == "none"
    assert step.ms_use_ply_cb.isChecked()

    normal_cmd = step._build_cubemap_cmd()
    normal_scale = float(_workflow_job(normal_cmd)["output_scale"])
    assert normal_scale == pytest.approx(2.0 / math.pi, rel=1e-5)


def test_external_import_scene_settings_do_not_arm_conversion(tmp_path: Path) -> None:
    _app()
    _write_output_dataset(tmp_path, output_shape="projected", legacy_root=True)
    settings = json.loads(step4_export_settings_path(tmp_path).read_text(encoding="utf-8"))
    settings["origin"] = {"kind": "external_import", "import_id": "import_test"}
    settings["export_method"] = "metashape"
    step4_export_settings_path(tmp_path).write_text(json.dumps(settings), encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step._export_method() == "metashape"
    assert step.pipeline_stage_intent("sfm") is False
    assert step.pipeline_stage_intent("conversion") is False
    assert step.primary_action_enabled() is False
    assert step._training_dataset_export_shape(tmp_path / "output") == "projected"


def test_external_import_with_metashape_inputs_arms_default_conversion(tmp_path: Path) -> None:
    _app()
    (tmp_path / "images").mkdir()
    _write_test_image(tmp_path / "images" / "frame_0001.jpg")
    _write_metashape_xml(tmp_path / "metashape.xml", labels=["frame_0001.jpg"])
    _write_ascii_ply(tmp_path / "metashape.ply", [(1.0, 2.0, 3.0)])
    _write_output_dataset(tmp_path, output_shape="projected", legacy_root=True)
    settings_path = step4_export_settings_path(tmp_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["origin"] = {"kind": "external_import", "import_id": "import_test"}
    settings["export_method"] = "metashape"
    settings["target_profile"] = "custom"
    settings["view_config"] = {"mode": "external_import", "yaw_slots": 0, "pitch_rows": [], "views": []}
    settings["conversion"] = {"write_images": False, "write_masks": False}
    settings["metashape_import"] = {"xml": "", "ply": "", "use_ply": False}
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step._export_method() == "metashape"
    assert step.profile_combo.currentData() == "lichtfeld"
    assert step.output_shape_combo.currentData() == "projected"
    assert step.view_config.view_mode() == "cube6"
    assert step.ms_xml_browse.text() == str(tmp_path / "metashape.xml")
    assert step.ms_ply_browse.text() == str(tmp_path / "metashape.ply")
    assert step.ms_use_ply_cb.isChecked()
    assert step.export_images_cb.isChecked()
    assert step.export_masks_cb.isChecked()
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent("conversion") is True
    assert step.primary_action_enabled() is True
    sfm_item = next(item for item in step.pipeline_nav_items() if item["stage"] == "sfm")
    assert sfm_item["intent_symbol"] == "●"
    assert sfm_item["intent_toggle_enabled"] is False
    assert sfm_item["status"] == "ready"


def test_step4_scene_settings_ignore_pre_v2_payload(tmp_path: Path) -> None:
    _app()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "images").mkdir()
    (tmp_path / "metashape.xml").write_text("<root />", encoding="utf-8")
    _write_ascii_ply(tmp_path / "metashape.ply", [(1.0, 2.0, 3.0)])
    settings_path = step4_export_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"output_shape": "equirect_3dgut"}), encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.output_shape_combo.currentData() == "projected"
