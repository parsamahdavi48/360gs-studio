from __future__ import annotations

from tests.helpers.step4 import (
    OUTPUT_SHAPE_EQUIRECT_3DGUT,
    OUTPUT_SHAPE_PROJECTED,
    PREVIEW_PROJECTION_EQUIRECT,
    PREVIEW_PROJECTION_PERSPECTIVE,
    SFM_ROUTE_COLMAP,
    SFM_ROUTE_IDS,
    SFM_ROUTE_METASHAPE,
    SFM_ROUTE_SPHERESFM,
    CubemapStep,
    Path,
    QPoint,
    _app,
    _ready_step,
    _workflow_job,
    _write_ascii_ply,
    _write_metashape_xml,
    _write_test_image,
    get_sfm_route_backend,
    get_sfm_route_spec,
    i18n,
    json,
    normalize_sfm_route,
    pytest,
    step4_views_config_path,
)


def test_sfm_route_registry_describes_current_routes() -> None:
    assert SFM_ROUTE_IDS == (SFM_ROUTE_METASHAPE, SFM_ROUTE_COLMAP, SFM_ROUTE_SPHERESFM)
    assert normalize_sfm_route("missing") == SFM_ROUTE_METASHAPE

    metashape = get_sfm_route_spec(SFM_ROUTE_METASHAPE)
    colmap = get_sfm_route_spec(SFM_ROUTE_COLMAP)
    spheresfm = get_sfm_route_spec(SFM_ROUTE_SPHERESFM)

    assert metashape.kind == "external_input"
    assert not metashape.runs_sfm_in_app
    assert metashape.supports_output_shape(OUTPUT_SHAPE_PROJECTED)
    assert metashape.supports_output_shape(OUTPUT_SHAPE_EQUIRECT_3DGUT)
    assert colmap.kind == "in_app"
    assert colmap.runs_sfm_in_app
    assert colmap.supports_output_shape(OUTPUT_SHAPE_PROJECTED)
    assert not colmap.supports_output_shape(OUTPUT_SHAPE_EQUIRECT_3DGUT)
    assert spheresfm.kind == "in_app"
    assert spheresfm.runs_sfm_in_app
    assert spheresfm.supports_output_shape(OUTPUT_SHAPE_PROJECTED)
    assert spheresfm.supports_output_shape(OUTPUT_SHAPE_EQUIRECT_3DGUT)
    for route_id in SFM_ROUTE_IDS:
        assert get_sfm_route_backend(route_id).spec.route_id == route_id
    assert get_sfm_route_backend("missing").spec.route_id == SFM_ROUTE_METASHAPE


def test_export_method_switch_keeps_fixed_tabs_and_swaps_route_sections() -> None:
    _app()
    _app()
    _app()
    step = CubemapStep(Path.cwd())

    assert step.settings_tabs.tabText(0) == i18n.t("STEP4_TAB_INPUT")
    assert step.settings_tabs.currentIndex() == step.input_tab_index
    assert step.metashape_section.isHidden()
    assert not step.metashape_sfm_input_widget.isHidden()
    assert step.colmap_section.isHidden()
    assert step.colmap_sfm_input_widget.isHidden()

    step.settings_tabs.setCurrentIndex(step.input_tab_index)
    step._set_export_method("colmap")

    assert step.settings_tabs.tabText(0) == i18n.t("STEP4_TAB_INPUT")
    assert step.metashape_section.isHidden()
    assert step.metashape_sfm_input_widget.isHidden()
    assert step.metashape_output_section.isHidden()
    assert step.colmap_section.isHidden()
    assert not step.colmap_sfm_input_widget.isHidden()
    step.set_pipeline_stage_intent("sfm", True)
    assert not step.colmap_section.isHidden()
    assert not step.colmap_sfm_input_widget.isHidden()
    assert step.settings_tabs.currentIndex() == step.input_tab_index

    step.settings_tabs.setCurrentIndex(step.view_export_tab_index)
    step._set_export_method("metashape")

    assert step.settings_tabs.tabText(0) == i18n.t("STEP4_TAB_INPUT")
    assert step.settings_tabs.currentIndex() == step.input_tab_index
    assert step.metashape_section.isHidden()
    assert not step.metashape_sfm_input_widget.isHidden()
    assert not step.metashape_output_section.isHidden()
    assert step.colmap_section.isHidden()
    assert step.colmap_sfm_input_widget.isHidden()


def test_sfm_input_section_stays_under_route_selector_across_routes() -> None:
    app = _app()
    step = CubemapStep(Path.cwd())
    step.resize(900, 720)
    step.show()
    step.settings_tabs.setCurrentIndex(step.output_tab_index)
    app.processEvents()

    positions = []
    for method in ("metashape", "colmap", "spheresfm"):
        step._set_export_method(method)
        step.settings_tabs.setCurrentIndex(step.input_tab_index)
        app.processEvents()
        position = step.sfm_input_section.mapTo(step.input_tab, QPoint(0, 0))
        positions.append((position.x(), position.y()))

    assert len(set(positions)) == 1
    margins = step.input_tab.layout().contentsMargins()
    route_position = step.export_method_row.mapTo(step.input_tab, QPoint(0, 0))
    assert positions[0][0] == margins.left()
    assert positions[0][1] > route_position.y()

    step.close()


def test_cubemap_step_does_not_count_repo_images_without_scene_dir() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert step.scene_dir == ""
    assert step._count_input_images() == 0


def test_cubemap_step_does_not_use_current_directory_without_scene_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    (tmp_path / "pointcloud.ply").write_text("ply\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    step = CubemapStep(Path.cwd())

    step.on_activated()

    assert step.preview.current_image_path() is None
    assert step._count_input_images() == 0
    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match=i18n.t("SCENE_REQUIRED_ACTION_HINT")):
        step._output_dir()
    with pytest.raises(ValueError, match=i18n.t("SCENE_REQUIRED_ACTION_HINT")):
        step._mask_dir()
    with pytest.raises(ValueError, match=i18n.t("SCENE_REQUIRED_ACTION_HINT")):
        step._metashape_images_dir()


def test_cubemap_step_refreshes_preview_when_activated_after_extraction(tmp_path: Path) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.preview.current_image_path() is None

    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "frame_0001.jpg"
    _write_test_image(image_path)

    step.on_activated()

    assert step.preview.current_image_path() == image_path
    assert step.preview.image_label._source_pixmap is not None
    assert step._count_input_images() == 1
    assert i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=6) in step.view_config.summary_text()


def test_metashape_preview_uses_xml_camera_images_not_entire_scene_folder(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    used = images / "erp_used.jpg"
    unused = images / "normal_unused.jpg"
    _write_test_image(used, size=(64, 32))
    _write_test_image(unused, size=(40, 30))
    _write_metashape_xml(tmp_path / "metashape.xml", labels=[used.name])
    _write_ascii_ply(tmp_path / "metashape.ply", [(0.0, 0.0, 0.0)])
    step = CubemapStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))
    step.on_activated()

    assert step.preview.preview_images == [used]
    assert step._input_image_count == 1
    assert i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=6) in step.view_config.summary_text()


def test_cubemap_step_projection_toggle_is_in_preview_header() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert not step.preview.isAncestorOf(step.preview.projection_toggle_btn)
    assert step.preview.projection_toggle_btn.parentWidget().objectName() == "workPane"


def test_cubemap_step_projection_toggle_is_enabled_only_for_erp_images(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "a_pano.jpg", size=(64, 32))
    _write_test_image(images / "z_normal.jpg", size=(40, 30))

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()

    assert step.preview.current_image_path() == images / "a_pano.jpg"
    assert step.preview.projection_toggle_btn.isEnabled()
    step.preview.projection_toggle_btn.click()
    assert step.preview.preview_projection() == PREVIEW_PROJECTION_PERSPECTIVE

    step.preview._set_index(1)

    assert step.preview.current_image_path() == images / "z_normal.jpg"
    assert step.preview.preview_projection() == PREVIEW_PROJECTION_EQUIRECT
    assert not step.preview.projection_toggle_btn.isEnabled()


def test_custom_grid_defaults_to_three_pitch_rows_all_enabled() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    assert idx >= 0
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    assert step.view_config.pitch_row_count() == 3
    assert step.view_config.pitch_values() == [-45.0, 0.0, 45.0]
    assert step.view_config.pitch_rows_text() == "-45,0,45"
    assert step.view_config.yaw_slot_count() == 6
    views = step.view_config.collect_views(include_disabled=True)

    assert len(views) == 18
    assert sum(1 for view in views if view["enabled"]) == 18
    assert {view["pitch"] for view in views} == {-45.0, 0.0, 45.0}


def test_editing_cube6_grid_switches_to_custom_without_resetting_grid() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    assert step.view_config.view_mode() == "cube6"
    first_enabled = next(view for view in step.view_config.collect_views(include_disabled=True) if view["name"] == "px")
    middle_row_index = step.view_config.pitch_values().index(0.0)
    step.view_config.pitch_rows[middle_row_index]["checks"][first_enabled["slot"]].setChecked(False)

    assert step.view_config.view_mode() == "custom_views"
    assert step.view_config.yaw_slot_count() == 4
    assert step.view_config.pitch_values() == [-90.0, 0.0, 90.0]
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 5


def test_custom_grid_pitch_rows_are_limited_to_five() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    step.view_config.set_pitch_row_count(99)
    assert step.view_config.pitch_row_count() == 5
    assert step.view_config._parse_pitches()[:3] == [-45.0, 0.0, 45.0]
    assert len(step.view_config._parse_pitches()) == len(set(step.view_config._parse_pitches()))

    step.view_config.set_pitch_row_count(0)
    assert step.view_config.pitch_row_count() == 1


def test_custom_grid_controls_apply_immediately_and_keep_pitch_unique() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    step.view_config.set_yaw_slot_count(8)
    assert len(step.view_config.collect_views(include_disabled=True)) == 24

    step.view_config.set_pitch_row_count(5)
    assert len(step.view_config.collect_views(include_disabled=True)) == 40
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 40

    step.view_config.pitch_rows[0]["pitch_edit"].setValue(0.0)
    pitches = step.view_config.pitch_values()
    assert len(pitches) == len({_pitch_key for _pitch_key in pitches})
    assert pitches[0] != 0.0


def test_custom_grid_resize_preserves_existing_rows_and_checks() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)

    step.view_config.pitch_rows[0]["pitch_edit"].setValue(-12.0)
    step.view_config.pitch_rows[0]["checks"][0].setChecked(False)
    step.view_config.pitch_rows[1]["checks"][4].setChecked(False)
    pitches_before = step.view_config.pitch_values()

    step.view_config.set_yaw_slot_count(8)

    assert step.view_config.pitch_values() == pitches_before
    assert not step.view_config.pitch_rows[0]["checks"][0].isChecked()
    assert not step.view_config.pitch_rows[1]["checks"][4].isChecked()
    assert all(row["checks"][6].isChecked() and row["checks"][7].isChecked() for row in step.view_config.pitch_rows)

    step.view_config.set_yaw_slot_count(4)

    assert step.view_config.pitch_values() == pitches_before
    assert not step.view_config.pitch_rows[0]["checks"][0].isChecked()
    assert len(step.view_config.pitch_rows[0]["checks"]) == 4

    step.view_config.add_pitch_row()

    assert step.view_config.pitch_values()[:3] == pitches_before
    assert step.view_config.pitch_row_count() == 4
    assert all(cb.isChecked() for cb in step.view_config.pitch_rows[-1]["checks"])

    removed_pitch = step.view_config.pitch_values()[1]
    step.view_config.remove_pitch_row(1)

    assert removed_pitch not in step.view_config.pitch_values()
    assert step.view_config.pitch_values()[0] == pitches_before[0]
    assert step.view_config.pitch_row_count() == 3


def test_pitch_row_controls_are_packed_left_without_clipping() -> None:
    app = _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.pitch_rows[0]["pitch_edit"].setValue(-90.0)
    step.settings_tabs.setCurrentIndex(step.view_export_tab_index)

    step.resize(720, 720)
    step.show()
    app.processEvents()

    first_row = step.view_config.pitch_rows[0]
    delete_btn = first_row["delete_btn"]
    pitch_edit = first_row["pitch_edit"]
    gap = pitch_edit.geometry().left() - delete_btn.geometry().right() - 1
    required_width = pitch_edit.fontMetrics().horizontalAdvance("-999.9") + 22

    assert gap <= 4
    assert pitch_edit.width() >= required_width

    step.close()


def test_yaw_slots_share_remaining_grid_width() -> None:
    app = _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.set_yaw_slot_count(4)
    step.settings_tabs.setCurrentIndex(step.view_export_tab_index)

    step.resize(720, 720)
    step.show()
    app.processEvents()

    first_row = step.view_config.pitch_rows[0]
    pitch_cell = first_row["pitch_edit"].parentWidget()
    grid = step.view_config.grid_widget
    checkboxes = first_row["checks"]
    centers = [cb.mapTo(grid, QPoint(0, 0)).x() + cb.width() / 2.0 for cb in checkboxes]
    gaps = [b - a for a, b in zip(centers, centers[1:], strict=False)]
    pitch_width = pitch_cell.width()
    available_width = grid.width() - pitch_width
    expected_first_center = pitch_width + available_width / (2.0 * len(checkboxes))

    assert max(gaps) - min(gaps) <= 2.0
    assert abs(centers[0] - expected_first_center) <= 4.0

    step.close()


def test_custom_grid_bulk_selection_emits_single_change() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.set_yaw_slot_count(8)
    step.view_config.set_pitch_row_count(5)

    emitted = 0

    def on_changed() -> None:
        nonlocal emitted
        emitted += 1

    step.view_config.views_changed.connect(on_changed)

    assert step.view_config.all_toggle_btn.isChecked()
    assert step.view_config.all_toggle_btn.toolTip() == i18n.t("DESELECT_ALL")

    step.view_config.all_toggle_btn.click()

    assert emitted == 1
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 0
    assert not step.view_config.all_toggle_btn.isChecked()
    assert step.view_config.all_toggle_btn.toolTip() == i18n.t("SELECT_ALL")

    step.view_config.all_toggle_btn.click()
    assert emitted == 2
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 40
    assert step.view_config.all_toggle_btn.isChecked()
    assert step.view_config.all_toggle_btn.toolTip() == i18n.t("DESELECT_ALL")

    step.view_config.pitch_rows[0]["checks"][0].setChecked(False)
    assert emitted == 3
    assert not step.view_config.all_toggle_btn.isChecked()
    assert step.view_config.all_toggle_btn.toolTip() == i18n.t("SELECT_ALL")

    step.view_config._all_off()
    assert emitted == 4

    step.view_config._all_on()
    assert emitted == 5
    assert sum(1 for view in step.view_config.collect_views(include_disabled=True) if view["enabled"]) == 40
    assert step.view_config.all_toggle_btn.isChecked()

    step.view_config._all_on()
    assert emitted == 5


def test_view_selection_reuses_cached_input_image_count(tmp_path: Path, monkeypatch) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(images / "frame_0002.jpg")
    step.on_activated()

    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    monkeypatch.setattr(step.preview, "render", lambda _views, _mask_dir: None)

    def fail_rescan() -> int:
        raise AssertionError("unexpected image rescan")

    monkeypatch.setattr(step, "_count_input_images", fail_rescan)

    first_check = step.view_config.pitch_rows[0]["checks"][0]
    first_check.setChecked(False)

    assert i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=34) in step.view_config.summary_text()


def test_custom_grid_hover_marks_view_and_rerenders_preview(tmp_path: Path, monkeypatch) -> None:
    step = _ready_step(tmp_path)
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    captured: list[list[dict]] = []

    def fake_render(views: list[dict], _mask_dir: str) -> None:
        captured.append(views)

    monkeypatch.setattr(step.preview, "render", fake_render)

    step.view_config._on_view_hover(0, 2, True)

    assert captured
    highlighted = [view for view in captured[-1] if view.get("highlighted")]
    assert len(highlighted) == 1
    assert highlighted[0]["name"] == "pitm45_s2"

    step.view_config._on_view_hover(0, 2, False)
    assert not any(view.get("highlighted") for view in captured[-1])


def test_custom_grid_yaw_labels_are_compact_without_degree_mark() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    idx = step.view_config.view_mode_combo.findData("custom_views")
    step.view_config.view_mode_combo.setCurrentIndex(idx)
    step.view_config.set_yaw_slot_count(7)

    texts = [label.text() for label in step.view_config.yaw_slot_labels]
    views = step.view_config.collect_views(include_disabled=True)

    assert "S0\n45.0" in texts
    assert "S1\n96.4" in texts
    assert all("°" not in text for text in texts)
    assert all(label.toolTip().endswith("°") for label in step.view_config.yaw_slot_labels)
    assert step.view_config.yaw_slot_labels[1].toolTip() == "96.4°"
    assert all("8pt" in label.styleSheet() for label in step.view_config.yaw_slot_labels)
    assert any(view["slot"] == 1 and view["yaw"] == pytest.approx(45.0 + 360.0 / 7.0) for view in views)


def test_cubemap_yaw_numeric_fields_are_clamped_and_used(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)

    step.view_config.yaw_offset_edit.setValue(999.0)
    step.yaw_per_frame_edit.setValue(-999.0)

    assert step.view_config.yaw_offset() == 180.0
    assert step.yaw_per_frame_edit.value() == -180.0

    cmd = step._build_cubemap_cmd()

    assert _workflow_job(cmd)["yaw_offset_per_frame"] == -180.0
    views = json.loads(step4_views_config_path(tmp_path).read_text(encoding="utf-8"))["views"]
    assert any(view["yaw"] == -90.0 for view in views)
