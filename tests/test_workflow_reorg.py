from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QSizePolicy

from core.artifact_registry import load_artifacts
from core.normal_camera_metadata import load_normal_camera_default
from core.scene_layout import source_image_sets_path
from gui import i18n
from gui.app import MainWindow
from gui.cubemap.view_config import VIEW_MODE_CUSTOM
from gui.steps.colmap_text_model_tool import ColmapTextModelTool
from gui.steps.realityscan_lfs_tool import RealityScanLfsTool


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _workflow_job(cmd: list[str]) -> dict:
    assert cmd[2].endswith("run_workflow_job.py")
    job_path = Path(cmd[cmd.index("--job") + 1])
    return json.loads(job_path.read_text(encoding="utf-8"))


def _write_colmap_sparse(root: Path) -> None:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse / name).write_text("", encoding="utf-8")


def _write_image(path: Path, size: tuple[int, int] = (40, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(110, 120, 130)).save(path)


def _write_mixed_metashape_xml(path: Path) -> None:
    identity = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical"><resolution width="64" height="32" /></sensor>
      <sensor id="1" type="frame"><resolution width="40" height="30" /><calibration><f>35</f><k1>0.1</k1></calibration></sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{identity}</transform></camera>
      <camera id="1" label="normal.jpg" sensor_id="1"><transform>{identity}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def test_sfm_cards_open_in_step_sfm_pages_and_external_route_goes_to_dataset(tmp_path: Path) -> None:
    _app()
    window = MainWindow(str(tmp_path))
    try:
        assert len(window.steps) == 6
        assert [button.text() for button in window.step_buttons] == [
            i18n.t("STEP1_NAV"),
            i18n.t("STEP2_NAV"),
            i18n.t("STEP3_NAV"),
            i18n.t("STEP4_NAV"),
            i18n.t("STEP5_NAV"),
            i18n.t("STEP6_NAV"),
        ]
        route_cards = [*window.sfm_step.card_grid.buttons.values(), *window.dataset_step.card_grid.buttons.values()]
        assert len({card.minimumHeight() for card in route_cards}) == 1
        assert all(card.minimumHeight() == card.maximumHeight() for card in route_cards)
        assert window.sfm_step.card_grid.buttons["colmap"].title_label.text() == i18n.t("SFM_ROUTE_COLMAP_TITLE")
        assert (
            window.dataset_step.card_grid.buttons["realityscan_lfs"].title_label.text()
            == i18n.t("DATASET_TOOL_RS_LFS_TITLE")
        )
        window._set_current_step(4)
        assert list(window.sfm_step.card_grid.buttons) == [
            "metashape",
            "realityscan_realign",
            "colmap",
            "spheresfm",
            "viewer",
        ]
        assert list(window.dataset_step.card_grid.buttons) == [
            "metashape_dataset",
            "realityscan_lfs",
            "spheresfm_dataset",
            "scale",
            "colmap_text_model",
        ]
        assert window.dataset_step.dataset_menu_note.text() == i18n.t("DATASET_MENU_DESC")
        dataset_menu_labels = window.dataset_step.stack.currentWidget().findChildren(QLabel)
        assert all(label.text() != i18n.t("DATASET_MENU_TITLE") for label in dataset_menu_labels)

        window._set_current_step(3)
        assert window.stack.currentWidget() is window.sfm_step
        assert window.run_btn.text().strip() == i18n.t("SFM_SELECT_ROUTE")
        assert not window.run_btn.isEnabled()

        window.sfm_step.card_grid.buttons["colmap"].click()

        assert window.stack.currentIndex() == window._sfm_step_index
        assert window.stack.currentWidget() is window.sfm_step
        assert window.sfm_step.current_route() == "colmap"
        assert window.step4._export_method() == "colmap"
        assert window.step4.pipeline_stage_intent("sfm") is True
        assert window.step4.pipeline_stage_intent("conversion") is True
        assert window.run_btn.text().strip() == i18n.t("SFM_RUN_COLMAP")

        window.sfm_step.show_menu()
        assert window.run_btn.text().strip() == i18n.t("SFM_SELECT_ROUTE")

        window.sfm_step.card_grid.buttons["spheresfm"].click()
        assert window.stack.currentIndex() == window._sfm_step_index
        assert window.sfm_step.current_route() == "spheresfm"
        assert window.step4._export_method() == "spheresfm"
        assert window.step4.pipeline_stage_intent("sfm") is True
        assert window.step4.pipeline_stage_intent("conversion") is False
        assert window.run_btn.text().strip() == i18n.t("SFM_RUN_SPHERESFM")
        spheresfm_buttons = window.sfm_step.stack.currentWidget().findChildren(QPushButton)
        assert all(button.text() != i18n.t("SPHERESFM_OPEN_GUI") for button in spheresfm_buttons)
        viewer_buttons = [button for button in spheresfm_buttons if button.text() == i18n.t("SFM_OPEN_VIEWER")]
        assert len(viewer_buttons) == 1
        assert viewer_buttons[0].sizePolicy().horizontalPolicy() == QSizePolicy.Fixed

        window.sfm_step.show_menu()
        assert window.run_btn.text().strip() == i18n.t("SFM_SELECT_ROUTE")

        window.sfm_step.card_grid.buttons["realityscan_realign"].click()
        assert window.stack.currentIndex() == window._sfm_step_index
        assert window.sfm_step.current_route() == "realityscan_realign"
        assert window.sfm_step.header_title() == i18n.t("SFM_REALITYSCAN_DETAIL_TITLE")
        assert window.step4._export_method() == "metashape"
        assert window.step4.profile_combo.currentData() == "realityscan"
        assert window.step4.profile_combo.isHidden()
        assert not window.step4.realityscan_options_row.isHidden()
        assert window.step4.realityscan_options_label is not None
        assert not window.step4.realityscan_options_label.isHidden()
        assert window.step4.pipeline_stage_intent("sfm") is True
        assert window.step4.pipeline_stage_intent("conversion") is True
        assert window.run_btn.text().strip() == i18n.t("SFM_RUN_REALITYSCAN")
        (tmp_path / "images").mkdir(exist_ok=True)
        metashape_xml = tmp_path / "metashape.xml"
        metashape_xml.write_text("<document/>", encoding="utf-8")
        window.step4.ms_xml_browse.set_text(str(metashape_xml))
        commands = window.sfm_step.build_commands()
        assert [phase for phase, _cmd in commands] == ["metashape", "cubemap"]
        job = _workflow_job(commands[1][1])
        assert job["realityscan_xmp"] is True
        assert job["output_dir"] == str(tmp_path / "output" / "realityscan")

        window.step_back_btn.click()
        assert window.sfm_step.current_route() == ""
        assert window.run_btn.text().strip() == i18n.t("SFM_SELECT_ROUTE")

        window.sfm_step.card_grid.buttons["viewer"].click()
        assert window.stack.currentIndex() == window._sfm_step_index
        assert window.stack.currentWidget() is window.sfm_step
        assert window.sfm_step.header_title() == i18n.t("SFM_ROUTE_VIEWER_TITLE")
        assert window.run_btn.text().strip() == i18n.t("SFM_OPEN_VIEWER")
        assert not window.run_btn.isEnabled()
        assert not window.step_back_btn.isHidden()
        assert window.sfm_step.scene_preview._scene_dir == tmp_path
        assert window.sfm_step.scene_preview.scene_label.isHidden()

        window.step_back_btn.click()
        assert window.sfm_step.header_title() == i18n.t("SFM_MENU_TITLE")
        assert window.run_btn.text().strip() == i18n.t("SFM_SELECT_ROUTE")

        window.sfm_step.card_grid.buttons["metashape"].click()

        assert window.stack.currentIndex() == window._dataset_step_index
        assert window.stack.currentWidget() is window.dataset_step
        assert window.dataset_step.current_tool() == "menu"
        assert window.run_btn.text().strip() == i18n.t("DATASET_SELECT_TOOL")
        assert not window.run_btn.isEnabled()
        assert not hasattr(window, "step4_sub_buttons")

        window.dataset_step.card_grid.buttons["metashape_dataset"].click()
        assert window.dataset_step.current_tool() == "metashape_dataset"
        assert window.step4._export_method() == "metashape"
        assert window.step4.pipeline_stage_intent("conversion") is True
        assert window.step4.profile_combo.currentData() != "realityscan"
        assert not window.step4.profile_combo.isHidden()
        assert window.step4.realityscan_options_row.isHidden()
        assert window.step4.realityscan_options_label is not None
        assert window.step4.realityscan_options_label.isHidden()
        assert window.step4.export_colmap_cb.isHidden()
        realityscan_profile_index = window.step4.profile_combo.findData("realityscan")
        assert window.step4.profile_combo.view().isRowHidden(realityscan_profile_index)
        custom_profile_index = window.step4.profile_combo.findData("custom")
        assert window.step4.profile_combo.view().isRowHidden(custom_profile_index)
        assert window.step4.settings_tabs.currentIndex() == window.step4.output_tab_index
        assert window.run_btn.text().strip() == i18n.t("DATASET_RUN_METASHAPE")

        window.dataset_step.show_tool("spheresfm_dataset")
        assert window.dataset_step.current_tool() == "spheresfm_dataset"
        assert window.step4._export_method() == "spheresfm"
        assert window.step4.pipeline_stage_intent("sfm") is False
        assert window.step4.pipeline_stage_intent("conversion") is True
        assert window.run_btn.text().strip() == i18n.t("DATASET_RUN_SPHERESFM")
        assert not window.step4.spheresfm_section.isVisible()
        spheresfm_custom_profile_index = window.step4.spheresfm_profile_combo.findData("custom")
        assert window.step4.spheresfm_profile_combo.view().isRowHidden(spheresfm_custom_profile_index)
        assert window.step4.settings_tabs.currentIndex() == window.step4.output_tab_index

        window.dataset_step.show_tool("colmap_text_model")
        assert window.dataset_step.current_tool() == "colmap_text_model"
        assert window.run_btn.text().strip() == i18n.t("DATASET_RUN_COLMAP_TEXT")
        assert window.dataset_step.colmap_text_tool.settings_tabs.currentIndex() == 1
        colmap_text_labels = window.dataset_step.stack.currentWidget().findChildren(QLabel)
        assert any(label.text() == i18n.t("DATASET_TOOL_COLMAP_TEXT_DESC") for label in colmap_text_labels)

        window._open_dataset_route("colmap")
        assert window.dataset_step.current_tool() == "colmap_ready"
        assert window.step4._export_method() == "colmap"
        assert window.run_btn.text().strip() == i18n.t("DATASET_COLMAP_READY_ACTION")
        assert not window.run_btn.isEnabled()
    finally:
        window.shutdown()


def test_colmap_sfm_route_saves_normal_camera_default(tmp_path: Path) -> None:
    _app()
    _write_image(tmp_path / "images" / "normal_0001.jpg", (40, 30))
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    window = MainWindow(str(tmp_path))
    try:
        window.sfm_step.show_route("colmap")
        window.sfm_step.colmap_exec_browse.set_text(str(fake_colmap))
        model_index = window.sfm_step.colmap_normal_camera_model_combo.findData("PINHOLE")
        assert model_index >= 0
        window.sfm_step.colmap_normal_camera_model_combo.setCurrentIndex(model_index)
        window.sfm_step.colmap_normal_camera_params_edit.setText("20,21,19.5,14.5")

        commands = window.sfm_step.build_commands()

        camera = load_normal_camera_default(tmp_path)
        assert camera.camera_model == "PINHOLE"
        assert camera.camera_params == (20.0, 21.0, 19.5, 14.5)
        normal_feature = commands[1][1]
        assert normal_feature[normal_feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
        assert normal_feature[normal_feature.index("--ImageReader.camera_params") + 1] == "20,21,19.5,14.5"
    finally:
        window.shutdown()


def test_colmap_sfm_route_saves_source_resolution_normal_camera_default(tmp_path: Path) -> None:
    _app()
    _write_image(tmp_path / "images" / "a.jpg", (40, 30))
    _write_image(tmp_path / "images" / "b.jpg", (80, 60))
    source_sets = source_image_sets_path(tmp_path)
    source_sets.parent.mkdir(parents=True, exist_ok=True)
    source_sets.write_text(
        json.dumps(
            {
                "version": 1,
                "image_sets": [
                    {
                        "id": "cam_a",
                        "source_type": "image_sequence",
                        "projection": "normal",
                        "files": [{"scene_path": "images/a.jpg"}],
                    },
                    {
                        "id": "cam_b",
                        "source_type": "image_sequence",
                        "projection": "normal",
                        "files": [{"scene_path": "images/b.jpg"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    window = MainWindow(str(tmp_path))
    try:
        window.sfm_step.show_route("colmap")
        window.sfm_step.colmap_exec_browse.set_text(str(fake_colmap))
        target_scope = ("group", "image_sequence", "cam_a", 40, 30)
        scope_index = window.sfm_step._find_combo_data(window.sfm_step.colmap_normal_camera_scope_combo, target_scope)
        assert scope_index >= 0
        window.sfm_step.colmap_normal_camera_scope_combo.setCurrentIndex(scope_index)
        model_index = window.sfm_step.colmap_normal_camera_model_combo.findData("PINHOLE")
        assert model_index >= 0
        window.sfm_step.colmap_normal_camera_model_combo.setCurrentIndex(model_index)
        window.sfm_step.colmap_normal_camera_params_edit.setText("20,21,19.5,14.5")
        window.sfm_step.colmap_normal_camera_apply_btn.click()

        commands = window.sfm_step.build_commands()

        normal_features = [cmd for phase, cmd in commands if phase.startswith("colmap_feature_normal")]
        assert len(normal_features) == 2
        camera_models = [cmd[cmd.index("--ImageReader.camera_model") + 1] for cmd in normal_features]
        assert "PINHOLE" in camera_models
        assert "SIMPLE_RADIAL" in camera_models
        pinhole_cmd = next(cmd for cmd in normal_features if cmd[cmd.index("--ImageReader.camera_model") + 1] == "PINHOLE")
        assert pinhole_cmd[pinhole_cmd.index("--ImageReader.camera_params") + 1] == "20,21,19.5,14.5"
        assert any("normal_image_list_cam_a_40x30_pinhole_20_21_19p5_14p5.txt" in part for part in pinhole_cmd)
    finally:
        window.shutdown()


def test_realityscan_lfs_tool_defaults_and_builds_cli_command(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    rs = scene / "output" / "realityscan"
    (rs / "images").mkdir(parents=True)
    (rs / "masks").mkdir()
    csv = rs / "rs_scene.csv"
    ply = rs / "rs_scene.ply"
    csv.write_text("header\n", encoding="utf-8")
    ply.write_text("ply\n", encoding="ascii")

    _app()
    tool = RealityScanLfsTool(Path.cwd())
    tool.set_scene_dir(str(scene))

    assert Path(tool.csv_browse.text()) == csv
    assert Path(tool.ply_browse.text()) == ply
    assert Path(tool.images_browse.text()) == rs / "images"
    assert Path(tool.masks_browse.text()) == rs / "masks"
    assert Path(tool.output_browse.text()) == rs / "lfs_colmap"
    assert tool.primary_action_enabled()
    assert not hasattr(tool, "undistort_alpha_edit")
    assert tool.pre_undistort_cb.text() == i18n.t("RS_LFS_PRE_UNDISTORT")
    assert not hasattr(tool, "open_output_btn")
    assert not hasattr(tool, "status_label")

    phase, cmd = tool.build_commands()[0]
    assert phase == "realityscan_lfs_colmap"
    assert cmd[2].endswith("realityscan_to_lfs_colmap.py")
    assert cmd[3] == "--job"
    job_path = Path(cmd[4])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["csv_path"] == str(csv)
    assert job["output_dir"] == str(rs / "lfs_colmap")
    assert job["images_dir"] == str(rs / "images")
    assert job["masks_dir"] == str(rs / "masks")
    assert job["ply_path"] == str(ply)

    tool.pre_undistort_cb.setChecked(True)
    assert Path(tool.output_browse.text()) == rs / "lfs_colmap_undistorted"
    _, undistort_cmd = tool.build_commands()[0]
    undistort_job = json.loads(Path(undistort_cmd[4]).read_text(encoding="utf-8"))
    assert undistort_job["pre_undistort_distorted_images"] is True
    assert undistort_job["undistort_alpha"] == 1.0

    _write_colmap_sparse(rs / "lfs_colmap_undistorted")
    tool.on_queue_finished(True)
    assert load_artifacts(scene, "sfm")[-1].kind == "realityscan_csv_ply"
    assert load_artifacts(scene, "dataset")[-1].kind == "lichtfeld_colmap"


def test_realityscan_lfs_tool_refreshes_defaults_when_scene_changes(tmp_path: Path) -> None:
    scenes: list[Path] = []
    for name in ("scene_a", "scene_b"):
        scene = tmp_path / name
        rs = scene / "output" / "realityscan"
        (rs / "images").mkdir(parents=True)
        (rs / "masks").mkdir()
        (rs / f"rs_{name}.csv").write_text("header\n", encoding="utf-8")
        (rs / f"rs_{name}.ply").write_text("ply\n", encoding="ascii")
        scenes.append(scene)

    _app()
    tool = RealityScanLfsTool(Path.cwd())
    tool.set_scene_dir(str(scenes[0]))
    tool.set_scene_dir(str(scenes[1]))

    rs_b = scenes[1] / "output" / "realityscan"
    assert Path(tool.csv_browse.text()) == rs_b / "rs_scene_b.csv"
    assert Path(tool.ply_browse.text()) == rs_b / "rs_scene_b.ply"
    assert Path(tool.images_browse.text()) == rs_b / "images"
    assert Path(tool.masks_browse.text()) == rs_b / "masks"
    assert Path(tool.output_browse.text()) == rs_b / "lfs_colmap"

    manual_csv = scenes[1] / "manual.csv"
    manual_csv.write_text("manual\n", encoding="utf-8")
    tool.csv_browse.set_text(str(manual_csv))
    tool.on_activated()

    assert Path(tool.csv_browse.text()) == manual_csv


def test_colmap_text_model_tool_ignores_cwd_images_without_scene(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "orphan.jpg").write_bytes(b"not a real image")
    monkeypatch.chdir(tmp_path)

    _app()
    tool = ColmapTextModelTool(Path.cwd())
    tool.on_activated()

    assert tool.scene_dir == ""
    assert tool.preview.preview_images == []
    assert tool.preview.current_image_path() is None


def test_colmap_text_model_tool_defaults_and_builds_cli_command(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"image")
    (masks / "frame_0001.png").write_bytes(b"mask")
    xml = scene / "metashape.xml"
    xml.write_text("<document/>", encoding="utf-8")
    ply = scene / "metashape.ply"
    ply.write_text("ply\n", encoding="ascii")

    _app()
    tool = ColmapTextModelTool(Path.cwd())
    tool.set_scene_dir(str(scene))

    assert Path(tool.images_browse.text()) == images
    assert Path(tool.masks_browse.text()) == masks
    assert Path(tool.xml_browse.text()) == xml
    assert Path(tool.ply_browse.text()) == ply
    output = scene / "output" / "metashape_colmap"
    assert not hasattr(tool, "output_browse")
    assert tool.primary_action_enabled()
    assert tool.settings_tabs.count() == 3
    assert tool.profile_combo.currentData() == "lichtfeld"

    commands = tool.build_commands()
    assert [phase for phase, _cmd in commands] == ["metashape", "cubemap", "colmap_text"]

    preprocess_cmd = commands[0][1]
    work = scene / "_stechdrive" / "step4" / "work" / "metashape_colmap_import"
    preprocess_job = _workflow_job(preprocess_cmd)
    assert preprocess_job["kind"] == "metashape_preprocess"
    assert preprocess_job["images_dir"] == str(images)
    assert preprocess_job["xml_path"] == str(xml)
    assert preprocess_job["output_dir"] == str(work)
    assert preprocess_job["ply_path"] == str(ply)

    cubemap_cmd = commands[1][1]
    cubemap_job = _workflow_job(cubemap_cmd)
    projected_work = work / "projected"
    assert cubemap_job["kind"] == "cubemap_conversion"
    assert cubemap_job["input_dir"] == str(work)
    assert cubemap_job["output_dir"] == str(projected_work)
    assert cubemap_job["axis_mode"] == "none"
    assert cubemap_job["final_orientation"] == "lichtfeld"
    assert cubemap_job["image_dir"] == str(images)
    assert cubemap_job["mask_dir"] == str(masks)

    colmap_cmd = commands[2][1]
    colmap_job = _workflow_job(colmap_cmd)
    assert colmap_job["kind"] == "transforms_to_colmap"
    assert colmap_job["input_dir"] == str(projected_work)
    assert colmap_job["output_dir"] == str(output / "sparse" / "0")
    assert colmap_job["ply_path"] == str(projected_work / "pointcloud.ply")
    assert colmap_job["dataset_root"] == str(output)
    assert colmap_job["asset_input_dir"] == str(projected_work)
    assert colmap_job["copy_images"] is True
    assert colmap_job["copy_masks"] is True

    brush_idx = tool.profile_combo.findData("brush")
    tool.profile_combo.setCurrentIndex(brush_idx)
    _, brush_cubemap_cmd = tool.build_commands()[1]
    brush_job = _workflow_job(brush_cubemap_cmd)
    assert brush_job["axis_mode"] == "brush"
    assert brush_job["final_orientation"] == "none"

    custom_idx = tool.view_config.view_mode_combo.findData(VIEW_MODE_CUSTOM)
    tool.view_config.view_mode_combo.setCurrentIndex(custom_idx)
    tool.view_config.set_yaw_slot_count(5)
    tool.view_config.set_pitch_row_count(2)
    _, custom_grid_cmd = tool.build_commands()[1]
    assert len(_workflow_job(custom_grid_cmd)["views"]) == 10

    _write_colmap_sparse(output)
    tool.on_queue_finished(True)
    assert load_artifacts(scene, "sfm")[-1].kind == "metashape_xml_ply"
    assert load_artifacts(scene, "dataset")[-1].kind == "colmap_dataset"


def test_colmap_text_model_tool_uses_mixed_writer_for_mixed_metashape_xml(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    (images / "pano.jpg").write_bytes(b"image")
    (images / "normal.jpg").write_bytes(b"image")
    xml = scene / "metashape.xml"
    _write_mixed_metashape_xml(xml)
    ply = scene / "metashape.ply"
    ply.write_text("ply\n", encoding="ascii")

    _app()
    tool = ColmapTextModelTool(Path.cwd())
    tool.set_scene_dir(str(scene))

    commands = tool.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_colmap_mixed"]
    cmd = commands[0][1]
    assert cmd[2].endswith("export_metashape_colmap_dataset.py")
    assert cmd[3] == "--job"
    job = json.loads(Path(cmd[4]).read_text(encoding="utf-8"))
    assert job["scene_dir"] == str(scene)
    assert job["output_dir"] == str(scene / "output" / "metashape_colmap")
    assert job["output_bit_depth"] == "8"
    assert job["jpg_quality"] == 95


def test_realityscan_realign_profile_is_step4_only(tmp_path: Path) -> None:
    _app()
    window = MainWindow(str(tmp_path))
    try:
        window._set_current_step(4)
        window.dataset_step.card_grid.buttons["metashape_dataset"].click()
        realityscan_profile_index = window.step4.profile_combo.findData("realityscan")
        assert window.step4.profile_combo.view().isRowHidden(realityscan_profile_index)

        window._set_current_step(3)
        window.sfm_step.card_grid.buttons["realityscan_realign"].click()
        assert not window.step4.profile_combo.view().isRowHidden(realityscan_profile_index)
        assert window.step4.profile_combo.currentData() == "realityscan"
        assert window.step4.profile_combo.isHidden()
    finally:
        window.shutdown()
