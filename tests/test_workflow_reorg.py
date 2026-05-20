from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QSizePolicy

from gui import i18n
from gui.app import MainWindow
from gui.cubemap.view_config import VIEW_MODE_CUSTOM
from gui.steps.colmap_text_model_tool import ColmapTextModelTool
from gui.steps.realityscan_lfs_tool import RealityScanLfsTool


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


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
        assert "--realityscan-xmp" in commands[1][1]
        assert str(tmp_path / "output" / "realityscan") in commands[1][1]

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
    assert cmd[3] == str(csv)
    assert cmd[4] == str(rs / "lfs_colmap")
    assert cmd[cmd.index("--images-dir") + 1] == str(rs / "images")
    assert cmd[cmd.index("--masks-dir") + 1] == str(rs / "masks")
    assert cmd[cmd.index("--ply") + 1] == str(ply)

    tool.pre_undistort_cb.setChecked(True)
    assert Path(tool.output_browse.text()) == rs / "lfs_colmap_undistorted"
    _, undistort_cmd = tool.build_commands()[0]
    assert "--pre-undistort-distorted-images" in undistort_cmd
    assert "--undistort-alpha" in undistort_cmd
    assert undistort_cmd[undistort_cmd.index("--undistort-alpha") + 1] == "1"


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
    assert preprocess_cmd[2].endswith("metashape_360_lfs.py")
    assert preprocess_cmd[preprocess_cmd.index("--images") + 1] == str(images)
    assert preprocess_cmd[preprocess_cmd.index("--xml") + 1] == str(xml)
    assert preprocess_cmd[preprocess_cmd.index("--output") + 1] == str(work)
    assert preprocess_cmd[preprocess_cmd.index("--ply") + 1] == str(ply)

    cubemap_cmd = commands[1][1]
    assert cubemap_cmd[2].endswith("cubemap_transforms_json.py")
    assert cubemap_cmd[3] == str(work)
    assert cubemap_cmd[4] == str(output)
    assert "--no_transform" in cubemap_cmd
    assert cubemap_cmd[cubemap_cmd.index("--final-orientation") + 1] == "lichtfeld"
    assert cubemap_cmd[cubemap_cmd.index("--image-dir") + 1] == str(images)
    assert cubemap_cmd[cubemap_cmd.index("--mask_dir") + 1] == str(masks)

    colmap_cmd = commands[2][1]
    assert colmap_cmd[2].endswith("transforms_to_colmap.py")
    assert colmap_cmd[3] == str(output)
    assert colmap_cmd[4] == str(output / "sparse" / "0")
    assert colmap_cmd[colmap_cmd.index("--ply") + 1] == str(output / "pointcloud.ply")

    brush_idx = tool.profile_combo.findData("brush")
    tool.profile_combo.setCurrentIndex(brush_idx)
    _, brush_cubemap_cmd = tool.build_commands()[1]
    assert "--brush" in brush_cubemap_cmd
    assert "--final-orientation" not in brush_cubemap_cmd

    custom_idx = tool.view_config.view_mode_combo.findData(VIEW_MODE_CUSTOM)
    tool.view_config.view_mode_combo.setCurrentIndex(custom_idx)
    tool.view_config.set_yaw_slot_count(5)
    tool.view_config.set_pitch_row_count(2)
    _, custom_grid_cmd = tool.build_commands()[1]
    views_path = Path(custom_grid_cmd[custom_grid_cmd.index("--views-json") + 1])
    views_payload = json.loads(views_path.read_text(encoding="utf-8"))
    assert len(views_payload["views"]) == 10


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
