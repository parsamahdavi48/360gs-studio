from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton

from core.app_job import AppJob
from core.artifact_registry import load_artifacts
from core.dataset_mask_policy import DATASET_MASK_GENERATE_TRAINING, DATASET_MASK_REUSE_EXISTING
from gui import i18n
from gui.app import MainWindow
from gui.common.perspective_preview import PREVIEW_PROJECTION_EQUIRECT, PREVIEW_PROJECTION_PERSPECTIVE
from gui.cubemap.view_config import VIEW_MODE_CUSTOM
from gui.steps.colmap_text_model_tool import ColmapTextModelTool
from gui.steps.realityscan_lfs_tool import RealityScanLfsTool


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _workflow_job(cmd: object) -> dict:
    if isinstance(cmd, AppJob):
        return cmd.payload
    raise AssertionError(f"Expected workflow AppJob, got {cmd!r}")


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


def _write_single_erp_metashape_xml(path: Path, label: str) -> None:
    identity = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical"><resolution width="64" height="32" /></sensor>
    </sensors>
    <cameras>
      <camera id="0" label="{label}" sensor_id="0"><transform>{identity}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def test_sfm_embedded_viewer_defers_scene_scan_until_viewer_is_opened(tmp_path: Path, monkeypatch) -> None:
    app = _app()
    calls = {"discover": 0}

    def fake_discover(_scene: Path) -> tuple:
        calls["discover"] += 1
        return ()

    monkeypatch.setattr("gui.scene_preview.window.discover_scene_preview_candidates", fake_discover)
    window = MainWindow("")
    try:
        window._set_current_step(window._sfm_step_index)

        assert window.sfm_step.scene_preview is None
        assert "viewer" not in window.sfm_step._page_indices
        anchors = window.sfm_step.findChildren(QOpenGLWidget, "sfmOpenGLSurfaceAnchor")
        assert len(anchors) == 1

        window.sfm_step.set_scene_dir(str(tmp_path))

        assert calls == {"discover": 0}
        assert window.sfm_step.scene_preview is None

        window.sfm_step.show_viewer()

        assert calls == {"discover": 0}
        app.processEvents()

        assert calls == {"discover": 1}
        assert window.sfm_step.scene_preview is not None

        window.sfm_step.show_menu()
        window.sfm_step.set_scene_dir(str(tmp_path))

        assert calls == {"discover": 1}
    finally:
        window.shutdown()


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
        colmap_link = window.sfm_step.stack.currentWidget().findChild(QLabel, "sfmColmapRepositoryLink")
        assert colmap_link is not None
        assert colmap_link.openExternalLinks()
        assert colmap_link.toolTip() == i18n.tip("COLMAP_REPOSITORY_LINK")
        assert i18n.t("COLMAP_REPOSITORY_LINK") in colmap_link.text()
        assert "https://github.com/colmap/colmap" in colmap_link.text()
        colmap_detail_layout = colmap_link.parentWidget().layout()
        assert colmap_detail_layout.indexOf(colmap_link) == colmap_detail_layout.count() - 1
        colmap_buttons = window.sfm_step.stack.currentWidget().findChildren(QPushButton)
        assert all(button.text() != i18n.t("SFM_OPEN_VIEWER") for button in colmap_buttons)
        assert not hasattr(window.sfm_step, "colmap_normal_camera_model_combo")

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
        assert all(button.text() != i18n.t("SFM_OPEN_VIEWER") for button in spheresfm_buttons)
        spheresfm_link = window.sfm_step.stack.currentWidget().findChild(QLabel, "sfmSpheresfmRepositoryLink")
        assert spheresfm_link is not None
        assert spheresfm_link.openExternalLinks()
        assert spheresfm_link.toolTip() == i18n.tip("SPHERESFM_REPOSITORY_LINK")
        assert i18n.t("SPHERESFM_REPOSITORY_LINK") in spheresfm_link.text()
        assert "https://github.com/json87/SphereSfM" in spheresfm_link.text()
        spheresfm_detail_layout = spheresfm_link.parentWidget().layout()
        assert spheresfm_detail_layout.indexOf(spheresfm_link) == spheresfm_detail_layout.count() - 1

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


def test_sfm_colmap_card_does_not_activate_hidden_cubemap_preview(tmp_path: Path, monkeypatch) -> None:
    _app()
    window = MainWindow(str(tmp_path))
    calls = {"activated": 0}

    def count_activation() -> None:
        calls["activated"] += 1

    try:
        monkeypatch.setattr(window.step4, "on_activated", count_activation)
        window._set_current_step(window._sfm_step_index)

        window.sfm_step.card_grid.buttons["colmap"].click()

        assert window.sfm_step.current_route() == "colmap"
        assert calls == {"activated": 0}

        window.sfm_step.show_menu()
        window.sfm_step.card_grid.buttons["realityscan_realign"].click()

        assert window.sfm_step.current_route() == "realityscan_realign"
        assert calls == {"activated": 1}
    finally:
        window.shutdown()


def test_realityscan_lfs_tool_defaults_and_builds_cli_command(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    rs = scene / "output" / "realityscan"
    (rs / "images" / "_geometry").mkdir(parents=True)
    (rs / "images" / "_mask").mkdir()
    (rs / "extra_images" / "_geometry").mkdir(parents=True)
    (rs / "extra_images" / "_mask").mkdir()
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
    assert Path(tool.masks_browse.text()) == rs / "images" / "_mask"
    assert Path(tool.output_browse.text()) == rs / "lfs_colmap"
    assert tool.images_extra_hint.text() == i18n.t("RS_LFS_ADDITIONAL_IMAGES_USED").format(
        folders="extra_images/_geometry"
    )
    assert tool.masks_extra_hint.text() == i18n.t("RS_LFS_ADDITIONAL_MASKS_USED").format(
        folders="extra_images/_mask"
    )
    assert len([label for label in tool.findChildren(QLabel) if label.objectName() == "workflowNote"]) == 1
    assert tool.primary_action_enabled()
    assert not hasattr(tool, "undistort_alpha_edit")
    assert tool.pre_undistort_cb.text() == i18n.t("RS_LFS_PRE_UNDISTORT")
    assert not hasattr(tool, "open_output_btn")
    assert not hasattr(tool, "status_label")

    phase, cmd = tool.build_commands()[0]
    assert phase == "realityscan_lfs_colmap"
    assert isinstance(cmd, AppJob)
    assert cmd.job_path is not None
    assert tool.on_line("[progress] 3/10") == (3, 10)
    assert tool.on_line("Images: 3") is None
    job = cmd.payload
    assert job["csv_path"] == str(csv)
    assert job["output_dir"] == str(rs / "lfs_colmap")
    assert job["images_dir"] == str(rs / "images")
    assert job["masks_dir"] == str(rs / "images" / "_mask")
    assert job["ply_path"] == str(ply)

    tool.pre_undistort_cb.setChecked(True)
    assert Path(tool.output_browse.text()) == rs / "lfs_colmap_undistorted"
    _, undistort_cmd = tool.build_commands()[0]
    assert isinstance(undistort_cmd, AppJob)
    undistort_job = undistort_cmd.payload
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
        (rs / "images" / "_geometry").mkdir(parents=True)
        (rs / "images" / "_mask").mkdir()
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
    assert Path(tool.masks_browse.text()) == rs_b / "images" / "_mask"
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


def test_colmap_text_model_tool_defaults_and_builds_dataset_job(tmp_path: Path) -> None:
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
    assert not hasattr(tool, "profile_combo")
    assert tool.primary_action_enabled()
    assert tool.settings_tabs.count() == 4
    assert tool.settings_tabs.tabText(3) == i18n.t("STEP4_TAB_MASK_SETTINGS")

    commands = tool.build_commands()
    assert [phase for phase, _cmd in commands] == ["metashape_colmap"]
    assert tool.on_line("[progress] 4/12") == (4, 12)
    assert tool.on_line("Images: 4") is None

    colmap_job = _workflow_job(commands[0][1])
    assert colmap_job["kind"] == "metashape_colmap_dataset"
    assert colmap_job["scene_dir"] == str(scene)
    assert colmap_job["images_dir"] == str(images)
    assert colmap_job["masks_dir"] == str(masks)
    assert colmap_job["xml_path"] == str(xml)
    assert colmap_job["ply_path"] == str(ply)
    assert colmap_job["output_dir"] == str(output)
    assert colmap_job["axis_transform"] == "none"
    assert colmap_job["final_orientation"] == "none"
    assert colmap_job["output_bit_depth"] == "8"
    assert colmap_job["jpg_quality"] == 95

    custom_idx = tool.view_config.view_mode_combo.findData(VIEW_MODE_CUSTOM)
    tool.view_config.view_mode_combo.setCurrentIndex(custom_idx)
    tool.view_config.set_yaw_slot_count(5)
    tool.view_config.set_pitch_row_count(2)
    custom_grid_job = _workflow_job(tool.build_commands()[0][1])
    assert len(custom_grid_job["views"]) == 10

    _write_colmap_sparse(output)
    tool.on_queue_finished(True)
    assert load_artifacts(scene, "sfm")[-1].kind == "metashape_xml_ply"
    assert load_artifacts(scene, "dataset")[-1].kind == "colmap_dataset"


def test_colmap_text_model_training_mask_modes_control_dataset_masks(tmp_path: Path, monkeypatch) -> None:
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
    assert tool._dataset_mask_step is not None
    none_index = tool._dataset_mask_step.dataset_mask_mode_combo.findData("none")
    assert tool._dataset_mask_step.dataset_mask_mode_combo.itemData(none_index, Qt.ToolTipRole) == i18n.tip(
        "COLMAP_TEXT_MASK_MODE_NONE"
    )

    tool._dataset_mask_step.set_mask_mode(DATASET_MASK_GENERATE_TRAINING)
    commands = tool.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_colmap", "yolo"]
    colmap_job = _workflow_job(commands[0][1])
    assert colmap_job["masks_dir"] == ""

    output_masks = scene / "output" / "metashape_colmap" / "masks"
    output_masks.mkdir(parents=True)
    (output_masks / "frame_0001.png").write_bytes(b"existing")
    (scene / "output" / "metashape_colmap" / "stale.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)

    tool._dataset_mask_step.set_mask_mode(DATASET_MASK_REUSE_EXISTING)
    commands = tool.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_colmap"]
    colmap_job = _workflow_job(commands[0][1])
    assert colmap_job["masks_dir"] == ""
    assert (output_masks / "frame_0001.png").is_file()
    assert not (scene / "output" / "metashape_colmap" / "stale.txt").exists()


def test_colmap_text_model_preview_projection_toggle_is_enabled_only_for_erp_images(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    _write_image(images / "a_pano.jpg", size=(64, 32))
    _write_image(images / "z_normal.jpg", size=(40, 30))
    (scene / "metashape.xml").write_text("<document/>", encoding="utf-8")
    (scene / "metashape.ply").write_text("ply\n", encoding="ascii")

    _app()
    tool = ColmapTextModelTool(Path.cwd())
    tool.set_scene_dir(str(scene))

    assert tool.preview.current_image_path() == images / "a_pano.jpg"
    assert tool.preview.projection_toggle_btn.isEnabled()
    tool.preview.projection_toggle_btn.click()
    assert tool.preview.preview_projection() == PREVIEW_PROJECTION_PERSPECTIVE

    tool.preview._set_index(1)

    assert tool.preview.current_image_path() == images / "z_normal.jpg"
    assert tool.preview.preview_projection() == PREVIEW_PROJECTION_EQUIRECT
    assert not tool.preview.projection_toggle_btn.isEnabled()


def test_colmap_text_model_preview_uses_metashape_xml_camera_images(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    used = images / "pano_used.jpg"
    unused = images / "normal_unused.jpg"
    _write_image(used, size=(64, 32))
    _write_image(unused, size=(40, 30))
    _write_single_erp_metashape_xml(scene / "metashape.xml", used.name)
    (scene / "metashape.ply").write_text("ply\n", encoding="ascii")

    _app()
    tool = ColmapTextModelTool(Path.cwd())
    tool.set_scene_dir(str(scene))

    assert tool.preview.preview_images == [used]
    assert i18n.t("OUTPUT_IMAGE_COUNT_FORMAT").format(count=6) in tool.view_config.summary_text()


def test_colmap_text_model_tool_uses_dataset_job_for_mixed_metashape_xml(tmp_path: Path) -> None:
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

    assert [phase for phase, _cmd in commands] == ["metashape_colmap"]
    cmd = commands[0][1]
    assert isinstance(cmd, AppJob)
    job = cmd.payload
    assert job["kind"] == "metashape_colmap_dataset"
    assert job["scene_dir"] == str(scene)
    assert job["output_dir"] == str(scene / "output" / "metashape_colmap")
    assert job["output_bit_depth"] == "8"
    assert job["jpg_quality"] == 95
    assert job["axis_transform"] == "none"
    assert job["final_orientation"] == "none"


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
