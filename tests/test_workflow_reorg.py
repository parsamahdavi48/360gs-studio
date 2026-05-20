from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui import i18n
from gui.app import MainWindow
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

        window.sfm_step.show_menu()
        assert window.run_btn.text().strip() == i18n.t("SFM_SELECT_ROUTE")

        opened: list[bool] = []
        window.step4.open_scene_preview = lambda: opened.append(True)  # type: ignore[method-assign]
        window.sfm_step.card_grid.buttons["viewer"].click()
        assert opened == [True]
        assert window.stack.currentIndex() == window._sfm_step_index

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
        assert window.run_btn.text().strip() == i18n.t("DATASET_RUN_METASHAPE")

        window.dataset_step.show_tool("spheresfm_dataset")
        assert window.dataset_step.current_tool() == "spheresfm_dataset"
        assert window.step4._export_method() == "spheresfm"
        assert window.step4.pipeline_stage_intent("sfm") is False
        assert window.step4.pipeline_stage_intent("conversion") is True
        assert window.run_btn.text().strip() == i18n.t("DATASET_RUN_SPHERESFM")
        assert not window.step4.spheresfm_section.isVisible()

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
