import json
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QMessageBox

import core.orientation_correction as orientation_correction
import gui.steps.step4_cubemap as step4_cubemap
from core.app_job import AppJob
from core.normal_camera_metadata import save_normal_camera_default
from core.scene_layout import (
    project_path,
    source_image_sets_path,
    step4_export_settings_path,
    step4_meta_dir,
    step4_training_runs_path,
    step4_views_config_path,
)
from core.transforms_to_colmap import read_ply_points
from core.workflow_artifacts import register_dataset_artifact
from gui import i18n
from gui.common.collapsible_section import CollapsibleSection
from gui.common.perspective_preview import PREVIEW_PROJECTION_EQUIRECT, PREVIEW_PROJECTION_PERSPECTIVE
from gui.steps.sfm_route_backends import get_sfm_route_backend
from gui.steps.sfm_route_specs import (
    OUTPUT_SHAPE_EQUIRECT_3DGUT,
    OUTPUT_SHAPE_PROJECTED,
    SFM_ROUTE_COLMAP,
    SFM_ROUTE_IDS,
    SFM_ROUTE_METASHAPE,
    SFM_ROUTE_SPHERESFM,
    get_sfm_route_spec,
    normalize_sfm_route,
)
from gui.steps.step4_cubemap import CubemapStep
from gui.steps.step4_settings import STEP4_SETTINGS_VERSION
from gui.steps.step5_training import TrainingStep
from gui.steps.training_backends import lichtfeld_defaults

_IDENTITY_MATRIX_TEXT = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def _app():
    return QApplication.instance() or QApplication([])


def _workflow_job(cmd: object) -> dict:
    if isinstance(cmd, AppJob):
        return cmd.payload
    raise AssertionError(f"Expected workflow AppJob, got {cmd!r}")


def _ready_step(scene: Path, *, metashape_inputs: bool = False) -> CubemapStep:
    _app()
    scene.mkdir(exist_ok=True)
    _write_ascii_ply(scene / "pointcloud.ply", [(0.0, 0.0, 0.0)])
    if metashape_inputs:
        (scene / "images").mkdir(exist_ok=True)
        _write_test_image(scene / "images" / "frame_0001.jpg", size=(64, 32))
        _write_metashape_xml(scene / "metashape.xml")
        _write_ascii_ply(scene / "metashape.ply", [(1.0, 2.0, 3.0)])
    _app()
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(scene))
    if metashape_inputs:
        step._approve_metashape_ply()
    return step


def _write_metashape_xml(path: Path, labels: list[str] | None = None) -> None:
    labels = labels or ["frame_0001.jpg"]
    cameras = "\n".join(
        f'        <camera id="{idx}" sensor_id="0" label="{label}">\n'
        "          <transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform>\n"
        "        </camera>"
        for idx, label in enumerate(labels)
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<document version="1.2.0">\n'
        "  <chunk>\n"
        "    <sensors>\n"
        '      <sensor id="0" label="camera" type="spherical">\n'
        '        <resolution width="64" height="32" />\n'
        "      </sensor>\n"
        "    </sensors>\n"
        "    <cameras>\n"
        f"{cameras}\n"
        "    </cameras>\n"
        "  </chunk>\n"
        "</document>\n",
        encoding="utf-8",
    )


def _write_mixed_metashape_xml(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<document>\n"
        "  <chunk>\n"
        "    <sensors>\n"
        '      <sensor id="0" type="spherical">\n'
        '        <resolution width="64" height="32" />\n'
        "      </sensor>\n"
        '      <sensor id="1" type="frame">\n'
        '        <resolution width="40" height="30" />\n'
        "        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>\n"
        "      </sensor>\n"
        "    </sensors>\n"
        "    <cameras>\n"
        f'      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY_MATRIX_TEXT}</transform></camera>\n'
        f'      <camera id="1" label="frame.jpg" sensor_id="1"><transform>{_IDENTITY_MATRIX_TEXT}</transform></camera>\n'
        "    </cameras>\n"
        "  </chunk>\n"
        "</document>\n",
        encoding="utf-8",
    )


def _write_ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    rows = "\n".join(f"{x:g} {y:g} {z:g}" for x, y, z in points)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        f"{rows}\n",
        encoding="ascii",
    )


def _write_test_image(path: Path, size: tuple[int, int] = (64, 32)) -> None:
    Image.new("RGB", size, (0, 0, 0)).save(path)


def _write_output_dataset(
    scene: Path,
    *,
    output_shape: str,
    pointcloud: bool = True,
    legacy_root: bool = False,
) -> Path:
    if legacy_root:
        output = scene / "output"
    else:
        name = "metashape_3dgut" if output_shape == "equirect_3dgut" else "metashape_cubemap"
        output = scene / "output" / name
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    _write_test_image(images / "frame_0001.jpg")
    camera_model = "EQUIRECTANGULAR" if output_shape == "equirect_3dgut" else "SIMPLE_PINHOLE"
    (output / "transforms.json").write_text(json.dumps({"camera_model": camera_model, "frames": []}), encoding="utf-8")
    if pointcloud:
        _write_ascii_ply(output / "pointcloud.ply", [(0.0, 0.0, 0.0)])
    settings_path = step4_export_settings_path(scene)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "settings_version": STEP4_SETTINGS_VERSION,
                "output_shape": output_shape,
                "output_dir": str(output),
                "portable_output": {
                    "root": output.relative_to(scene).as_posix(),
                    "dataset_kind": "3dgut" if output_shape == "equirect_3dgut" else "projection_views",
                    "active": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return output


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


def _write_spheresfm_sparse_stub(scene: Path) -> Path:
    sparse_model = scene / "output" / "spheresfm" / "sparse" / "0"
    sparse_model.mkdir(parents=True, exist_ok=True)
    (sparse_model / "cameras.txt").write_text("# cameras\n", encoding="ascii")
    (sparse_model / "images.txt").write_text("# images\n", encoding="ascii")
    (sparse_model / "points3D.txt").write_text("# points\n", encoding="ascii")
    return sparse_model


def _is_descendant(widget, ancestor) -> bool:
    current = widget
    while current is not None:
        if current is ancestor:
            return True
        current = current.parentWidget()
    return False


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


def _ready_lichtfeld_training_step(scene: Path) -> CubemapStep:
    step = _ready_step(scene, metashape_inputs=True)
    _write_output_dataset(scene, output_shape="projected")
    fake_lfs = scene / "LichtFeld-Studio.exe"
    fake_lfs.write_text("", encoding="utf-8")
    step.run_training_cb.setChecked(True)
    step.training_executable_browse.set_text(str(fake_lfs))
    step.lfs_auto_steps_scaler_cb.setChecked(False)
    return step


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
    assert set(step.training_backend_buttons) == {"lichtfeld", "postshot"}
    assert set(step.training_backend_selector.primary_backend_buttons) == {"lichtfeld", "postshot"}
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
    assert stale.exists()
    assert not (tmp_path / "transforms.json").exists()


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


def test_spheresfm_output_shape_change_keeps_conversion_tab_focused() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step._set_export_method("spheresfm")
    step.settings_tabs.setCurrentIndex(step.spheresfm_convert_tab_index)

    direct_idx = step.spheresfm_output_shape_combo.findData("equirect_3dgut")
    projected_idx = step.spheresfm_output_shape_combo.findData("projected")
    assert direct_idx >= 0
    assert projected_idx >= 0

    step.spheresfm_output_shape_combo.setCurrentIndex(direct_idx)
    assert step.settings_tabs.currentIndex() == step.spheresfm_convert_tab_index

    step.spheresfm_output_shape_combo.setCurrentIndex(projected_idx)
    assert step.settings_tabs.currentIndex() == step.spheresfm_convert_tab_index


def test_spheresfm_erp_output_is_disabled_outside_lichtfeld_profile(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    step._set_export_method("spheresfm")
    direct_idx = step.spheresfm_output_shape_combo.findData("equirect_3dgut")
    postshot_idx = step.spheresfm_profile_combo.findData("postshot")
    lichtfeld_idx = step.spheresfm_profile_combo.findData("lichtfeld")
    assert direct_idx >= 0
    assert postshot_idx >= 0
    assert lichtfeld_idx >= 0

    assert step.spheresfm_output_shape_combo.isItemEnabled(direct_idx)
    step.spheresfm_profile_combo.setCurrentIndex(postshot_idx)

    assert step.spheresfm_output_shape_combo.currentData() == "projected"
    assert step.spheresfm_profile_combo.currentData() == "postshot"
    assert not step.spheresfm_output_shape_combo.isItemEnabled(direct_idx)
    assert step.spheresfm_output_shape_combo.itemToolTip(direct_idx) == i18n.tip(
        "OUTPUT_SHAPE_EQUIRECT_3DGUT_DISABLED"
    )

    step.spheresfm_output_shape_combo.setCurrentIndex(direct_idx)
    assert step.spheresfm_output_shape_combo.currentData() == "projected"
    assert step.spheresfm_profile_combo.currentData() == "postshot"

    step.spheresfm_profile_combo.setCurrentIndex(lichtfeld_idx)
    assert step.spheresfm_output_shape_combo.isItemEnabled(direct_idx)
    assert step.spheresfm_output_shape_combo.itemToolTip(direct_idx) == i18n.tip("OUTPUT_SHAPE_EQUIRECT_3DGUT")


def test_spheresfm_visible_tabs_follow_projection_conversion_sfm_order() -> None:
    _app()
    step = CubemapStep(Path.cwd())

    step._set_export_method("spheresfm")

    assert [step.settings_tabs.tabText(i) for i in range(step.settings_tabs.count())] == [
        i18n.t("STEP4_TAB_INPUT"),
        i18n.t("STEP4_TAB_OUTPUT"),
        i18n.t("STEP4_TAB_DETAILS"),
    ]
    assert step.metashape_section.isHidden()
    assert step.metashape_sfm_input_widget.isHidden()
    assert step.colmap_section.isHidden()
    assert step.colmap_sfm_input_widget.isHidden()
    assert not step.spheresfm_sfm_input_widget.isHidden()
    assert not step.spheresfm_section.isHidden()
    assert not step.spheresfm_convert_section.isHidden()
    assert step.settings_tabs.isTabEnabled(step.input_tab_index)
    assert step.settings_tabs.isTabEnabled(step.output_tab_index)


def test_spheresfm_conversion_rows_keep_axis_transform_internal_only() -> None:
    _app()
    step = CubemapStep(Path.cwd())
    form = step.spheresfm_convert_section.layout().itemAt(0).layout()

    profile_row, _profile_role = form.getWidgetPosition(step.spheresfm_profile_combo)
    shape_row, _shape_role = form.getWidgetPosition(step.spheresfm_output_shape_combo)
    axis_row, _axis_role = form.getWidgetPosition(step.spheresfm_axis_transform_combo)
    axis_label = form.labelForField(step.spheresfm_axis_transform_combo)

    assert profile_row < shape_row < axis_row
    assert step.spheresfm_axis_transform_combo.toolTip() == i18n.tip("SPHERESFM_AXIS_TRANSFORM")
    assert step.spheresfm_axis_transform_combo.isHidden()
    assert axis_label is not None
    assert axis_label.toolTip() == i18n.tip("SPHERESFM_AXIS_TRANSFORM")
    assert axis_label.isHidden()


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


def test_colmap_export_method_uses_image_only_conversion(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    step._set_export_method("colmap")

    assert not step.metashape_section.isVisible()
    assert step.export_method_buttons["colmap"].isChecked()
    assert step.yaw_per_frame_edit.value() == 0.0
    assert not step.yaw_per_frame_edit.isEnabled()
    assert step.yaw_per_frame_edit.toolTip() == i18n.t("YAW_OFFSET_PER_FRAME_COLMAP_HINT")
    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["colmap_rig_export"]
    cmd = commands[0][1]
    job = _workflow_job(cmd)
    assert job["image_only"] is True
    assert job["colmap_rig"] is True
    assert job["yaw_offset_per_frame"] == 0.0
    assert job["final_orientation"] == "none"
    assert job["write_images"] is True
    assert job["write_masks"] is True


def test_colmap_export_method_prepares_normal_only_project(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "perspective_0001.jpg", size=(40, 30))
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["colmap_mixed_prepare"]
    cmd = commands[0][1]
    assert isinstance(cmd, AppJob)
    assert cmd.job_type == "sfm"
    job = cmd.payload
    assert job["scene_dir"] == str(tmp_path)
    assert job["output_dir"] == str(tmp_path / "output")
    assert job["views"]


def test_colmap_export_can_queue_mixed_erp_and_normal_sfm(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "pano_0001.jpg", size=(64, 32))
    _write_test_image(images / "perspective_0001.jpg", size=(40, 30))
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "colmap_mixed_prepare",
        "colmap_feature_rig",
        "colmap_rig_config",
        "colmap_feature_normal",
        "colmap_match",
        "colmap_mapper",
    ]
    assert isinstance(commands[0][1], AppJob)
    assert commands[0][1].job_type == "sfm"
    rig_feature = commands[1][1]
    normal_feature = commands[3][1]
    assert rig_feature[1] == "feature_extractor"
    assert rig_feature[rig_feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    assert rig_feature[rig_feature.index("--image_list_path") + 1] == str(
        tmp_path / "output" / "colmap_rig" / "rig_image_list.txt"
    )
    assert normal_feature[1] == "feature_extractor"
    assert normal_feature[normal_feature.index("--ImageReader.camera_model") + 1] == "SIMPLE_RADIAL"
    assert normal_feature[normal_feature.index("--image_list_path") + 1] == str(
        tmp_path / "output" / "colmap_rig" / "normal_image_list_unknown_40x30_simple_radial.txt"
    )


def test_colmap_export_can_queue_normal_only_sfm_without_rig_config(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "perspective_0001.jpg", size=(40, 30))
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "colmap_mixed_prepare",
        "colmap_feature_normal",
        "colmap_match",
        "colmap_mapper",
    ]
    assert isinstance(commands[0][1], AppJob)
    assert commands[0][1].job_type == "sfm"
    assert all(phase != "colmap_rig_config" for phase, _cmd in commands)
    normal_feature = commands[1][1]
    assert normal_feature[normal_feature.index("--ImageReader.camera_model") + 1] == "SIMPLE_RADIAL"
    assert "--Mapper.ba_refine_sensor_from_rig" not in commands[-1][1]


def test_colmap_export_uses_normal_camera_metadata_for_feature_group(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "perspective_0001.jpg", size=(40, 30))
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
                        "files": [
                            {
                                "scene_path": "images/perspective_0001.jpg",
                                "camera": {
                                    "model": "PINHOLE",
                                    "params": [20.0, 21.0, 19.5, 14.5],
                                    "source": "manual",
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    normal_feature = commands[1][1]
    assert normal_feature[normal_feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    assert normal_feature[normal_feature.index("--ImageReader.camera_params") + 1] == "20,21,19.5,14.5"


def test_colmap_export_uses_normal_camera_default_for_feature_group(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "perspective_0001.jpg", size=(40, 30))
    save_normal_camera_default(
        tmp_path,
        camera_model="PINHOLE",
        camera_params=(20.0, 21.0, 19.5, 14.5),
        camera_source="test_default",
    )
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    normal_feature = commands[1][1]
    assert normal_feature[normal_feature.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    assert normal_feature[normal_feature.index("--ImageReader.camera_params") + 1] == "20,21,19.5,14.5"


def test_colmap_export_method_restores_yaw_step_when_leaving_route(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    step.yaw_per_frame_edit.setValue(45.0)

    step._set_export_method("colmap")

    assert step.yaw_per_frame_edit.value() == 0.0
    assert not step.yaw_per_frame_edit.isEnabled()

    step._set_export_method("metashape")

    assert step.yaw_per_frame_edit.isEnabled()
    assert step.yaw_per_frame_edit.value() == 45.0
    assert step.yaw_per_frame_edit.toolTip() == i18n.t("YAW_OFFSET_PER_FRAME_HINT")


def test_colmap_export_method_validates_images_before_resetting_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir(parents=True)
    old_file = output / "old.txt"
    old_file.write_text("old", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    with pytest.raises(ValueError, match="画像フォルダ"):
        step.build_commands()

    assert old_file.is_file()


def test_colmap_export_finalize_writes_export_method_settings(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")

    commands = step.build_commands()
    assert [phase for phase, _cmd in commands] == ["colmap_rig_export"]
    step._finalize_bundle()

    settings_path = step4_export_settings_path(tmp_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["export_method"] == "colmap"
    assert settings["conversion"]["no_image"] is False
    assert settings["conversion"]["export_colmap"] is False
    assert settings["conversion"]["yaw_offset_per_frame"] == 0.0
    assert settings["colmap_rig"]["enabled"] is True
    assert settings["colmap_rig"]["dir"] == str(tmp_path / "output" / "colmap_rig")
    assert settings["colmap_rig"]["project_dir"] == str(tmp_path / "output" / "colmap_rig")

    manifest_path = step4_meta_dir(tmp_path) / "sfm" / "stechdrive_colmap_project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["export_type"] == "colmap_project"
    assert manifest["project_dir"] == str(tmp_path / "output" / "colmap_rig")
    assert manifest["ready_for_import"] is False


def test_colmap_export_manifest_marks_sparse_project_ready(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = tmp_path / "output" / "colmap_rig" / "sparse" / "0"
    sparse_model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse_model / name).write_bytes(b"model")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")

    step._finalize_bundle()

    manifest = json.loads(
        (step4_meta_dir(tmp_path) / "sfm" / "stechdrive_colmap_project.json").read_text(encoding="utf-8")
    )
    assert manifest["ready_for_import"] is True
    assert manifest["sparse_model_dir"] == "sparse/0"


def test_colmap_export_method_displays_colmap_project_folder_summary(tmp_path: Path) -> None:
    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step.cubemap_path_summary_value.full_text() == "output/metashape_cubemap/"

    step._set_export_method("colmap")

    assert step.sfm_path_summary_kind.text() == i18n.t("STEP4_SUMMARY_INPUT")
    assert step.sfm_path_summary_value.full_text() == "output/colmap_rig/sparse/"

    step.set_pipeline_stage_intent("sfm", True)

    assert step.sfm_path_summary_kind.text() == i18n.t("STEP4_SUMMARY_OUTPUT")
    assert step.sfm_path_summary_value.full_text() == "output/colmap_rig/"
    assert step.cubemap_path_summary_value.full_text() == "output/colmap_rig/"


def test_colmap_export_can_queue_colmap_sfm_with_custom_executable(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "colmap_rig_export",
        "colmap_feature",
        "colmap_rig_config",
        "colmap_match",
        "colmap_mapper",
    ]
    assert commands[1][1][0] == str(fake_colmap)
    assert commands[1][1][1] == "feature_extractor"
    assert "--ImageReader.single_camera_per_folder" in commands[1][1]
    assert commands[1][1][commands[1][1].index("--ImageReader.camera_params") + 1] == "10,10,9.5,9.5"
    assert commands[2][1][1] == "rig_configurator"
    assert commands[3][1][1] == "sequential_matcher"
    assert commands[4][1][1] == "global_mapper"


def test_colmap_export_can_queue_colmap_global_mapper(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))
    idx = step.colmap_mapper_combo.findData("global")
    assert idx >= 0
    step.colmap_mapper_combo.setCurrentIndex(idx)

    commands = step.build_commands()

    assert commands[-1][1][0] == str(fake_colmap)
    assert commands[-1][1][1] == "global_mapper"


def test_spheresfm_method_can_queue_3dgut_export_without_projection_views(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step._set_combo_data(step.spheresfm_output_shape_combo, "equirect_3dgut")
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    assert step.sfm_path_summary_value.full_text() == "output/spheresfm/"
    assert step.cubemap_path_summary_value.full_text() == "output/spheresfm_3dgut/"
    assert not step.export_targets_row.isEnabled()
    assert not step.view_config.settings_widget.isEnabled()

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "spheresfm_preflight",
        "spheresfm_prepare",
        "spheresfm_database",
        "spheresfm_feature",
        "spheresfm_match",
        "spheresfm_mapper",
        "spheresfm_transforms",
    ]
    preflight_job = _workflow_job(commands[0][1])
    prepare_job = _workflow_job(commands[1][1])
    transforms_job = _workflow_job(commands[6][1])
    assert preflight_job["colmap"] == str(fake_colmap)
    assert prepare_job["use_masks"] is True
    assert commands[3][1][commands[3][1].index("--ImageReader.camera_model") + 1] == "SPHERE"
    assert commands[3][1][commands[3][1].index("--ImageReader.camera_params") + 1] == "1,32,16"
    assert commands[3][1][commands[3][1].index("--ImageReader.mask_path") + 1] == str(
        tmp_path / "output" / "spheresfm" / "masks_colmap"
    )
    assert commands[4][1][commands[4][1].index("--SequentialMatching.overlap") + 1] == "10"
    assert commands[5][1][commands[5][1].index("--Mapper.sphere_camera") + 1] == "1"
    assert commands[5][1][commands[5][1].index("--Mapper.multiple_models") + 1] == "0"
    assert commands[5][1][commands[5][1].index("--Mapper.ba_global_max_num_iterations") + 1] == "33"
    assert transforms_job["sparse_dir"] == str(tmp_path / "output" / "spheresfm" / "sparse")
    assert transforms_job["output_dir"] == str(tmp_path / "output" / "spheresfm_3dgut")
    assert transforms_job["image_path_mode"] == "images-prefix"
    assert os.path.samefile(
        images / "frame_0001.jpg",
        tmp_path / "output" / "spheresfm_3dgut" / "images" / "frame_0001.jpg",
    )
    assert os.path.samefile(
        masks / "frame_0001.png",
        tmp_path / "output" / "spheresfm_3dgut" / "masks" / "frame_0001.png",
    )


def test_spheresfm_method_can_queue_projected_cubemap_export(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    assert step.sfm_path_summary_value.full_text() == "output/spheresfm/"
    assert step.cubemap_path_summary_value.full_text() == "output/spheresfm_cubemap/"
    assert step.export_targets_row.isEnabled()
    assert step.view_config.settings_widget.isEnabled()

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "spheresfm_preflight",
        "spheresfm_prepare",
        "spheresfm_database",
        "spheresfm_feature",
        "spheresfm_match",
        "spheresfm_mapper",
        "spheresfm_transforms",
        "spheresfm_cubemap",
    ]
    transform_cmd = commands[6][1]
    cubemap_cmd = commands[7][1]
    transform_job = _workflow_job(transform_cmd)
    cubemap_job = _workflow_job(cubemap_cmd)
    assert transform_job["output_dir"] == str(tmp_path / "output" / "spheresfm" / "equirect")
    assert transform_job["image_path_mode"] == "relative"
    assert cubemap_job["input_dir"] == str(tmp_path / "output" / "spheresfm" / "equirect")
    assert cubemap_job["output_dir"] == str(tmp_path / "output" / "spheresfm_cubemap")
    assert step4_views_config_path(tmp_path).is_file()
    assert cubemap_job["image_dir"] == str(images)
    assert cubemap_job["mask_dir"] == str(masks)
    assert cubemap_job["axis_mode"] == "none"
    assert cubemap_job["final_orientation"] == "lichtfeld"


def test_spheresfm_method_can_queue_sfm_only(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step.set_pipeline_stage_intent("conversion", False)
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "spheresfm_preflight",
        "spheresfm_prepare",
        "spheresfm_database",
        "spheresfm_feature",
        "spheresfm_match",
        "spheresfm_mapper",
    ]
    assert not step.settings_tabs.isTabEnabled(step.spheresfm_convert_tab_index)


def test_spheresfm_convert_only_requires_existing_sparse(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step.set_pipeline_stage_intent("sfm", False)

    with pytest.raises(ValueError, match="sparse"):
        step.build_commands()


def test_spheresfm_convert_only_queues_3dgut_without_colmap_binary(tmp_path: Path) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step.set_pipeline_stage_intent("sfm", False)
    step._set_combo_data(step.spheresfm_output_shape_combo, "equirect_3dgut")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["spheresfm_transforms"]
    job = _workflow_job(commands[0][1])
    assert job["sparse_dir"] == str(sparse_model)
    assert job["output_dir"] == str(tmp_path / "output" / "spheresfm_3dgut")
    assert job["image_path_mode"] == "images-prefix"
    assert sparse_model.is_dir()


def test_spheresfm_open_gui_warns_when_selected_binary_has_no_gui_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)
    fake_colmap = tmp_path / "colmap.exe"
    fake_colmap.write_text("", encoding="utf-8")
    warnings: list[tuple[str, str]] = []

    class FakeGuiLessQProcess:
        MergedChannels = object()
        NormalExit = object()

        def __init__(self, parent=None) -> None:
            self.parent = parent
            self.program = ""
            self.arguments: list[str] = []

        def setProgram(self, program: str) -> None:
            self.program = program

        def setArguments(self, arguments: list[str]) -> None:
            self.arguments = arguments

        def setProcessChannelMode(self, _mode) -> None:
            pass

        def start(self) -> None:
            pass

        def waitForStarted(self, _msecs: int) -> bool:
            return True

        def waitForFinished(self, _msecs: int) -> bool:
            return True

        def readAllStandardOutput(self) -> bytes:
            return b"ERROR: Cannot start colmap GUI; colmap was built without GUI support or QT dependency is missing."

        def readAllStandardError(self) -> bytes:
            return b""

        def errorString(self) -> str:
            return ""

        def exitStatus(self):
            return self.NormalExit

        def exitCode(self) -> int:
            return 1

    monkeypatch.setattr(
        CubemapStep,
        "_create_spheresfm_gui_process",
        lambda self: FakeGuiLessQProcess(self),
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, text: warnings.append((title, text)))

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.spheresfm_exec_browse.set_text(str(fake_colmap))

    step._open_spheresfm_result()

    assert len(warnings) == 1
    assert warnings[0][0] == i18n.t("SPHERESFM_OPEN_GUI")
    assert str(fake_colmap) in warnings[0][1]
    assert str(sparse_model) in warnings[0][1]
    assert "Qt GUI" in warnings[0][1]


def test_spheresfm_3dgut_convert_only_confirms_output_dataset_targets(tmp_path: Path, monkeypatch) -> None:
    _app()
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    _write_test_image(masks / "frame_0001.png")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)
    gut_output = tmp_path / "output" / "spheresfm_3dgut"
    transforms = gut_output / "transforms.json"
    pointcloud = gut_output / "pointcloud.ply"
    old_linked_image = gut_output / "images" / "old.jpg"
    old_linked_mask = gut_output / "masks" / "old.png"
    old_linked_image.parent.mkdir(parents=True)
    old_linked_mask.parent.mkdir(parents=True)
    transforms.write_text("old", encoding="utf-8")
    pointcloud.write_text("old", encoding="utf-8")
    old_linked_image.write_text("old", encoding="utf-8")
    old_linked_mask.write_text("old", encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step.set_pipeline_stage_intent("sfm", False)
    step._set_combo_data(step.spheresfm_output_shape_combo, "equirect_3dgut")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["spheresfm_transforms"]
    assert not transforms.exists()
    assert not pointcloud.exists()
    assert not old_linked_image.exists()
    assert not old_linked_mask.exists()
    assert os.path.samefile(images / "frame_0001.jpg", gut_output / "images" / "frame_0001.jpg")
    assert os.path.samefile(masks / "frame_0001.png", gut_output / "masks" / "frame_0001.png")
    assert sparse_model.is_dir()
    assert images.is_dir()
    assert masks.is_dir()


def test_spheresfm_convert_only_resets_conversion_outputs_only(tmp_path: Path, monkeypatch) -> None:
    _app()
    images = tmp_path / "images"
    images.mkdir()
    _write_test_image(images / "frame_0001.jpg")
    sparse_model = _write_spheresfm_sparse_stub(tmp_path)
    database = tmp_path / "output" / "spheresfm" / "database.db"
    database.write_text("db", encoding="utf-8")
    old_equirect = tmp_path / "output" / "spheresfm" / "equirect" / "old.txt"
    old_views = step4_views_config_path(tmp_path)
    old_images = tmp_path / "output" / "spheresfm_cubemap" / "images" / "old.jpg"
    old_masks = tmp_path / "output" / "spheresfm_cubemap" / "masks" / "old.png"
    old_equirect.parent.mkdir(parents=True)
    old_views.parent.mkdir(parents=True)
    old_images.parent.mkdir(parents=True)
    old_masks.parent.mkdir(parents=True)
    old_equirect.write_text("old", encoding="utf-8")
    old_views.write_text("old", encoding="utf-8")
    old_images.write_text("old", encoding="utf-8")
    old_masks.write_text("old", encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("spheresfm")
    step.set_pipeline_stage_intent("sfm", False)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["spheresfm_transforms", "spheresfm_cubemap"]
    assert not old_equirect.exists()
    assert old_views.is_file()
    assert old_views.read_text(encoding="utf-8") != "old"
    assert not old_images.exists()
    assert not old_masks.exists()
    assert sparse_model.is_dir()
    assert database.is_file()


def test_colmap_user_preferences_restore_executable_and_pipeline_choices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("STECHDRIVE_USER_SETTINGS_PATH", str(settings_path))

    first = CubemapStep(Path.cwd())
    first.enable_user_preferences()
    first.colmap_exec_browse.set_text(str(tmp_path / "colmap.exe"))
    first.glomap_exec_browse.set_text(str(tmp_path / "glomap.exe"))
    matcher_idx = first.colmap_matcher_combo.findData("exhaustive")
    mapper_idx = first.colmap_mapper_combo.findData("incremental")
    assert matcher_idx >= 0
    assert mapper_idx >= 0
    first.colmap_matcher_combo.setCurrentIndex(matcher_idx)
    first.colmap_mapper_combo.setCurrentIndex(mapper_idx)

    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["step4_colmap"]["colmap_executable"].endswith("colmap.exe")
    assert stored["step4_colmap"]["glomap_executable"].endswith("glomap.exe")
    assert stored["step4_colmap"]["matcher"] == "exhaustive"
    assert stored["step4_colmap"]["mapper"] == "incremental"

    second = CubemapStep(Path.cwd())
    second.enable_user_preferences()

    assert second.colmap_exec_browse.text().endswith("colmap.exe")
    assert second.glomap_exec_browse.text().endswith("glomap.exe")
    assert second.colmap_matcher_combo.currentData() == "exhaustive"
    assert second.colmap_mapper_combo.currentData() == "incremental"


def test_spheresfm_user_preferences_restore_quality_preset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _app()
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("STECHDRIVE_USER_SETTINGS_PATH", str(settings_path))
    settings_path.write_text(
        json.dumps({"step4_colmap": {"spheresfm_quality_preset": "quality"}}),
        encoding="utf-8",
    )

    step = CubemapStep(Path.cwd())
    step.enable_user_preferences()

    assert step.spheresfm_quality_combo.currentData() == "quality"


def test_spheresfm_scene_settings_restore_stage_intents(tmp_path: Path) -> None:
    _app()
    settings_path = step4_export_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "settings_version": STEP4_SETTINGS_VERSION,
                "export_method": "spheresfm",
                "spheresfm": {"run_scope": "convert_only"},
            }
        ),
        encoding="utf-8",
    )

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step._export_method() == "spheresfm"
    assert step.pipeline_stage_intent("sfm") is False
    assert step.pipeline_stage_intent("conversion") is True
    assert step._spheresfm_run_scope() == "convert_only"


def test_colmap_scene_settings_restore_stage_intents(tmp_path: Path) -> None:
    _app()
    settings_path = step4_export_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "settings_version": STEP4_SETTINGS_VERSION,
                "export_method": "colmap",
                "colmap_rig": {"run_sfm": True},
            }
        ),
        encoding="utf-8",
    )

    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))

    assert step._export_method() == "colmap"
    assert step.pipeline_stage_intent("sfm") is True
    assert step.pipeline_stage_intent("conversion") is True
    assert step.colmap_exec_browse.isEnabled()
    assert step.take_pipeline_notice() == ""


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


def test_postshot_does_not_use_lichtfeld_pointcloud(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    idx = step.profile_combo.findData("postshot")
    assert idx >= 0
    step.profile_combo.setCurrentIndex(idx)

    assert step._resolve_ply_source() is None
    with pytest.raises(ValueError, match="pointcloud"):
        step.build_commands()


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


def test_realityscan_profile_builds_xmp_export_command(tmp_path: Path) -> None:
    step = _ready_step(tmp_path, metashape_inputs=True)

    realityscan_idx = step.profile_combo.findData("realityscan")
    assert realityscan_idx >= 0
    step.profile_combo.setCurrentIndex(realityscan_idx)
    assert step.realityscan_pose_prior_combo.currentData() == "exact"
    assert step.realityscan_calibration_prior_combo.currentData() == "exact"
    assert not step.realityscan_include_rig_cb.isChecked()
    assert not step.realityscan_options_row.isHidden()
    assert step.realityscan_options_label is not None
    assert not step.realityscan_options_label.isHidden()
    assert not step.axis_transform_combo.isEnabled()
    assert step.axis_transform_combo.currentText() == i18n.t("AXIS_TRANSFORM_REALITYSCAN_AUTO")
    step._set_combo_data(step.realityscan_pose_prior_combo, "locked")

    cmd = step._build_cubemap_cmd()

    assert step._display_output_dir() == tmp_path / "output" / "realityscan"
    job = _workflow_job(cmd)
    assert job["output_dir"] == str(tmp_path / "output" / "realityscan")
    assert job["realityscan_xmp"] is True
    assert job["realityscan_pose_prior"] == "locked"
    assert job["realityscan_calibration_prior"] == "exact"
    assert job["realityscan_coordinates"] == "auto"
    assert job["realityscan_include_rig"] is False
    assert job["realityscan_unposed_images"] is True
    assert job["realityscan_unposed_scene_dir"] == str(tmp_path)
    assert job["axis_mode"] == "brush"
    assert job["final_orientation"] == "realityscan"
    assert job["yaw_offset_per_frame"] == 0.0
    assert step.axis_transform_combo.currentData() == "brush"
    assert not step.ms_use_ply_cb.isChecked()
    assert not step.export_colmap_cb.isEnabled()
    assert step.export_colmap_cb.isHidden()
    assert not step.ms_no_fix_rot_cb.isChecked()

    settings = step._collect_export_settings()
    assert settings["postprocess"]["final_orientation"] == "realityscan"
    assert settings["postprocess"]["final_orientation_stage"] == "cubemap_cli"
    assert settings["postprocess"]["realityscan_final_orientation_correction"] is True
    assert settings["realityscan"]["include_rig"] is False

    step.realityscan_include_rig_cb.setChecked(True)
    rig_cmd = step._build_cubemap_cmd()
    rig_job = _workflow_job(rig_cmd)
    assert rig_job["realityscan_include_rig"] is True
    assert rig_job["realityscan_rig_name"] == "stechdrive-cubemap"


def test_realityscan_profile_preserves_existing_shared_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir(parents=True)
    old_transforms = output / "transforms.json"
    old_transforms.write_text("old lichtfeld", encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    realityscan_idx = step.profile_combo.findData("realityscan")
    assert realityscan_idx >= 0
    step.profile_combo.setCurrentIndex(realityscan_idx)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("confirmation should not open")),
    )

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape", "cubemap"]
    assert old_transforms.read_text(encoding="utf-8") == "old lichtfeld"
    assert (output / "realityscan").is_dir()
    assert _workflow_job(commands[1][1])["output_dir"] == str(output / "realityscan")


def test_realityscan_profile_resets_only_realityscan_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir(parents=True)
    old_shared = output / "lichtfeld.txt"
    old_shared.write_text("old lichtfeld", encoding="utf-8")
    realityscan_output = output / "realityscan"
    realityscan_output.mkdir()
    old_realityscan = realityscan_output / "old.txt"
    old_realityscan.write_text("old realityscan", encoding="utf-8")
    questions: list[str] = []
    step = _ready_step(tmp_path, metashape_inputs=True)
    realityscan_idx = step.profile_combo.findData("realityscan")
    assert realityscan_idx >= 0
    step.profile_combo.setCurrentIndex(realityscan_idx)

    def confirm(_parent, _title, text, *_args, **_kwargs):
        questions.append(text)
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape", "cubemap"]
    assert questions == [i18n.t("OUTPUT_RESET_MESSAGE").format(path=str(realityscan_output))]
    assert old_shared.read_text(encoding="utf-8") == "old lichtfeld"
    assert not old_realityscan.exists()
    assert realityscan_output.is_dir()


def test_realityscan_finalize_does_not_touch_shared_output_or_copy_ply(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir(parents=True)
    shared_transforms = output / "transforms.json"
    shared_transforms.write_text(json.dumps({"existing": True}), encoding="utf-8")
    step = _ready_step(tmp_path, metashape_inputs=True)
    realityscan_idx = step.profile_combo.findData("realityscan")
    assert realityscan_idx >= 0
    step.profile_combo.setCurrentIndex(realityscan_idx)
    assert step.ms_ply_browse.text()
    assert step._resolve_ply_source() is None
    realityscan_output = output / "realityscan"
    realityscan_output.mkdir()
    realityscan_transforms = realityscan_output / "transforms.json"
    realityscan_transforms.write_text(json.dumps({"frames": []}), encoding="utf-8")

    step._finalize_bundle()

    assert not (output / "metashape.ply").exists()
    assert json.loads(shared_transforms.read_text(encoding="utf-8")) == {"existing": True}
    assert not (realityscan_output / "metashape.ply").exists()
    assert "ply_file_path" not in json.loads(realityscan_transforms.read_text(encoding="utf-8"))


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


def test_postshot_accepts_raw_ply_with_custom_name(tmp_path: Path) -> None:
    step = _ready_step(tmp_path)
    raw_ply = tmp_path / "raw_scan.ply"
    raw_ply.write_text("ply\n", encoding="utf-8")
    step.ms_ply_browse.set_text(str(raw_ply))
    idx = step.profile_combo.findData("postshot")
    assert idx >= 0
    step.profile_combo.setCurrentIndex(idx)

    assert step._resolve_ply_source() == raw_ply


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

    assert step.export_images_cb.isChecked() is True
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


def test_colmap_route_splits_conversion_and_step5_postshot_training(tmp_path: Path) -> None:
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

    _app()
    step = CubemapStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step._set_export_method("colmap")
    step.set_pipeline_stage_intent("sfm", True)
    step.colmap_exec_browse.set_text(str(fake_colmap))
    step._set_training_backend("postshot")
    assert step.postshot_project_name_edit.text() == f"{tmp_path.name}.psht"
    step.training_executable_browse.set_text(str(fake_postshot))
    step.run_training_cb.setChecked(True)
    step.postshot_project_name_edit.setText("scene.psht")
    step.postshot_ksteps_auto_cb.setChecked(False)
    step.postshot_ksteps_edit.setText("42")
    step.postshot_max_image_size_edit.setText("2048")

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == [
        "colmap_rig_export",
        "colmap_feature",
        "colmap_rig_config",
        "colmap_match",
        "colmap_mapper",
    ]
    assert all(not phase.startswith("training_") for phase, _cmd in commands)

    rig_images = tmp_path / "output" / "colmap_rig" / "images"
    rig_images.mkdir(parents=True, exist_ok=True)
    _write_test_image(rig_images / "frame_0001.jpg")
    sparse = tmp_path / "output" / "colmap_rig" / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text("", encoding="utf-8")
    (sparse / "images.txt").write_text("", encoding="utf-8")
    (sparse / "points3D.txt").write_text("", encoding="utf-8")

    cmd = step.build_training_launch_commands()[0][1]
    assert cmd == [
        str(fake_postshot),
        "train",
        "--import",
        str(tmp_path / "output" / "colmap_rig" / "images"),
        str(tmp_path / "output" / "colmap_rig" / "sparse" / "0"),
        "--output",
        str(tmp_path / "output" / "scene.psht"),
        "--profile",
        "Splat3",
        "-s",
        "42",
        "--max-image-size",
        "2048",
        "--image-select",
        "all",
        "--max-sh-degree",
        "3",
    ]


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


def test_metashape_dataset_writer_resets_existing_output_even_if_images_toggle_is_off(
    tmp_path: Path, monkeypatch
) -> None:
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
    step.export_images_cb.setChecked(False)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    commands = step.build_commands()

    assert [phase for phase, _cmd in commands] == ["metashape_nerf"]
    job = _workflow_job(commands[0][1])
    assert job["kind"] == "metashape_nerf_dataset"
    assert not old_file.exists()
    assert not old_mask.exists()
    assert old_settings.read_text(encoding="utf-8") == '{"old": true}\n'


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
    assert settings["output_files"]["pointcloud"] == "pointcloud.ply"
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
