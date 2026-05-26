from __future__ import annotations

from tests.helpers.step4 import (
    Path,
    QMessageBox,
    _ready_step,
    _workflow_job,
    i18n,
    json,
)


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

    commands = step.build_commands()
    preprocess_job = _workflow_job(commands[0][1])
    assert preprocess_job["kind"] == "metashape_preprocess"
    assert preprocess_job["lichtfeld_camera_y180"] is False


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
