from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.dataset_export_plan import (
    EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
    EXPORT_ACTION_LINK_PINHOLE,
    EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE,
    build_metashape_dataset_export_plan,
)
from core.metashape_model import (
    CAMERA_MODEL_EQUIRECTANGULAR,
    CAMERA_MODEL_OPENCV,
    CAMERA_MODEL_PINHOLE,
    parse_metashape_model,
)
from core.metashape_preview_targets import build_metashape_preview_targets, metashape_output_count_for_actions
from core.scene_inventory import build_scene_inventory

_IDENTITY = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 120, 140)).save(path)


def _write_xml(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical" label="pano">
        <resolution width="64" height="32" />
      </sensor>
      <sensor id="1" type="frame" label="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>
      </sensor>
      <sensor id="2" type="frame" label="distorted">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy><k1>0.1</k1></calibration>
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
      <camera id="1" label="frame.jpg" sensor_id="1"><transform>{_IDENTITY}</transform></camera>
      <camera id="2" label="distorted.jpg" sensor_id="2"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def test_parse_metashape_model_preserves_mixed_sensor_types(tmp_path: Path) -> None:
    xml = tmp_path / "cameras.xml"
    _write_xml(xml)

    model = parse_metashape_model(xml)

    assert len(model.sensors) == 3
    assert model.sensors["0"].camera_model == CAMERA_MODEL_EQUIRECTANGULAR
    assert model.sensors["1"].camera_model == CAMERA_MODEL_PINHOLE
    assert model.sensors["2"].camera_model == CAMERA_MODEL_OPENCV
    assert [camera.label for camera in model.cameras] == ["pano.jpg", "frame.jpg", "distorted.jpg"]


def test_metashape_export_plan_expands_only_erp_and_undistorts_distorted_frames(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32))
    _write_image(scene / "images" / "frame.jpg", (40, 30))
    _write_image(scene / "images" / "distorted.jpg", (40, 30))
    xml = tmp_path / "cameras.xml"
    _write_xml(xml)

    plan = build_metashape_dataset_export_plan(parse_metashape_model(xml), build_scene_inventory(scene))

    assert plan.warnings == ()
    assert [item.action for item in plan.items] == [
        EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
        EXPORT_ACTION_LINK_PINHOLE,
        EXPORT_ACTION_UNDISTORT_FRAME_TO_PINHOLE,
    ]
    assert [item.image_rel_path for item in plan.items] == [
        "images/pano.jpg",
        "images/frame.jpg",
        "images/distorted.jpg",
    ]


def test_metashape_preview_targets_follow_xml_cameras_not_whole_folder(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    pano = scene / "images" / "pano.jpg"
    frame = scene / "images" / "frame.jpg"
    unused = scene / "images" / "unused.jpg"
    _write_image(pano, (64, 32))
    _write_image(frame, (40, 30))
    _write_image(unused, (40, 30))
    xml = tmp_path / "cameras.xml"
    xml.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical"><resolution width="64" height="32" /></sensor>
      <sensor id="1" type="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
      <camera id="1" label="frame.jpg" sensor_id="1"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )

    targets = build_metashape_preview_targets(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=None,
        xml_path=xml,
    )

    assert targets.image_paths == (pano, frame)
    assert targets.equirect_paths == (pano,)
    assert metashape_output_count_for_actions(targets.action_counts, enabled_view_count=6) == 7
    assert metashape_output_count_for_actions(targets.action_counts, enabled_view_count=6, direct_output=True) == 2


def test_metashape_preview_targets_default_uses_lightweight_lookup(tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path / "scene"
    pano = scene / "images" / "pano.jpg"
    _write_image(pano, (64, 32))
    xml = tmp_path / "cameras.xml"
    xml.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical"><resolution width="64" height="32" /></sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )

    def fail_inventory(*_args, **_kwargs):
        raise AssertionError("preview target discovery should not build strict scene inventory")

    monkeypatch.setattr("core.metashape_preview_targets.build_scene_inventory", fail_inventory)

    targets = build_metashape_preview_targets(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=None,
        xml_path=xml,
    )

    assert targets.image_paths == (pano,)
    assert targets.equirect_paths == (pano,)
