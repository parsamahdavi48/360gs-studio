from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import core.metashape_dataset_assets as asset_mod
from core.dataset_export_plan import EXPORT_ACTION_EXPAND_ERP_TO_VIEWS, DatasetExportPlan, DatasetExportPlanItem
from core.metashape_coordinates import metashape_camera_matrix_to_output_world, metashape_pointcloud_matrix
from core.metashape_model import parse_metashape_model
from core.metashape_nerf_dataset import (
    analyze_metashape_nerf_compatibility,
    axis_transform_matrix,
    export_metashape_nerf_dataset,
    metashape_model_requires_mixed_nerf_writer,
)
from core.orientation_correction import final_orientation_matrix

_IDENTITY = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def _write_ply(path: Path) -> None:
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "1 2 3\n",
        encoding="ascii",
    )


def _write_mixed_xml(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical">
        <resolution width="64" height="32" />
      </sensor>
      <sensor id="1" type="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy></calibration>
      </sensor>
      <sensor id="2" type="frame">
        <resolution width="40" height="30" />
        <calibration><f>35</f><cx>0</cx><cy>0</cy><k1>0.05</k1></calibration>
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


def _write_single_spherical_xml(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical">
        <resolution width="64" height="32" />
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def _write_translated_spherical_xml(path: Path, transform: np.ndarray) -> None:
    text = " ".join(str(float(value)) for value in transform.reshape(-1))
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical">
        <resolution width="64" height="32" />
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano.jpg" sensor_id="0"><transform>{text}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def _write_two_spherical_xml(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor id="0" type="spherical">
        <resolution width="64" height="32" />
      </sensor>
    </sensors>
    <cameras>
      <camera id="0" label="pano_a.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
      <camera id="1" label="pano_b.jpg" sensor_id="0"><transform>{_IDENTITY}</transform></camera>
    </cameras>
  </chunk>
</document>
""",
        encoding="utf-8",
    )


def test_export_metashape_nerf_dataset_expands_links_and_undistorts(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    _write_image(scene / "images" / "frame.jpg", (40, 30), (10, 20, 30))
    _write_image(scene / "images" / "distorted.jpg", (40, 30), (40, 50, 60))
    _write_mask(scene / "masks" / "pano.png", (64, 32))
    _write_mask(scene / "masks" / "frame.png", (40, 30))
    xml = scene / "metashape.xml"
    ply = scene / "metashape.ply"
    _write_mixed_xml(xml)
    _write_ply(ply)

    output = scene / "output" / "metashape_cubemap"
    progress: list[tuple[int, int]] = []
    result = export_metashape_nerf_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=scene / "masks",
        xml_path=xml,
        ply_path=ply,
        output_dir=output,
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}, {"name": "px", "yaw": 90.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
        axis_transform="none",
        final_orientation="none",
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    data = json.loads(result.transforms_json.read_text(encoding="utf-8"))
    frames = data["frames"]
    assert progress == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert result.frame_count == 4
    assert len(frames) == 4
    assert data["source"]["per_frame_intrinsics"] is True
    assert data["source"]["per_frame_camera_model"] is True
    assert result.action_counts == {
        "expand_erp_to_views": 1,
        "link_pinhole": 1,
        "undistort_frame_to_pinhole": 1,
    }
    assert (output / "images" / "pano_pz.jpg").is_file()
    assert (output / "images" / "pano_px.jpg").is_file()
    assert (output / "images" / "frame.jpg").is_file()
    assert (output / "images" / "distorted_undistorted.jpg").is_file()
    assert (output / "masks" / "pano_pz.png").is_file()
    assert (output / "masks" / "frame.png").is_file()
    assert (output / "masks" / "distorted_undistorted.png").is_file()
    assert result.transforms_json == output / "transforms_custom.json"
    assert (output / "pointcloud_custom.ply").is_file()
    assert all(frame["camera_model"] == "PINHOLE" for frame in frames)
    assert {frame["file_path"] for frame in frames} == {
        "images/pano_pz.jpg",
        "images/pano_px.jpg",
        "images/frame.jpg",
        "images/distorted_undistorted.jpg",
    }


def test_export_metashape_nerf_dataset_reuses_remap_tables_for_same_size_erp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano_a.jpg", (64, 32), (80, 120, 160))
    _write_image(scene / "images" / "pano_b.jpg", (64, 32), (90, 130, 170))
    xml = scene / "metashape.xml"
    _write_two_spherical_xml(xml)
    original_build_remap = asset_mod.build_remap
    calls: list[tuple[tuple[int, int], float, float, int]] = []

    def count_build_remap(
        input_size: tuple[int, int],
        fov_deg: float,
        yaw_deg: float,
        pitch_deg: float,
        output_size: int,
    ):
        calls.append((input_size, yaw_deg, pitch_deg, output_size))
        return original_build_remap(input_size, fov_deg, yaw_deg, pitch_deg, output_size)

    monkeypatch.setattr(asset_mod, "build_remap", count_build_remap)

    result = export_metashape_nerf_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=None,
        xml_path=xml,
        ply_path=None,
        output_dir=scene / "output" / "metashape_cubemap",
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}, {"name": "px", "yaw": 90.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
        axis_transform="none",
        final_orientation="none",
    )

    assert result.frame_count == 4
    assert len(calls) == 2


def test_metashape_asset_image_size_uses_metadata_without_full_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    _write_image(image, (64, 32), (80, 120, 160))

    def fail_load_equirect(_path: str):
        raise AssertionError("full image decode should not be needed for size metadata")

    monkeypatch.setattr(asset_mod, "load_equirect", fail_load_equirect)

    assert asset_mod.image_size(image) == (64, 32)


@pytest.mark.parametrize("axis_transform", ["postshot", "brush"])
def test_export_metashape_nerf_postshot_brush_apply_import_axis_to_cameras_only(
    tmp_path: Path,
    axis_transform: str,
) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    raw_transform = np.eye(4, dtype=np.float64)
    raw_transform[:3, 3] = [1.25, -2.5, 3.75]
    xml = scene / "metashape.xml"
    ply = scene / "metashape.ply"
    output = scene / "output" / "metashape_cubemap"
    _write_translated_spherical_xml(xml, raw_transform)
    _write_ply(ply)
    output.mkdir(parents=True)
    _write_ply(output / "pointcloud.ply")

    result = export_metashape_nerf_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=None,
        xml_path=xml,
        ply_path=ply,
        output_dir=output,
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
        axis_transform=axis_transform,
        final_orientation="none",
    )

    transforms_json = output / f"transforms_{axis_transform}.json"
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    frame_transform = np.array(data["frames"][0]["transform_matrix"], dtype=np.float64)
    expected_center = (axis_transform_matrix(axis_transform) @ metashape_pointcloud_matrix() @ raw_transform)[:3, 3]
    lichtfeld_precompensated_center = (
        axis_transform_matrix(axis_transform)
        @ np.diag([-1.0, 1.0, -1.0, 1.0])
        @ metashape_pointcloud_matrix()
        @ raw_transform
    )[:3, 3]

    assert np.allclose(frame_transform[:3, 3], expected_center)
    assert not np.allclose(frame_transform[:3, 3], lichtfeld_precompensated_center)
    raw_output_ply = output / f"pointcloud_{axis_transform}.ply"
    assert data["source"]["pointcloud_policy"] == "raw_metashape_ply"
    assert data["source"]["pointcloud_world_transform"] is None
    assert data["source"]["raw_metashape_pointcloud_path"] == raw_output_ply.name
    assert "ply_file_path" not in data
    assert raw_output_ply.read_bytes() == ply.read_bytes()
    assert result.pointcloud == raw_output_ply
    assert not (output / "pointcloud.ply").exists()
    assert not (output / "metashape.ply").exists()


def test_export_metashape_nerf_can_reuse_existing_images_and_masks_for_pose_only_update(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    _write_mask(scene / "masks" / "pano.png", (64, 32))
    xml = scene / "metashape.xml"
    ply = scene / "metashape.ply"
    _write_single_spherical_xml(xml)
    _write_ply(ply)
    output = scene / "output" / "metashape_cubemap"

    export_metashape_nerf_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=scene / "masks",
        xml_path=xml,
        ply_path=ply,
        output_dir=output,
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
        axis_transform="postshot",
        final_orientation="none",
    )
    output_image = output / "images" / "pano_pz.jpg"
    output_mask = output / "masks" / "pano_pz.png"
    image_bytes = output_image.read_bytes()
    mask_bytes = output_mask.read_bytes()

    export_metashape_nerf_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=scene / "masks",
        xml_path=xml,
        ply_path=ply,
        output_dir=output,
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
        axis_transform="brush",
        final_orientation="none",
        write_images=False,
        write_masks=False,
    )

    assert output_image.read_bytes() == image_bytes
    assert output_mask.read_bytes() == mask_bytes
    data = json.loads((output / "transforms_brush.json").read_text(encoding="utf-8"))
    assert data["source"]["axis_transform"] == "brush"
    assert data["frames"][0]["file_path"] == "images/pano_pz.jpg"
    assert data["frames"][0]["mask_path"] == "masks/pano_pz.png"


def test_export_metashape_nerf_reuse_mode_requires_existing_images(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    xml = scene / "metashape.xml"
    _write_single_spherical_xml(xml)

    with pytest.raises(ValueError, match="Existing output image"):
        export_metashape_nerf_dataset(
            scene_dir=scene,
            images_dir=scene / "images",
            masks_dir=None,
            xml_path=xml,
            ply_path=None,
            output_dir=scene / "output" / "metashape_cubemap",
            views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
            output_scale=0.5,
            output_format="jpg",
            axis_transform="postshot",
            final_orientation="none",
            write_images=False,
            write_masks=False,
        )


def test_export_metashape_nerf_lichtfeld_keeps_camera_y180_precompensation(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    raw_transform = np.eye(4, dtype=np.float64)
    raw_transform[:3, 3] = [1.25, -2.5, 3.75]
    xml = scene / "metashape.xml"
    _write_translated_spherical_xml(xml, raw_transform)

    export_metashape_nerf_dataset(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=None,
        xml_path=xml,
        ply_path=None,
        output_dir=scene / "output" / "metashape_cubemap",
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
        output_scale=0.5,
        output_format="jpg",
        axis_transform="none",
        final_orientation="lichtfeld",
    )

    data = json.loads(
        (scene / "output" / "metashape_cubemap" / "transforms_lichtfeld.json").read_text(encoding="utf-8")
    )
    frame_transform = np.array(data["frames"][0]["transform_matrix"], dtype=np.float64)
    expected_center = (
        final_orientation_matrix("lichtfeld")
        @ np.diag([-1.0, 1.0, -1.0, 1.0])
        @ metashape_pointcloud_matrix()
        @ raw_transform
    )[:3, 3]

    assert np.allclose(frame_transform[:3, 3], expected_center)


def test_metashape_nerf_lichtfeld_compatibility_uses_xml_sensor_sizes(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    xml = scene / "metashape.xml"
    _write_two_spherical_xml(xml)
    model = parse_metashape_model(xml)
    camera_by_id = {camera.camera_id: camera for camera in model.cameras}
    plan = DatasetExportPlan(
        source_kind="metashape_xml_ply",
        items=(
            DatasetExportPlanItem(
                action=EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
                camera_id="0",
                camera_label="pano_a.jpg",
                sensor_id="0",
                camera_model="EQUIRECTANGULAR",
                image_rel_path="images/missing_a.jpg",
                mask_rel_path="",
            ),
            DatasetExportPlanItem(
                action=EXPORT_ACTION_EXPAND_ERP_TO_VIEWS,
                camera_id="1",
                camera_label="pano_b.jpg",
                sensor_id="0",
                camera_model="EQUIRECTANGULAR",
                image_rel_path="images/missing_b.jpg",
                mask_rel_path="",
            ),
        ),
    )

    compatibility = analyze_metashape_nerf_compatibility(
        scene_dir=scene,
        images_dir=scene / "images",
        masks_dir=None,
        xml_path=xml,
        views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}, {"name": "px", "yaw": 90.0, "pitch": 0.0}],
        output_scale=0.5,
        _model=model,
        _plan=plan,
        _camera_by_id=camera_by_id,
    )

    assert compatibility.lichtfeld_nerf_supported is True
    assert compatibility.frame_count == 4
    assert compatibility.camera_group_count == 1


def test_metashape_nerf_camera_transform_matches_coordinate_contract() -> None:
    transform = np.array(
        [
            [0.36, -0.48, 0.80, 1.25],
            [0.80, 0.60, 0.00, -2.5],
            [-0.48, 0.64, 0.60, 3.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected = np.array(
        [
            [-0.48, -0.64, -0.60, 3.75],
            [-0.80, 0.60, 0.00, 2.50],
            [0.36, 0.48, -0.80, 1.25],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(metashape_camera_matrix_to_output_world(transform), expected)


def test_export_metashape_nerf_dataset_blocks_multicamera_lichtfeld_target(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32), (80, 120, 160))
    _write_image(scene / "images" / "frame.jpg", (40, 30), (10, 20, 30))
    xml = scene / "metashape.xml"
    _write_mixed_xml(xml)

    with pytest.raises(ValueError, match="LichtFeld"):
        export_metashape_nerf_dataset(
            scene_dir=scene,
            images_dir=scene / "images",
            masks_dir=None,
            xml_path=xml,
            ply_path=None,
            output_dir=scene / "output" / "metashape_cubemap",
            views=[{"name": "pz", "yaw": 0.0, "pitch": 0.0}],
            output_scale=0.5,
            output_format="jpg",
            axis_transform="none",
            final_orientation="lichtfeld",
        )

    assert not (scene / "output" / "metashape_cubemap" / "transforms_lichtfeld.json").exists()


def test_metashape_nerf_writer_detection_flags_models_requiring_projected_output(tmp_path: Path) -> None:
    simple = tmp_path / "simple.xml"
    mixed = tmp_path / "mixed.xml"
    _write_single_spherical_xml(simple)
    _write_mixed_xml(mixed)

    assert not metashape_model_requires_mixed_nerf_writer(simple)
    assert metashape_model_requires_mixed_nerf_writer(mixed)
