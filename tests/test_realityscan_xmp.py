import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.cubemap_image_io import remap_with_channels
from core.cubemap_remap import build_remap, rot4, rotation_matrix
from core.orientation_correction import (
    REALITYSCAN_FINAL_ORIENTATION_MATRIX,
    final_orientation_writes_pointcloud,
)
from core.realityscan_xmp import (
    REALITYSCAN_XMP_NAMESPACE,
    append_realityscan_unposed_scene_images,
    c2w_to_xmp_rotation_position,
    cubemap_c2w_to_xmp_rotation_position,
    write_realityscan_mask_layers,
    write_realityscan_xmp_sidecars,
)


def _write_transforms(output: Path) -> None:
    images = output / "images"
    masks = output / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    (images / "frame_0001_pz.jpg").write_bytes(b"image")
    Image.fromarray(np.array([[255, 0], [128, 64]], dtype=np.uint8)).save(masks / "frame_0001_pz.png")

    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [1.0, 2.0, 3.0]
    data = {
        "camera_model": "PINHOLE",
        "w": 100,
        "h": 100,
        "fl_x": 50.0,
        "fl_y": 50.0,
        "cx": 49.5,
        "cy": 49.5,
        "frames": [
            {
                "file_path": "images/frame_0001_pz.jpg",
                "source_file_path": "images/frame_0001.jpg",
                "source_image_index": 0,
                "view_name": "pz",
                "view_index": 4,
                "transform_matrix": transform.tolist(),
            }
        ],
    }
    (output / "transforms.json").write_text(json.dumps(data), encoding="utf-8")


def _xcr_attr(element: ET.Element, name: str) -> str:
    return element.attrib[f"{{{REALITYSCAN_XMP_NAMESPACE}}}{name}"]


def _has_xcr_attr(element: ET.Element, name: str) -> bool:
    return f"{{{REALITYSCAN_XMP_NAMESPACE}}}{name}" in element.attrib


def test_write_realityscan_xmp_sidecars(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)

    manifest = write_realityscan_xmp_sidecars(
        output,
        pose_prior="exact",
        calibration_prior="exact",
        rig_name="test-rig",
    )

    xmp = output / "images" / "frame_0001_pz.xmp"
    assert xmp.is_file()
    assert manifest["xmp_count"] == 1
    assert manifest["focal_length_35mm"] == 18.0

    root = ET.parse(xmp).getroot()
    desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert desc is not None
    assert _xcr_attr(desc, "PosePrior") == "exact"
    assert _xcr_attr(desc, "Coordinates") == "relative"
    assert _xcr_attr(desc, "CalibrationPrior") == "exact"
    assert not _has_xcr_attr(desc, "Rig")
    assert not _has_xcr_attr(desc, "RigInstance")
    assert not _has_xcr_attr(desc, "RigPoseIndex")
    assert not _has_xcr_attr(desc, "ComponentId")
    assert _xcr_attr(desc, "DistortionModel") == "division"
    assert _xcr_attr(desc, "FocalLength35mm") == "18"
    assert _xcr_attr(desc, "PrincipalPointU") == "0"
    assert _xcr_attr(desc, "PrincipalPointV") == "0"
    assert _xcr_attr(desc, "Rotation") == "1 0 0 0 1 0 0 0 1"

    position = desc.find("{http://www.capturingreality.com/ns/xcr/1.1#}Position")
    assert position is not None
    assert position.text == "1 2 3"


def test_write_realityscan_xmp_sidecars_can_include_experimental_rig_metadata(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)

    manifest = write_realityscan_xmp_sidecars(output, include_rig=True, rig_name="test-rig")

    root = ET.parse(output / "images" / "frame_0001_pz.xmp").getroot()
    desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert desc is not None
    assert manifest["rig_metadata"] is True
    assert _xcr_attr(desc, "Rig").startswith("{")
    assert _xcr_attr(desc, "RigInstance").startswith("{")
    assert _xcr_attr(desc, "RigPoseIndex") == "4"
    assert not _has_xcr_attr(desc, "ComponentId")


def test_write_realityscan_xmp_sidecars_defaults_to_exact_relative_pose_prior(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)

    write_realityscan_xmp_sidecars(output)

    root = ET.parse(output / "images" / "frame_0001_pz.xmp").getroot()
    desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert desc is not None
    assert _xcr_attr(desc, "PosePrior") == "exact"
    assert _xcr_attr(desc, "Coordinates") == "relative"
    assert _xcr_attr(desc, "CalibrationPrior") == "exact"


def test_write_realityscan_mask_layers(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)
    write_realityscan_xmp_sidecars(output)

    manifest = write_realityscan_mask_layers(output)

    layer = output / "images" / "frame_0001_pz.jpg.mask.png"
    assert np.array_equal(np.asarray(Image.open(layer)), np.array([[255, 0], [128, 64]], dtype=np.uint8))
    assert manifest["mask_layer_count"] == 1
    assert manifest["mask_layer_files"] == ["images/frame_0001_pz.jpg.mask.png"]
    assert manifest["mask_layer_polarity"] == "white_used_black_excluded"
    assert manifest["source_mask_polarity"] == "white_keep_black_exclude"
    assert manifest["mask_layers_inverted_for_realityscan"] is False


def test_append_realityscan_unposed_scene_images_excludes_metashape_sources(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene_images = scene / "images"
    scene_masks = scene / "masks"
    output = scene / "output" / "realityscan"
    scene_images.mkdir(parents=True)
    scene_masks.mkdir()
    output.mkdir(parents=True)

    Image.new("RGB", (16, 8), color=(10, 20, 30)).save(scene_images / "pano.jpg")
    Image.new("RGB", (12, 10), color=(80, 90, 100)).save(scene_images / "normal.jpg")
    Image.fromarray(np.full((10, 12), 255, dtype=np.uint8)).save(scene_masks / "normal.png")
    _write_transforms(output)
    write_realityscan_xmp_sidecars(output)
    manifest = write_realityscan_mask_layers(output)

    manifest = append_realityscan_unposed_scene_images(
        output,
        scene_dir=scene,
        exclude_source_files=["images/pano.jpg"],
        exclude_root=scene,
        manifest=manifest,
    )

    extra = output / "extra_images" / "extra_normal.jpg"
    assert extra.is_file()
    assert (output / "extra_masks" / "extra_normal.png").is_file()
    assert (output / "extra_images" / "extra_normal.jpg.mask.png").is_file()
    assert not (output / "images" / "extra_pano.jpg").exists()
    assert manifest["unposed_images_dir"] == "extra_images"
    assert manifest["unposed_masks_dir"] == "extra_masks"
    assert manifest["unposed_image_count"] == 1
    assert manifest["unposed_mask_layer_count"] == 1
    assert manifest["unposed_pose"] == "none"
    assert manifest["mask_layer_count"] == 2
    assert "extra_images/extra_normal.jpg.mask.png" in manifest["mask_layer_files"]


def test_c2w_to_xmp_rotation_position_uses_world_to_camera_rotation() -> None:
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = [4.0, 5.0, 6.0]

    rotation, position = c2w_to_xmp_rotation_position(transform)

    assert np.allclose(rotation, transform[:3, :3].T)
    assert np.allclose(position, [4.0, 5.0, 6.0])
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_cubemap_c2w_to_xmp_rotation_position_flips_camera_basis() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    rotation, position = cubemap_c2w_to_xmp_rotation_position(transform)

    assert np.allclose(rotation, np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(position, [1.0, 2.0, 3.0])
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_realityscan_final_orientation_maps_metashape_y_up_to_realityscan_z_up() -> None:
    rotation = REALITYSCAN_FINAL_ORIENTATION_MATRIX[:3, :3]

    assert np.allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0])
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert not final_orientation_writes_pointcloud("realityscan")


def test_realityscan_xmp_camera_axes_match_cubemap_image_rays() -> None:
    axis_transform = (
        REALITYSCAN_FINAL_ORIENTATION_MATRIX
        @ rot4(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ]
            )
        )
        @ rot4(
            np.array(
                [
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]
            )
        )
    )
    image_to_cubemap_camera = np.diag([1.0, 1.0, -1.0])
    basis_checks = [
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])),
    ]
    views = [
        (90.0, 0.0),
        (-90.0, 0.0),
        (0.0, -90.0),
        (0.0, 90.0),
        (0.0, 0.0),
        (180.0, 0.0),
    ]

    for yaw, pitch in views:
        remap_rotation = rotation_matrix(yaw, pitch, False)
        pose_rotation = rotation_matrix(yaw, pitch, True)
        cubemap_transform = axis_transform @ rot4(pose_rotation.T)
        xmp_rotation, _position = cubemap_c2w_to_xmp_rotation_position(cubemap_transform)
        xmp_camera_to_world = xmp_rotation.T

        for image_local_ray, xmp_camera_ray in basis_checks:
            expected_world_ray = axis_transform[:3, :3] @ image_to_cubemap_camera @ remap_rotation @ image_local_ray
            xmp_world_ray = xmp_camera_to_world @ xmp_camera_ray
            assert np.allclose(xmp_world_ray, expected_world_ray, atol=1e-10)


def test_cubemap_faces_roundtrip_to_source_equirect_without_face_name_assumptions() -> None:
    source_w, source_h = 192, 96
    face_size = 96
    xs = (np.arange(source_w, dtype=np.float64) + 0.5) / source_w
    ys = (np.arange(source_h, dtype=np.float64) + 0.5) / source_h
    lon = (xs * 2.0 - 1.0) * np.pi
    lat = (0.5 - ys) * np.pi
    cos_lat = np.cos(lat)[:, None]
    source = np.dstack(
        [
            ((np.sin(lon)[None, :] * cos_lat * 0.5 + 0.5) * 255).astype(np.uint8),
            ((np.sin(lat)[:, None] * 0.5 + 0.5) * 255).repeat(source_w, axis=1).astype(np.uint8),
            ((np.cos(lon)[None, :] * cos_lat * 0.5 + 0.5) * 255).astype(np.uint8),
        ]
    )
    views = [
        {"name": "northward", "yaw": 0.0, "pitch": 0.0},
        {"name": "eastward", "yaw": 90.0, "pitch": 0.0},
        {"name": "backward", "yaw": 180.0, "pitch": 0.0},
        {"name": "westward", "yaw": -90.0, "pitch": 0.0},
        {"name": "ceiling", "yaw": 0.0, "pitch": -90.0},
        {"name": "floor", "yaw": 0.0, "pitch": 90.0},
    ]
    faces: list[tuple[dict, np.ndarray]] = []
    for view in views:
        map_x, map_y = build_remap(
            (source_w, source_h),
            90.0,
            float(view["yaw"]),
            float(view["pitch"]),
            face_size,
        )
        faces.append((view, remap_with_channels(source, map_x, map_y)))

    uu = (np.arange(source_w, dtype=np.float64) + 0.5) / source_w
    vv = (np.arange(source_h, dtype=np.float64) + 0.5) / source_h
    lon_grid = (uu * 2.0 - 1.0) * np.pi
    lat_grid = (0.5 - vv) * np.pi
    cos_grid = np.cos(lat_grid)[:, None]
    rays = np.stack(
        [
            np.sin(lon_grid)[None, :] * cos_grid,
            np.sin(lat_grid)[:, None].repeat(source_w, axis=1),
            np.cos(lon_grid)[None, :] * cos_grid,
        ],
        axis=-1,
    )

    reconstructed = np.full_like(source, 16)
    best_z = np.full((source_h, source_w), -np.inf, dtype=np.float64)
    focal = 0.5 * face_size
    center = (face_size - 1) / 2.0
    for view, face in faces:
        rotation = rotation_matrix(float(view["yaw"]), float(view["pitch"]), False)
        local = rays @ rotation
        z = local[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            map_x = focal * (local[:, :, 0] / z) + center
            map_y = center - focal * (local[:, :, 1] / z)
        valid = (
            (z > 1e-8)
            & (map_x >= -0.5)
            & (map_y >= -0.5)
            & (map_x <= face_size - 0.5)
            & (map_y <= face_size - 0.5)
            & (z > best_z)
        )
        sampled = cv2.remap(
            face,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        reconstructed[valid] = sampled[valid]
        best_z[valid] = z[valid]

    assert np.isfinite(best_z).all()
    assert float(np.mean(np.abs(reconstructed.astype(np.int16) - source.astype(np.int16)))) < 2.5


def test_mask_layer_overwrites_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "realityscan"
    _write_transforms(output)
    layer = output / "images" / "frame_0001_pz.jpg.mask.png"
    layer.write_bytes(b"old")

    write_realityscan_mask_layers(output)

    assert np.array_equal(np.asarray(Image.open(layer)), np.array([[255, 0], [128, 64]], dtype=np.uint8))
