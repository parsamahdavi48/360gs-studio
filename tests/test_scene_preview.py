from __future__ import annotations

import json
from inspect import signature
from pathlib import Path

import numpy as np

from core.pointcloud_io import load_point_cloud_sample
from core.scene_preview import (
    SCENE_PREVIEW_WORLD_UP,
    ScenePreviewPointCloud,
    load_colmap_preview_dataset,
    load_metashape_preview_dataset,
    load_ply_preview_pointcloud,
    load_realityscan_preview_dataset,
    load_transforms_preview_dataset,
    transform_preview_dataset,
)
from core.scene_preview_profiles import (
    COORDINATE_PROFILE_LICHTFELD_CUBE6,
    COORDINATE_PROFILE_POSTSHOT_CUBE6,
    COORDINATE_PROFILE_REALITYSCAN,
    LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT,
    REALITYSCAN_LFS_FILE_TO_PREVIEW_Y_UP,
    REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP,
    step4_output_display_transform,
    transforms_dataset_display_transform,
)
from core.scene_preview_sources import discover_scene_preview_candidates
from core.workflow_artifacts import (
    DATASET_KIND_COLMAP_DATASET,
    SFM_KIND_REALITYSCAN_CSV_PLY,
    register_dataset_artifact,
    register_sfm_artifact,
)


def test_preview_pointcloud_defaults_load_all_points() -> None:
    assert signature(load_point_cloud_sample).parameters["max_points"].default is None
    assert signature(load_ply_preview_pointcloud).parameters["max_points"].default is None


def test_transforms_preview_projection_maps_camera_plus_y_to_screen_up(tmp_path: Path) -> None:
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 100,
                "h": 100,
                "fl_x": 50.0,
                "fl_y": 50.0,
                "cx": 49.5,
                "cy": 49.5,
                "frames": [
                    {
                        "file_path": "images/frame_0001.png",
                        "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_transforms_preview_dataset(transforms)
    camera = dataset.cameras[0]

    projected = camera.project_world_points(np.array([[0.0, 1.0, 4.0]], dtype=np.float64))

    assert projected is not None
    assert projected[0, 0] == 49.5
    assert projected[0, 1] < 49.5
    assert camera.world_up_roll_radians(SCENE_PREVIEW_WORLD_UP) == 0.0


def test_transforms_preview_loads_equirectangular_without_pinhole_intrinsics(tmp_path: Path) -> None:
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "EQUIRECTANGULAR",
                "w": 4000,
                "h": 2000,
                "frames": [
                    {
                        "file_path": "images/frame_0001.jpg",
                        "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_transforms_preview_dataset(transforms)
    camera = dataset.cameras[0]

    assert camera.projection == "equirectangular"
    assert camera.width == 4000
    assert camera.fl_x is None
    assert camera.project_world_points(np.array([[0.0, 0.0, 4.0]], dtype=np.float64)) is None


def test_metashape_preview_applies_component_transform_and_resolves_image(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"not an image header for this parser")
    xml = tmp_path / "cameras.xml"
    xml.write_text(
        """
        <document>
          <chunk>
            <sensors>
              <sensor id="0" type="frame">
                <calibration>
                  <resolution width="100" height="80"/>
                  <f>45</f>
                </calibration>
              </sensor>
            </sensors>
            <components>
              <component id="1">
                <transform>
                  <rotation>1 0 0 0 1 0 0 0 1</rotation>
                  <translation>10 0 0</translation>
                  <scale>2</scale>
                </transform>
              </component>
            </components>
            <cameras>
              <camera id="7" sensor_id="0" component_id="1" label="frame_0001.jpg">
                <transform>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</transform>
              </camera>
            </cameras>
          </chunk>
        </document>
        """,
        encoding="utf-8",
    )

    dataset = load_metashape_preview_dataset(xml, images_dir=images)
    camera = dataset.cameras[0]

    assert camera.camera_id == "7"
    assert camera.image_path == images / "frame_0001.jpg"
    assert camera.width == 100
    assert camera.fl_x == 45.0
    assert np.allclose(camera.position, np.array([12.0, 4.0, 6.0]))
    assert np.allclose(camera.up, np.array([0.0, -1.0, 0.0]))

    projected = camera.project_world_points(np.array([[12.0, 3.0, 10.0]], dtype=np.float64))

    assert projected is not None
    assert projected[0, 1] < 40.0


def test_colmap_preview_uses_image_up_for_screen_up_and_loads_points(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "cameras.txt").write_text(
        "\n".join(
            [
                "# cameras",
                "1 PINHOLE 100 100 50 50 49.5 49.5",
            ]
        ),
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "\n".join(
            [
                "# images",
                "1 1 0 0 0 0 0 0 1 frame_0001.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text(
        "\n".join(
            [
                "# points",
                "1 0 0 3 255 128 64 0",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_colmap_preview_dataset(sparse)
    camera = dataset.cameras[0]

    projected = camera.project_world_points(np.array([[0.0, -1.0, 4.0]], dtype=np.float64))

    assert projected is not None
    assert projected[0, 1] < 49.5
    assert dataset.pointcloud is not None
    assert dataset.pointcloud.points.shape == (1, 3)
    assert dataset.pointcloud.colors is not None
    assert dataset.pointcloud.colors[0].tolist() == [255, 128, 64]


def test_colmap_preview_can_read_app_opengl_camera_axes(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 100 100 50 50 49.5 49.5\n", encoding="utf-8")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_0001.jpg\n\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text("# Number of points: 0\n", encoding="utf-8")

    dataset = load_colmap_preview_dataset(sparse, opengl_camera=True)
    camera = dataset.cameras[0]

    projected = camera.project_world_points(np.array([[0.0, -1.0, -4.0]], dtype=np.float64))

    assert projected is not None
    assert projected[0, 1] < 49.5


def test_colmap_preview_uses_points3d_ply_when_points_txt_is_empty(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 100 100 50 50 49.5 49.5\n", encoding="utf-8")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_0001.jpg\n\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text("# Number of points: 0\n", encoding="utf-8")
    (sparse / "points3D.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "1 2 3",
            ]
        ),
        encoding="ascii",
    )

    dataset = load_colmap_preview_dataset(sparse)

    assert dataset.pointcloud is not None
    assert dataset.pointcloud.source_path == sparse / "points3D.ply"
    assert dataset.pointcloud.points.tolist() == [[1.0, 2.0, 3.0]]


def test_realityscan_preview_loads_csv_and_ply(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_png(images / "a.jpg", size=(100, 80))
    csv = tmp_path / "rs.csv"
    csv.write_text(
        "#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"
        "a.jpg,1,2,3,0,0,0,18,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    ply = tmp_path / "rs.ply"
    _write_empty_ply(ply)

    dataset = load_realityscan_preview_dataset(
        csv,
        images_dir=images,
        pointcloud=load_ply_preview_pointcloud(ply),
    )

    assert dataset.source_kind == "realityscan"
    assert len(dataset.cameras) == 1
    assert dataset.cameras[0].image_path == images / "a.jpg"
    assert np.allclose(dataset.cameras[0].position, [1.0, 2.0, 3.0])
    assert dataset.pointcloud is not None


def test_ply_preview_pointcloud_loads_ascii_colors(tmp_path: Path) -> None:
    ply = tmp_path / "pointcloud.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 1 2 10 20 30",
                "3 4 5 40 50 60",
            ]
        ),
        encoding="ascii",
    )

    pointcloud = load_ply_preview_pointcloud(ply, max_points=None)

    assert pointcloud.source_path == ply
    assert pointcloud.points.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert pointcloud.colors is not None
    assert pointcloud.colors.tolist() == [[10, 20, 30], [40, 50, 60]]


def test_transform_preview_dataset_applies_camera_and_pointcloud_matrices(tmp_path: Path) -> None:
    transforms = tmp_path / "transforms.json"
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    transforms.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 100,
                "h": 100,
                "frames": [{"file_path": "frame.png", "transform_matrix": matrix.tolist()}],
            }
        ),
        encoding="utf-8",
    )
    dataset = load_transforms_preview_dataset(
        transforms,
        pointcloud=ScenePreviewPointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            colors=None,
        ),
    )
    display_matrix = np.diag([-1.0, 1.0, -1.0, 1.0])

    transformed = transform_preview_dataset(
        dataset,
        camera_matrix=display_matrix,
        pointcloud_matrix=display_matrix,
        coordinate_note="test",
    )

    assert transformed.coordinate_note == "test"
    assert np.allclose(transformed.cameras[0].position, [-1.0, 2.0, -3.0])
    assert np.allclose(transformed.cameras[0].right, [-1.0, 0.0, 0.0])
    assert transformed.pointcloud is not None
    assert np.allclose(transformed.pointcloud.points[0], [-1.0, 2.0, -3.0])


def test_step4_output_display_transform_keeps_lichtfeld_ply_raw_and_aligns_cameras() -> None:
    transform = step4_output_display_transform(
        {
            "effective_profile": "lichtfeld",
            "axis_transform": "none",
            "postprocess": {"final_orientation": "lichtfeld"},
        }
    )

    assert transform is not None
    assert transform.profile == COORDINATE_PROFILE_LICHTFELD_CUBE6
    assert np.allclose(transform.camera_matrix, LICHTFELD_CAMERA_POINTCLOUD_ALIGNMENT)
    assert transform.pointcloud_matrix is None


def test_step4_output_display_transform_leaves_postshot_output_raw() -> None:
    transform = step4_output_display_transform(
        {
            "effective_profile": "postshot",
            "axis_transform": "postshot",
            "postprocess": {"final_orientation": "none"},
        }
    )

    assert transform is not None
    assert transform.profile == COORDINATE_PROFILE_POSTSHOT_CUBE6
    assert transform.camera_matrix is None
    assert transform.pointcloud_matrix is None


def test_transforms_dataset_display_transform_unwinds_realityscan_final_orientation(tmp_path: Path) -> None:
    transforms = tmp_path / "transforms.json"
    transforms.write_text(
        json.dumps(
            {
                "postprocess": {
                    "final_orientation": "realityscan",
                    "final_orientation_matrix": np.linalg.inv(REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP).tolist(),
                },
                "frames": [],
            }
        ),
        encoding="utf-8",
    )

    transform = transforms_dataset_display_transform(transforms)

    assert transform is not None
    assert transform.profile == COORDINATE_PROFILE_REALITYSCAN
    assert np.allclose(transform.camera_matrix, REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP)
    assert transform.pointcloud_matrix is None


def test_discover_scene_preview_candidates_finds_step4_sources(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "masks").mkdir()
    (output / "transforms.json").write_text("{}", encoding="utf-8")
    (output / "pointcloud.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
    settings = tmp_path / "_stechdrive" / "step4" / "export_settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "effective_profile": "lichtfeld",
                "axis_transform": "none",
                "postprocess": {"final_orientation": "none"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "cameras.xml").write_text("<document/>", encoding="utf-8")
    colmap = output / "colmap_rig" / "sparse" / "0"
    colmap.mkdir(parents=True)
    (output / "colmap_rig" / "masks").mkdir()
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (colmap / name).write_text("", encoding="utf-8")
    spheresfm = output / "spheresfm" / "sparse"
    spheresfm.mkdir(parents=True)
    (output / "spheresfm" / "masks_colmap").mkdir()
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (spheresfm / name).write_text("", encoding="utf-8")

    candidates = discover_scene_preview_candidates(tmp_path)

    by_kind = {candidate.kind: candidate for candidate in candidates}
    assert by_kind["output"].path == output / "transforms.json"
    assert by_kind["output"].mask_root == output / "masks"
    assert by_kind["output"].pointcloud_path == output / "pointcloud.ply"
    assert by_kind["output"].display_transform is not None
    assert by_kind["output"].display_transform.profile == COORDINATE_PROFILE_LICHTFELD_CUBE6
    assert by_kind["metashape"].path == tmp_path / "cameras.xml"
    assert by_kind["metashape"].mask_root == tmp_path / "masks"
    assert by_kind["colmap"].path == colmap
    assert by_kind["colmap"].mask_root == output / "colmap_rig" / "masks"
    assert by_kind["spheresfm"].path == spheresfm
    assert by_kind["spheresfm"].mask_root == output / "spheresfm" / "masks_colmap"


def test_discover_scene_preview_candidates_finds_registered_refactor_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    metashape_colmap = output / "metashape_colmap"
    sparse = metashape_colmap / "sparse" / "0"
    sparse.mkdir(parents=True)
    (metashape_colmap / "images").mkdir(parents=True)
    (metashape_colmap / "masks").mkdir()
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse / name).write_text("", encoding="utf-8")
    (sparse / "points3D.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
    register_dataset_artifact(
        tmp_path,
        artifact_id="metashape_colmap",
        root=metashape_colmap,
        kind=DATASET_KIND_COLMAP_DATASET,
    )

    rs_root = tmp_path / "realityscan"
    images = rs_root / "images"
    masks = rs_root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    csv = rs_root / "rs.csv"
    csv.write_text(
        "#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"
        "a.jpg,1,2,3,0,0,0,18,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    ply = rs_root / "rs.ply"
    _write_empty_ply(ply)
    register_sfm_artifact(
        tmp_path,
        artifact_id="rs_sfm",
        kind=SFM_KIND_REALITYSCAN_CSV_PLY,
        root=rs_root,
        files={"csv": csv, "ply": ply, "images_dir": images, "masks_dir": masks},
    )

    candidates = discover_scene_preview_candidates(tmp_path)

    colmap_candidate = next(candidate for candidate in candidates if candidate.label == "Dataset: COLMAP (metashape_colmap)")
    assert colmap_candidate.kind == "colmap"
    assert colmap_candidate.path == sparse
    assert colmap_candidate.image_root == metashape_colmap / "images"
    assert colmap_candidate.mask_root == metashape_colmap / "masks"
    assert colmap_candidate.pointcloud_path == sparse / "points3D.ply"
    assert colmap_candidate.colmap_opengl_camera is False
    realityscan = next(candidate for candidate in candidates if candidate.kind == "realityscan")
    assert realityscan.path == csv
    assert realityscan.pointcloud_path == ply
    assert realityscan.display_transform is not None
    assert np.allclose(realityscan.display_transform.camera_matrix, REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP)


def test_discover_scene_preview_candidates_marks_realityscan_colmap_profiles(tmp_path: Path) -> None:
    output = tmp_path / "output"
    direct_sparse = output / "realityscan" / "sparse" / "0"
    lfs_sparse = output / "realityscan" / "lfs_colmap" / "sparse" / "0"
    for sparse in (direct_sparse, lfs_sparse):
        sparse.mkdir(parents=True)
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            (sparse / name).write_text("", encoding="utf-8")
        (sparse / "points3D.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
    (output / "realityscan" / "lfs_colmap" / "stechdrive_dataset_manifest.json").write_text(
        json.dumps({"kind": "lichtfeld_colmap", "source_kind": "realityscan_csv_ply"}),
        encoding="utf-8",
    )

    candidates = discover_scene_preview_candidates(tmp_path)

    direct = next(candidate for candidate in candidates if candidate.label == "RealityScan COLMAP Export")
    lfs = next(candidate for candidate in candidates if candidate.label == "RealityScan LichtFeld COLMAP")
    assert direct.display_transform is not None
    assert np.allclose(direct.display_transform.camera_matrix, REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP)
    assert np.allclose(direct.display_transform.pointcloud_matrix, REALITYSCAN_LFS_FILE_TO_PREVIEW_Y_UP)
    assert direct.colmap_opengl_camera is True
    assert lfs.display_transform is not None
    assert np.allclose(lfs.display_transform.camera_matrix, REALITYSCAN_LFS_FILE_TO_PREVIEW_Y_UP)
    assert np.allclose(lfs.display_transform.pointcloud_matrix, REALITYSCAN_LFS_FILE_TO_PREVIEW_Y_UP)
    assert lfs.colmap_opengl_camera is True


def test_realityscan_colmap_export_uses_raw_point_transform_without_points_ply(tmp_path: Path) -> None:
    sparse = tmp_path / "output" / "realityscan" / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse / name).write_text("", encoding="utf-8")

    candidates = discover_scene_preview_candidates(tmp_path)

    direct = next(candidate for candidate in candidates if candidate.label == "RealityScan COLMAP Export")
    assert direct.display_transform is not None
    assert np.allclose(direct.display_transform.camera_matrix, REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP)
    assert np.allclose(direct.display_transform.pointcloud_matrix, REALITYSCAN_Z_UP_TO_PREVIEW_Y_UP)


def test_discover_scene_preview_candidates_applies_lfs_profile_per_transforms_dataset(tmp_path: Path) -> None:
    output = tmp_path / "output" / "custom_lfs"
    output.mkdir(parents=True)
    (output / "transforms.json").write_text(
        json.dumps({"postprocess": {"final_orientation": "lichtfeld"}, "frames": []}),
        encoding="utf-8",
    )

    candidates = discover_scene_preview_candidates(tmp_path)

    candidate = next(item for item in candidates if item.path == output / "transforms.json")
    assert candidate.display_transform is not None
    assert candidate.display_transform.profile == COORDINATE_PROFILE_LICHTFELD_CUBE6


def _write_png(path: Path, *, size: tuple[int, int]) -> None:
    from PIL import Image

    Image.new("RGB", size, color=(80, 120, 160)).save(path)


def _write_empty_ply(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 0",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
            ]
        ),
        encoding="ascii",
    )
