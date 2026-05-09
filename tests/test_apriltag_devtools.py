from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.apriltag_geometry import PinholeFrame
from core.cubemap_transforms_json import build_remap
from devtools.apriltag.case import (
    AprilTagPlacement,
    create_case,
    load_case,
    load_placement,
    run_dir_for_placement,
    save_placement,
)
from devtools.apriltag.coordinates import (
    COORDINATE_PROFILE_LICHTFELD_CUBE6,
    COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY,
)
from devtools.apriltag.cubemap_preview import (
    CubemapFrameGroup,
    face_view_params,
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    project_sfm_points_to_preview,
    project_sfm_points_to_preview_points,
    render_cubemap_equirect,
    split_cubemap_face,
    view_pixel_to_world_ray,
    view_pixel_to_world_ray_and_up,
)
from devtools.apriltag.printable import create_printable_target


def _write_transforms(path: Path, *, absolute_image: bool = False) -> Path:
    image_dir = path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "frame_0001.jpg"
    image_path.write_bytes(b"fake image bytes")
    file_path = str(image_path) if absolute_image else "images/frame_0001.jpg"
    data = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 100,
        "h": 100,
        "fl_x": 50.0,
        "fl_y": 50.0,
        "cx": 50.0,
        "cy": 50.0,
        "frames": [
            {
                "file_path": file_path,
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_cubemap_transforms(path: Path) -> Path:
    image_dir = path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in ("px", "nx", "pz", "nz", "top", "bottom"):
        image_path = image_dir / f"frame_0001_{name}.jpg"
        image_path.write_bytes(b"fake image bytes")
        frames.append(
            {
                "file_path": f"images/frame_0001_{name}.jpg",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    path.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 100,
                "h": 100,
                "fl_x": 50.0,
                "fl_y": 50.0,
                "cx": 50.0,
                "cy": 50.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return path


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=float,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=float,
    )
    return ry @ rx


def _write_generated_cube6_transforms(path: Path) -> Path:
    image_dir = path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "px": (90.0, 0.0),
        "nx": (-90.0, 0.0),
        "pz": (0.0, 0.0),
        "nz": (180.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }
    frames = []
    for name, view in views.items():
        (image_dir / f"frame_0001_{name}.jpg").write_bytes(b"fake image bytes")
        transform = np.eye(4)
        transform[:3, :3] = _rotation(*view).T
        frames.append({"file_path": f"images/frame_0001_{name}.jpg", "transform_matrix": transform.tolist()})
    path.write_text(
        json.dumps(
            {
                "camera_model": "SIMPLE_PINHOLE",
                "w": 100,
                "h": 100,
                "fl_x": 50.0,
                "fl_y": 50.0,
                "cx": 49.5,
                "cy": 49.5,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_create_case_reference_mode_copies_metadata_only(tmp_path: Path) -> None:
    source = _write_transforms(tmp_path / "source" / "transforms.json")
    ply = tmp_path / "source" / "pointcloud.ply"
    ply.write_text("ply\n", encoding="utf-8")

    case = create_case(
        case_root=tmp_path / "cases",
        case_name="My Case",
        source_transforms=source,
        source_pointcloud=ply,
        copy_images=False,
    )

    assert case.input_mode == "reference"
    assert case.transforms_for_processing() == source.resolve()
    assert (case.input_dir / "transforms.json").is_file()
    assert not (case.input_dir / "images" / "frame_0001.jpg").exists()
    assert (case.input_dir / "pointcloud.ply").is_file()

    loaded = load_case(case.case_dir)
    assert loaded.source_transforms == source.resolve()
    assert loaded.source_pointcloud == ply.resolve()
    assert loaded.coordinate_profile == COORDINATE_PROFILE_LICHTFELD_CUBE6


def test_create_case_saves_coordinate_profile(tmp_path: Path) -> None:
    source = _write_transforms(tmp_path / "source" / "transforms.json")

    case = create_case(
        case_root=tmp_path / "cases",
        case_name="profile",
        source_transforms=source,
        coordinate_profile=COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY,
    )

    loaded = load_case(case.case_dir)
    assert loaded.coordinate_profile == COORDINATE_PROFILE_LICHTFELD_CUBE6_PRE_FINAL_PLY


def test_create_case_copy_mode_rewrites_absolute_image_paths(tmp_path: Path) -> None:
    source = _write_transforms(tmp_path / "source" / "transforms.json", absolute_image=True)

    case = create_case(
        case_root=tmp_path / "cases",
        case_name="copy",
        source_transforms=source,
        copy_images=True,
    )

    copied_transforms = json.loads((case.input_dir / "transforms.json").read_text(encoding="utf-8"))
    assert case.input_mode == "copy"
    assert case.transforms_for_processing() == case.input_dir / "transforms.json"
    assert copied_transforms["frames"][0]["file_path"] == "images/frame_0001.jpg"
    assert (case.input_dir / "images" / "frame_0001.jpg").is_file()


def test_save_and_load_placement(tmp_path: Path) -> None:
    source = _write_transforms(tmp_path / "source" / "transforms.json")
    case = create_case(case_root=tmp_path / "cases", case_name="case", source_transforms=source)
    tag = tmp_path / "tag.png"
    tag.write_bytes(b"tag")
    placement = AprilTagPlacement(
        name="placement A",
        tag_family="tag36h11",
        tag_id=7,
        tag_image=tag,
        tag_size_m=0.16,
        true_scale=0.25,
        tag_center_sfm=(1.0, 2.0, 3.0),
        tag_normal_sfm=(0.0, 0.0, 1.0),
        tag_up_sfm=(0.0, 1.0, 0.0),
    )

    path = save_placement(case, placement)
    loaded = load_placement(path)

    assert path.name == "placement_A.json"
    assert loaded.tag_center_sfm == (1.0, 2.0, 3.0)
    assert run_dir_for_placement(case, loaded) == case.runs_dir / "placement_A"


def test_create_printable_target_outputs_marker_and_a4_files(tmp_path: Path) -> None:
    target = create_printable_target(tmp_path, family="tag36h11", tag_id=7, tag_size_m=0.160, dpi=300)

    assert target.marker_png.is_file()
    assert target.a4_png.is_file()
    assert target.a4_pdf.is_file()
    assert target.spec_json.is_file()
    assert target.page_pixels == (2480, 3508)
    assert target.marker_pixels == (1890, 1890)


def test_printable_crop_marks_stay_outside_tag_square(tmp_path: Path) -> None:
    target = create_printable_target(tmp_path, family="tag36h11", tag_id=7, tag_size_m=0.160, dpi=300)
    image = Image.open(target.a4_png).convert("RGB")
    tag_w, tag_h = target.marker_pixels
    page_w, page_h = target.page_pixels
    x = (page_w - tag_w) // 2
    y = max(260, int(page_h * 0.12))

    # The margin immediately outside the marker must remain white. Crop marks are
    # drawn farther out so they cannot be mistaken for part of the detected tag.
    assert image.getpixel((x + tag_w + 10, y)) == (255, 255, 255)
    assert image.getpixel((x + tag_w + 10, y + tag_h - 1)) == (255, 255, 255)
    assert image.getpixel((x - 10, y)) == (255, 255, 255)
    assert image.getpixel((x - 10, y + tag_h - 1)) == (255, 255, 255)


def test_cubemap_preview_groups_faces_by_frame_prefix(tmp_path: Path) -> None:
    transforms = _write_cubemap_transforms(tmp_path / "transforms.json")

    groups = load_cubemap_frame_groups(transforms)

    assert split_cubemap_face("images/frame_0001_px.jpg") == ("frame_0001", "px")
    assert len(groups) == 1
    assert groups[0].name == "frame_0001"
    assert set(groups[0].frames_by_face) == {"px", "nx", "pz", "nz", "top", "bottom"}


def test_metashape_labels_can_order_cubemap_preview_groups(tmp_path: Path) -> None:
    xml = tmp_path / "metashape.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document><chunk><cameras>
  <camera id="0" label="frame_b.jpg"/>
  <camera id="1" label="frame_a.jpg"/>
</cameras></chunk></document>
""",
        encoding="utf-8",
    )
    groups = (
        type("Group", (), {"name": "frame_a"})(),
        type("Group", (), {"name": "frame_b"})(),
    )

    labels = load_metashape_camera_labels(xml)
    ordered = order_groups_by_labels(groups, labels)

    assert labels == ("frame_b", "frame_a")
    assert [group.name for group in ordered] == ["frame_b", "frame_a"]


def test_face_view_params_uses_transform_relationships(tmp_path: Path) -> None:
    transforms = _write_cubemap_transforms(tmp_path / "transforms.json")
    group = load_cubemap_frame_groups(transforms)[0]

    params = face_view_params(group, "pz")

    assert params is not None
    yaw, pitch, fov = params
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
    assert fov == 90.0


def test_standard_cube6_face_view_params_use_preview_convention(tmp_path: Path) -> None:
    transforms = _write_cubemap_transforms(tmp_path / "transforms.json")
    group = load_cubemap_frame_groups(transforms)[0]

    assert face_view_params(group, "pz") == (0.0, 0.0, 90.0)
    assert face_view_params(group, "px") == (90.0, 0.0, 90.0)
    assert face_view_params(group, "nx") == (-90.0, 0.0, 90.0)
    assert face_view_params(group, "top") == (0.0, -90.0, 90.0)
    assert face_view_params(group, "bottom") == (0.0, 90.0, 90.0)


def test_face_view_params_matches_preview_pitch_sign(tmp_path: Path) -> None:
    identity = np.eye(4)
    top_transform = np.eye(4)
    top_transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    bottom_transform = np.eye(4)
    bottom_transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    def frame(name: str, transform: np.ndarray) -> PinholeFrame:
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/{name}.jpg",
            image_path=tmp_path / f"{name}.jpg",
            width=100,
            height=100,
            fl_x=50.0,
            fl_y=50.0,
            cx=50.0,
            cy=50.0,
            transform_matrix=transform,
        )

    group = CubemapFrameGroup(
        name="frame",
        frames_by_face={
            "pz": frame("pz", identity),
            "top": frame("top", top_transform),
            "bottom": frame("bottom", bottom_transform),
        },
    )

    assert face_view_params(group, "top") == (0.0, -90.0, 90.0)
    assert face_view_params(group, "bottom") == (0.0, 90.0, 90.0)


def test_render_cubemap_equirect_uses_standard_cube6_face_layout(tmp_path: Path) -> None:
    source_w, source_h = 192, 96
    face_size = 64
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
    views = {
        "pz": (0.0, 0.0),
        "px": (90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (-90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }

    def frame(name: str) -> PinholeFrame:
        yaw, pitch = views[name]
        map_x, map_y = build_remap((source_w, source_h), 90.0, yaw, pitch, face_size)
        image = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        image_path = tmp_path / f"frame_{name}.png"
        assert cv2.imwrite(str(image_path), image)
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/frame_{name}.png",
            image_path=image_path,
            width=face_size,
            height=face_size,
            fl_x=face_size / 2.0,
            fl_y=face_size / 2.0,
            cx=(face_size - 1) / 2.0,
            cy=(face_size - 1) / 2.0,
            transform_matrix=np.eye(4),
        )

    group = CubemapFrameGroup(
        name="frame",
        frames_by_face={name: frame(name) for name in views},
    )

    rendered = render_cubemap_equirect(group, output_width=source_w, output_height=source_h)

    assert float(np.mean(np.abs(rendered.astype(np.int16) - source.astype(np.int16)))) < 3.0


def test_standard_cube6_preview_click_ray_uses_matching_face_transform(tmp_path: Path) -> None:
    px_transform = np.eye(4)
    px_transform[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    def frame(name: str, transform: np.ndarray | None = None) -> PinholeFrame:
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/frame_{name}.png",
            image_path=tmp_path / f"frame_{name}.png",
            width=100,
            height=100,
            fl_x=50.0,
            fl_y=50.0,
            cx=49.5,
            cy=49.5,
            transform_matrix=np.eye(4) if transform is None else transform,
        )

    group = CubemapFrameGroup(
        name="frame",
        frames_by_face={
            "pz": frame("pz"),
            "px": frame("px", px_transform),
            "nx": frame("nx"),
            "nz": frame("nz"),
            "top": frame("top"),
            "bottom": frame("bottom"),
        },
    )

    ray = view_pixel_to_world_ray(
        group,
        x_px=49.5,
        y_px=49.5,
        output_size=100,
        yaw_deg=90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert np.allclose(ray, np.array([1.0, 0.0, 0.0]))


def test_project_sfm_points_to_standard_cube6_preview_center(tmp_path: Path) -> None:
    views = {
        "pz": (0.0, 0.0),
        "px": (90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (-90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }

    def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        ry = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ]
        )
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch), -np.sin(pitch)],
                [0.0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        return ry @ rx

    def frame(name: str) -> PinholeFrame:
        transform = np.eye(4)
        transform[:3, :3] = rotation(*views[name])
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/frame_{name}.png",
            image_path=tmp_path / f"frame_{name}.png",
            width=100,
            height=100,
            fl_x=50.0,
            fl_y=50.0,
            cx=49.5,
            cy=49.5,
            transform_matrix=transform,
        )

    group = CubemapFrameGroup(
        name="frame",
        frames_by_face={
            name: frame(name) for name in views
        },
    )

    projected = project_sfm_points_to_preview(
        group,
        np.array([[0.0, 0.0, 10.0]], dtype=float),
        output_size=100,
        yaw_deg=0.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert projected is not None
    assert np.allclose(projected[0], np.array([49.5, 49.5]))

    projected_px = project_sfm_points_to_preview(
        group,
        np.array([[10.0, 0.0, 0.0]], dtype=float),
        output_size=100,
        yaw_deg=90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert projected_px is not None
    assert np.allclose(projected_px[0], np.array([49.5, 49.5]))

    point_projected = project_sfm_points_to_preview_points(
        group,
        np.array([[10.0, 0.0, 0.0]], dtype=float),
        output_size=100,
        yaw_deg=90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert point_projected is not None
    assert np.allclose(point_projected[0], np.array([49.5, 49.5]))


def test_generated_cube6_transforms_are_normalized_for_preview_projection(tmp_path: Path) -> None:
    transforms = _write_generated_cube6_transforms(tmp_path / "transforms.json")
    group = load_cubemap_frame_groups(transforms)[0]

    ray, _up, face = view_pixel_to_world_ray_and_up(
        group,
        x_px=49.5,
        y_px=49.5,
        output_size=100,
        yaw_deg=90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )
    projected = project_sfm_points_to_preview_points(
        group,
        np.array([[10.0, 0.0, 0.0]], dtype=float),
        output_size=100,
        yaw_deg=90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert face == "px"
    assert np.allclose(ray, np.array([1.0, 0.0, 0.0]))
    assert projected is not None
    assert np.allclose(projected[0], np.array([49.5, 49.5]))


def test_standard_cube6_click_ray_and_up_use_clicked_face(tmp_path: Path) -> None:
    px_transform = np.eye(4)
    px_transform[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    def frame(name: str, transform: np.ndarray | None = None) -> PinholeFrame:
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/frame_{name}.png",
            image_path=tmp_path / f"frame_{name}.png",
            width=100,
            height=100,
            fl_x=50.0,
            fl_y=50.0,
            cx=49.5,
            cy=49.5,
            transform_matrix=np.eye(4) if transform is None else transform,
        )

    group = CubemapFrameGroup(
        name="frame",
        frames_by_face={
            "pz": frame("pz"),
            "px": frame("px", px_transform),
            "nx": frame("nx"),
            "nz": frame("nz"),
            "top": frame("top"),
            "bottom": frame("bottom"),
        },
    )

    ray, up, face = view_pixel_to_world_ray_and_up(
        group,
        x_px=49.5,
        y_px=49.5,
        output_size=100,
        yaw_deg=90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert face == "px"
    assert np.allclose(ray, np.array([1.0, 0.0, 0.0]))
    assert abs(float(ray @ up)) < 1e-9
