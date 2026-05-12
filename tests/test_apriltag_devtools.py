from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from core.apriltag_geometry import PinholeFrame, load_pinhole_frames
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
    axis_face_view_params,
    cubemap_preview_sampler_faces,
    face_view_params,
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    preview_frustum_rays,
    project_sfm_points_to_preview,
    project_sfm_points_to_preview_points,
    render_cubemap_axis_equirect,
    render_cubemap_direct_preview,
    render_cubemap_equirect,
    render_cubemap_world_perspective,
    split_cubemap_face,
    view_pixel_to_world_ray,
    view_pixel_to_world_ray_and_up,
    virtual_camera_rotation,
    visible_cubemap_preview_face_indices,
)
from devtools.apriltag.printable import create_printable_target
from gui.common.perspective_preview import PerspectiveParams, equirect_to_perspective


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


def _rotation_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
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
    rz = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return ry @ rx @ rz


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


def _export_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
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
    return rx @ ry


def _write_gui_cube6_generated_transforms(path: Path) -> Path:
    image_dir = path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    views = (
        ("py", -90.0, -90.0),
        ("px", 0.0, 0.0),
        ("nz", 90.0, 0.0),
        ("nx", 180.0, 0.0),
        ("pz", -90.0, 0.0),
        ("ny", -90.0, 90.0),
    )
    frames = []
    for name, yaw, pitch in views:
        (image_dir / f"frame_0001_{name}.jpg").write_bytes(b"fake image bytes")
        transform = np.eye(4)
        transform[:3, :3] = _export_rotation(yaw, pitch).T
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
    assert face_view_params(group, "top") == (0.0, 90.0, 90.0)
    assert face_view_params(group, "bottom") == (0.0, -90.0, 90.0)


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


def test_render_cubemap_axis_equirect_preserves_world_aligned_face_roll(tmp_path: Path) -> None:
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
    faces = {
        "pz": (0.0, 0.0),
        "px": (90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (-90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }
    base = _rotation_ypr(37.0, -6.0, 11.0)

    def render_face(rotation: np.ndarray) -> np.ndarray:
        coords = (np.arange(face_size, dtype=np.float64) + 0.5) / face_size
        u = coords * 2.0 - 1.0
        v = 1.0 - coords * 2.0
        uu, vv = np.meshgrid(u, v)
        rays = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        rays = rays @ rotation.T
        map_x = ((np.arctan2(rays[..., 0], rays[..., 2]) / np.pi + 1.0) * 0.5 * source_w).astype(np.float32)
        map_y = ((0.5 - np.arcsin(np.clip(rays[..., 1], -1.0, 1.0)) / np.pi) * source_h).astype(np.float32)
        return cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def frame(name: str, rotation: np.ndarray) -> PinholeFrame:
        image = render_face(rotation)
        image_path = tmp_path / f"axis_{name}.png"
        assert cv2.imwrite(str(image_path), image)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/axis_{name}.png",
            image_path=image_path,
            width=face_size,
            height=face_size,
            fl_x=face_size / 2.0,
            fl_y=face_size / 2.0,
            cx=(face_size - 1) / 2.0,
            cy=(face_size - 1) / 2.0,
            transform_matrix=transform,
        )

    frames = {
        name: frame(name, base @ _rotation(yaw, pitch))
        for name, (yaw, pitch) in faces.items()
    }
    group = CubemapFrameGroup(name="axis", frames_by_face=frames)

    rendered = render_cubemap_axis_equirect(group, output_width=source_w, output_height=source_h)

    assert float(np.mean(np.abs(rendered.astype(np.int16) - source.astype(np.int16)))) < 4.0
    for face_name, source_frame in frames.items():
        yaw, pitch, roll, fov = axis_face_view_params(group, face_name)
        reprojected = equirect_to_perspective(
            rendered,
            PerspectiveParams(yaw_deg=yaw, pitch_deg=pitch, fov_deg=fov, roll_deg=roll),
            output_size=face_size,
        )
        source_image = cv2.imread(str(source_frame.image_path), cv2.IMREAD_COLOR)
        assert float(np.mean(np.abs(reprojected.astype(np.int16) - source_image.astype(np.int16)))) < 10.0


def test_render_cubemap_equirect_uses_transform_relative_face_layout(tmp_path: Path) -> None:
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
        "px": (-90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }

    def frame(name: str) -> PinholeFrame:
        yaw, pitch = views[name]
        map_x, map_y = build_remap((source_w, source_h), 90.0, yaw, pitch, face_size)
        image = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        image_path = tmp_path / f"custom_{name}.png"
        assert cv2.imwrite(str(image_path), image)
        transform = np.eye(4)
        transform[:3, :3] = _rotation(yaw, pitch)
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/custom_{name}.png",
            image_path=image_path,
            width=face_size,
            height=face_size,
            fl_x=face_size / 2.0,
            fl_y=face_size / 2.0,
            cx=(face_size - 1) / 2.0,
            cy=(face_size - 1) / 2.0,
            transform_matrix=transform,
        )

    group = CubemapFrameGroup(name="custom", frames_by_face={name: frame(name) for name in views})

    rendered = render_cubemap_equirect(group, output_width=source_w, output_height=source_h)
    px_yaw, px_pitch, _fov = face_view_params(group, "px")
    nx_yaw, nx_pitch, _fov = face_view_params(group, "nx")

    assert float(np.mean(np.abs(rendered.astype(np.int16) - source.astype(np.int16)))) < 3.0
    assert px_yaw == pytest.approx(-90.0)
    assert px_pitch == pytest.approx(0.0)
    assert nx_yaw == pytest.approx(90.0)
    assert nx_pitch == pytest.approx(0.0)


def _constant_color_cube_group(
    tmp_path: Path,
    *,
    views: dict[str, tuple[float, float]] | None = None,
) -> tuple[CubemapFrameGroup, dict[str, tuple[int, int, int]]]:
    face_size = 32
    colors = {
        "pz": (10, 20, 30),
        "px": (40, 50, 60),
        "nz": (70, 80, 90),
        "nx": (100, 110, 120),
        "top": (130, 140, 150),
        "bottom": (160, 170, 180),
    }
    transforms = views or {face: (0.0, 0.0) for face in colors}

    def frame(name: str) -> PinholeFrame:
        image_path = tmp_path / f"direct_{name}.png"
        image = np.full((face_size, face_size, 3), colors[name], dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        transform = np.eye(4)
        yaw, pitch = transforms[name]
        transform[:3, :3] = _rotation(yaw, pitch)
        return PinholeFrame(
            frame_id=name,
            file_path=f"images/direct_{name}.png",
            image_path=image_path,
            width=face_size,
            height=face_size,
            fl_x=face_size / 2.0,
            fl_y=face_size / 2.0,
            cx=(face_size - 1) / 2.0,
            cy=(face_size - 1) / 2.0,
            transform_matrix=transform,
        )

    return CubemapFrameGroup(name="direct", frames_by_face={face: frame(face) for face in colors}), colors


def test_render_cubemap_direct_preview_samples_standard_side_faces(tmp_path: Path) -> None:
    group, colors = _constant_color_cube_group(tmp_path)

    cases = {
        "pz": 0.0,
        "px": 90.0,
        "nz": 180.0,
        "nx": -90.0,
    }
    for face, yaw in cases.items():
        rendered = render_cubemap_direct_preview(
            group,
            yaw_deg=yaw,
            pitch_deg=0.0,
            output_size=33,
        )

        assert tuple(int(value) for value in rendered[16, 16]) == colors[face]


def test_render_cubemap_direct_preview_applies_roll(tmp_path: Path) -> None:
    group, _colors = _constant_color_cube_group(tmp_path)
    pz_frame = group.frames_by_face["pz"]
    gradient = np.zeros((pz_frame.height, pz_frame.width, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, pz_frame.width, dtype=np.uint8)[None, :]
    assert cv2.imwrite(str(pz_frame.image_path), gradient)

    no_roll = render_cubemap_direct_preview(
        group,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        output_size=33,
    )
    rolled = render_cubemap_direct_preview(
        group,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=180.0,
        output_size=33,
    )

    assert int(no_roll[16, 24, 0]) > 170
    assert int(rolled[16, 24, 0]) < 90


def test_render_cubemap_world_perspective_uses_frame_world_pose(tmp_path: Path) -> None:
    group, _colors = _constant_color_cube_group(tmp_path)
    pz_frame = group.frames_by_face["pz"]
    gradient = np.zeros((pz_frame.height, pz_frame.width, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, pz_frame.width, dtype=np.uint8)[None, :]
    gradient[:, :, 1] = np.linspace(0, 255, pz_frame.height, dtype=np.uint8)[:, None]
    assert cv2.imwrite(str(pz_frame.image_path), gradient)

    yaw, pitch, roll, fov = axis_face_view_params(group, "pz", fov_deg=90.0)
    rendered = render_cubemap_world_perspective(
        group,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        fov_deg=fov,
        output_size=pz_frame.width,
    )

    direct_error = float(np.mean(np.abs(rendered.astype(np.int16) - gradient.astype(np.int16))))
    mirrored_error = float(np.mean(np.abs(rendered.astype(np.int16) - gradient[:, ::-1].astype(np.int16))))
    assert direct_error < 2.0
    assert direct_error * 8.0 < mirrored_error


def test_render_cubemap_direct_preview_uses_transform_relative_face_layout(tmp_path: Path) -> None:
    views = {
        "pz": (0.0, 0.0),
        "px": (-90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }
    group, colors = _constant_color_cube_group(tmp_path, views=views)

    rendered_px = render_cubemap_direct_preview(group, yaw_deg=-90.0, pitch_deg=0.0, output_size=33)
    rendered_nx = render_cubemap_direct_preview(group, yaw_deg=90.0, pitch_deg=0.0, output_size=33)
    faces = cubemap_preview_sampler_faces(group)

    assert tuple(int(value) for value in rendered_px[16, 16]) == colors["px"]
    assert tuple(int(value) for value in rendered_nx[16, 16]) == colors["nx"]
    assert {face.face for face in faces} == {"pz", "px", "nz", "nx", "top", "bottom"}


def test_direct_preview_visible_faces_are_culled_for_current_view(tmp_path: Path) -> None:
    group, _colors = _constant_color_cube_group(tmp_path)
    faces = cubemap_preview_sampler_faces(group)

    front_indices = visible_cubemap_preview_face_indices(
        faces,
        yaw_deg=0.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )
    seam_indices = visible_cubemap_preview_face_indices(
        faces,
        yaw_deg=45.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert [faces[index].face for index in front_indices] == ["pz"]
    assert {"pz", "px"}.issubset({faces[index].face for index in seam_indices})
    assert len(seam_indices) < len(faces)


def test_direct_preview_visible_faces_follow_transform_relative_layout(tmp_path: Path) -> None:
    views = {
        "pz": (0.0, 0.0),
        "px": (-90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }
    group, _colors = _constant_color_cube_group(tmp_path, views=views)
    faces = cubemap_preview_sampler_faces(group)
    indices = visible_cubemap_preview_face_indices(
        faces,
        yaw_deg=-90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert [faces[index].face for index in indices] == ["px"]


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


def test_virtual_camera_rotation_matches_preview_ray_with_tilted_base(tmp_path: Path) -> None:
    base = _rotation(35.0, 12.0)
    views = {
        "pz": (0.0, 0.0),
        "px": (90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (-90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }

    def frame(name: str, view: tuple[float, float]) -> PinholeFrame:
        transform = np.eye(4)
        transform[:3, :3] = base @ _rotation(*view)
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

    group = CubemapFrameGroup(name="frame", frames_by_face={name: frame(name, view) for name, view in views.items()})

    for yaw, pitch in ((0.0, 0.0), (90.0, 0.0), (45.0, -15.0), (-60.0, 20.0)):
        ray = view_pixel_to_world_ray(
            group,
            x_px=49.5,
            y_px=49.5,
            output_size=100,
            yaw_deg=yaw,
            pitch_deg=pitch,
            fov_deg=90.0,
        )
        rotation = virtual_camera_rotation(group, yaw_deg=yaw, pitch_deg=pitch)
        frustum_forward = np.array([0.0, 0.0, 1.0]) @ rotation.T

        assert np.allclose(ray, frustum_forward)


def test_preview_frustum_rays_use_same_mapping_as_click_ray(tmp_path: Path) -> None:
    base = _rotation(35.0, 12.0)
    views = {
        "pz": (0.0, 0.0),
        "px": (90.0, 0.0),
        "nz": (180.0, 0.0),
        "nx": (-90.0, 0.0),
        "top": (0.0, 90.0),
        "bottom": (0.0, -90.0),
    }

    def frame(name: str, view: tuple[float, float]) -> PinholeFrame:
        transform = np.eye(4)
        transform[:3, :3] = base @ _rotation(*view)
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

    group = CubemapFrameGroup(name="frame", frames_by_face={name: frame(name, view) for name, view in views.items()})

    center_ray, corner_rays = preview_frustum_rays(
        group,
        output_size=101,
        yaw_deg=42.0,
        pitch_deg=-18.0,
        fov_deg=90.0,
    )
    click_ray = view_pixel_to_world_ray(
        group,
        x_px=50.0,
        y_px=50.0,
        output_size=101,
        yaw_deg=42.0,
        pitch_deg=-18.0,
        fov_deg=90.0,
    )

    assert np.allclose(center_ray, click_ray)
    assert corner_rays.shape == (4, 3)
    assert np.allclose(np.linalg.norm(corner_rays, axis=1), np.ones(4))
    assert np.all(corner_rays @ center_ray > 0.0)


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


def test_gui_cube6_generated_transforms_use_gui_face_layout(tmp_path: Path) -> None:
    transforms = _write_gui_cube6_generated_transforms(tmp_path / "transforms.json")
    group = load_cubemap_frame_groups(transforms)[0]

    assert face_view_params(group, "px") == (0.0, -0.0, 90.0)
    assert face_view_params(group, "nz") == (90.0, -0.0, 90.0)
    assert face_view_params(group, "nx") == (180.0, -0.0, 90.0)
    assert face_view_params(group, "pz") == (-90.0, -0.0, 90.0)

    ray, _up, face = view_pixel_to_world_ray_and_up(
        group,
        x_px=49.5,
        y_px=49.5,
        output_size=100,
        yaw_deg=-90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )
    projected = project_sfm_points_to_preview_points(
        group,
        np.array([[-10.0, 0.0, 0.0]], dtype=float),
        output_size=100,
        yaw_deg=-90.0,
        pitch_deg=0.0,
        fov_deg=90.0,
    )

    assert face == "pz"
    assert np.allclose(ray, np.array([-1.0, 0.0, 0.0]))
    assert projected is not None
    assert np.allclose(projected[0], np.array([49.5, 49.5]))


def test_generated_cube6_normalization_uses_view_metadata(tmp_path: Path) -> None:
    view_params = {
        "bottom": (-45.0, -90.0),
        "px": (45.0, 0.0),
        "nz": (135.0, 0.0),
        "nx": (-135.0, 0.0),
        "pz": (-45.0, 0.0),
        "top": (-45.0, 90.0),
    }
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    frames_json = []
    for face, (yaw, pitch) in view_params.items():
        (image_dir / f"frame_0001_{face}.jpg").write_bytes(b"fake image bytes")
        transform = np.eye(4)
        transform[:3, :3] = _export_rotation(yaw, pitch).T
        frames_json.append({"file_path": f"images/frame_0001_{face}.jpg", "transform_matrix": transform.tolist()})
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
                "frames": frames_json,
            }
        ),
        encoding="utf-8",
    )

    frames = {
        Path(frame.file_path).stem.rsplit("_", 1)[-1]: frame
        for frame in load_pinhole_frames(transforms, cubemap_view_params=view_params)
    }

    for face, (yaw, pitch) in view_params.items():
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ _rotation(yaw, pitch).T
        actual = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ frames[face].camera_to_world_rotation.T
        assert np.allclose(actual, expected, atol=1e-6)


def test_generated_cube6_normalization_discovers_step4_view_metadata(tmp_path: Path) -> None:
    view_params = {
        "bottom": (-45.0, -90.0),
        "px": (45.0, 0.0),
        "nz": (135.0, 0.0),
        "nx": (-135.0, 0.0),
        "pz": (-45.0, 0.0),
        "top": (-45.0, 90.0),
    }
    output = tmp_path / "output"
    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    frames_json = []
    for frame_index, prefix in enumerate(("frame_0001", "frame_0002")):
        yaw_offset = frame_index * 30.0
        for face, (yaw, pitch) in view_params.items():
            (image_dir / f"{prefix}_{face}.jpg").write_bytes(b"fake image bytes")
            transform = np.eye(4)
            transform[:3, :3] = _export_rotation(yaw + yaw_offset, pitch).T
            frames_json.append({"file_path": f"images/{prefix}_{face}.jpg", "transform_matrix": transform.tolist()})
    transforms = output / "transforms.json"
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
                "frames": frames_json,
            }
        ),
        encoding="utf-8",
    )
    settings = tmp_path / "_stechdrive" / "step4" / "export_settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "views_config_snapshot": {
                    "views": [{"name": k, "yaw": v[0], "pitch": v[1]} for k, v in view_params.items()]
                },
                "conversion": {"yaw_offset_per_frame": 30.0},
            }
        ),
        encoding="utf-8",
    )

    frames = {Path(frame.file_path).stem: frame for frame in load_pinhole_frames(transforms)}

    actual = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ frames["frame_0002_px"].camera_to_world_rotation.T
    expected = np.array([0.0, 0.0, 1.0], dtype=np.float64) @ _rotation(75.0, 0.0).T
    assert np.allclose(actual, expected, atol=1e-6)


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
