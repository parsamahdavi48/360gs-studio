from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from devtools.apriltag.case import (
    AprilTagPlacement,
    create_case,
    load_case,
    load_placement,
    run_dir_for_placement,
    save_placement,
)
from devtools.apriltag.cubemap_preview import (
    face_view_params,
    load_cubemap_frame_groups,
    load_metashape_camera_labels,
    order_groups_by_labels,
    split_cubemap_face,
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
