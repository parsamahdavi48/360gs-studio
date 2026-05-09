from __future__ import annotations

import json
from pathlib import Path

from devtools.apriltag.case import (
    AprilTagPlacement,
    create_case,
    load_case,
    load_placement,
    run_dir_for_placement,
    save_placement,
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

