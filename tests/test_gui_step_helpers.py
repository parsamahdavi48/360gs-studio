import json
from pathlib import Path

from gui.steps.cubemap_commands import (
    ColmapSfmCommand,
    CubemapConversionCommand,
    build_colmap_sfm_commands,
    build_cubemap_conversion_cmd,
    write_views_config,
)
from gui.steps.mask_commands import MaskCommandContext, build_sam31_prompt_cmd
from gui.steps.mask_image_import import import_external_images


def _mask_context(base_dir: Path) -> MaskCommandContext:
    return MaskCommandContext(
        python_executable="python.exe",
        base_dir=base_dir,
        projection="equirect",
        quality="high",
        yolo_expand="2",
        sky_inference_size="768",
        sky_min_score="0.25",
        sky_min_area_ratio="0.01",
        sky_top_connected=False,
        stitch_boundary_width=5.0,
        stitch_workers="4",
        overexposure_threshold="254",
        overexposure_dilate="1",
        sam31_merge_mode="replace",
    )


def test_mask_command_builder_keeps_sam31_safe_batch_directory_only(tmp_path: Path) -> None:
    (tmp_path / "sky_mask.py").write_text("", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    image_file = images / "frame_0001.jpg"
    image_file.write_bytes(b"image")
    masks = tmp_path / "masks"

    dir_cmd = build_sam31_prompt_cmd(
        _mask_context(tmp_path),
        images,
        masks,
        prompts=["person", "sky"],
    )
    file_cmd = build_sam31_prompt_cmd(
        _mask_context(tmp_path),
        image_file,
        masks,
        prompts=["person"],
    )

    assert dir_cmd[0:3] == ["python.exe", "-u", str(tmp_path / "sky_mask.py")]
    assert dir_cmd[dir_cmd.index("--backend") + 1] == "sam31"
    assert "--replace" in dir_cmd
    assert "--safe-batch" in dir_cmd
    assert "--safe-batch" not in file_cmd


def test_external_image_import_helper_skips_existing_names(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "scene" / "images"
    (source / "a.JPG").write_bytes(b"a")
    (source / "b.png").write_bytes(b"b")
    (source / "ignore.txt").write_text("ignore", encoding="utf-8")

    assert import_external_images(source, target) == (2, 0)
    assert import_external_images(source, target) == (0, 2)
    assert sorted(path.name for path in target.iterdir()) == ["a.JPG", "b.png"]


def test_cubemap_command_builder_writes_views_and_flags(tmp_path: Path) -> None:
    script = tmp_path / "cubemap_transforms_json.py"
    script.write_text("", encoding="utf-8")
    views_json = write_views_config(
        tmp_path / "output",
        [{"name": "px", "yaw": 0, "pitch": 0, "enabled": True}],
    )

    payload = json.loads(views_json.read_text(encoding="utf-8"))
    assert payload["views"][0] == {"name": "px", "yaw": 0.0, "pitch": 0.0, "enabled": True}

    cmd = build_cubemap_conversion_cmd(
        CubemapConversionCommand(
            python_executable="python.exe",
            script=script,
            scene=tmp_path,
            output=tmp_path / "output",
            views_json=views_json,
            scale=1.0,
            axis_mode="brush",
            image_only=False,
            colmap_rig=False,
            invert_masks=True,
            writes_images=False,
            writes_masks=True,
            yaw_offset_per_frame=30.0,
            output_format="jpg",
            output_bit_depth="8",
            jpg_quality=92,
        )
    )

    assert "--brush" in cmd
    assert "--invert_masks" in cmd
    assert "--skip-images" in cmd
    assert "--skip-masks" not in cmd
    assert cmd[cmd.index("--jpg-quality") + 1] == "92"


def test_colmap_sfm_builder_keeps_mapper_contract(tmp_path: Path) -> None:
    images = tmp_path / "rig" / "images"
    masks = tmp_path / "rig" / "masks"
    sparse = tmp_path / "rig" / "sparse"
    images.mkdir(parents=True)

    commands = build_colmap_sfm_commands(
        ColmapSfmCommand(
            colmap="colmap.exe",
            glomap="glomap.exe",
            rig_dir=tmp_path / "rig",
            images_dir=images,
            masks_dir=masks,
            database=tmp_path / "rig" / "database.db",
            sparse=sparse,
            camera_params="16,16,8,8",
            writes_images=True,
            writes_masks=False,
            matcher="exhaustive",
            mapper="incremental",
        )
    )

    assert [phase for phase, _cmd in commands] == [
        "colmap_feature",
        "colmap_rig_config",
        "colmap_match",
        "colmap_mapper",
    ]
    assert commands[2][1][1] == "exhaustive_matcher"
    assert commands[3][1][0:2] == ["colmap.exe", "mapper"]
    assert sparse.is_dir()
