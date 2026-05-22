from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from core.sfm_job_spec import colmap_mixed_project_job, write_sfm_job


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 130, 140)).save(path)


def _write_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_prepare_colmap_mixed_project_writes_rig_and_normal_lists(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "pano.jpg", (64, 32))
    _write_image(scene / "images" / "normal.jpg", (40, 30))
    _write_mask(scene / "masks" / "normal.png", (40, 30))
    views = tmp_path / "views.json"
    views.write_text(
        json.dumps(
            {
                "views": [
                    {"name": "front", "yaw": 0.0, "pitch": 0.0},
                    {"name": "right", "yaw": 90.0, "pitch": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_colmap_mixed_project.py",
            str(scene),
            str(output),
            "--views-json",
            str(views),
            "--output-scale",
            "0.5",
            "--workers",
            "1",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    project = output / "colmap_rig"
    rig_names = (project / "rig_image_list.txt").read_text(encoding="utf-8").splitlines()
    normal_names = (project / "normal_image_list.txt").read_text(encoding="utf-8").splitlines()
    assert rig_names == ["rig1/cam01/frame_00001.jpg", "rig1/cam02/frame_00001.jpg"]
    assert len(normal_names) == 1
    assert (project / "images" / rig_names[0]).is_file()
    assert (project / "images" / normal_names[0]).is_file()
    assert (project / "masks" / f"{normal_names[0]}.png").is_file()

    manifest = json.loads((project / "stechdrive_colmap_mixed_project.json").read_text(encoding="utf-8"))
    assert manifest["export_type"] == "colmap_mixed_project"
    assert manifest["erp_source_count"] == 1
    assert manifest["normal_source_count"] == 1
    assert manifest["rig_image_count"] == 2
    assert manifest["normal_camera_model"] == "SIMPLE_RADIAL"
    assert manifest["normal_camera_groups"] == [
        {
            "id": "unknown_40x30",
            "image_dir": "normal/unknown_40x30",
            "camera_model": "SIMPLE_RADIAL",
            "width": 40,
            "height": 30,
            "source_kind": "unknown",
            "source_id": "unknown",
            "image_count": 1,
        }
    ]


def test_prepare_colmap_mixed_project_accepts_job_json(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_image(scene / "images" / "normal.jpg", (40, 30))
    output = tmp_path / "output"
    job_path = write_sfm_job(
        tmp_path / "job.json",
        colmap_mixed_project_job(
            scene_dir=scene,
            output_dir=output,
            views=[{"name": "front", "yaw": 0.0, "pitch": 0.0, "enabled": True}],
            output_scale=0.5,
            output_format="jpg",
            output_bit_depth="8",
            jpg_quality=95,
            write_images=True,
            write_masks=True,
            invert_masks=False,
            workers="1",
            remap_cache_limit="auto",
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_colmap_mixed_project.py",
            "--job",
            str(job_path),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "colmap_rig" / "stechdrive_colmap_mixed_project.json").read_text(encoding="utf-8"))
    assert manifest["normal_source_count"] == 1
    assert (output / "colmap_rig" / "normal_image_list.txt").read_text(encoding="utf-8").strip()
