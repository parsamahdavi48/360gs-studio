from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gui.steps.step3_mask_manifests import write_mask_target_manifest, write_projection_manifests
from gui.steps.step3_mask_plan import PROJECTION_EQUIRECT, PROJECTION_NORMAL


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color=(1, 2, 3)).save(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_write_projection_manifests_groups_by_projection(tmp_path: Path) -> None:
    scene = tmp_path
    erp = scene / "images" / "erp.jpg"
    normal = scene / "images" / "normal.jpg"
    _write_image(erp)
    _write_image(normal)
    projections = {erp: PROJECTION_EQUIRECT, normal: PROJECTION_NORMAL}

    manifests = write_projection_manifests(
        scene_dir=scene,
        image_paths=[erp, normal],
        projection_for_image=lambda path: projections[path],
        mask_path_for_image=lambda path: scene / "masks" / f"{path.stem}.png",
        run_id_factory=lambda prefix: f"{prefix}_fixed",
    )

    assert set(manifests) == {"all", PROJECTION_EQUIRECT, PROJECTION_NORMAL}
    assert [item["image"] for item in _read_jsonl(manifests[PROJECTION_EQUIRECT])] == ["images/erp.jpg"]
    assert [item["image"] for item in _read_jsonl(manifests[PROJECTION_NORMAL])] == ["images/normal.jpg"]
    assert [item["image"] for item in _read_jsonl(manifests["all"])] == ["images/erp.jpg", "images/normal.jpg"]


def test_write_mask_target_manifest_writes_scene_relative_masks(tmp_path: Path) -> None:
    scene = tmp_path
    image = scene / "images" / "sub" / "frame.jpg"
    _write_image(image)

    manifest = write_mask_target_manifest(
        scene_dir=scene,
        image_paths=[image],
        projection_for_image=lambda _path: PROJECTION_EQUIRECT,
        mask_path_for_image=lambda path: scene / "masks" / "sub" / f"{path.stem}.png",
        run_id_factory=lambda prefix: f"{prefix}_fixed",
    )

    assert manifest == scene / "_stechdrive" / "masks" / "work" / "mask_targets_fixed" / "targets.jsonl"
    assert _read_jsonl(manifest) == [
        {
            "image": "images/sub/frame.jpg",
            "mask": "masks/sub/frame.png",
            "projection": PROJECTION_EQUIRECT,
        }
    ]
