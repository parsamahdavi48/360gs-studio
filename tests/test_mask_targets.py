from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.init_masks import run as init_masks_run
from core.mask_targets import collect_image_targets, load_mask_paths_from_image_list


def test_image_list_resolves_scene_relative_image_and_mask_paths(tmp_path: Path) -> None:
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    (images / "sub").mkdir()
    image_path = images / "sub" / "frame.jpg"
    cv2.imwrite(str(image_path), np.full((8, 16, 3), 128, dtype=np.uint8))
    manifest = scene / "_stechdrive" / "masks" / "work" / "list.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "image": "images/sub/frame.jpg",
                "mask": "masks/sub/frame.png",
                "projection": "equirect",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _root, targets = collect_image_targets(images, masks, image_list=manifest)

    assert len(targets) == 1
    assert targets[0].image_path == image_path
    assert targets[0].mask_path == masks / "sub" / "frame.png"
    assert targets[0].projection == "equirect"
    assert load_mask_paths_from_image_list(manifest, masks_root=masks) == [masks / "sub" / "frame.png"]


def test_image_list_accepts_multiple_jsonl_object_lines(tmp_path: Path) -> None:
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "a.jpg"), np.full((8, 16, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(images / "b.jpg"), np.full((8, 16, 3), 64, dtype=np.uint8))
    manifest = scene / "_stechdrive" / "masks" / "work" / "list.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"image": "images/a.jpg", "mask": "masks/a.png", "projection": "equirect"}),
                json.dumps({"image": "images/b.jpg", "mask": "masks/b.png", "projection": "equirect"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _root, targets = collect_image_targets(images, masks, image_list=manifest)

    assert [target.image_path for target in targets] == [images / "a.jpg", images / "b.jpg"]
    assert [target.mask_path for target in targets] == [masks / "a.png", masks / "b.png"]


def test_init_masks_image_list_writes_only_listed_outputs(tmp_path: Path) -> None:
    scene = tmp_path
    images = scene / "images"
    masks = scene / "masks"
    images.mkdir()
    cv2.imwrite(str(images / "a.jpg"), np.full((8, 16, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(images / "b.jpg"), np.full((8, 16, 3), 64, dtype=np.uint8))
    manifest = scene / "targets.jsonl"
    manifest.write_text(
        json.dumps({"image": "images/b.jpg", "mask": "masks/custom/b.png", "projection": "normal"}) + "\n",
        encoding="utf-8",
    )

    assert init_masks_run(images, masks, image_list=manifest) == 0

    assert not (masks / "a.png").exists()
    assert (masks / "custom" / "b.png").is_file()
