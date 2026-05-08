from __future__ import annotations

from pathlib import Path

import pytest

from mask_targets import load_image_targets
from path_safety import safe_clear_path
from scene_layout import scene_images_dir, scene_masks_dir, scene_output_dir
from scene_project import load_json, write_json


def test_scene_layout_names_primary_scene_folders(tmp_path: Path) -> None:
    assert scene_images_dir(tmp_path) == tmp_path / "images"
    assert scene_masks_dir(tmp_path) == tmp_path / "masks"
    assert scene_output_dir(tmp_path) == tmp_path / "output"


def test_write_json_replaces_atomically_without_leaving_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "meta" / "project.json"

    write_json(path, {"version": 1, "name": "first"})
    write_json(path, {"version": 1, "name": "second"})

    assert load_json(path) == {"version": 1, "name": "second"}
    assert list(path.parent.glob(".*.tmp")) == []


def test_safe_clear_path_rejects_allowed_root_itself(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="allowed root"):
        safe_clear_path(tmp_path, allowed_roots=[tmp_path])


def test_safe_clear_path_rejects_outside_allowed_roots(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()

    with pytest.raises(RuntimeError, match="outside allowed roots"):
        safe_clear_path(outside, allowed_roots=[inside])

    assert outside.is_dir()


def test_image_list_resolution_uses_manifest_directory_before_optional_cwd(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    manifest_dir = tmp_path / "lists"
    manifest_dir.mkdir()
    (manifest_dir / "local.jpg").write_bytes(b"image")
    manifest = manifest_dir / "targets.jsonl"
    manifest.write_text('"local.jpg"\n', encoding="utf-8")

    targets = load_image_targets(manifest, images_root=images, masks_root=masks)

    assert len(targets) == 1
    assert targets[0].image_path == manifest_dir / "local.jpg"
