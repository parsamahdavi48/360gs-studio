"""JSONL manifest writing for Step 3 mask jobs."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from core.scene_project import scene_relative
from gui.steps.step3_mask_plan import PROJECTION_ALL, PROJECTION_EQUIRECT, PROJECTION_NORMAL

ProjectionResolver = Callable[[Path], str]
MaskPathResolver = Callable[[Path], Path]
RunIdFactory = Callable[[str], str]


def write_projection_manifests(
    *,
    scene_dir: str | Path,
    image_paths: Sequence[Path],
    projection_for_image: ProjectionResolver,
    mask_path_for_image: MaskPathResolver,
    run_id_factory: RunIdFactory,
) -> dict[str, Path]:
    if not image_paths:
        return {}

    scene = Path(scene_dir)
    manifest_dir = _mask_work_dir(scene, run_id_factory("mask_list"))
    groups: dict[str, list[Path]] = {
        PROJECTION_EQUIRECT: [],
        PROJECTION_NORMAL: [],
        PROJECTION_ALL: [],
    }
    for image_path in image_paths:
        projection = projection_for_image(image_path)
        if projection not in {PROJECTION_EQUIRECT, PROJECTION_NORMAL}:
            projection = PROJECTION_EQUIRECT
        groups[projection].append(image_path)
        groups[PROJECTION_ALL].append(image_path)

    manifests: dict[str, Path] = {}
    for key, paths in groups.items():
        if not paths:
            continue
        manifests[key] = _write_manifest(
            manifest_dir / f"{key}.jsonl",
            scene=scene,
            image_paths=paths,
            projection_for_image=projection_for_image,
            mask_path_for_image=mask_path_for_image,
        )
    return manifests


def write_mask_target_manifest(
    *,
    scene_dir: str | Path,
    image_paths: Sequence[Path],
    projection_for_image: ProjectionResolver,
    mask_path_for_image: MaskPathResolver,
    run_id_factory: RunIdFactory,
) -> Path:
    scene = Path(scene_dir)
    manifest_dir = _mask_work_dir(scene, run_id_factory("mask_targets"))
    return _write_manifest(
        manifest_dir / "targets.jsonl",
        scene=scene,
        image_paths=image_paths,
        projection_for_image=projection_for_image,
        mask_path_for_image=mask_path_for_image,
    )


def _mask_work_dir(scene: Path, run_id: str) -> Path:
    manifest_dir = scene / "_stechdrive" / "masks" / "work" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir


def _write_manifest(
    path: Path,
    *,
    scene: Path,
    image_paths: Sequence[Path],
    projection_for_image: ProjectionResolver,
    mask_path_for_image: MaskPathResolver,
) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for image_path in image_paths:
            record = {
                "image": scene_relative(scene, image_path),
                "mask": scene_relative(scene, mask_path_for_image(image_path)),
                "projection": projection_for_image(image_path),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
