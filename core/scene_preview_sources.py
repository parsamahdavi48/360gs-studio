"""Discover read-only scene preview inputs under a managed scene folder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.scene_layout import (
    scene_images_dir,
    scene_masks_dir,
    scene_output_dir,
    step4_export_settings_path,
    step4_metashape_import_work_dir,
)
from core.scene_preview_profiles import ScenePreviewDisplayTransform, step4_output_display_transform

ScenePreviewCandidateKind = Literal["output", "metashape", "colmap", "spheresfm"]


@dataclass(frozen=True)
class ScenePreviewCandidate:
    kind: ScenePreviewCandidateKind
    label: str
    path: Path
    image_root: Path | None = None
    mask_root: Path | None = None
    pointcloud_path: Path | None = None
    display_transform: ScenePreviewDisplayTransform | None = None


def discover_scene_preview_candidates(scene_dir: Path) -> tuple[ScenePreviewCandidate, ...]:
    scene = Path(scene_dir)
    candidates: list[ScenePreviewCandidate] = []
    output = scene_output_dir(scene)
    output_transforms = output / "transforms.json"
    settings = _load_step4_settings(scene)
    if output_transforms.is_file():
        candidates.append(
            ScenePreviewCandidate(
                kind="output",
                label="Step 4 output",
                path=output_transforms,
                image_root=output,
                mask_root=_existing_dir(output / "masks"),
                pointcloud_path=_existing_file(output / "pointcloud.ply"),
                display_transform=step4_output_display_transform(settings),
            )
        )

    metashape_xml, metashape_ply = _metashape_inputs(scene)
    if metashape_xml is not None:
        candidates.append(
            ScenePreviewCandidate(
                kind="metashape",
                label="Metashape SfM",
                path=metashape_xml,
                image_root=scene_images_dir(scene),
                mask_root=_step4_input_masks_dir(scene, settings),
                pointcloud_path=metashape_ply,
            )
        )

    colmap_sparse = _resolve_colmap_model(output / "colmap_rig" / "sparse")
    if colmap_sparse is not None:
        candidates.append(
            ScenePreviewCandidate(
                kind="colmap",
                label="COLMAP SfM",
                path=colmap_sparse,
                image_root=output / "colmap_rig" / "images",
                mask_root=_first_existing_dir(
                    _settings_path_or_none(scene, _settings_dict(settings, "colmap_rig").get("masks_dir")),
                    output / "colmap_rig" / "masks",
                ),
            )
        )

    spheresfm_sparse = _resolve_colmap_model(output / "spheresfm" / "sparse")
    if spheresfm_sparse is not None:
        image_root = output / "spheresfm" / "equirect"
        candidates.append(
            ScenePreviewCandidate(
                kind="spheresfm",
                label="SphereSfM SfM",
                path=spheresfm_sparse,
                image_root=image_root if image_root.is_dir() else scene_images_dir(scene),
                mask_root=_first_existing_dir(
                    _settings_path_or_none(scene, _settings_dict(settings, "spheresfm").get("prepared_masks_dir")),
                    output / "spheresfm" / "masks_colmap",
                    scene_masks_dir(scene),
                ),
            )
        )
    return tuple(candidates)


def _metashape_inputs(scene: Path) -> tuple[Path | None, Path | None]:
    settings = _load_step4_settings(scene)
    metashape = settings.get("metashape_import") if isinstance(settings.get("metashape_import"), dict) else {}
    xml = _existing_file(_path_or_none(metashape.get("xml")))
    ply = _existing_file(_path_or_none(metashape.get("ply")))
    if xml is None:
        xml = _first_existing(
            step4_metashape_import_work_dir(scene) / "metashape.xml",
            scene / "metashape.xml",
            scene / "cameras.xml",
        )
    if ply is None:
        ply = _first_existing(scene / "pointcloud.ply", scene_output_dir(scene) / "pointcloud.ply")
    return xml, ply


def _load_step4_settings(scene: Path) -> dict:
    path = step4_export_settings_path(scene)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _path_or_none(value: object) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None


def _settings_path_or_none(scene: Path, value: object) -> Path | None:
    path = _path_or_none(value)
    if path is None:
        return None
    return path if path.is_absolute() else scene / path


def _settings_dict(settings: dict, key: str) -> dict:
    value = settings.get(key)
    return value if isinstance(value, dict) else {}


def _existing_file(path: Path | None) -> Path | None:
    return path if path is not None and path.is_file() else None


def _existing_dir(path: Path | None) -> Path | None:
    return path if path is not None and path.is_dir() else None


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _first_existing_dir(*paths: Path | None) -> Path | None:
    return next((path for path in paths if path is not None and path.is_dir()), None)


def _step4_input_masks_dir(scene: Path, settings: dict) -> Path | None:
    inputs = _settings_dict(settings, "inputs")
    return _first_existing_dir(
        _settings_path_or_none(scene, inputs.get("masks_dir")),
        scene_masks_dir(scene),
    )


def _has_colmap_model(path: Path) -> bool:
    return path.is_dir() and (
        all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))
        or all((path / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))
    )


def _resolve_colmap_model(root: Path) -> Path | None:
    if _has_colmap_model(root):
        return root
    if not root.is_dir():
        return None
    candidates = [path for path in root.iterdir() if _has_colmap_model(path)]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, int | str]:
        if path.name.isdigit():
            return (0, int(path.name))
        return (1, path.name.lower())

    return sorted(candidates, key=sort_key)[0]
