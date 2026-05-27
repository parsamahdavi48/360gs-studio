"""Discover read-only scene preview inputs under a managed scene folder."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.artifact_registry import ArtifactRecord, load_artifacts
from core.nerf_dataset_paths import (
    find_nerf_pointcloud_path,
    iter_nerf_transforms_paths,
    profile_from_transforms_name,
)
from core.scene_layout import (
    scene_images_dir,
    scene_masks_dir,
    scene_metashape_3dgut_dir,
    scene_metashape_cubemap_dir,
    scene_output_dir,
    scene_spheresfm_3dgut_dir,
    scene_spheresfm_cubemap_dir,
    step4_export_settings_path,
    step4_metashape_import_work_dir,
)
from core.scene_preview_profiles import (
    ScenePreviewDisplayTransform,
    realityscan_colmap_export_display_transform,
    realityscan_csv_display_transform,
    realityscan_lfs_colmap_display_transform,
    transforms_dataset_display_transform,
)
from core.workflow_artifacts import (
    DATASET_KIND_COLMAP_DATASET,
    DATASET_KIND_LICHTFELD_COLMAP,
    DATASET_KIND_NERF_JSON_PLY,
    DATASET_KIND_REALITYSCAN_REALIGN_INPUT,
    SFM_KIND_COLMAP_SPARSE,
    SFM_KIND_METASHAPE_XML_PLY,
    SFM_KIND_REALITYSCAN_CSV_PLY,
    SFM_KIND_SPHERESFM_SPARSE,
    artifact_root_path,
)

ScenePreviewCandidateKind = Literal["output", "metashape", "colmap", "spheresfm", "realityscan"]


@dataclass(frozen=True)
class ScenePreviewCandidate:
    kind: ScenePreviewCandidateKind
    label: str
    path: Path
    image_root: Path | None = None
    mask_root: Path | None = None
    pointcloud_path: Path | None = None
    display_transform: ScenePreviewDisplayTransform | None = None
    colmap_opengl_camera: bool = False


def discover_scene_preview_candidates(scene_dir: Path) -> tuple[ScenePreviewCandidate, ...]:
    scene = Path(scene_dir)
    candidates: list[ScenePreviewCandidate] = []
    settings = _load_step4_settings(scene)
    current_output = _current_step4_output_root(scene, settings)
    dataset_records = load_artifacts(scene, "dataset")
    sfm_records = load_artifacts(scene, "sfm")
    for dataset_root, output_transforms, record in _transforms_dataset_roots(scene, settings, dataset_records):
        candidates.append(
            ScenePreviewCandidate(
                kind="output",
                label=_dataset_label(dataset_root, current_output, record, transforms_json=output_transforms),
                path=output_transforms,
                image_root=dataset_root,
                mask_root=_existing_dir(dataset_root / "masks"),
                pointcloud_path=_transforms_pointcloud_path(dataset_root, output_transforms),
                display_transform=transforms_dataset_display_transform(
                    output_transforms,
                    fallback_settings=settings if _same_path(dataset_root, current_output) else None,
                ),
            )
        )

    for metashape_xml, metashape_ply, image_root, mask_root, label in _metashape_inputs(scene, settings, sfm_records):
        candidates.append(
            ScenePreviewCandidate(
                kind="metashape",
                label=label,
                path=metashape_xml,
                image_root=image_root,
                mask_root=mask_root,
                pointcloud_path=metashape_ply,
            )
        )

    for colmap_root, sparse, label, record in _colmap_dataset_roots(scene, settings, dataset_records, sfm_records):
        manifest = _artifact_metadata(record)
        candidates.append(
            ScenePreviewCandidate(
                kind="colmap",
                label=label,
                path=sparse,
                image_root=_colmap_images_dir(scene, colmap_root, sparse, manifest),
                mask_root=_colmap_masks_dir(scene, colmap_root, sparse, manifest, settings, record),
                display_transform=_colmap_display_transform(scene, colmap_root, sparse, manifest, record),
                colmap_opengl_camera=_colmap_uses_app_camera_axes(scene, colmap_root, manifest, record),
            )
        )

    for sparse, record in _spheresfm_sparse_models(scene, sfm_records):
        root = _dataset_root_from_sparse(sparse)
        image_root = root / "equirect"
        candidates.append(
            ScenePreviewCandidate(
                kind="spheresfm",
                label=_sfm_label("SphereSfM SfM", record),
                path=sparse,
                image_root=image_root if image_root.is_dir() else scene_images_dir(scene),
                mask_root=_first_existing_dir(
                    _settings_path_or_none(scene, _settings_dict(settings, "spheresfm").get("prepared_masks_dir")),
                    root / "masks_colmap",
                    scene_masks_dir(scene),
                ),
            )
        )
    for csv_path, ply_path, image_root, mask_root, label in _realityscan_inputs(scene, sfm_records):
        candidates.append(
            ScenePreviewCandidate(
                kind="realityscan",
                label=label,
                path=csv_path,
                image_root=image_root,
                mask_root=mask_root,
                pointcloud_path=ply_path,
                display_transform=realityscan_csv_display_transform(),
            )
        )
    return tuple(_dedupe_candidates(candidates))


def _metashape_inputs(
    scene: Path,
    settings: dict,
    records: list[ArtifactRecord],
) -> tuple[tuple[Path, Path | None, Path, Path | None, str], ...]:
    result: list[tuple[Path, Path | None, Path, Path | None, str]] = []
    for record in records:
        if record.kind != SFM_KIND_METASHAPE_XML_PLY or record.status != "ready":
            continue
        xml = _existing_file(_artifact_file_path(scene, record, "xml"))
        if xml is None:
            continue
        result.append(
            (
                xml,
                _existing_file(_artifact_file_path(scene, record, "ply")),
                _existing_dir(_artifact_file_path(scene, record, "images_dir")) or scene_images_dir(scene),
                _existing_dir(_artifact_file_path(scene, record, "masks_dir")) or _step4_input_masks_dir(scene, settings),
                _sfm_label("Metashape SfM", record),
            )
        )

    metashape = settings.get("metashape_import") if isinstance(settings.get("metashape_import"), dict) else {}
    xml = _existing_file(_path_or_none(metashape.get("xml")))
    ply = _existing_file(_path_or_none(metashape.get("ply")))
    if xml is None:
        xml = _first_existing(
            step4_metashape_import_work_dir(scene) / "metashape.xml",
            scene / "metashape.xml",
            scene / "cameras.xml",
        )
    if xml is None:
        xml = _single_scene_metashape_xml(scene)
    if ply is None:
        ply = _single_scene_raw_ply(scene)
    if ply is None:
        ply = _first_existing(scene / "pointcloud.ply", scene_output_dir(scene) / "pointcloud.ply")
    if xml is not None:
        result.append((xml, ply, scene_images_dir(scene), _step4_input_masks_dir(scene, settings), "Metashape SfM"))
    return tuple(result)


def _single_scene_metashape_xml(scene: Path) -> Path | None:
    candidates = [path for path in _scene_xml_candidates(scene) if _looks_like_metashape_xml(path)]
    return candidates[0] if len(candidates) == 1 else None


def _scene_xml_candidates(scene: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in scene.glob("*.xml") if path.is_file()), key=lambda path: path.name.lower()))


def _looks_like_metashape_xml(path: Path) -> bool:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return False
    if _xml_tag_name(root.tag) != "document":
        return False
    for chunk in root.iter():
        if _xml_tag_name(chunk.tag) != "chunk":
            continue
        has_sensor = any(_xml_tag_name(child.tag) == "sensor" for child in chunk.iter())
        has_camera = any(_xml_tag_name(child.tag) == "camera" for child in chunk.iter())
        if has_sensor and has_camera:
            return True
    return False


def _single_scene_raw_ply(scene: Path) -> Path | None:
    candidates = _scene_raw_ply_candidates(scene)
    return candidates[0] if len(candidates) == 1 else None


def _scene_raw_ply_candidates(scene: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in scene.glob("*.ply")
                if path.is_file() and path.name.casefold() != "pointcloud.ply"
            ),
            key=lambda path: path.name.lower(),
        )
    )


def _xml_tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def _current_step4_output_root(scene: Path, settings: dict) -> Path | None:
    output_dir = _settings_path_or_none(scene, settings.get("output_dir"))
    if output_dir is not None:
        return output_dir
    portable = _settings_dict(settings, "portable_output")
    portable_root = _settings_path_or_none(scene, portable.get("root"))
    return portable_root if portable_root is not None else scene_output_dir(scene)


def _step4_dataset_roots(scene: Path, settings: dict) -> tuple[Path, ...]:
    roots = [
        _current_step4_output_root(scene, settings),
        scene_metashape_cubemap_dir(scene),
        scene_metashape_3dgut_dir(scene),
        scene_spheresfm_cubemap_dir(scene),
        scene_spheresfm_3dgut_dir(scene),
        scene_output_dir(scene),
    ]
    return tuple(_dedupe_paths(path for path in roots if path is not None))


def _transforms_dataset_roots(
    scene: Path,
    settings: dict,
    records: list[ArtifactRecord],
) -> tuple[tuple[Path, Path, ArtifactRecord | None], ...]:
    roots: list[tuple[Path, Path, ArtifactRecord | None]] = []
    for record in records:
        if record.kind not in {DATASET_KIND_NERF_JSON_PLY, DATASET_KIND_REALITYSCAN_REALIGN_INPUT} or record.status != "ready":
            continue
        root = artifact_root_path(scene, record)
        roots.extend((root, transforms, record) for transforms in iter_nerf_transforms_paths(root))
    for root in _step4_dataset_roots(scene, settings):
        roots.extend((root, transforms, None) for transforms in iter_nerf_transforms_paths(root))
    roots.extend((root, transforms, None) for root, transforms in _shallow_transforms_dataset_roots(scene_output_dir(scene)))
    return tuple(_dedupe_transform_records(roots))


def _colmap_dataset_roots(
    scene: Path,
    settings: dict,
    dataset_records: list[ArtifactRecord],
    sfm_records: list[ArtifactRecord],
) -> tuple[tuple[Path, Path, str, ArtifactRecord | None], ...]:
    roots: list[tuple[Path, ArtifactRecord | None, str]] = []
    for record in dataset_records:
        if record.kind not in {DATASET_KIND_COLMAP_DATASET, DATASET_KIND_LICHTFELD_COLMAP} or record.status != "ready":
            continue
        roots.append((artifact_root_path(scene, record), record, _dataset_label(artifact_root_path(scene, record), None, record)))
    for record in sfm_records:
        if record.kind != SFM_KIND_COLMAP_SPARSE or record.status != "ready":
            continue
        root = artifact_root_path(scene, record)
        roots.append((_dataset_root_from_sparse(root), record, _sfm_label("COLMAP SfM", record)))

    output = scene_output_dir(scene)
    roots.extend(
        [
            (output / "colmap_rig", None, "COLMAP SfM"),
            (output / "metashape_colmap", None, "Metashape COLMAP Dataset"),
            (output / "realityscan" / "lfs_colmap", None, "RealityScan LichtFeld COLMAP"),
            (output / "realityscan" / "lfs_colmap_undistorted", None, "RealityScan LichtFeld COLMAP"),
        ]
    )
    for root in _shallow_colmap_dataset_dirs(output):
        roots.append((root, None, _shallow_colmap_label(output, root)))

    result: list[tuple[Path, Path, str, ArtifactRecord | None]] = []
    seen: set[str] = set()
    for root, record, label in roots:
        sparse = _resolve_colmap_model(root / "sparse") or _resolve_colmap_model(root)
        if sparse is None:
            continue
        key = _path_key(sparse)
        if key in seen:
            continue
        seen.add(key)
        result.append((_dataset_root_from_sparse(sparse), sparse, label, record))
    return tuple(result)


def _spheresfm_sparse_models(scene: Path, records: list[ArtifactRecord]) -> tuple[tuple[Path, ArtifactRecord | None], ...]:
    sparse_models: list[tuple[Path, ArtifactRecord | None]] = []
    for record in records:
        if record.kind != SFM_KIND_SPHERESFM_SPARSE or record.status != "ready":
            continue
        sparse = _resolve_colmap_model(artifact_root_path(scene, record))
        if sparse is not None:
            sparse_models.append((sparse, record))
    fallback = _resolve_colmap_model(scene_output_dir(scene) / "spheresfm" / "sparse")
    if fallback is not None:
        sparse_models.append((fallback, None))
    result: list[tuple[Path, ArtifactRecord | None]] = []
    seen: set[str] = set()
    for sparse, record in sparse_models:
        key = _path_key(sparse)
        if key in seen:
            continue
        seen.add(key)
        result.append((sparse, record))
    return tuple(result)


def _realityscan_inputs(
    scene: Path,
    records: list[ArtifactRecord],
) -> tuple[tuple[Path, Path | None, Path | None, Path | None, str], ...]:
    result: list[tuple[Path, Path | None, Path | None, Path | None, str]] = []
    for record in records:
        if record.kind != SFM_KIND_REALITYSCAN_CSV_PLY or record.status != "ready":
            continue
        csv = _existing_file(_artifact_file_path(scene, record, "csv"))
        if csv is None:
            continue
        result.append(
            (
                csv,
                _existing_file(_artifact_file_path(scene, record, "ply")),
                _existing_dir(_artifact_file_path(scene, record, "images_dir")),
                _existing_dir(_artifact_file_path(scene, record, "masks_dir")),
                _sfm_label("RealityScan CSV/PLY", record),
            )
        )
    return tuple(result)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not isinstance(path, Path):
            continue
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _dedupe_transform_records(
    items: Iterable[tuple[Path, Path, ArtifactRecord | None]],
) -> list[tuple[Path, Path, ArtifactRecord | None]]:
    result: list[tuple[Path, Path, ArtifactRecord | None]] = []
    seen: set[str] = set()
    for root, transforms, record in items:
        key = _path_key(transforms)
        if key in seen:
            continue
        seen.add(key)
        result.append((root, transforms, record))
    return result


def _dedupe_candidates(candidates: Iterable[ScenePreviewCandidate]) -> list[ScenePreviewCandidate]:
    result: list[ScenePreviewCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.kind, _path_key(candidate.path))
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path.absolute()).casefold()


def _same_path(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a.absolute() == b.absolute()


def _existing_file(path: Path | None) -> Path | None:
    return path if path is not None and path.is_file() else None


def _existing_dir(path: Path | None) -> Path | None:
    return path if path is not None and path.is_dir() else None


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _first_existing_dir(*paths: Path | None) -> Path | None:
    return next((path for path in paths if path is not None and path.is_dir()), None)


def _artifact_file_path(scene: Path, record: ArtifactRecord, key: str) -> Path | None:
    raw = str(record.files.get(key) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else scene / path


def _dataset_label(
    root: Path,
    current_output: Path | None,
    record: ArtifactRecord | None,
    *,
    transforms_json: Path | None = None,
) -> str:
    if record is not None:
        if record.kind == DATASET_KIND_REALITYSCAN_REALIGN_INPUT or _is_realityscan_realign_record(record):
            return f"RealityScan realign input ({record.id})"
        if record.kind == DATASET_KIND_NERF_JSON_PLY:
            profile = _transforms_profile_label(transforms_json)
            suffix = "JSON/PLY" if _transforms_pointcloud_path(root, transforms_json) is not None else "JSON"
            prefix = f"NeRF {profile} " if profile else "NeRF "
            return f"Dataset: {prefix}{suffix} ({record.id})"
        if record.kind == DATASET_KIND_LICHTFELD_COLMAP:
            return f"Dataset: LichtFeld COLMAP ({record.id})"
        if record.kind == DATASET_KIND_COLMAP_DATASET:
            return f"Dataset: COLMAP ({record.id})"
        return f"Dataset: {record.kind} ({record.id})"
    return "Step 4 output" if _same_path(root, current_output) else f"Step 4 output ({root.name})"


def _sfm_label(fallback: str, record: ArtifactRecord | None) -> str:
    return fallback if record is None else f"{fallback} ({record.id})"


def _is_realityscan_realign_record(record: ArtifactRecord) -> bool:
    settings = record.settings if isinstance(record.settings, dict) else {}
    realityscan = settings.get("realityscan") if isinstance(settings.get("realityscan"), dict) else {}
    if realityscan.get("enabled") is True:
        return True
    output_files = settings.get("output_files") if isinstance(settings.get("output_files"), dict) else {}
    return str(settings.get("effective_profile") or "").strip().lower() == "realityscan" and not str(
        output_files.get("pointcloud") or ""
    ).strip()


def _transforms_pointcloud_path(root: Path, transforms_json: Path | None = None) -> Path | None:
    return find_nerf_pointcloud_path(root, transforms_json=transforms_json)


def _transforms_profile_label(transforms_json: Path | None) -> str:
    if transforms_json is None:
        return ""
    profile = profile_from_transforms_name(transforms_json)
    return profile.capitalize() if profile else ""


def _artifact_metadata(record: ArtifactRecord | None) -> dict:
    if record is None:
        return {}
    return dict(record.metadata)


def _manifest_dir(scene: Path, dataset_root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = dataset_root / path
    if candidate.exists():
        return candidate
    return scene / path


def _colmap_images_dir(scene: Path, root: Path, sparse: Path, manifest: dict) -> Path | None:
    return _first_existing_dir(
        _manifest_dir(scene, root, manifest.get("images_dir")),
        root / "images",
        sparse.parent.parent / "images" if sparse.name == "0" else sparse.parent / "images",
    )


def _colmap_masks_dir(
    scene: Path,
    root: Path,
    sparse: Path,
    manifest: dict,
    settings: dict,
    record: ArtifactRecord | None,
) -> Path | None:
    record_mask = _artifact_file_path(scene, record, "masks_dir") if record is not None else None
    return _first_existing_dir(
        _manifest_dir(scene, root, manifest.get("masks_dir")),
        record_mask,
        _settings_path_or_none(scene, _settings_dict(settings, "colmap_rig").get("masks_dir")),
        root / "masks",
        sparse.parent.parent / "masks" if sparse.name == "0" else sparse.parent / "masks",
    )


def _colmap_display_transform(
    scene: Path,
    root: Path,
    sparse: Path,
    manifest: dict,
    record: ArtifactRecord | None,
) -> ScenePreviewDisplayTransform | None:
    if _is_realityscan_lfs_colmap(scene, root, manifest, record):
        return realityscan_lfs_colmap_display_transform()
    if _same_path(root, scene_output_dir(scene) / "realityscan"):
        return realityscan_colmap_export_display_transform()
    return None


def _colmap_uses_app_camera_axes(
    scene: Path,
    root: Path,
    manifest: dict,
    record: ArtifactRecord | None,
) -> bool:
    # Scene preview reads every COLMAP candidate with COLMAP/OpenCV camera axes.
    # Dataset-specific display transforms below handle world-axis differences.
    return False


def _is_realityscan_lfs_colmap(
    scene: Path,
    root: Path,
    manifest: dict,
    record: ArtifactRecord | None,
) -> bool:
    if record is not None and record.kind == DATASET_KIND_LICHTFELD_COLMAP:
        return True
    manifest_kind = str(manifest.get("kind") or "").strip().lower()
    manifest_source = str(manifest.get("source_kind") or "").strip().lower()
    if manifest_kind == "lichtfeld_colmap" and manifest_source == "realityscan_csv_ply":
        return True

    output = scene_output_dir(scene)
    return _same_path(root, output / "realityscan" / "lfs_colmap") or _same_path(
        root,
        output / "realityscan" / "lfs_colmap_undistorted",
    )


def _dataset_root_from_sparse(sparse: Path) -> Path:
    if sparse.name == "0" and sparse.parent.name.lower() == "sparse":
        return sparse.parent.parent
    if sparse.name.lower() == "sparse":
        return sparse.parent
    return sparse


def _shallow_dataset_dirs(output: Path, *, marker: str) -> tuple[Path, ...]:
    if not output.is_dir():
        return ()
    roots: list[Path] = []
    for child in output.iterdir():
        if _is_preview_scan_excluded(child):
            continue
        if child.is_dir() and (child / marker).is_file():
            roots.append(child)
        if child.is_dir():
            for grandchild in child.iterdir():
                if _is_preview_scan_excluded(grandchild):
                    continue
                if grandchild.is_dir() and (grandchild / marker).is_file():
                    roots.append(grandchild)
    return tuple(_dedupe_paths(roots))


def _shallow_transforms_dataset_roots(output: Path) -> tuple[tuple[Path, Path], ...]:
    if not output.is_dir():
        return ()
    roots: list[tuple[Path, Path]] = []

    def add_root(root: Path) -> None:
        roots.extend((root, transforms) for transforms in iter_nerf_transforms_paths(root))

    for child in output.iterdir():
        if _is_preview_scan_excluded(child):
            continue
        if child.is_dir():
            add_root(child)
            for grandchild in child.iterdir():
                if _is_preview_scan_excluded(grandchild):
                    continue
                if grandchild.is_dir():
                    add_root(grandchild)
    result: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for root, transforms in roots:
        key = _path_key(transforms)
        if key in seen:
            continue
        seen.add(key)
        result.append((root, transforms))
    return tuple(result)


def _shallow_colmap_dataset_dirs(output: Path) -> tuple[Path, ...]:
    if not output.is_dir():
        return ()
    roots: list[Path] = []
    for child in output.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() == "spheresfm" or _is_preview_scan_excluded(child):
            continue
        if _resolve_colmap_model(child / "sparse") is not None or _resolve_colmap_model(child) is not None:
            roots.append(child)
        for grandchild in child.iterdir():
            if _is_preview_scan_excluded(grandchild):
                continue
            if grandchild.is_dir() and (
                _resolve_colmap_model(grandchild / "sparse") is not None
                or _resolve_colmap_model(grandchild) is not None
            ):
                roots.append(grandchild)
    return tuple(_dedupe_paths(roots))


def _shallow_colmap_label(output: Path, root: Path) -> str:
    if _same_path(root, output / "realityscan"):
        return "RealityScan COLMAP Export"
    return f"COLMAP Dataset ({root.name})"


def _is_preview_scan_excluded(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith("apriltag_scale_backup_") or name.startswith("backup")


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
