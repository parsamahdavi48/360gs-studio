from __future__ import annotations

from pathlib import Path
from typing import Any

from core.artifact_registry import ArtifactRecord, load_artifacts, make_artifact_record, upsert_artifact
from core.nerf_dataset_paths import find_nerf_pointcloud_path, find_nerf_transforms_path, load_json_object

SFM_KIND_METASHAPE_XML_PLY = "metashape_xml_ply"
SFM_KIND_COLMAP_SPARSE = "colmap_sparse"
SFM_KIND_SPHERESFM_SPARSE = "spheresfm_sparse"
SFM_KIND_REALITYSCAN_CSV_PLY = "realityscan_csv_ply"

DATASET_KIND_NERF_JSON_PLY = "nerf_json_ply"
DATASET_KIND_COLMAP_DATASET = "colmap_dataset"
DATASET_KIND_LICHTFELD_COLMAP = "lichtfeld_colmap"
DATASET_KIND_REALITYSCAN_REALIGN_INPUT = "realityscan_realign_input"

SFM_ARTIFACT_KINDS = {
    SFM_KIND_METASHAPE_XML_PLY,
    SFM_KIND_COLMAP_SPARSE,
    SFM_KIND_SPHERESFM_SPARSE,
    SFM_KIND_REALITYSCAN_CSV_PLY,
}
DATASET_ARTIFACT_KINDS = {
    DATASET_KIND_NERF_JSON_PLY,
    DATASET_KIND_COLMAP_DATASET,
    DATASET_KIND_LICHTFELD_COLMAP,
    DATASET_KIND_REALITYSCAN_REALIGN_INPUT,
}


def detect_dataset_kind(root: str | Path, *, preferred_kind: str = "") -> str:
    dataset_root = Path(root)
    if preferred_kind:
        return preferred_kind
    if _has_colmap_sparse(dataset_root):
        return DATASET_KIND_LICHTFELD_COLMAP if "lfs_colmap" in dataset_root.name.lower() else DATASET_KIND_COLMAP_DATASET
    if find_nerf_transforms_path(dataset_root) is not None:
        return DATASET_KIND_NERF_JSON_PLY
    return ""


def register_dataset_artifact(
    scene_dir: str | Path,
    *,
    artifact_id: str,
    root: str | Path,
    kind: str = "",
    source_artifact_id: str = "",
    source_inputs: list[str | Path] | tuple[str | Path, ...] | None = None,
    settings: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
) -> ArtifactRecord | None:
    detected_kind = detect_dataset_kind(root, preferred_kind=kind)
    if not detected_kind:
        return None
    _require_known_kind(detected_kind, DATASET_ARTIFACT_KINDS, group="dataset")
    metadata_payload = _dataset_metadata(Path(root), detected_kind)
    metadata_payload.update(dict(metadata or {}))
    record = make_artifact_record(
        scene_dir,
        artifact_id=artifact_id,
        kind=detected_kind,
        root=root,
        files=_dataset_files(Path(root)),
        source_artifact_id=source_artifact_id,
        source_inputs=source_inputs,
        producer=detected_kind,
        settings=settings,
        metadata=metadata_payload,
        warnings=warnings,
    )
    return upsert_artifact(scene_dir, "dataset", record)


def register_sfm_artifact(
    scene_dir: str | Path,
    *,
    artifact_id: str,
    kind: str,
    root: str | Path,
    files: dict[str, str | Path] | None = None,
    source_inputs: list[str | Path] | tuple[str | Path, ...] | None = None,
    settings: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
) -> ArtifactRecord:
    _require_known_kind(kind, SFM_ARTIFACT_KINDS, group="sfm")
    record = make_artifact_record(
        scene_dir,
        artifact_id=artifact_id,
        kind=kind,
        root=root,
        files=_existing_files(files or {}),
        source_inputs=source_inputs,
        producer=kind,
        settings=settings,
        metadata=metadata,
        warnings=warnings,
    )
    return upsert_artifact(scene_dir, "sfm", record)


def artifact_root_path(scene_dir: str | Path, record: ArtifactRecord) -> Path:
    root = Path(record.root)
    return root if root.is_absolute() else Path(scene_dir) / root


def latest_dataset_artifact(
    scene_dir: str | Path,
    *,
    accepted_kinds: set[str] | None = None,
) -> ArtifactRecord | None:
    records = load_artifacts(scene_dir, "dataset")
    if accepted_kinds:
        records = [record for record in records if record.kind in accepted_kinds]
    records = [record for record in records if artifact_root_path(scene_dir, record).exists()]
    if not records:
        return None
    return sorted(records, key=lambda record: record.created_at or record.id)[-1]


def latest_dataset_root(
    scene_dir: str | Path,
    *,
    accepted_kinds: set[str] | None = None,
) -> Path | None:
    record = latest_dataset_artifact(scene_dir, accepted_kinds=accepted_kinds)
    return artifact_root_path(scene_dir, record) if record is not None else None


def _dataset_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    transforms = find_nerf_transforms_path(root)
    if transforms is not None:
        files["transforms_json"] = transforms
        pointcloud = find_nerf_pointcloud_path(root, transforms_json=transforms)
        if pointcloud is not None:
            files["pointcloud_file"] = pointcloud
        raw_metashape_pointcloud = _declared_raw_metashape_pointcloud(transforms)
        if raw_metashape_pointcloud is not None:
            files["raw_metashape_pointcloud_file"] = raw_metashape_pointcloud
    for key, rel in (
        ("images_dir", "images"),
        ("masks_dir", "masks"),
        ("colmap_sparse_dir", "sparse/0"),
        ("colmap_sparse_root", "sparse"),
    ):
        candidate = root / rel
        if candidate.exists():
            files[key] = candidate
    return files


def _dataset_metadata(root: Path, kind: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
    }
    transforms = find_nerf_transforms_path(root)
    if transforms is not None:
        metadata["transforms_json"] = transforms.relative_to(root).as_posix()
        pointcloud = find_nerf_pointcloud_path(root, transforms_json=transforms)
        if pointcloud is not None:
            metadata["pointcloud"] = pointcloud.relative_to(root).as_posix()
    for key, rel in (
        ("images_dir", "images"),
        ("masks_dir", "masks"),
        ("sparse_dir", "sparse/0"),
        ("sparse_root", "sparse"),
    ):
        candidate = root / rel
        if candidate.exists():
            metadata[key] = rel
    return metadata


def _declared_raw_metashape_pointcloud(transforms: Path) -> Path | None:
    if not transforms.is_file():
        return None
    data = load_json_object(transforms)
    source = data.get("source") if isinstance(data, dict) else None
    if not isinstance(source, dict):
        return None
    raw = str(source.get("raw_metashape_pointcloud_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    candidate = path if path.is_absolute() else transforms.parent / path
    return candidate if candidate.is_file() else None


def _require_known_kind(kind: str, accepted: set[str], *, group: str) -> None:
    if kind not in accepted:
        raise ValueError(f"Unknown {group} artifact kind: {kind}")


def _existing_files(files: dict[str, str | Path]) -> dict[str, str | Path]:
    result: dict[str, str | Path] = {}
    for key, value in files.items():
        if not str(value).strip():
            continue
        path = Path(value)
        if path.exists():
            result[key] = value
    return result


def _has_colmap_sparse(root: Path) -> bool:
    sparse = root / "sparse" / "0"
    return all((sparse / name).is_file() for name in ("cameras.txt", "images.txt", "points3D.txt"))
