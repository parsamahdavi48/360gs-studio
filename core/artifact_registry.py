from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.scene_layout import dataset_artifacts_path, sfm_artifacts_path
from core.scene_project import load_json, scene_relative, utc_now_iso, write_json

ARTIFACT_SCHEMA_VERSION = 1
ArtifactGroup = Literal["sfm", "dataset"]


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    id: str
    kind: str
    root: str
    created_at: str
    status: str = "ready"
    producer: str = ""
    files: dict[str, str] = field(default_factory=dict)
    source_artifact_id: str = ""
    source_inputs: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "id": self.id,
            "kind": self.kind,
            "created_at": self.created_at,
            "status": self.status,
            "producer": self.producer,
            "root": self.root,
            "files": dict(self.files),
            "source_artifact_id": self.source_artifact_id,
            "source_inputs": list(self.source_inputs),
            "settings": dict(self.settings),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ArtifactRecord:
        files = payload.get("files")
        source_inputs = payload.get("source_inputs")
        settings = payload.get("settings")
        warnings = payload.get("warnings")
        return cls(
            id=str(payload.get("id") or ""),
            kind=str(payload.get("kind") or ""),
            created_at=str(payload.get("created_at") or ""),
            status=str(payload.get("status") or "ready"),
            producer=str(payload.get("producer") or ""),
            root=str(payload.get("root") or ""),
            files={str(k): str(v) for k, v in files.items()} if isinstance(files, dict) else {},
            source_artifact_id=str(payload.get("source_artifact_id") or ""),
            source_inputs=tuple(str(item) for item in source_inputs) if isinstance(source_inputs, list) else (),
            settings=dict(settings) if isinstance(settings, dict) else {},
            warnings=tuple(str(item) for item in warnings) if isinstance(warnings, list) else (),
        )


def artifact_path(scene_dir: Path, group: ArtifactGroup) -> Path:
    return sfm_artifacts_path(scene_dir) if group == "sfm" else dataset_artifacts_path(scene_dir)


def load_artifacts(scene_dir: str | Path, group: ArtifactGroup) -> list[ArtifactRecord]:
    scene = Path(scene_dir)
    data = load_json(artifact_path(scene, group), {"schema_version": ARTIFACT_SCHEMA_VERSION, "artifacts": []})
    raw_items = data.get("artifacts")
    if not isinstance(raw_items, list):
        return []
    records: list[ArtifactRecord] = []
    for item in raw_items:
        if isinstance(item, dict):
            record = ArtifactRecord.from_json(item)
            if record.id:
                records.append(record)
    return records


def save_artifacts(scene_dir: str | Path, group: ArtifactGroup, records: list[ArtifactRecord]) -> None:
    scene = Path(scene_dir)
    write_json(
        artifact_path(scene, group),
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifacts": [record.to_json() for record in records],
        },
    )


def upsert_artifact(scene_dir: str | Path, group: ArtifactGroup, record: ArtifactRecord) -> ArtifactRecord:
    records = load_artifacts(scene_dir, group)
    kept = [item for item in records if item.id != record.id]
    kept.append(record)
    save_artifacts(scene_dir, group, kept)
    return record


def make_artifact_record(
    scene_dir: str | Path,
    *,
    artifact_id: str,
    kind: str,
    root: str | Path,
    files: dict[str, str | Path] | None = None,
    source_artifact_id: str = "",
    source_inputs: list[str | Path] | tuple[str | Path, ...] | None = None,
    status: str = "ready",
    producer: str = "",
    settings: dict[str, Any] | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
) -> ArtifactRecord:
    scene = Path(scene_dir)
    root_text = _portable_path(scene, root)
    file_payload = {
        name: _portable_path(scene, value)
        for name, value in (files or {}).items()
    }
    return ArtifactRecord(
        id=artifact_id,
        kind=kind,
        created_at=utc_now_iso(),
        status=status,
        producer=producer,
        root=root_text,
        files=file_payload,
        source_artifact_id=source_artifact_id,
        source_inputs=tuple(_portable_path(scene, value) for value in (source_inputs or ())),
        settings=dict(settings or {}),
        warnings=tuple(warnings or ()),
    )


def _portable_path(scene_dir: Path, value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return scene_relative(scene_dir, path).replace("\\", "/")
    return str(path).replace("\\", "/")
