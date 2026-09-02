"""360GS project storage and non-destructive legacy migration."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gs360studio.domain.models import ProjectManifest, atomic_write_json, utc_now
from gs360studio.version import __version__

APP_DIR_NAME = "_360gs"
PROJECT_FILE_NAME = "project.json"
LEGACY_APP_DIR_NAME = "_stechdrive"


@dataclass(frozen=True, slots=True)
class MigrationReport:
    scene_dir: Path
    source_path: Path | None
    target_path: Path
    created: bool
    migrated: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_dir": str(self.scene_dir),
            "source_path": str(self.source_path) if self.source_path else None,
            "target_path": str(self.target_path),
            "created": self.created,
            "migrated": self.migrated,
            "warnings": list(self.warnings),
        }


def project_path(scene_dir: str | Path) -> Path:
    return Path(scene_dir) / APP_DIR_NAME / PROJECT_FILE_NAME


def legacy_project_path(scene_dir: str | Path) -> Path:
    return Path(scene_dir) / LEGACY_APP_DIR_NAME / PROJECT_FILE_NAME


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read project manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"project manifest must contain a JSON object: {path}")
    return payload


def _legacy_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate_legacy_project(scene_dir: str | Path, *, write: bool = True) -> tuple[ProjectManifest, MigrationReport]:
    """Create schema v2 metadata without modifying the schema-v1 source."""

    scene = Path(scene_dir).resolve()
    target = project_path(scene)
    legacy = legacy_project_path(scene)
    if target.exists():
        manifest = ProjectManifest.from_dict(_read_mapping(target))
        return manifest, MigrationReport(scene, legacy if legacy.exists() else None, target, False, False)

    warnings: list[str] = []
    legacy_payload = _read_mapping(legacy) if legacy.exists() else {}
    migrated = bool(legacy_payload)
    if legacy.exists() and not legacy_payload:
        warnings.append("Legacy metadata exists but contains no usable project fields.")

    reserved = {"version", "schema_version", "app", "created_at", "updated_at"}
    stage_configuration = {key: value for key, value in legacy_payload.items() if key not in reserved}
    created_at = str(legacy_payload.get("created_at") or utc_now())
    extensions: dict[str, Any] = {}
    if legacy.exists():
        extensions["legacy_migration"] = {
            "source": str(legacy.relative_to(scene)).replace("\\", "/"),
            "source_schema_version": int(legacy_payload.get("version", 1) or 1),
            "source_sha256": _legacy_digest(legacy),
            "migrated_at": utc_now(),
            "source_preserved": True,
        }

    manifest = ProjectManifest(
        name=scene.name,
        application_version=__version__,
        created_at=created_at,
        stage_configuration=stage_configuration,
        extensions=extensions,
    )
    if write:
        atomic_write_json(target, manifest.to_dict())
        atomic_write_json(target.parent / "migration_report.json", MigrationReport(scene, legacy if legacy.exists() else None, target, True, migrated, tuple(warnings)).to_dict())
    return manifest, MigrationReport(scene, legacy if legacy.exists() else None, target, True, migrated, tuple(warnings))


def load_project(scene_dir: str | Path, *, migrate: bool = True) -> tuple[ProjectManifest, MigrationReport | None]:
    target = project_path(scene_dir)
    if target.exists():
        return ProjectManifest.from_dict(_read_mapping(target)), None
    if not migrate:
        raise FileNotFoundError(target)
    return migrate_legacy_project(scene_dir)


def save_project(scene_dir: str | Path, manifest: ProjectManifest) -> Path:
    manifest.updated_at = utc_now()
    manifest.application_version = __version__
    target = project_path(scene_dir)
    atomic_write_json(target, manifest.to_dict())
    return target


def rollback_metadata_migration(scene_dir: str | Path, *, backup_dir: str | Path | None = None) -> Path | None:
    """Remove only 360GS-owned metadata, optionally moving it to a backup."""

    app_dir = project_path(scene_dir).parent
    if not app_dir.exists():
        return None
    if app_dir.name != APP_DIR_NAME:
        raise ValueError("refusing to remove a directory that is not owned by 360GS Studio")
    if backup_dir is not None:
        destination = Path(backup_dir).resolve() / app_dir.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(app_dir), str(destination))
        return destination
    shutil.rmtree(app_dir)
    return None
