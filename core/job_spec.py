from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.scene_layout import jobs_dir
from core.scene_project import load_json, utc_now_iso, write_json

JOB_SPEC_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class WorkflowJobSpec:
    id: str
    kind: str
    created_at: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": JOB_SPEC_SCHEMA_VERSION,
            "id": self.id,
            "kind": self.kind,
            "created_at": self.created_at,
            "params": dict(self.params),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> WorkflowJobSpec:
        params = payload.get("params")
        return cls(
            id=str(payload.get("id") or ""),
            kind=str(payload.get("kind") or ""),
            created_at=str(payload.get("created_at") or ""),
            params=dict(params) if isinstance(params, dict) else {},
        )


def new_job_spec(kind: str, params: dict[str, Any] | None = None) -> WorkflowJobSpec:
    return WorkflowJobSpec(
        id=f"job_{uuid.uuid4().hex[:12]}",
        kind=kind,
        created_at=utc_now_iso(),
        params=dict(params or {}),
    )


def write_job_spec(scene_dir: str | Path, spec: WorkflowJobSpec) -> Path:
    path = jobs_dir(Path(scene_dir)) / f"{spec.id}.json"
    write_json(path, spec.to_json())
    return path


def read_job_spec(path: str | Path) -> WorkflowJobSpec:
    return WorkflowJobSpec.from_json(load_json(Path(path)))
