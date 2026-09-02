"""Small, dependency-free, versioned domain models.

The application deliberately keeps these contracts free from Qt, OpenCV, and
PyTorch so they can be used by the CLI, migration tools, tests, and future
plugins without importing the desktop stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

_INTERPOLATIONS = frozenset({"nearest", "linear", "cubic", "lanczos"})
_JOB_STATUSES = frozenset({"queued", "running", "completed", "failed", "canceled", "interrupted"})
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _safe_identifier(value: str, name: str) -> str:
    value = str(value).strip()
    if not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"{name} must contain only letters, numbers, '.', '_' or '-'")
    return value


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """Perspective projection definition shared by every export path."""

    SCHEMA_VERSION: ClassVar[int] = 1

    id: str
    name: str
    enabled: bool = True
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float | None = None
    width: int = 1920
    height: int = 1920
    interpolation: str = "cubic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _safe_identifier(self.id, "view id"))
        name = self.name.strip()
        if not name or len(name) > 128:
            raise ValueError("view name must contain 1-128 characters")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "yaw_deg", ((_finite(self.yaw_deg, "yaw_deg") + 180.0) % 360.0) - 180.0)
        pitch = _finite(self.pitch_deg, "pitch_deg")
        if not -90.0 <= pitch <= 90.0:
            raise ValueError("pitch_deg must be between -90 and 90")
        object.__setattr__(self, "pitch_deg", pitch)
        object.__setattr__(self, "roll_deg", ((_finite(self.roll_deg, "roll_deg") + 180.0) % 360.0) - 180.0)
        hfov = _finite(self.hfov_deg, "hfov_deg")
        if not 1.0 <= hfov < 180.0:
            raise ValueError("hfov_deg must be in [1, 180)")
        object.__setattr__(self, "hfov_deg", hfov)
        if self.vfov_deg is not None:
            vfov = _finite(self.vfov_deg, "vfov_deg")
            if not 1.0 <= vfov < 180.0:
                raise ValueError("vfov_deg must be in [1, 180)")
            object.__setattr__(self, "vfov_deg", vfov)
        if not 16 <= int(self.width) <= 32768 or not 16 <= int(self.height) <= 32768:
            raise ValueError("view dimensions must be between 16 and 32768 pixels")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        interpolation = self.interpolation.strip().lower()
        if interpolation not in _INTERPOLATIONS:
            raise ValueError(f"unsupported interpolation: {interpolation}")
        object.__setattr__(self, "interpolation", interpolation)

    @property
    def effective_vfov_deg(self) -> float:
        if self.vfov_deg is not None:
            return self.vfov_deg
        horizontal_tan = math.tan(math.radians(self.hfov_deg) / 2.0)
        vertical_tan = horizontal_tan * self.height / self.width
        return math.degrees(2.0 * math.atan(vertical_tan))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, **asdict(self)}

    def to_legacy_view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "yaw": self.yaw_deg,
            "pitch": self.pitch_deg,
            "roll": self.roll_deg,
            "fov": self.hfov_deg,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, index: int = 0) -> ViewSpec:
        name = str(value.get("name") or f"view_{index + 1:02d}")
        view_id = str(value.get("id") or re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or f"view_{index + 1:02d}")
        fov = value.get("hfov_deg", value.get("fov", 90.0))
        if value.get("vfov_deg") is not None:
            vfov = float(value["vfov_deg"])
        elif "hfov_deg" not in value and value.get("fov") is not None:
            # Legacy viewpoint profiles stored one FOV value. Preserve their
            # original square-FOV meaning even when a new rectangular output
            # size is selected later.
            vfov = float(fov)
        else:
            vfov = None
        return cls(
            id=view_id,
            name=name,
            enabled=bool(value.get("enabled", True)),
            yaw_deg=float(value.get("yaw_deg", value.get("yaw", 0.0))),
            pitch_deg=float(value.get("pitch_deg", value.get("pitch", 0.0))),
            roll_deg=float(value.get("roll_deg", value.get("roll", 0.0))),
            hfov_deg=float(fov),
            vfov_deg=vfov,
            width=int(value.get("width", value.get("output_width", 1920))),
            height=int(value.get("height", value.get("output_height", value.get("width", 1920)))),
            interpolation=str(value.get("interpolation", value.get("interp", "cubic"))),
        )


def cubemap_view_specs(size: int = 1920) -> list[ViewSpec]:
    axes = (
        ("front", 0.0, 0.0),
        ("right", 90.0, 0.0),
        ("back", 180.0, 0.0),
        ("left", -90.0, 0.0),
        ("up", 0.0, -90.0),
        ("down", 0.0, 90.0),
    )
    return [ViewSpec(id=name, name=name.title(), yaw_deg=yaw, pitch_deg=pitch, width=size, height=size) for name, yaw, pitch in axes]


def grid_view_specs(
    *,
    yaw_count: int = 8,
    pitches: tuple[float, ...] = (0.0,),
    hfov_deg: float = 100.0,
    size: int = 1920,
) -> list[ViewSpec]:
    if not 1 <= yaw_count <= 64:
        raise ValueError("yaw_count must be between 1 and 64")
    result: list[ViewSpec] = []
    for pitch_index, pitch in enumerate(pitches):
        for yaw_index in range(yaw_count):
            yaw = ((yaw_index * 360.0 / yaw_count + 180.0) % 360.0) - 180.0
            token = f"p{pitch:g}_y{yaw:g}".replace("-", "m").replace(".", "d")
            result.append(
                ViewSpec(
                    id=f"grid_{pitch_index}_{yaw_index}_{token}",
                    name=f"Pitch {pitch:g} / Yaw {yaw:g}",
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    hfov_deg=hfov_deg,
                    width=size,
                    height=size,
                )
            )
    return result


@dataclass(slots=True)
class ProjectManifest:
    SCHEMA_VERSION: ClassVar[int] = 2

    project_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Untitled Scene"
    application: str = "360gs-studio"
    application_version: str = "0.1.0"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    sources: list[dict[str, Any]] = field(default_factory=list)
    projection_types: dict[str, str] = field(default_factory=dict)
    scene_paths: dict[str, str] = field(default_factory=lambda: {"images": "images", "masks": "masks", "output": "output"})
    stage_configuration: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProjectManifest:
        version = int(value.get("schema_version", value.get("version", 0)) or 0)
        if version != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema version: {version}")
        allowed = {name for name in cls.__dataclass_fields__}
        return cls(**{key: item for key, item in value.items() if key in allowed})


@dataclass(slots=True)
class JobSpec:
    SCHEMA_VERSION: ClassVar[int] = 1

    job_type: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    dependency_ids: list[str] = field(default_factory=list)
    input_signatures: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    progress_current: int = 0
    progress_total: int = 0
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.job_type = _safe_identifier(self.job_type, "job_type")
        self.job_id = _safe_identifier(self.job_id, "job_id")
        if self.status not in _JOB_STATUSES:
            raise ValueError(f"unsupported job status: {self.status}")

    @property
    def configuration_hash(self) -> str:
        body = json.dumps(self.configuration, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, "configuration_hash": self.configuration_hash, **asdict(self)}


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    SCHEMA_VERSION: ClassVar[int] = 1

    component_id: str
    version: str
    display_name: str
    download_url: str = ""
    sha256: str = ""
    license_id: str = ""
    platforms: tuple[str, ...] = ("windows-x86_64",)
    executable: str = ""
    capability_probe: tuple[str, ...] = ()
    optional: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _safe_identifier(self.component_id, "component_id"))
        if self.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("component sha256 must be a 64-character hexadecimal digest")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["platforms"] = list(self.platforms)
        data["capability_probe"] = list(self.capability_probe)
        return {"schema_version": self.SCHEMA_VERSION, **data}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ComponentManifest:
        return cls(
            component_id=str(value["component_id"]),
            version=str(value["version"]),
            display_name=str(value.get("display_name") or value["component_id"]),
            download_url=str(value.get("download_url") or ""),
            sha256=str(value.get("sha256") or ""),
            license_id=str(value.get("license_id") or ""),
            platforms=tuple(str(item) for item in value.get("platforms", ("windows-x86_64",))),
            executable=str(value.get("executable") or ""),
            capability_probe=tuple(str(item) for item in value.get("capability_probe", ())),
            optional=bool(value.get("optional", True)),
        )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
