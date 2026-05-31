from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.scene_layout import normal_camera_defaults_path
from core.scene_project import load_json, write_json

NORMAL_CAMERA_DEFAULTS_SCHEMA_VERSION = 1
COLMAP_NORMAL_CAMERA_MODELS = ("SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV")
COLMAP_CAMERA_PARAM_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
}

# This metadata contract is intentionally kept out of the normal Step 4 UI.
# In practice, most users of this app do not have reliable per-source,
# per-resolution camera calibration for smartphone/JPEG/video frames, and a
# visible free-form intrinsics field made the workflow feel more fragile than
# it is. Keep this module for source manifests, scene import, tests, and a
# future calibration-file import path. If a GUI returns, it should be
# group-aware and import-driven, not a scene-wide manual params textbox.


@dataclass(frozen=True, slots=True)
class NormalCameraDefault:
    camera_model: str = ""
    camera_params: tuple[float, ...] = ()
    camera_source: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.camera_model)


@dataclass(frozen=True, slots=True)
class NormalCameraGroupDefault:
    source_kind: str
    source_id: str
    width: int
    height: int
    camera_model: str = ""
    camera_params: tuple[float, ...] = ()
    camera_source: str = ""

    @property
    def key(self) -> tuple[str, str, int, int]:
        return normal_camera_group_key(self.source_kind, self.source_id, self.width, self.height)

    @property
    def camera(self) -> NormalCameraDefault:
        return NormalCameraDefault(
            camera_model=self.camera_model,
            camera_params=self.camera_params,
            camera_source=self.camera_source,
        )


@dataclass(frozen=True, slots=True)
class NormalCameraDefaults:
    default: NormalCameraDefault
    groups: tuple[NormalCameraGroupDefault, ...] = ()


def normalize_camera_model(value: object) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def parse_camera_params(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.replace(",", " ").replace(";", " ").split()]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raise ValueError("camera parameters must be text or a list")

    parsed: list[float] = []
    for raw in raw_values:
        if raw == "":
            continue
        try:
            parsed.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid camera parameter: {raw}") from exc
    return tuple(parsed)


def validate_camera_params_for_model(model: object, params: tuple[float, ...]) -> None:
    normalized = normalize_camera_model(model)
    if not normalized or not params:
        return
    expected = COLMAP_CAMERA_PARAM_COUNTS.get(normalized)
    if expected is None:
        return
    if len(params) != expected:
        raise ValueError(f"{normalized} expects {expected} parameters, got {len(params)}")


def normal_camera_group_key(
    source_kind: object,
    source_id: object,
    width: object,
    height: object,
) -> tuple[str, str, int, int]:
    return (
        str(source_kind or "unknown").strip() or "unknown",
        str(source_id or "").strip(),
        _int_or_zero(width),
        _int_or_zero(height),
    )


def load_normal_camera_defaults(scene_dir: str | Path) -> NormalCameraDefaults:
    data = _load_payload(Path(scene_dir))
    default = _camera_from_mapping(data.get("normal_camera"))
    groups: list[NormalCameraGroupDefault] = []
    raw_groups = data.get("normal_camera_groups")
    if isinstance(raw_groups, list):
        for item in raw_groups:
            if not isinstance(item, dict):
                continue
            camera = _camera_from_mapping(item)
            source_kind = str(item.get("source_kind") or "unknown").strip() or "unknown"
            source_id = str(item.get("source_id") or "").strip()
            width = _int_or_zero(item.get("width"))
            height = _int_or_zero(item.get("height"))
            if width <= 0 or height <= 0 or not camera.enabled:
                continue
            groups.append(
                NormalCameraGroupDefault(
                    source_kind=source_kind,
                    source_id=source_id,
                    width=width,
                    height=height,
                    camera_model=camera.camera_model,
                    camera_params=camera.camera_params,
                    camera_source=camera.camera_source,
                )
            )
    return NormalCameraDefaults(default=default, groups=tuple(groups))


def normal_camera_default_for_group(
    defaults: NormalCameraDefaults,
    *,
    source_kind: object,
    source_id: object,
    width: object,
    height: object,
    fallback: bool = True,
) -> NormalCameraDefault:
    key = normal_camera_group_key(source_kind, source_id, width, height)
    for group in defaults.groups:
        if group.key == key:
            return group.camera
    return defaults.default if fallback else NormalCameraDefault()


def load_normal_camera_default(scene_dir: str | Path) -> NormalCameraDefault:
    return load_normal_camera_defaults(scene_dir).default


def save_normal_camera_default(
    scene_dir: str | Path,
    *,
    camera_model: str,
    camera_params: tuple[float, ...] = (),
    camera_source: str = "gui_default",
) -> None:
    model = normalize_camera_model(camera_model)
    params = tuple(float(value) for value in camera_params)
    validate_camera_params_for_model(model, params)
    if not model:
        clear_normal_camera_default(scene_dir)
        return
    scene = Path(scene_dir)
    payload = _load_payload(scene)
    payload["normal_camera"] = _camera_payload(model, params, camera_source)
    _write_or_delete_payload(scene, payload)


def save_normal_camera_group_default(
    scene_dir: str | Path,
    *,
    source_kind: str,
    source_id: str,
    width: int,
    height: int,
    camera_model: str,
    camera_params: tuple[float, ...] = (),
    camera_source: str = "gui_group",
) -> None:
    model = normalize_camera_model(camera_model)
    params = tuple(float(value) for value in camera_params)
    validate_camera_params_for_model(model, params)
    if not model:
        clear_normal_camera_group_default(
            scene_dir,
            source_kind=source_kind,
            source_id=source_id,
            width=width,
            height=height,
        )
        return

    scene = Path(scene_dir)
    payload = _load_payload(scene)
    target = normal_camera_group_key(source_kind, source_id, width, height)
    groups = _payload_groups_except(payload, target)
    groups.append(
        {
            "source_kind": target[0],
            "source_id": target[1],
            "width": target[2],
            "height": target[3],
            **_camera_payload(model, params, camera_source),
        }
    )
    payload["normal_camera_groups"] = sorted(
        groups,
        key=lambda item: (
            str(item.get("source_kind") or ""),
            str(item.get("source_id") or ""),
            int(item.get("width") or 0),
            int(item.get("height") or 0),
        ),
    )
    _write_or_delete_payload(scene, payload)


def clear_normal_camera_default(scene_dir: str | Path) -> None:
    scene = Path(scene_dir)
    payload = _load_payload(scene)
    payload.pop("normal_camera", None)
    _write_or_delete_payload(scene, payload)


def clear_normal_camera_group_default(
    scene_dir: str | Path,
    *,
    source_kind: str,
    source_id: str,
    width: int,
    height: int,
) -> None:
    scene = Path(scene_dir)
    payload = _load_payload(scene)
    target = normal_camera_group_key(source_kind, source_id, width, height)
    payload["normal_camera_groups"] = _payload_groups_except(payload, target)
    _write_or_delete_payload(scene, payload)


def _load_payload(scene_dir: Path) -> dict[str, Any]:
    data = load_json(normal_camera_defaults_path(scene_dir), {"schema_version": NORMAL_CAMERA_DEFAULTS_SCHEMA_VERSION})
    data["schema_version"] = NORMAL_CAMERA_DEFAULTS_SCHEMA_VERSION
    return data


def _write_or_delete_payload(scene_dir: Path, payload: dict[str, Any]) -> None:
    groups = payload.get("normal_camera_groups")
    if isinstance(groups, list):
        payload["normal_camera_groups"] = [item for item in groups if isinstance(item, dict)]
    else:
        payload.pop("normal_camera_groups", None)

    if payload.get("normal_camera") or payload.get("normal_camera_groups"):
        payload["schema_version"] = NORMAL_CAMERA_DEFAULTS_SCHEMA_VERSION
        write_json(normal_camera_defaults_path(scene_dir), payload)
        return

    path = normal_camera_defaults_path(scene_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _payload_groups_except(payload: dict[str, Any], target: tuple[str, str, int, int]) -> list[dict[str, Any]]:
    raw_groups = payload.get("normal_camera_groups")
    if not isinstance(raw_groups, list):
        return []
    groups: list[dict[str, Any]] = []
    for item in raw_groups:
        if not isinstance(item, dict):
            continue
        key = normal_camera_group_key(
            item.get("source_kind"),
            item.get("source_id"),
            item.get("width"),
            item.get("height"),
        )
        if key != target:
            groups.append(dict(item))
    return groups


def _camera_payload(model: str, params: tuple[float, ...], source: str) -> dict[str, Any]:
    return {
        "model": model,
        "params": list(params),
        "source": str(source or "gui_default"),
    }


def _camera_from_mapping(value: object) -> NormalCameraDefault:
    raw_camera = value if isinstance(value, dict) else {}
    model = normalize_camera_model(raw_camera.get("model"))
    try:
        params = parse_camera_params(raw_camera.get("params"))
    except ValueError:
        params = ()
    source = str(raw_camera.get("source") or "").strip()
    return NormalCameraDefault(camera_model=model, camera_params=params, camera_source=source)


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
