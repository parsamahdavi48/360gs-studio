from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class NormalCameraDefault:
    camera_model: str = ""
    camera_params: tuple[float, ...] = ()
    camera_source: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.camera_model)


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


def load_normal_camera_default(scene_dir: str | Path) -> NormalCameraDefault:
    data = load_json(normal_camera_defaults_path(Path(scene_dir)), {"schema_version": NORMAL_CAMERA_DEFAULTS_SCHEMA_VERSION})
    raw_camera = data.get("normal_camera")
    if not isinstance(raw_camera, dict):
        raw_camera = {}
    model = normalize_camera_model(raw_camera.get("model"))
    try:
        params = parse_camera_params(raw_camera.get("params"))
    except ValueError:
        params = ()
    source = str(raw_camera.get("source") or "").strip()
    return NormalCameraDefault(camera_model=model, camera_params=params, camera_source=source)


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
    write_json(
        normal_camera_defaults_path(Path(scene_dir)),
        {
            "schema_version": NORMAL_CAMERA_DEFAULTS_SCHEMA_VERSION,
            "normal_camera": {
                "model": model,
                "params": list(params),
                "source": str(camera_source or "gui_default"),
            },
        },
    )


def clear_normal_camera_default(scene_dir: str | Path) -> None:
    path = normal_camera_defaults_path(Path(scene_dir))
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
