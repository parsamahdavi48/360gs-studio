"""Training backend registry shared by the Step 4 UI and command wiring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrainingBackendCategory = Literal["primary", "other"]

TRAINING_BACKEND_LICHTFELD = "lichtfeld"
TRAINING_BACKEND_POSTSHOT = "postshot"
TRAINING_BACKEND_CUSTOM = "custom"
DEFAULT_TRAINING_BACKEND = TRAINING_BACKEND_LICHTFELD


@dataclass(frozen=True, slots=True)
class TrainingBackendSpec:
    backend_id: str
    label_key: str
    short_label_key: str
    tooltip_key: str
    category: TrainingBackendCategory
    stack_order: int
    phase_name: str
    default_executable_windows: str
    default_executable_posix: str
    supports_headless: bool = False

    def default_executable(self, *, windows: bool) -> str:
        return self.default_executable_windows if windows else self.default_executable_posix


_SPECS: tuple[TrainingBackendSpec, ...] = (
    TrainingBackendSpec(
        backend_id=TRAINING_BACKEND_LICHTFELD,
        label_key="TRAINING_BACKEND_LICHTFELD",
        short_label_key="TRAINING_BACKEND_LICHTFELD_SHORT",
        tooltip_key="TRAINING_BACKEND_LICHTFELD",
        category="primary",
        stack_order=0,
        phase_name="training_lichtfeld",
        default_executable_windows="LichtFeld-Studio.exe",
        default_executable_posix="LichtFeld-Studio",
        supports_headless=True,
    ),
    TrainingBackendSpec(
        backend_id=TRAINING_BACKEND_POSTSHOT,
        label_key="TRAINING_BACKEND_POSTSHOT",
        short_label_key="TRAINING_BACKEND_POSTSHOT_SHORT",
        tooltip_key="TRAINING_BACKEND_POSTSHOT",
        category="primary",
        stack_order=1,
        phase_name="training_postshot",
        default_executable_windows="postshot-cli.exe",
        default_executable_posix="postshot-cli",
    ),
    TrainingBackendSpec(
        backend_id=TRAINING_BACKEND_CUSTOM,
        label_key="TRAINING_BACKEND_CUSTOM",
        short_label_key="TRAINING_BACKEND_CUSTOM_SHORT",
        tooltip_key="TRAINING_BACKEND_CUSTOM",
        category="other",
        stack_order=2,
        phase_name="training_custom",
        default_executable_windows="",
        default_executable_posix="",
    ),
)

TRAINING_BACKEND_SPECS: dict[str, TrainingBackendSpec] = {spec.backend_id: spec for spec in _SPECS}
TRAINING_BACKEND_IDS: tuple[str, ...] = tuple(spec.backend_id for spec in _SPECS)
PRIMARY_TRAINING_BACKEND_IDS: tuple[str, ...] = tuple(
    spec.backend_id for spec in _SPECS if spec.category == "primary"
)
OTHER_TRAINING_BACKEND_IDS: tuple[str, ...] = tuple(
    spec.backend_id for spec in _SPECS if spec.category == "other"
)


def training_backend_specs(*, category: TrainingBackendCategory | None = None) -> tuple[TrainingBackendSpec, ...]:
    if category is None:
        return _SPECS
    return tuple(spec for spec in _SPECS if spec.category == category)


def get_training_backend_spec(backend_id: str) -> TrainingBackendSpec:
    return TRAINING_BACKEND_SPECS[normalize_training_backend(backend_id)]


def normalize_training_backend(backend_id: str | None) -> str:
    normalized = (backend_id or "").strip().lower()
    if normalized in TRAINING_BACKEND_SPECS:
        return normalized
    return DEFAULT_TRAINING_BACKEND


def training_backend_default_executable(backend_id: str, *, windows: bool) -> str:
    return get_training_backend_spec(backend_id).default_executable(windows=windows)


def training_backend_phase_name(backend_id: str) -> str:
    return get_training_backend_spec(backend_id).phase_name
