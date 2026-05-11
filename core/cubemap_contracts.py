"""Shared cubemap face and output-profile contracts.

The constants in this module describe the saved image contract, not a viewer
preference. GUI export, AprilTag processing, and development viewers should use
the same face names and yaw/pitch assignments.
"""

from __future__ import annotations

CUBEMAP_CONTRACT_VERSION = 2

PROFILE_POSTSHOT = "postshot"
PROFILE_BRUSH = "brush"
PROFILE_LICHTFELD = "lichtfeld"
PROFILE_CUSTOM = "custom"

CUBEMAP_FACE_NAMES = ("px", "nx", "pz", "nz", "top", "bottom", "py", "ny")
CUBEMAP_REFERENCE_FACE_ORDER = ("pz", "px", "nz", "nx", "top", "bottom", "py", "ny")
CUBEMAP_SIDE_FACES = frozenset({"px", "nx", "pz", "nz"})
CUBEMAP_VERTICAL_FACE_SETS = (frozenset({"top", "bottom"}), frozenset({"py", "ny"}))

CUBEMAP_STANDARD_FACE_VIEW_PARAMS: dict[str, tuple[float, float]] = {
    "pz": (0.0, 0.0),
    "px": (90.0, 0.0),
    "nz": (180.0, 0.0),
    "nx": (-90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    "py": (0.0, -90.0),
    "ny": (0.0, 90.0),
}

CUBE6_YAW_SLOTS = 4
CUBE6_PITCHES = (-90.0, 0.0, 90.0)
CUBE6_ENABLED_CELLS = frozenset({(0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)})
CUBE6_DEFAULT_YAW_OFFSET = 45.0
CUBE6_DEFAULT_YAW_OFFSET_PER_FRAME = 30.0
CUBE6_VIEW_CELLS = (
    ("px", 1, 0),
    ("nx", 1, 2),
    ("pz", 1, 3),
    ("nz", 1, 1),
    ("top", 2, 3),
    ("bottom", 0, 3),
)


def normalize_angle(angle_deg: float) -> float:
    raw = float(angle_deg)
    normalized = ((raw + 180.0) % 360.0) - 180.0
    if normalized == -180.0 and raw > 0.0:
        return 180.0
    return normalized


def cube6_view_params(yaw_offset: float = 0.0) -> dict[str, tuple[float, float]]:
    step = 360.0 / float(CUBE6_YAW_SLOTS)
    return {
        name: (normalize_angle(float(yaw_offset) + slot * step), CUBE6_PITCHES[row])
        for name, row, slot in CUBE6_VIEW_CELLS
    }


def cube6_views(yaw_offset: float = 0.0) -> list[dict]:
    params = cube6_view_params(yaw_offset)
    return [
        {
            "name": name,
            "yaw": params[name][0],
            "pitch": params[name][1],
            "enabled": True,
            "slot": slot,
            "label": name,
        }
        for name, row, slot in CUBE6_VIEW_CELLS
    ]
