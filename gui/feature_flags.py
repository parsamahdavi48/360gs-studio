"""Small feature flags for local-only experimental GUI work."""

from __future__ import annotations

import os

_TRUE_VALUES = {"1", "true", "yes", "on"}


def apriltag_scale_enabled() -> bool:
    """Return whether the experimental AprilTag scale UI should be shown."""
    return os.environ.get("STECHDRIVE_ENABLE_APRILTAG", "").strip().lower() in _TRUE_VALUES
