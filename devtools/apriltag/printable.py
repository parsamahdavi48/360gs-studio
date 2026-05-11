"""Development-tool compatibility wrapper for printable AprilTag targets."""

from __future__ import annotations

from core.apriltag_markers import marker_image_array
from core.apriltag_printable import PrintableTarget, available_pages, create_printable_target

__all__ = [
    "PrintableTarget",
    "available_pages",
    "create_printable_target",
    "marker_image_array",
]
