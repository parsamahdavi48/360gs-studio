#!/usr/bin/env python3
# ruff: noqa: E402
"""Open the AprilTag Cube6 scene viewer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.apriltag.case import DEFAULT_CASE_ROOT
from devtools.apriltag.scene_viewer import AprilTagSceneViewerWindow
from gui.theme import apply_theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the AprilTag Cube6 scene viewer.")
    parser.add_argument(
        "--case",
        type=Path,
        default=DEFAULT_CASE_ROOT / "current",
        help="AprilTag dev case directory containing case.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("STECHDRIVE_ENABLE_APRILTAG", "1")
    os.environ.setdefault("STECHDRIVE_APRILTAG_DEVTOOLS", "1")
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = AprilTagSceneViewerWindow(initial_case=args.case)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
