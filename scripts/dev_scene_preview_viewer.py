#!/usr/bin/env python3
# ruff: noqa: E402
"""Open the read-only scene preview viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.scene_preview.window import ScenePreviewWindow
from gui.theme import apply_theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the read-only scene preview viewer.")
    parser.add_argument("scene", nargs="?", type=Path, help="Managed scene folder to open")
    parser.add_argument("--scene", dest="scene_option", type=Path, default=None, help="Managed scene folder to open")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = args.scene_option or args.scene
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = ScenePreviewWindow(scene_dir=scene)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
