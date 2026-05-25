#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Thin developer entry point for AprilTag scale estimation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.apriltag_scale_estimate import main


if __name__ == "__main__":
    raise SystemExit(main())
