from __future__ import annotations

# ruff: noqa: E402, I001
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metashape_dataset_cli import main_nerf


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_nerf())
