"""Compatibility wrapper for :mod:`core.realityscan_to_lfs_colmap`."""

from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module

_impl = _import_module("core.realityscan_to_lfs_colmap")

if __name__ == "__main__":
    raise SystemExit(_impl.main())

_sys.modules[__name__] = _impl
