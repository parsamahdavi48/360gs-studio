"""Compatibility wrapper for :mod:`core.cubemap_transforms_json`."""

from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module

_impl = _import_module("core.cubemap_transforms_json")

if __name__ == "__main__":
    raise SystemExit(_impl.main())

_sys.modules[__name__] = _impl
