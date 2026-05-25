from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.cubemap_remap import build_remap
from core.cubemap_transforms_json import load_custom_views, make_default_views
from core.cubemap_view_spec import build_remap_spec, load_views_json, make_default_cube6_views, normalize_views


def test_default_cube6_views_match_legacy_dicts() -> None:
    specs = make_default_cube6_views(45.0, 2.5)
    legacy = make_default_views(45.0, 2.5, False, False)

    assert [view.as_dict() for view in specs] == legacy


def test_default_cube6_views_can_remove_poles() -> None:
    views = make_default_cube6_views(45.0, 0.0, no_top=True, no_bottom=True)

    assert {view.name for view in views} == {"px", "nx", "pz", "nz"}


def test_load_views_json_normalizes_enabled_views(tmp_path: Path) -> None:
    path = tmp_path / "views.json"
    path.write_text(
        json.dumps(
            {
                "views": [
                    {"name": "front", "yaw": "0", "pitch": 0, "enabled": True},
                    {"name": "skip", "yaw": 90, "pitch": 0, "enabled": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_views_json(path)

    assert tuple(view.name for view in specs) == ("front",)
    assert load_custom_views(str(path)) == [{"name": "front", "yaw": 0.0, "pitch": 0.0}]


def test_normalize_views_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        normalize_views(
            [
                {"name": "front", "yaw": 0.0, "pitch": 0.0},
                {"name": "front", "yaw": 90.0, "pitch": 0.0},
            ]
        )


def test_remap_spec_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="input_size"):
        build_remap_spec(input_size=(0, 1024), output_size=512, fov_deg=90.0, yaw_deg=0.0, pitch_deg=0.0)


def test_build_remap_uses_shared_remap_spec_validation() -> None:
    with pytest.raises(ValueError, match="fov_deg"):
        build_remap((2048, 1024), 180.0, 0.0, 0.0, 512)
