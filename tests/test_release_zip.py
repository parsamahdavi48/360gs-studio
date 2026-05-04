from __future__ import annotations

import pytest

from scripts.create_release_zip import include_in_release, validate_release_member


def test_release_zip_excludes_tests_but_keeps_runtime_scripts() -> None:
    assert not include_in_release("tests/test_smoke.py")
    assert not include_in_release("scripts/create_release_zip.py")
    assert include_in_release("scripts/update_venv.py")
    assert include_in_release("scripts/check_venv.py")
    assert include_in_release("sky_mask.py")
    assert include_in_release("mask_view_recipes.py")
    assert include_in_release("models/README.md")
    assert include_in_release("run_gui.bat")


@pytest.mark.parametrize(
    "path",
    [
        ".venv/Scripts/python.exe",
        ".cache/release/app.zip",
        "sam2.1_l.pt",
        "models/mask2former-swin-large-ade-semantic/model.safetensors",
        "models/mask2former-swin-large-ade-semantic/pytorch_model.bin",
        "models/mask2former-swin-large-ade-semantic/config.json",
        "AGENTS.md",
        "scene/selected_frames.csv",
        "pkg/__pycache__/mod.pyc",
    ],
)
def test_release_zip_rejects_unwanted_members(path: str) -> None:
    with pytest.raises(ValueError):
        validate_release_member(path)
