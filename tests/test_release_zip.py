from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts import create_release_zip as release_zip
from scripts.create_release_zip import include_in_release, release_setup_preflight_command, validate_release_member


def test_release_zip_excludes_tests_but_keeps_runtime_scripts() -> None:
    assert not include_in_release("tests/test_smoke.py")
    assert not include_in_release("devtools/apriltag/synthetic.py")
    assert not include_in_release("scripts/create_release_zip.py")
    assert include_in_release("scripts/update_venv.py")
    assert include_in_release("scripts/check_venv.py")
    assert include_in_release("scripts/estimate_apriltag_scale.py")
    assert include_in_release("core/apriltag_scale_apply.py")
    assert include_in_release("core/apriltag_printable.py")
    assert include_in_release("core/scene_preview_cubemap.py")
    assert include_in_release("core/sky_mask.py")
    assert include_in_release("core/mask_view_recipes.py")
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
        "user_settings.json",
        "stechdrive_export_settings.json",
        "export_settings.json",
        "export_settings_3dgut.json",
        "views_config.json",
        ".cache/update_venv.log",
        "scene/_stechdrive/frames/selected_frames.csv",
        "pkg/__pycache__/mod.pyc",
    ],
)
def test_release_zip_rejects_unwanted_members(path: str) -> None:
    with pytest.raises(ValueError):
        validate_release_member(path)


def test_release_setup_preflight_command_uses_extracted_batch_on_windows() -> None:
    cmd = release_setup_preflight_command(Path("pkg"), windows=True)

    assert cmd == [
        "cmd",
        "/c",
        "update_venv.bat",
        "--no-pause",
        "--dry-run",
        "--locked",
        "--candidates",
        "3.12",
        "--no-install-python",
    ]


def test_release_setup_preflight_command_uses_extracted_script_on_posix() -> None:
    root = Path("pkg")
    cmd = release_setup_preflight_command(root, windows=False, python_executable="python")

    assert cmd == [
        "python",
        str(root / "scripts" / "update_venv.py"),
        "--dry-run",
        "--locked",
        "--candidates",
        "3.12",
        "--no-install-python",
        "--repo-root",
        str(root),
    ]


def test_verify_release_zip_setup_runs_from_extracted_root(monkeypatch, tmp_path: Path) -> None:
    zip_path = tmp_path / "release.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("stechdrive-3dgs-utils-v9.9.9/update_venv.bat", "")

    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(cmd: list[str], *, cwd: Path, check: bool) -> None:
        calls.append((cmd, cwd, check))

    monkeypatch.setattr(release_zip.subprocess, "run", fake_run)

    release_zip.verify_release_zip_setup(zip_path, version="9.9.9")

    assert len(calls) == 1
    _cmd, cwd, check = calls[0]
    assert cwd.name == "stechdrive-3dgs-utils-v9.9.9"
    assert check is True
