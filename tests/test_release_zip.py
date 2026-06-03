from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts import create_release_zip as release_zip
from scripts.create_release_zip import include_in_release, release_setup_preflight_command, validate_release_member


def test_release_zip_excludes_tests_but_keeps_setup_scripts() -> None:
    assert not include_in_release("tests/test_smoke.py")
    assert not include_in_release("devtools/apriltag/synthetic.py")
    assert not include_in_release("doc/architecture.md")
    assert not include_in_release("doc/architecture.ja.md")
    assert not include_in_release("scripts/create_release_zip.py")
    assert include_in_release("scripts/update_venv.py")
    assert include_in_release("scripts/update_app.py")
    assert include_in_release("scripts/sync_venv.py")
    assert include_in_release("scripts/check_venv.py")
    assert not include_in_release("scripts/estimate_apriltag_scale.py")
    assert include_in_release("core/apriltag_scale_apply.py")
    assert include_in_release("core/apriltag_printable.py")
    assert include_in_release("core/scene_preview_cubemap.py")
    assert include_in_release("core/sky_mask.py")
    assert include_in_release("core/mask_view_recipes.py")
    assert include_in_release("requirements/core.txt")
    assert include_in_release("requirements/ml.txt")
    assert include_in_release("requirements/sam31.txt")
    assert include_in_release("requirements/torch-cu128.txt")
    assert not include_in_release("requirements/test.txt")
    assert not include_in_release("requirements/dev.txt")
    assert include_in_release("models/README.md")
    assert include_in_release("run_gui.bat")
    assert include_in_release("update.bat")
    assert not include_in_release("update_venv.bat")


def test_release_zip_script_surface_is_explicit() -> None:
    tracked_scripts = {path.as_posix() for path in Path("scripts").glob("*.py")}

    assert release_zip.RELEASE_SCRIPT_PATHS <= tracked_scripts
    for path in sorted(tracked_scripts):
        assert include_in_release(path) == (path in release_zip.RELEASE_SCRIPT_PATHS)


def test_build_release_manifest_records_release_files(tmp_path: Path) -> None:
    (tmp_path / "run_gui.bat").write_text("run", encoding="utf-8")
    (tmp_path / "update.bat").write_text("update", encoding="utf-8")

    manifest = json.loads(
        release_zip.build_release_manifest(
            tmp_path,
            version="9.9.9",
            root_prefix="stechdrive-3dgs-utils-v9.9.9/",
            files=["update.bat", "run_gui.bat"],
        )
    )

    assert manifest["schema_version"] == 1
    assert manifest["app"] == "stechdrive-3dgs-utils"
    assert manifest["version"] == "9.9.9"
    assert manifest["root"] == "stechdrive-3dgs-utils-v9.9.9"
    assert [item["path"] for item in manifest["files"]] == ["run_gui.bat", "update.bat"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


@pytest.mark.parametrize(
    "path",
    [
        ".venv/Scripts/python.exe",
        ".cache/release/app.zip",
        "sam2.1_l.pt",
        "models/ultralytics/yolo26s-sem.pt",
        "models/ultralytics/yolo26n-sem.pt",
        "AGENTS.md",
        "user_settings.json",
        "stechdrive_export_settings.json",
        "export_settings.json",
        "export_settings_3dgut.json",
        "views_config.json",
        ".cache/update_venv.log",
        "doc/architecture.md",
        "doc/architecture.ja.md",
        "requirements/test.txt",
        "requirements/dev.txt",
        "update_venv.bat",
        "scene/_stechdrive/frames/selected_frames.csv",
        "pkg/__pycache__/mod.pyc",
        "scripts/estimate_apriltag_scale.py",
        "scripts/prepare_spheresfm_project.py",
    ],
)
def test_release_zip_rejects_unwanted_members(path: str) -> None:
    with pytest.raises(ValueError):
        validate_release_member(path)


def test_release_zip_rejects_local_development_paths_in_text_members(tmp_path: Path) -> None:
    member = tmp_path / "gui" / "bad_default.py"
    member.parent.mkdir(parents=True)
    local_checkout = "D:" + "\\GitHub\\gsplat\\examples\\simple_trainer.py"
    member.write_text(f"DEFAULT_SCRIPT = {local_checkout!r}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="local developer path"):
        release_zip.validate_release_file_contents(tmp_path, "gui/bad_default.py")


def test_release_zip_allows_generic_windows_example_paths(tmp_path: Path) -> None:
    member = tmp_path / "README.md"
    generic_scene = "D:" + "\\work\\scene01"
    member.write_text(f"Use a short working path such as `{generic_scene}`.\n", encoding="utf-8")

    release_zip.validate_release_file_contents(tmp_path, "README.md")


def test_tracked_release_files_do_not_contain_local_development_paths() -> None:
    repo_root = Path.cwd()

    for path in release_zip.git_tracked_files(repo_root):
        if include_in_release(path):
            release_zip.validate_release_file_contents(repo_root, path)


def test_release_setup_preflight_command_uses_extracted_batch_on_windows() -> None:
    cmd = release_setup_preflight_command(Path("pkg"), windows=True)

    assert cmd == [
        "cmd",
        "/c",
        "update.bat",
        "--no-pause",
        "--deps-only",
        "--dry-run",
    ]


def test_update_batch_uses_delayed_python_command_expansion() -> None:
    text = Path("update.bat").read_text(encoding="utf-8")

    assert '%PYTHON_CMD% "%~dp0scripts\\update_app.py"' not in text
    assert '%PYTHON_CMD% "%~dp0scripts\\update_venv.py"' not in text
    assert '!PYTHON_CMD! "%~dp0scripts\\update_app.py" !APP_ARGS!' in text
    assert '!PYTHON_CMD! "%~dp0scripts\\update_venv.py" !PREFLIGHT_ARGS!' in text


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
        zf.writestr("stechdrive-3dgs-utils-v9.9.9/update.bat", "")

    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(cmd: list[str], *, cwd: Path, check: bool) -> None:
        calls.append((cmd, cwd, check))

    monkeypatch.setattr(release_zip.subprocess, "run", fake_run)

    release_zip.verify_release_zip_setup(zip_path, version="9.9.9")

    assert len(calls) == 1
    _cmd, cwd, check = calls[0]
    assert cwd.name == "stechdrive-3dgs-utils-v9.9.9"
    assert check is True
