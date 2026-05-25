import argparse
from pathlib import Path

import pytest

import scripts.update_venv as update_venv
from scripts.update_venv import parse_py_launcher_output, parse_version_label, parse_winget_python_versions


def test_parse_py_launcher_output_handles_starred_and_plain_entries() -> None:
    text = """
 -V:3.14 *        C:\\Python314\\python.exe
 -V:3.13          C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python313\\python.exe
"""

    parsed = parse_py_launcher_output(text)

    assert parsed[(3, 14)] == [Path("C:\\Python314\\python.exe")]
    assert parsed[(3, 13)] == [Path("C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python313\\python.exe")]


def test_parse_version_label() -> None:
    assert parse_version_label("3.12") == (3, 12)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_version_label("3")


def test_parse_winget_python_versions() -> None:
    text = """
Name                 Id                 Version
------------------------------------------------
Python 3.14          Python.Python.3.14 3.14.0
Python 3.13          Python.Python.3.13 3.13.4
Python 3.12          Python.Python.3.12 3.12.10
"""

    assert parse_winget_python_versions(text) == {(3, 12), (3, 13), (3, 14)}


def test_has_pytest_suite_detects_optional_tests_dir(tmp_path: Path) -> None:
    assert not update_venv.has_pytest_suite(tmp_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    assert not update_venv.has_pytest_suite(tmp_path)

    (tests_dir / "test_smoke.py").write_text("def test_smoke(): pass\n", encoding="utf-8")

    assert update_venv.has_pytest_suite(tmp_path)


def test_pytest_is_not_a_runtime_requirement() -> None:
    assert all(not req.startswith("pytest") for req in update_venv.CORE_REQUIREMENTS)
    assert update_venv.TEST_REQUIREMENTS == ["pytest", "ruff"]
    assert update_venv.LOCKED_TEST_REQUIREMENTS == ["pytest==9.0.3", "ruff==0.15.12"]


def test_optional_test_requirements_can_be_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(update_venv, "REQUIREMENTS_DIR", tmp_path)

    assert update_venv.read_requirements_file("test.txt", required=False) == []
    with pytest.raises(FileNotFoundError):
        update_venv.read_requirements_file("core.txt")


def test_default_requirements_are_unpinned_for_latest_updates() -> None:
    requirements = (
        update_venv.CORE_REQUIREMENTS
        + update_venv.TORCH_REQUIREMENTS
        + update_venv.ML_REQUIREMENTS
        + update_venv.TEST_REQUIREMENTS
    )

    assert requirements
    assert all("==" not in req for req in requirements)
    assert update_venv.TORCH_REQUIREMENTS == ["torch", "torchvision", "torchaudio"]


def test_locked_requirements_are_pinned() -> None:
    requirements = update_venv.requirements_for_mode(locked=True)

    assert all("==" in req for req in requirements.core + requirements.torch + requirements.ml + requirements.test)
    assert "torch==2.11.0+cu128" in requirements.torch
    assert requirements.label == "locked pinned requirements"


def test_dry_run_does_not_install_missing_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = argparse.Namespace(
        repo_root=tmp_path,
        candidates="",
        refresh_python_cache=False,
        python_cache_days=7,
        install_fallback_python=False,
        no_install_python=False,
        allow_cpu_torch=False,
        skip_pytest=False,
        keep_temp=False,
        keep_backup=False,
        locked=False,
        dry_run=True,
    )
    monkeypatch.setattr(update_venv, "parse_args", lambda: args)
    monkeypatch.setattr(update_venv, "build_target_versions", lambda **_: ([(3, 14)], {(3, 14)}))
    monkeypatch.setattr(update_venv, "preflight_version", lambda *_: (True, "preflight passed"))
    monkeypatch.setattr(update_venv, "detect_python_candidates", lambda *_: [])
    monkeypatch.setattr(
        update_venv,
        "install_python_with_winget",
        lambda *_: pytest.fail("dry-run must not install Python"),
    )
    monkeypatch.setattr(
        update_venv,
        "create_candidate_venv",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not create a venv"),
    )

    assert update_venv.main() == 0


def test_repair_activation_scripts_after_venv_promotion(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    old = tmp_path / ".venv-candidate-py312"

    (scripts / "activate.bat").write_text(
        "\n".join(
            [
                f'set "VIRTUAL_ENV={old}"',
                "set PROMPT=(.venv-candidate-py312) %PROMPT%",
                'set "VIRTUAL_ENV_PROMPT=(.venv-candidate-py312) "',
            ]
        ),
        encoding="utf-8",
    )
    (scripts / "activate").write_text(
        "\n".join(
            [
                f"VIRTUAL_ENV=$(cygpath '{old}')",
                f"export VIRTUAL_ENV='{old}'",
                "VIRTUAL_ENV_PROMPT='(.venv-candidate-py312) '",
                'PS1="("\'(.venv-candidate-py312) \'") ${PS1:-}"',
            ]
        ),
        encoding="utf-8",
    )

    update_venv.repair_activation_scripts(venv)

    activate_bat = (scripts / "activate.bat").read_text(encoding="utf-8")
    activate_posix = (scripts / "activate").read_text(encoding="utf-8")
    assert f'set "VIRTUAL_ENV={venv}"' in activate_bat
    assert "set PROMPT=(.venv) %PROMPT%" in activate_bat
    assert f"export VIRTUAL_ENV='{venv}'" in activate_posix
    assert ".venv-candidate-py312" not in activate_bat
    assert ".venv-candidate-py312" not in activate_posix
