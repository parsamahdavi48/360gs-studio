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
