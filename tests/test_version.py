import re
from pathlib import Path

from gui.version import APP_VERSION, app_version_label, read_app_version


def test_read_app_version_from_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8")

    assert read_app_version(pyproject) == "1.2.3"


def test_read_app_version_falls_back_for_missing_file(tmp_path: Path) -> None:
    assert read_app_version(tmp_path / "missing.toml") == "0.0.0"


def test_read_app_version_falls_back_for_short_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nversion = "1.2"\n', encoding="utf-8")

    assert read_app_version(pyproject) == "0.0.0"


def test_app_version_label_is_prefixed() -> None:
    assert re.fullmatch(r"v\d+\.\d+\.\d+", app_version_label())


def test_app_version_uses_semver_shape() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", APP_VERSION)
