from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import update_app


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(version: str, files: dict[str, bytes], *, bad_hash_for: str | None = None) -> dict[str, object]:
    root = update_app.release_root_name(version)
    entries = []
    for path, data in sorted(files.items()):
        digest = "0" * 64 if path == bad_hash_for else _hash(data)
        entries.append({"path": path, "size": len(data), "sha256": digest})
    return {
        "schema_version": 1,
        "app": update_app.APP_NAME,
        "version": version,
        "root": root,
        "files": entries,
    }


def _write_release_zip(
    zip_path: Path,
    *,
    version: str,
    files: dict[str, bytes],
    bad_hash_for: str | None = None,
) -> None:
    root = update_app.release_root_name(version)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path, data in files.items():
            zf.writestr(f"{root}/{path}", data)
        zf.writestr(
            f"{root}/{update_app.RELEASE_MANIFEST_NAME}",
            json.dumps(_manifest(version, files, bad_hash_for=bad_hash_for)),
        )


def test_validate_release_zip_rejects_traversal(tmp_path: Path) -> None:
    version = "9.9.9"
    root = update_app.release_root_name(version)
    zip_path = tmp_path / update_app.release_zip_name(version)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{root}/pyproject.toml", f'[project]\nname = "x"\nversion = "{version}"\n')
        zf.writestr(f"{root}/../bad.txt", "bad")

    with pytest.raises(RuntimeError, match="unsafe member path"):
        update_app.validate_release_zip(zip_path, expected_version=version, extract_parent=tmp_path / "stage")


def test_validate_release_zip_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    version = "9.9.9"
    files = {
        "pyproject.toml": f'[project]\nname = "x"\nversion = "{version}"\n'.encode(),
        "run_gui.bat": b"run",
    }
    zip_path = tmp_path / update_app.release_zip_name(version)
    _write_release_zip(zip_path, version=version, files=files, bad_hash_for="run_gui.bat")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        update_app.validate_release_zip(zip_path, expected_version=version, extract_parent=tmp_path / "stage")


def test_apply_release_preserves_local_state_and_removes_old_manifest_files(tmp_path: Path) -> None:
    repo = tmp_path / "app"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.0.0"\n', encoding="utf-8")
    (repo / "old.txt").write_text("old", encoding="utf-8")
    (repo / ".venv" / "Scripts").mkdir(parents=True)
    (repo / ".venv" / "Scripts" / "python.exe").write_text("venv", encoding="utf-8")
    (repo / ".cache").mkdir()
    (repo / ".cache" / "local.txt").write_text("cache", encoding="utf-8")
    (repo / "models").mkdir()
    (repo / "models" / "local.pt").write_text("weights", encoding="utf-8")

    old_manifest = update_app.ReleaseManifest(
        version="1.0.0",
        root=update_app.release_root_name("1.0.0"),
        files=(update_app.ManifestFile(path="old.txt", size=3, sha256="0" * 64),),
    )
    update_app.write_installed_manifest(repo, old_manifest)

    stage = tmp_path / update_app.release_root_name("1.1.0")
    stage.mkdir()
    files = {
        "pyproject.toml": b'[project]\nname = "x"\nversion = "1.1.0"\n',
        "run_gui.bat": b"run",
        "core/new.py": b"print('new')\n",
    }
    manifest_files = []
    for relative, data in files.items():
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest_files.append(update_app.ManifestFile(path=relative, size=len(data), sha256=_hash(data)))
    release = update_app.ValidatedRelease(
        version="1.1.0",
        root_name=update_app.release_root_name("1.1.0"),
        stage_root=stage,
        manifest=update_app.ReleaseManifest(
            version="1.1.0",
            root=update_app.release_root_name("1.1.0"),
            files=tuple(manifest_files),
        ),
    )

    backup = update_app.apply_release(repo, release)

    assert backup is not None and (backup / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (repo / "old.txt").exists()
    assert (repo / "run_gui.bat").read_text(encoding="utf-8") == "run"
    assert (repo / "core" / "new.py").read_text(encoding="utf-8") == "print('new')\n"
    assert (repo / ".venv" / "Scripts" / "python.exe").read_text(encoding="utf-8") == "venv"
    assert (repo / ".cache" / "local.txt").read_text(encoding="utf-8") == "cache"
    assert (repo / "models" / "local.pt").read_text(encoding="utf-8") == "weights"


def test_apply_release_normalizes_managed_files_without_installed_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "app"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.1.0"\n', encoding="utf-8")
    (repo / "update.bat").write_text("new updater", encoding="utf-8")
    (repo / "update_venv.bat").write_text("old updater", encoding="utf-8")
    (repo / "core").mkdir()
    (repo / "core" / "new.py").write_text("new", encoding="utf-8")
    (repo / "core" / "legacy.py").write_text("legacy", encoding="utf-8")
    (repo / "scene").mkdir()
    (repo / "scene" / "user.txt").write_text("user", encoding="utf-8")
    (repo / "models").mkdir()
    (repo / "models" / "local.pt").write_text("weights", encoding="utf-8")
    (repo / ".venv" / "Scripts").mkdir(parents=True)
    (repo / ".venv" / "Scripts" / "python.exe").write_text("venv", encoding="utf-8")

    manifest_files = []
    for relative in ("pyproject.toml", "update.bat", "core/new.py"):
        path = repo / relative
        manifest_files.append(
            update_app.ManifestFile(path=relative, size=path.stat().st_size, sha256=update_app.sha256_file(path))
        )
    release = update_app.ValidatedRelease(
        version="1.1.0",
        root_name=update_app.release_root_name("1.1.0"),
        stage_root=repo,
        manifest=update_app.ReleaseManifest(
            version="1.1.0",
            root=update_app.release_root_name("1.1.0"),
            files=tuple(manifest_files),
        ),
    )

    backup = update_app.apply_release(repo, release, remove_obsolete_managed=True)

    assert backup is not None
    assert not (repo / "update_venv.bat").exists()
    assert not (repo / "core" / "legacy.py").exists()
    assert (repo / "scene" / "user.txt").read_text(encoding="utf-8") == "user"
    assert (repo / "models" / "local.pt").read_text(encoding="utf-8") == "weights"
    assert (repo / ".venv" / "Scripts" / "python.exe").read_text(encoding="utf-8") == "venv"
    assert (repo / update_app.RELEASE_MANIFEST_NAME).is_file()
    assert (repo / update_app.INSTALLED_MANIFEST).is_file()


def test_load_local_release_validates_manifest_hashes(tmp_path: Path) -> None:
    repo = tmp_path / "app"
    repo.mkdir()
    version = "1.1.0"
    files = {
        "pyproject.toml": f'[project]\nname = "x"\nversion = "{version}"\n'.encode(),
        "update.bat": b"update",
    }
    for relative, data in files.items():
        (repo / relative).write_bytes(data)
    (repo / update_app.RELEASE_MANIFEST_NAME).write_text(
        json.dumps(_manifest(version, files)),
        encoding="utf-8",
    )

    release = update_app.load_local_release(repo)

    assert release is not None
    assert release.version == version
    assert release.stage_root == repo


def test_select_zip_asset_requires_exact_release_asset_name() -> None:
    release = update_app.ReleaseInfo(
        tag_name="v1.2.3",
        version="1.2.3",
        html_url="https://example.test/release",
        assets=(
            update_app.ReleaseAsset(name="other.zip", browser_download_url="https://example.test/other.zip", size=1),
            update_app.ReleaseAsset(
                name="stechdrive-3dgs-utils-v1.2.3.zip",
                browser_download_url="https://example.test/app.zip",
                size=2,
            ),
        ),
    )

    assert update_app.select_zip_asset(release).browser_download_url == "https://example.test/app.zip"


@pytest.mark.parametrize("path", ["/bad.txt", "bad//name.txt", ".venv/Scripts/python.exe", "models/local.pt"])
def test_safe_relative_path_rejects_local_state_and_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        update_app.safe_relative_path(path)
