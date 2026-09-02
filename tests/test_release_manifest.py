from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_release_manifest import file_sha256, write_release_metadata


def test_release_metadata_contains_sorted_assets_and_valid_hashes(tmp_path: Path) -> None:
    second = tmp_path / "zeta.zip"
    first = tmp_path / "Alpha.exe"
    second.write_bytes(b"portable")
    first.write_bytes(b"installer")

    manifest_path, checksums_path = write_release_metadata("0.1.0-dev.1", tmp_path, [second, first])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "0.1.0-dev.1"
    assert manifest["platform"] == "windows-x86_64"
    assert [item["name"] for item in manifest["assets"]] == ["Alpha.exe", "zeta.zip"]
    assert manifest["assets"][0]["sha256"] == file_sha256(first)
    assert checksums_path.read_text(encoding="utf-8").splitlines() == [
        f"{file_sha256(first)}  Alpha.exe",
        f"{file_sha256(second)}  zeta.zip",
    ]


def test_release_metadata_rejects_missing_asset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="release asset does not exist"):
        write_release_metadata("0.1.0-dev.1", tmp_path, [tmp_path / "missing.zip"])
