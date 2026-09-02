"""Create deterministic checksums and metadata for release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(version: str, assets: list[Path]) -> dict[str, object]:
    records = [
        {
            "name": asset.name,
            "size": asset.stat().st_size,
            "sha256": file_sha256(asset),
        }
        for asset in sorted(assets, key=lambda item: item.name.lower())
    ]
    return {
        "schema_version": 1,
        "application": "360GS Studio",
        "version": version,
        "platform": "windows-x86_64",
        "assets": records,
    }


def write_release_metadata(version: str, output_dir: Path, assets: list[Path]) -> tuple[Path, Path]:
    missing = [path for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release asset does not exist: {missing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(version, assets)
    manifest_path = output_dir / "release-manifest.json"
    checksums_path = output_dir / "SHA256SUMS.txt"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f"{item['sha256']}  {item['name']}" for item in manifest["assets"]]
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest_path, checksums_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("assets", nargs="+", type=Path)
    args = parser.parse_args()
    write_release_metadata(args.version, args.output_dir, args.assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
