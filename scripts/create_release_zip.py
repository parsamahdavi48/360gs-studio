from __future__ import annotations

import argparse
import subprocess
import tomllib
import zipfile
from pathlib import Path

EXCLUDED_PREFIXES = (
    "tests/",
)

EXCLUDED_PATHS = {
    "scripts/create_release_zip.py",
}

UNWANTED_PARTS = (
    ".git/",
    ".venv/",
    ".cache/",
    ".claude/",
    ".codex/",
    "__pycache__/",
    ".pytest_cache/",
)

UNWANTED_SUFFIXES = (
    ".pt",
    ".safetensors",
    ".pyc",
    ".pyo",
)

UNWANTED_NAMES = {
    "AGENTS.md",
    "extract_cache.npz",
    "extract_report.json",
    "pytorch_model.bin",
    "stechdrive_export_settings.json",
    "update_venv.log",
    "user_settings.json",
    "views_config.json",
}


def read_version(repo_root: Path) -> str:
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = data["project"]["version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("project.version is missing")
    return version.strip()


def git_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def include_in_release(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in EXCLUDED_PATHS:
        return False
    return not any(normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def validate_release_member(path: str) -> None:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name
    parts = set(normalized.split("/"))
    if normalized.startswith("models/") and normalized != "models/README.md":
        raise ValueError(f"unwanted local model file would be included: {path}")
    if normalized in UNWANTED_NAMES or name in UNWANTED_NAMES:
        raise ValueError(f"unwanted generated/local file would be included: {path}")
    if name.startswith("selected_frames") and name.endswith(".csv"):
        raise ValueError(f"unwanted generated/local file would be included: {path}")
    if normalized.endswith(UNWANTED_SUFFIXES):
        raise ValueError(f"unwanted binary/cache file would be included: {path}")
    for part in UNWANTED_PARTS:
        clean = part.strip("/")
        if clean in parts:
            raise ValueError(f"unwanted local directory would be included: {path}")


def create_release_zip(repo_root: Path, output: Path | None = None) -> Path:
    version = read_version(repo_root)
    prefix = f"stechdrive-3dgs-utils-v{version}/"
    output_path = output or repo_root / ".cache" / "release" / f"stechdrive-3dgs-utils-v{version}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = [path for path in git_tracked_files(repo_root) if include_in_release(path)]
    if not files:
        raise ValueError("no tracked files found for release ZIP")

    for path in files:
        validate_release_member(path)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(repo_root / path, prefix + path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a distribution ZIP without local state, model weights, or tests.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ZIP path. Defaults to .cache/release/stechdrive-3dgs-utils-vVERSION.zip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = create_release_zip(args.repo_root.resolve(), args.output)
    print(output)
    print(f"Size: {output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
