from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
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


def release_setup_preflight_command(
    extracted_root: Path,
    *,
    windows: bool | None = None,
    python_executable: str | None = None,
) -> list[str]:
    use_windows = os.name == "nt" if windows is None else windows
    if use_windows:
        return [
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
    return [
        python_executable or sys.executable,
        str(extracted_root / "scripts" / "update_venv.py"),
        "--dry-run",
        "--locked",
        "--candidates",
        "3.12",
        "--no-install-python",
        "--repo-root",
        str(extracted_root),
    ]


def verify_release_zip_setup(
    zip_path: Path,
    *,
    version: str,
    python_executable: str | None = None,
) -> None:
    expected_root_name = f"stechdrive-3dgs-utils-v{version}"
    with tempfile.TemporaryDirectory(prefix="stechdrive-release-verify-") as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(temp_root)
        extracted_root = temp_root / expected_root_name
        if not extracted_root.is_dir():
            raise RuntimeError(f"release ZIP did not contain expected root folder: {expected_root_name}")
        cmd = release_setup_preflight_command(extracted_root, python_executable=python_executable)
        subprocess.run(cmd, cwd=extracted_root, check=True)


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
    parser.add_argument(
        "--skip-setup-verify",
        action="store_true",
        help=(
            "Only create the ZIP. By default the script extracts the ZIP and runs "
            "the locked setup dry-run from inside the extracted package."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = create_release_zip(repo_root, args.output)
    print(output, flush=True)
    print(f"Size: {output.stat().st_size} bytes", flush=True)
    if not args.skip_setup_verify:
        print("[INFO] Verifying extracted release ZIP setup preflight...", flush=True)
        verify_release_zip_setup(output, version=read_version(repo_root))
        print("[INFO] Release ZIP setup preflight passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
