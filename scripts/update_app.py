from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

APP_NAME = "360gs-studio"
GITHUB_OWNER = "stechdrive"
GITHUB_REPO = "360gs-studio"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
LATEST_RELEASE_API = f"{GITHUB_API_BASE}/releases/latest"
RELEASE_MANIFEST_NAME = "release_manifest.json"
INSTALLED_MANIFEST = ".cache/updater/installed_manifest.json"
USER_AGENT = f"{APP_NAME}-updater"

MANAGED_TOP_LEVEL_DIRS = {
    "core",
    "doc",
    "gui",
    "gs360studio",
    "images",
    "requirements",
    "scripts",
    "packaging",
}

MANAGED_TOP_LEVEL_FILES = {
    ".gitignore",
    "LICENSE",
    "NOTICE.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "ROADMAP.md",
    "README.ja.md",
    "README.md",
    "THIRD_PARTY_LICENSES.md",
    "pyproject.toml",
    "run_gui.bat",
    "setup_windows.bat",
    "update.bat",
    "update_venv.bat",
}

PROTECTED_TOP_LEVEL = {
    ".cache",
    ".claude",
    ".codex",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}

UNWANTED_SUFFIXES = (
    ".pt",
    ".safetensors",
    ".pyc",
    ".pyo",
)

UNWANTED_NAMES = {
    "AGENTS.md",
    "extract_report.json",
    "export_settings.json",
    "export_settings_3dgut.json",
    "pytorch_model.bin",
    "stechdrive_export_settings.json",
    "update_venv.log",
    "user_settings.json",
    "views_config.json",
}


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    browser_download_url: str
    size: int


@dataclass(frozen=True)
class ReleaseInfo:
    tag_name: str
    version: str
    html_url: str
    assets: tuple[ReleaseAsset, ...]


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseManifest:
    version: str
    root: str
    files: tuple[ManifestFile, ...]

    @property
    def file_paths(self) -> set[str]:
        return {item.path for item in self.files}


@dataclass(frozen=True)
class ValidatedRelease:
    version: str
    root_name: str
    stage_root: Path
    manifest: ReleaseManifest


def emit(message: str) -> None:
    print(message, flush=True)


def read_app_version(repo_root: Path) -> str:
    try:
        data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        version = data["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return "0.0.0"
    return version.strip() if isinstance(version, str) and version.strip() else "0.0.0"


def parse_version(value: str) -> tuple[int, int, int] | None:
    text = value.strip()
    if text.startswith("v"):
        text = text[1:]
    parts = text.split(".")
    if len(parts) != 3:
        return None
    try:
        version = tuple(int(part) for part in parts)
    except ValueError:
        return None
    return version if len(version) == 3 else None


def version_label(version: str) -> str:
    return version if version.startswith("v") else f"v{version}"


def release_zip_name(version: str) -> str:
    return f"{APP_NAME}-{version_label(version)}.zip"


def release_root_name(version: str) -> str:
    return f"{APP_NAME}-{version_label(version)}"


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected JSON response from {url}")
    return parsed


def parse_release_info(data: dict[str, Any]) -> ReleaseInfo:
    tag_name = str(data.get("tag_name", "")).strip()
    parsed = parse_version(tag_name)
    if parsed is None:
        raise RuntimeError(f"Release tag is not a vX.Y.Z version: {tag_name or '<missing>'}")
    assets: list[ReleaseAsset] = []
    for item in data.get("assets", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        url = str(item.get("browser_download_url", "")).strip()
        size = item.get("size", 0)
        if name and url and isinstance(size, int):
            assets.append(ReleaseAsset(name=name, browser_download_url=url, size=size))
    return ReleaseInfo(
        tag_name=tag_name,
        version=tag_name[1:] if tag_name.startswith("v") else tag_name,
        html_url=str(data.get("html_url", "")).strip(),
        assets=tuple(assets),
    )


def fetch_release(version: str | None = None) -> ReleaseInfo:
    url = f"{GITHUB_API_BASE}/releases/tags/{version_label(version)}" if version else LATEST_RELEASE_API
    try:
        return parse_release_info(request_json(url))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub release lookup failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub release lookup failed: {exc.reason}") from exc


def select_zip_asset(release: ReleaseInfo) -> ReleaseAsset:
    expected = release_zip_name(release.version)
    for asset in release.assets:
        if asset.name == expected:
            return asset
    available = ", ".join(asset.name for asset in release.assets) or "none"
    raise RuntimeError(f"Release asset not found: {expected} (available: {available})")


def select_sha256_asset(release: ReleaseInfo, zip_asset: ReleaseAsset) -> ReleaseAsset | None:
    expected_names = {f"{zip_asset.name}.sha256", zip_asset.name.removesuffix(".zip") + ".sha256"}
    for asset in release.assets:
        if asset.name in expected_names:
            return asset
    return None


def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha256_text(text: str, expected_name: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1 and len(parts[0]) == 64:
            return parts[0].lower()
        if len(parts) >= 2 and parts[1].lstrip("*") == expected_name and len(parts[0]) == 64:
            return parts[0].lower()
    raise RuntimeError(f"SHA256 file did not contain a checksum for {expected_name}")


def safe_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or normalized != normalized.strip("/")
        or "//" in normalized
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe release path: {path}")
    top_level = pure.parts[0]
    if top_level in PROTECTED_TOP_LEVEL:
        raise ValueError(f"release path targets protected local state: {path}")
    if PurePosixPath(normalized).name in UNWANTED_NAMES:
        raise ValueError(f"release path targets generated or local state: {path}")
    if normalized.endswith(UNWANTED_SUFFIXES):
        raise ValueError(f"release path targets generated binary/cache data: {path}")
    return pure.as_posix()


def manifest_from_json(data: dict[str, Any], *, expected_version: str, expected_root: str) -> ReleaseManifest:
    if data.get("schema_version") != 1:
        raise RuntimeError("Unsupported release manifest schema")
    if data.get("app") != APP_NAME:
        raise RuntimeError("Release manifest is for a different application")
    version = str(data.get("version", "")).strip()
    root = str(data.get("root", "")).strip()
    if version != expected_version:
        raise RuntimeError(f"Release manifest version mismatch: expected {expected_version}, found {version}")
    if root != expected_root:
        raise RuntimeError(f"Release manifest root mismatch: expected {expected_root}, found {root}")
    files: list[ManifestFile] = []
    for item in data.get("files", []):
        if not isinstance(item, dict):
            raise RuntimeError("Release manifest contains an invalid file entry")
        path = safe_relative_path(str(item.get("path", "")))
        size = item.get("size")
        sha256 = str(item.get("sha256", "")).strip().lower()
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"Release manifest contains an invalid size for {path}")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise RuntimeError(f"Release manifest contains an invalid sha256 for {path}")
        files.append(ManifestFile(path=path, size=size, sha256=sha256))
    if not files:
        raise RuntimeError("Release manifest does not contain any files")
    return ReleaseManifest(version=version, root=root, files=tuple(files))


def manifest_from_zip_members(zip_path: Path, *, expected_version: str, expected_root: str) -> ReleaseManifest:
    files: list[ManifestFile] = []
    prefix = f"{expected_root}/"
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            if not name.startswith(prefix):
                raise RuntimeError(f"Release ZIP contains a file outside {expected_root}/: {name}")
            relative = name[len(prefix) :]
            if relative == RELEASE_MANIFEST_NAME:
                continue
            path = safe_relative_path(relative)
            digest = hashlib.sha256(zf.read(info)).hexdigest()
            files.append(ManifestFile(path=path, size=info.file_size, sha256=digest))
    if not files:
        raise RuntimeError("Release ZIP does not contain application files")
    return ReleaseManifest(version=expected_version, root=expected_root, files=tuple(files))


def validate_release_zip(zip_path: Path, *, expected_version: str, extract_parent: Path) -> ValidatedRelease:
    expected_root = release_root_name(expected_version)
    prefix = f"{expected_root}/"
    with zipfile.ZipFile(zip_path) as zf:
        file_names: list[str] = []
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name.rstrip("/"))
            if "//" in name or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise RuntimeError(f"Release ZIP contains an unsafe member path: {name}")
            if not name.startswith(prefix):
                raise RuntimeError(f"Release ZIP must contain a single root folder named {expected_root}/")
            if not info.is_dir():
                file_names.append(name)
        if not file_names:
            raise RuntimeError("Release ZIP is empty")
        manifest_name = prefix + RELEASE_MANIFEST_NAME
        if manifest_name in file_names:
            manifest_data = json.loads(zf.read(manifest_name).decode("utf-8"))
            if not isinstance(manifest_data, dict):
                raise RuntimeError("Release manifest is not a JSON object")
            manifest = manifest_from_json(
                manifest_data,
                expected_version=expected_version,
                expected_root=expected_root,
            )
        else:
            manifest = manifest_from_zip_members(zip_path, expected_version=expected_version, expected_root=expected_root)

        for item in manifest.files:
            member = prefix + item.path
            try:
                info = zf.getinfo(member)
            except KeyError as exc:
                raise RuntimeError(f"Release manifest references a missing file: {item.path}") from exc
            if info.file_size != item.size:
                raise RuntimeError(f"Release manifest size mismatch for {item.path}")
            digest = hashlib.sha256(zf.read(info)).hexdigest()
            if digest != item.sha256:
                raise RuntimeError(f"Release manifest hash mismatch for {item.path}")

        pyproject_member = prefix + "pyproject.toml"
        try:
            pyproject = tomllib.loads(zf.read(pyproject_member).decode("utf-8"))
        except KeyError as exc:
            raise RuntimeError("Release ZIP is missing pyproject.toml") from exc
        version = pyproject.get("project", {}).get("version")
        if version != expected_version:
            raise RuntimeError(f"pyproject.toml version mismatch: expected {expected_version}, found {version}")

        zf.extractall(extract_parent)

    stage_root = extract_parent / expected_root
    if not stage_root.is_dir():
        raise RuntimeError(f"Release ZIP did not extract {expected_root}/")
    return ValidatedRelease(
        version=expected_version,
        root_name=expected_root,
        stage_root=stage_root,
        manifest=manifest,
    )


def load_local_release(repo_root: Path) -> ValidatedRelease | None:
    manifest_path = repo_root / RELEASE_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Local {RELEASE_MANIFEST_NAME} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Local {RELEASE_MANIFEST_NAME} is not a JSON object")

    expected_version = str(data.get("version", "")).strip()
    expected_root = str(data.get("root", "")).strip()
    manifest = manifest_from_json(data, expected_version=expected_version, expected_root=expected_root)

    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeError("Local release manifest is present, but pyproject.toml is missing")
    version = read_app_version(repo_root)
    if version != manifest.version:
        raise RuntimeError(f"Local release manifest version mismatch: expected {version}, found {manifest.version}")

    for item in manifest.files:
        target = resolve_target(repo_root, item.path)
        if not target.is_file():
            raise RuntimeError(f"Local release manifest references a missing file: {item.path}")
        if target.stat().st_size != item.size:
            raise RuntimeError(f"Local release manifest size mismatch for {item.path}")
        if sha256_file(target) != item.sha256:
            raise RuntimeError(f"Local release manifest hash mismatch for {item.path}")

    return ValidatedRelease(
        version=manifest.version,
        root_name=manifest.root,
        stage_root=repo_root,
        manifest=manifest,
    )


def load_installed_manifest(repo_root: Path) -> ReleaseManifest | None:
    path = repo_root / INSTALLED_MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        root = str(data.get("root", release_root_name(str(data.get("version", "")))))
        return manifest_from_json(data, expected_version=str(data.get("version", "")), expected_root=root)
    except Exception:
        return None


def write_installed_manifest(repo_root: Path, manifest: ReleaseManifest) -> None:
    path = repo_root / INSTALLED_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "app": APP_NAME,
        "version": manifest.version,
        "root": manifest.root,
        "installed_at": int(time.time()),
        "files": [
            {
                "path": item.path,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in sorted(manifest.files, key=lambda item: item.path)
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def release_manifest_payload(manifest: ReleaseManifest) -> dict[str, object]:
    return {
        "schema_version": 1,
        "app": APP_NAME,
        "version": manifest.version,
        "root": manifest.root,
        "files": [
            {
                "path": item.path,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in sorted(manifest.files, key=lambda item: item.path)
        ],
    }


def write_local_release_manifest(repo_root: Path, manifest: ReleaseManifest) -> None:
    path = repo_root / RELEASE_MANIFEST_NAME
    text = json.dumps(release_manifest_payload(manifest), indent=2) + "\n"
    with tempfile.NamedTemporaryFile(prefix=f".{RELEASE_MANIFEST_NAME}.", suffix=".update", dir=repo_root, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(text.encode("utf-8"))
    try:
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def resolve_target(repo_root: Path, relative_path: str) -> Path:
    safe = safe_relative_path(relative_path)
    target = (repo_root / safe).resolve()
    try:
        target.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"release path escapes the application folder: {relative_path}") from exc
    return target


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".update", dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def remove_empty_parents(path: Path, stop_at: Path) -> None:
    stop = stop_at.resolve()
    current = path.parent
    while current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def iter_managed_local_files(repo_root: Path) -> set[str]:
    files: set[str] = set()
    for name in MANAGED_TOP_LEVEL_FILES:
        path = repo_root / name
        if path.is_file():
            files.add(name)

    for dirname in MANAGED_TOP_LEVEL_DIRS:
        root = repo_root / dirname
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root).as_posix()
            try:
                files.add(safe_relative_path(relative))
            except ValueError:
                continue
    return files


def obsolete_managed_files(repo_root: Path, manifest: ReleaseManifest) -> set[str]:
    return iter_managed_local_files(repo_root) - manifest.file_paths


def apply_release(
    repo_root: Path,
    release: ValidatedRelease,
    *,
    dry_run: bool = False,
    remove_obsolete_managed: bool = False,
) -> Path | None:
    old_manifest = load_installed_manifest(repo_root)
    old_files = old_manifest.file_paths if old_manifest else set()
    new_files = release.manifest.file_paths
    removals = old_files - new_files
    if remove_obsolete_managed:
        removals |= obsolete_managed_files(repo_root, release.manifest)
    removals = sorted(removals)
    writes = sorted(new_files)

    backup_dir = repo_root / ".cache" / "updater" / "backups" / (
        f"v{read_app_version(repo_root)}-to-v{release.version}-{time.strftime('%Y%m%d-%H%M%S')}"
    )

    emit(f"[INFO] Files to write: {len(writes)}")
    if removals:
        emit(f"[INFO] Obsolete files to remove: {len(removals)}")
    if dry_run:
        emit("[DRY-RUN] Application files were not changed.")
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up: set[str] = set()
    added: set[str] = set()

    try:
        for relative in sorted(set(writes) | set(removals)):
            target = resolve_target(repo_root, relative)
            if target.exists():
                backup_target = backup_dir / relative
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_target)
                backed_up.add(relative)

        for relative in removals:
            target = resolve_target(repo_root, relative)
            if target.exists():
                target.unlink()
                remove_empty_parents(target, repo_root)

        for relative in writes:
            source = release.stage_root / relative
            target = resolve_target(repo_root, relative)
            existed = target.exists()
            copy_atomic(source, target)
            if not existed:
                added.add(relative)

        write_installed_manifest(repo_root, release.manifest)
        write_local_release_manifest(repo_root, release.manifest)
    except Exception:
        emit("[WARN] Update failed while applying files; restoring previous files.")
        for relative in added:
            target = resolve_target(repo_root, relative)
            if target.exists():
                target.unlink()
                remove_empty_parents(target, repo_root)
        for relative in backed_up:
            backup_source = backup_dir / relative
            target = resolve_target(repo_root, relative)
            copy_atomic(backup_source, target)
        raise

    return backup_dir


def verify_sha256_sidecar(release: ReleaseInfo, zip_asset: ReleaseAsset, zip_path: Path, *, require: bool) -> None:
    sidecar = select_sha256_asset(release, zip_asset)
    if sidecar is None:
        if require:
            raise RuntimeError("Release does not provide a SHA256 sidecar asset")
        emit("[INFO] Release SHA256 sidecar was not found; ZIP contents will still be validated.")
        return
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    emit(f"[INFO] Downloading checksum: {sidecar.name}")
    download_file(sidecar.browser_download_url, checksum_path)
    expected = parse_sha256_text(checksum_path.read_text(encoding="utf-8"), zip_asset.name)
    actual = sha256_file(zip_path)
    if actual != expected:
        raise RuntimeError(f"Downloaded ZIP checksum mismatch: expected {expected}, found {actual}")
    emit("[INFO] Release ZIP checksum passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update this app from the official GitHub release ZIP.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Application folder. Defaults to this script's parent repository.",
    )
    parser.add_argument("--version", default="", help="Target release version or tag. Defaults to the latest release.")
    parser.add_argument("--zip", type=Path, default=None, help="Use a local release ZIP instead of GitHub.")
    parser.add_argument("--dry-run", action="store_true", help="Check and stage the update without changing files.")
    parser.add_argument("--force", action="store_true", help="Install even when the target version is not newer.")
    parser.add_argument(
        "--require-sha256",
        action="store_true",
        help="Require a SHA256 sidecar asset when downloading from GitHub.",
    )
    parser.add_argument(
        "--allow-dev-checkout",
        action="store_true",
        help="Allow updating a folder that contains .git. Intended only for maintainer testing.",
    )
    parser.add_argument(
        "--skip-local-normalize",
        action="store_true",
        help="Skip normalization from a local release_manifest.json before checking GitHub.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    current_version = read_app_version(repo_root)

    if (repo_root / ".git").exists() and not args.allow_dev_checkout:
        emit("[ERROR] This looks like a Git checkout. Release self-update is disabled here.")
        emit("[INFO] Use git pull for development checkouts, or pass --allow-dev-checkout for maintainer testing.")
        return 2

    emit("========== App Update ==========")
    emit(f"Application folder: {repo_root}")
    emit(f"Current version: v{current_version}")

    remove_obsolete_managed = not (repo_root / ".git").exists()
    if not args.skip_local_normalize:
        local_release = load_local_release(repo_root)
        if local_release is not None:
            emit(f"[INFO] Normalizing local release files from {RELEASE_MANIFEST_NAME}.")
            apply_release(
                repo_root,
                local_release,
                dry_run=args.dry_run,
                remove_obsolete_managed=remove_obsolete_managed,
            )
            current_version = read_app_version(repo_root)

    if args.zip is None:
        release = fetch_release(args.version.strip() or None)
        target_version = release.version
        target_parsed = parse_version(target_version)
        current_parsed = parse_version(current_version)
        if current_parsed is not None and target_parsed is not None and target_parsed <= current_parsed and not args.force:
            emit(f"Result: already up to date ({version_label(current_version)})")
            emit("================================")
            return 0

        zip_asset = select_zip_asset(release)
        downloads = repo_root / ".cache" / "updater" / "downloads"
        zip_path = downloads / zip_asset.name
        emit(f"[INFO] Latest release: {release.tag_name}")
        emit(f"[INFO] Downloading: {zip_asset.name}")
        download_file(zip_asset.browser_download_url, zip_path)
        verify_sha256_sidecar(release, zip_asset, zip_path, require=args.require_sha256)
    else:
        zip_path = args.zip.resolve()
        target_version = (args.version.strip() or "").lstrip("v")
        if not target_version:
            name = zip_path.stem
            prefix = f"{APP_NAME}-v"
            if not name.startswith(prefix):
                raise RuntimeError("--version is required when --zip does not use the standard release name")
            target_version = name[len(prefix) :]
        emit(f"[INFO] Using local release ZIP: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="360gs-update-stage-") as temp_dir:
        staged = validate_release_zip(zip_path, expected_version=target_version, extract_parent=Path(temp_dir))
        backup_dir = apply_release(
            repo_root,
            staged,
            dry_run=args.dry_run,
            remove_obsolete_managed=remove_obsolete_managed,
        )

    if args.dry_run:
        emit(f"Result: update to {version_label(target_version)} is ready")
    else:
        emit(f"Result: updated to {version_label(target_version)}")
        if backup_dir is not None:
            emit(f"Backup: {backup_dir}")
    emit("================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit(f"[ERROR] {exc}")
        raise SystemExit(1) from exc
