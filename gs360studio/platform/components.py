"""Checksum-verified optional component registry and installer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gs360studio.domain.models import ComponentManifest, atomic_write_json


def default_component_root() -> Path:
    configured = os.environ.get("GS360_COMPONENT_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "360GS Studio" / "components"
    return Path.home() / ".360gs-studio" / "components"


@dataclass(frozen=True, slots=True)
class ComponentState:
    manifest: ComponentManifest
    installed: bool
    executable_path: Path | None
    verified: bool
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.manifest.to_dict(),
            "installed": self.installed,
            "executable_path": str(self.executable_path) if self.executable_path else None,
            "verified": self.verified,
            "message": self.message,
        }


class ComponentManager:
    def __init__(self, manifests: list[ComponentManifest], root: str | Path | None = None) -> None:
        self.root = Path(root) if root else default_component_root()
        self._manifests = {manifest.component_id: manifest for manifest in manifests}

    @classmethod
    def from_registry(cls, path: str | Path, root: str | Path | None = None) -> ComponentManager:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported component registry schema")
        items = payload.get("components")
        if not isinstance(items, list):
            raise ValueError("component registry must contain a components list")
        return cls([ComponentManifest.from_dict(item) for item in items], root=root)

    def manifests(self) -> tuple[ComponentManifest, ...]:
        return tuple(self._manifests[key] for key in sorted(self._manifests))

    def manifest(self, component_id: str) -> ComponentManifest:
        try:
            return self._manifests[component_id]
        except KeyError as exc:
            raise KeyError(f"unknown component: {component_id}") from exc

    def component_dir(self, manifest: ComponentManifest) -> Path:
        return self.root / manifest.component_id / manifest.version

    def state(self, component_id: str) -> ComponentState:
        manifest = self.manifest(component_id)
        if manifest.version == "system" and manifest.executable:
            discovered = shutil.which(manifest.executable)
            if discovered:
                path = Path(discovered).resolve()
                return ComponentState(manifest, True, path, True, "Discovered on PATH")
            return ComponentState(manifest, False, None, False, "Not found on PATH")
        executable = self.component_dir(manifest) / manifest.executable if manifest.executable else None
        installed_manifest = self.component_dir(manifest) / "installed.json"
        installed = installed_manifest.is_file() and (executable is None or executable.is_file())
        verified = False
        message = "Not installed"
        if installed:
            try:
                payload = json.loads(installed_manifest.read_text(encoding="utf-8"))
                verified = payload.get("sha256") == manifest.sha256 and payload.get("version") == manifest.version
                message = "Installed and verified" if verified else "Installed metadata does not match registry"
            except (OSError, json.JSONDecodeError):
                message = "Installed metadata is unreadable"
        return ComponentState(manifest, installed, executable, verified, message)

    def install(
        self,
        component_id: str,
        *,
        accept_license: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> ComponentState:
        manifest = self.manifest(component_id)
        if manifest.license_id and not accept_license:
            raise PermissionError(f"license acceptance is required for {manifest.display_name} ({manifest.license_id})")
        if not manifest.download_url or not manifest.sha256:
            raise ValueError(f"{manifest.display_name} is user-managed and has no automatic download")
        destination = self.component_dir(manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="360gs-component-") as temp_text:
            temp = Path(temp_text)
            archive = temp / "download"
            request = urllib.request.Request(manifest.download_url, headers={"User-Agent": "360GS-Studio-component-manager"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
                total = int(response.headers.get("Content-Length") or 0)
                current = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    current += len(chunk)
                    if progress:
                        progress(current, total)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest.lower() != manifest.sha256.lower():
                raise ValueError(f"checksum verification failed for {manifest.display_name}")
            staged = temp / "staged"
            staged.mkdir()
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as package:
                    for info in package.infolist():
                        target = (staged / info.filename).resolve()
                        if staged.resolve() not in target.parents and target != staged.resolve():
                            raise ValueError("component archive contains an unsafe path")
                    package.extractall(staged)
            else:
                target_name = Path(manifest.download_url).name or manifest.executable or "component.bin"
                shutil.copy2(archive, staged / target_name)
            atomic_write_json(
                staged / "installed.json",
                {
                    "schema_version": 1,
                    "component_id": manifest.component_id,
                    "version": manifest.version,
                    "sha256": manifest.sha256,
                    "license_id": manifest.license_id,
                },
            )
            if destination.exists():
                backup = destination.with_name(f"{destination.name}.previous")
                if backup.exists():
                    shutil.rmtree(backup)
                destination.replace(backup)
            staged.replace(destination)
        return self.state(component_id)

    def remove(self, component_id: str) -> None:
        manifest = self.manifest(component_id)
        target = self.component_dir(manifest).resolve()
        root = self.root.resolve()
        if root not in target.parents:
            raise ValueError("refusing to remove a component outside the managed root")
        if target.exists():
            shutil.rmtree(target)


def bundled_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "resources" / "components.json"


def bundled_component_manager(root: str | Path | None = None) -> ComponentManager:
    return ComponentManager.from_registry(bundled_registry_path(), root=root)
