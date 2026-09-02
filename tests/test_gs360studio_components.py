from __future__ import annotations

from pathlib import Path

from gs360studio.domain.models import ComponentManifest
from gs360studio.platform.components import ComponentManager


def test_system_component_is_discovered_on_path(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "tool.exe"
    executable.write_bytes(b"tool")
    manager = ComponentManager(
        [ComponentManifest(component_id="tool", version="system", display_name="Tool", executable="tool.exe")],
        root=tmp_path / "components",
    )
    monkeypatch.setattr("gs360studio.platform.components.shutil.which", lambda _name: str(executable))

    state = manager.state("tool")

    assert state.installed and state.verified
    assert state.executable_path == executable.resolve()
