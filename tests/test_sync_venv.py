from __future__ import annotations

from pathlib import Path

from scripts import sync_venv


def test_build_install_steps_syncs_pinned_runtime_groups_in_order() -> None:
    steps = sync_venv.build_install_steps(Path(".venv/Scripts/python.exe"), locked=True, dry_run=True)

    labels = [step.label for step in steps]
    assert labels[:4] == ["pip", "core requirements", "PyTorch CUDA requirements", "ML requirements"]
    assert "SAM3.1 requirements" in labels
    assert "SAM3.1 source package" in labels

    for step in steps:
        assert "--dry-run" in step.command

    torch_step = steps[2]
    assert "--index-url" in torch_step.command
    assert sync_venv.TORCH_INDEX_URL in torch_step.command

    sam_source = next(step for step in steps if step.label == "SAM3.1 source package")
    assert "--no-deps" in sam_source.command
    assert any("github.com/facebookresearch/sam3" in part for part in sam_source.command)
