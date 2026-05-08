from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.sam31_download import SAM31_CHECKPOINT_NAME, SAM31_REPO_ID, download_sam31_checkpoint


def test_download_sam31_checkpoint_requires_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token"):
        download_sam31_checkpoint("", tmp_path)


def test_download_sam31_checkpoint_uses_token_without_persisting(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["local_dir"]) / kwargs["filename"]
        output.write_bytes(b"checkpoint")
        return str(output)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    path = download_sam31_checkpoint(" hf_token ", tmp_path)

    assert path == tmp_path / SAM31_CHECKPOINT_NAME
    assert path.read_bytes() == b"checkpoint"
    assert calls == [
        {
            "repo_id": SAM31_REPO_ID,
            "filename": SAM31_CHECKPOINT_NAME,
            "local_dir": tmp_path,
            "token": "hf_token",
        }
    ]
