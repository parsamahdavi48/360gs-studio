from __future__ import annotations

from pathlib import Path

SAM31_REPO_ID = "facebook/sam3.1"
SAM31_CHECKPOINT_NAME = "sam3.1_multiplex.pt"


def download_sam31_checkpoint(token: str, target_dir: str | Path) -> Path:
    """Download the gated SAM3.1 checkpoint into the app-local models directory."""
    token = token.strip()
    if not token:
        raise ValueError("Hugging Face access token is required.")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional runtime setup
        raise RuntimeError(
            "huggingface_hub is not installed. Run setup_windows.bat, then try SAM3.1 again."
        ) from exc

    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=SAM31_REPO_ID,
        filename=SAM31_CHECKPOINT_NAME,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    checkpoint = Path(downloaded)
    if not checkpoint.is_file():
        raise RuntimeError(f"SAM3.1 checkpoint download did not create a file: {checkpoint}")
    return checkpoint
