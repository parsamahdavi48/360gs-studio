"""Capability diagnostics used by both the GUI status bar and CLI doctor."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

from gs360studio.adapters.tools import ColmapAdapter, FFmpegAdapter


@dataclass(frozen=True, slots=True)
class Diagnostic:
    diagnostic_id: str
    status: str
    summary: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nvidia_diagnostic() -> Diagnostic:
    executable = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi")
    if not executable:
        return Diagnostic("nvidia", "unavailable", "NVIDIA driver tools were not found", {})
    query = [
        executable,
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        fallback = subprocess.run([executable, "-L"], capture_output=True, text=True, timeout=10, check=False, encoding="utf-8", errors="replace")
        return Diagnostic(
            "nvidia",
            "available" if fallback.returncode == 0 else "warning",
            (fallback.stdout or fallback.stderr).strip() or "NVIDIA GPU detected",
            {"executable": executable, "compute_capability_query": "unsupported"},
        )
    gpus = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        gpus.append(
            {
                "name": fields[0],
                "driver": fields[1],
                "memory_mb": int(float(fields[2])),
                "compute_capability": fields[3],
            }
        )
    return Diagnostic("nvidia", "available", f"Detected {len(gpus)} NVIDIA GPU(s)", {"gpus": gpus, "executable": executable})


def _lichtfeld_diagnostic(nvidia: Diagnostic) -> Diagnostic:
    gpus = nvidia.details.get("gpus")
    if not isinstance(gpus, list) or not gpus or not isinstance(gpus[0], dict):
        return Diagnostic("lichtfeld", "unavailable", "LichtFeld GPU requirements could not be verified", {})
    compatible = []
    for gpu in gpus:
        try:
            compute = float(gpu.get("compute_capability", 0))
            driver = int(str(gpu.get("driver", "0")).split(".", 1)[0])
        except (TypeError, ValueError):
            continue
        if compute >= 7.5 and driver >= 570:
            compatible.append(str(gpu.get("name") or "NVIDIA GPU"))
    if compatible:
        return Diagnostic(
            "lichtfeld",
            "available",
            "LichtFeld hardware preflight passed",
            {"compatible_gpus": compatible, "minimum_compute_capability": 7.5, "minimum_driver": 570},
        )
    return Diagnostic(
        "lichtfeld",
        "unavailable",
        "GPU does not meet LichtFeld's documented compute capability 7.5+ and driver 570+ requirements",
        {"minimum_compute_capability": 7.5, "minimum_driver": 570},
    )


def _pytorch_diagnostic() -> Diagnostic:
    if importlib.util.find_spec("torch") is None:
        return Diagnostic(
            "cuda_ml",
            "optional",
            "PyTorch is not installed; CPU conversion and non-ML masks remain available",
            {},
        )
    try:
        import torch

        available = bool(torch.cuda.is_available())
        details: dict[str, Any] = {"torch_version": str(torch.__version__), "cuda_available": available}
        if available:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            details.update({"device": torch.cuda.get_device_name(0), "free_vram_mb": free_bytes // (1024 * 1024), "total_vram_mb": total_bytes // (1024 * 1024)})
            enough_memory = free_bytes >= 2 * 1024**3
            return Diagnostic(
                "cuda_ml",
                "available" if enough_memory else "warning",
                "CUDA ML masking is available" if enough_memory else "CUDA is available but free VRAM is below the 2 GB preflight threshold",
                details,
            )
        return Diagnostic("cuda_ml", "optional", "PyTorch cannot use this GPU; CPU conversion remains available", details)
    except Exception as exc:
        return Diagnostic("cuda_ml", "warning", "PyTorch capability probe failed; CPU conversion remains available", {"error": str(exc)})


def run_diagnostics() -> list[Diagnostic]:
    ffmpeg = FFmpegAdapter().probe()
    colmap = ColmapAdapter().probe()
    nvidia = _nvidia_diagnostic()
    return [
        Diagnostic(
            "platform",
            "available" if sys.platform == "win32" else "warning",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
            {"python": platform.python_version(), "executable": sys.executable},
        ),
        nvidia,
        _pytorch_diagnostic(),
        _lichtfeld_diagnostic(nvidia),
        Diagnostic("ffmpeg", "available" if ffmpeg.available else "unavailable", ffmpeg.version or ffmpeg.message, ffmpeg.to_dict()),
        Diagnostic("colmap", "available" if colmap.available else "optional", colmap.version or colmap.message, colmap.to_dict()),
    ]
