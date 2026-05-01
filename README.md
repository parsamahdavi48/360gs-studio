# stechdrive-3dgs-utils

A studio-grade toolkit for building 3D Gaussian Splatting (3DGS) training assets from 360-degree video, with an integrated Japanese GUI.

Forked from [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils) and extended with a unified workflow GUI, additional masking features, and multi-framework export support.

[JP 日本語の説明](README.ja.md)

## Workflow Overview

```
360 Video  ──  Frame Extraction  ──  Review & Selection  ──  Mask Generation
                                                                    │
                                                          Metashape SfM (manual, with masks)
                                                                    │
                                                          Cubemap Conversion  ──  3DGS Training
                                                          (Postshot / Brush /
                                                           LichtFeld Studio)
```

Masks are generated **before** Metashape SfM and imported into Metashape so that moving subjects (people, vehicles), stitching seams, and blown-out highlights are excluded from feature matching. This significantly improves SfM accuracy and downstream 3DGS quality.

| Step | Description |
|------|-------------|
| **1. Frame Extraction** | Extract equirectangular stills from 360 video at fixed intervals (recommended) or change-based selection |
| **2. Frame Review** | Review frames with blur worst-order navigation, bulk drop by threshold |
| **3. Mask Generation** | YOLO+SAM2.1 person detection, stitch seam masking, overexposure (blown-out pixel) masking. **Import the resulting `masks/` folder into Metashape as per-image masks before running SfM.** |
| **4. Cubemap Conversion** | After Metashape SfM, convert the equirectangular result (XML + PLY) to cubemap views with transforms.json for Postshot, Brush, or LichtFeld Studio. Masks are propagated to cubemap faces |

## Quick Start (Windows)

```bat
setup_windows.bat
start_gui.bat
```

`setup_windows.bat` installs Python 3.11 (via winget if needed), creates a venv, and installs all dependencies including PyTorch with CUDA 12.8.

`start_gui.bat` launches the unified GUI application (dark modern theme, Japanese UI).

## Requirements

- Windows 10/11
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)
- [Python 3.11](https://www.python.org/) (3.11.8 confirmed)
- [FFmpeg / FFprobe](https://ffmpeg.org/)
- GPU with CUDA support (for YOLO/SAM2 and PyTorch)

### Python Dependencies

Installed automatically by `setup_windows.bat`:

```
torch==2.8.0 (CUDA 12.8), torchvision, torchaudio
numpy, opencv-python, Pillow, open3d, ultralytics, tqdm, PySide6
```

### ML Model Files

YOLO and SAM2 model weights are downloaded automatically by ultralytics on first use:
- `yolo26m.pt` / `yolo26l.pt` (YOLO v26)
- `sam2.1_l.pt` (SAM 2.1 Large)

## CLI Tools

All CLI engines can be used independently without the GUI:

| Script | Description | Docs |
|--------|-------------|------|
| `extract_frames.py` | Extract frames from 360 video with change/fixed selection + blur replacement | [EN](doc/extract_frames.md) |
| `apply_frame_decisions.py` | Apply keep/drop decisions from CSV | [EN](doc/apply_frame_decisions.md) |
| `review_frames.py` | Frame review GUI with blur worst-order navigation | [EN](doc/review_frames.md) |
| `yolo_mask.py` | YOLO+SAM2.1 person detection masks for 360 images | [EN](doc/yolo_mask.md) |
| `stitch_mask.py` | Angular stitch seam masks for dual-fisheye cameras | [EN](doc/stitch_mask.md) |
| `overexposure_mask.py` | Blown-out pixel detection and mask merging | - |
| `cubemap_transforms_json.py` | Equirectangular to cubemap transforms.json conversion | [EN](doc/cubemap_transforms_json.md) |

## Changes from Upstream

This fork adds:
- **Unified GUI** (`gui/`) with dark modern theme and Japanese UI (PySide6)
- **4-step workflow** in a single tabbed window
- **Overexposure mask** detection (`overexposure_mask.py`)
- **Blur worst-order navigation** and bulk threshold drop in review GUI
- **Brush profile** support (`--brush` coordinate transform)
- **Removed**: COLMAP rig export, RealityScan rig export (Metashape-based SfM workflow only)

Upstream changes are periodically synced from [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils).

## License

MIT License. See [LICENSE](LICENSE).

Original code by [tetraface Inc.](https://github.com/tetraface)
Fork extensions by [stechdrive](https://github.com/stechdrive)
