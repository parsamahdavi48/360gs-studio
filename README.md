# stechdrive-3dgs-utils

**v0.3.0**

A Windows desktop toolkit for preparing equirectangular 360 video for Metashape SfM and 3D Gaussian Splatting (3DGS) training. The current workflow is centered on the integrated Japanese PySide6 GUI and is designed to run from `setup_windows.bat` and `start_gui.bat` without requiring users to manually assemble a Python environment.

[JP 日本語の説明](README.ja.md)

Forked from [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils).

## What This Tool Does

```text
360 video
  -> frame extraction
  -> frame review and keep/drop decisions
  -> mask generation
  -> Metashape SfM
  -> cubemap conversion
  -> 3DGS training
```

The intended pipeline keeps the source frames as equirectangular images for Metashape SfM, then converts the SfM result to perspective training images, masks, and `transforms.json` for downstream 3DGS tools. Masks are generated before Metashape SfM so that people, vehicles, stitching seams, and blown-out highlights can be excluded from feature matching.

## Quick Start

```bat
setup_windows.bat
start_gui.bat
```

`setup_windows.bat` detects Python 3.11, installs Python 3.11 through winget when needed, creates `.venv`, and installs PyTorch CUDA 12.8 plus the project dependencies.

`start_gui.bat` activates the venv and launches the integrated GUI.

## GUI Workflow

| Step | Purpose | Current Default |
| --- | --- | --- |
| 1. Frame Extraction | Extract equirectangular still images from 360 video | Fixed interval or change-based selection |
| 2. Frame Review | Review extracted frames and apply keep/drop decisions | Blur worst-order review support |
| 3. Mask Generation | Generate YOLO, stitch seam, and overexposure masks | YOLO enabled, person detection as the baseline |
| 4. Cubemap Conversion | Convert Metashape output to 3DGS perspective images and JSON | LichtFeld / Full / Cube6 |

### Current GUI Direction

- Workflow navigation uses a compact vertical step tab on the left.
- Run and Cancel actions are unified at the bottom of the main window.
- The left settings pane in each step has a fixed width, vertical-only scrolling, and consistent padding near the preview splitter.
- Long or rarely used settings are collapsible. The YOLO 80-class list, stitch/overexposure settings, and advanced cubemap view grid stay folded until needed.
- Step 4 defaults to `LichtFeld`, `Full (1.0x)`, and `Cube6 (4 sides + top/bottom)`.
- Step 4 displays `Selected Views` and `Output Images`. `Output Images` is the deterministic output image count from input image count multiplied by enabled view count.

## Recommended Workflow

1. Prepare a 360 video from an Insta360 or similar camera.
2. Extract frames in Step 1.
3. Review and drop unusable frames in Step 2.
4. Generate person masks in Step 3. Enable stitch seam and overexposure masks only when they match the source material.
5. Import the generated `masks/` folder into Metashape as per-image masks, then run SfM.
6. Use Step 4 with the Metashape XML/PLY result to export training images, masks, and `transforms.json`.

Stitch seam masks are useful when the seam position is stable in the equirectangular image. If FlowState stabilization, direction lock, AI stitching, or similar processing moves the seam, leave seam masking disabled unless you have verified the preview.

## Requirements

- Windows 10/11
- Python 3.11 (3.11.8 confirmed)
- CUDA-capable GPU
- CUDA Toolkit 12.8
- FFmpeg / FFprobe

Main Python packages installed by `setup_windows.bat`:

```text
torch==2.8.0 (CUDA 12.8), torchvision, torchaudio
numpy, opencv-python, Pillow, open3d, ultralytics, tqdm, PySide6
```

YOLO/SAM2 model weights are downloaded automatically by ultralytics on first use.

## CLI Tools

The GUI wraps these CLI engines, which can also be used directly.

| Script | Purpose | Docs |
| --- | --- | --- |
| `extract_frames.py` | Extract frames from 360 video | [EN](doc/extract_frames.md) |
| `apply_frame_decisions.py` | Apply keep/drop decisions from CSV | [EN](doc/apply_frame_decisions.md) |
| `review_frames.py` | Frame review GUI | [EN](doc/review_frames.md) |
| `yolo_mask.py` | YOLO+SAM2.1 mask generation | [EN](doc/yolo_mask.md) |
| `stitch_mask.py` | Stitch seam mask generation | [EN](doc/stitch_mask.md) |
| `overexposure_mask.py` | Overexposure mask generation | - |
| `cubemap_transforms_json.py` | Convert equirectangular images to cubemap views | [EN](doc/cubemap_transforms_json.md) |
| `transforms_to_colmap.py` | Export COLMAP files from `transforms.json` | [EN](doc/transforms_to_colmap.md) |

## License

MIT License. See [LICENSE](LICENSE).

Original code by [tetraface Inc.](https://github.com/tetraface)
Fork extensions by [stechdrive](https://github.com/stechdrive)
