# stechdrive-3dgs-utils

**v1.0.0**

A Windows desktop toolkit for preparing frames, masks, and camera data from 360 video or image sequences for SfM and 3D Gaussian Splatting (3DGS) workflows. The current workflow is centered on the integrated PySide6 GUI with Japanese and English UI support, and is designed to run from `setup_windows.bat` and `run_gui.bat` without requiring users to manually assemble a Python environment.

[JP 日本語の説明](README.ja.md)

Forked from [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils).

## What This Tool Does

```text
360 video
  -> frame extraction
  -> frame review and keep/drop decisions
  -> mask generation
  -> export
      -> build 3DGS-ready outputs from Metashape SfM results
      -> export COLMAP Rig viewpoint images and optionally run COLMAP/GLOMAP
```

The main goal is to turn equirectangular 360 video into practical SfM/3DGS datasets. The high-accuracy route keeps source frames as equirectangular images for Metashape SfM, then converts the SfM result to viewpoint images, masks, and `transforms.json` for downstream 3DGS tools. A COLMAP route can also export COLMAP Rig viewpoint images from extracted equirectangular frames and optionally run COLMAP/GLOMAP. Masks are generated before SfM so that people, vehicles, stitching seams, and blown-out highlights can be excluded from feature matching.

## Quick Start

```bat
setup_windows.bat
run_gui.bat
```

`setup_windows.bat` detects Python 3.12, installs Python 3.12 through winget when needed, and creates a verified `.venv`. If an existing `.venv` is already healthy, setup reports that state and does not rebuild it. Use `setup_windows.bat --force` to rebuild intentionally. Package versions are resolved at setup time, with PyTorch installed from the CUDA 12.8 wheel index.

The setup window stays open at the end so the summary can be read. Use `setup_windows.bat --no-pause` when running from an existing terminal.

`run_gui.bat` activates the venv and launches the integrated GUI.

To update an existing `.venv` to the latest compatible package set, run:

```bat
update_venv.bat
```

`update_venv.bat` discovers installed and winget-available Python candidates from newest to oldest. It first runs pip dry-run compatibility checks for the target Python ABI, then installs a missing Python through winget only when that candidate is likely to work. The first candidate that builds a temporary venv and passes `pip check`, import/CUDA smoke tests, and pytest is promoted to `.venv`.

The update window stays open at the end so the summary can be read. Use `update_venv.bat --no-pause` when running from an existing terminal.

## GUI Workflow

| Step | Purpose | Current Default |
| --- | --- | --- |
| 1. Frame Extraction | Extract equirectangular still images from 360 video | Fixed interval + motion adjustment |
| 2. Frame Review | Review extracted frames and apply keep/drop decisions | Representative replacements and low-quality review flags |
| 3. Mask Generation | Generate YOLO, stitch seam, and overexposure masks | YOLO enabled, person detection as the baseline |
| 4. Export | Export 3DGS outputs from SfM results, or export COLMAP Rig viewpoint images | Metashape Import / LichtFeld / Full (Quality) / Cube6 |

### Current GUI Direction

- Workflow navigation uses a compact vertical step tab on the left.
- Run and Cancel actions are unified at the bottom of the main window.
- The left settings pane in each step has a fixed width, vertical-only scrolling, and consistent padding near the preview splitter.
- Long or rarely used settings are collapsible. The YOLO 80-class list, stitch/overexposure settings, and advanced cubemap view grid stay folded until needed.
- The Step 4 Metashape route defaults to `LichtFeld`, `Full (Quality)`, and `Cube6 (4 sides + top/bottom)`.
- Step 4 displays `Selected Views` and `Output Images`. `Output Images` is the deterministic output image count from input image count multiplied by enabled view count.
- The Step 4 COLMAP route writes a COLMAP Rig dataset under `output/colmap_rig/` and can optionally run COLMAP/GLOMAP from a user-selected executable.

## Recommended Workflow: Metashape Route

1. Prepare a 360 video from an Insta360 or similar camera.
2. Extract frames in Step 1.
3. Review and drop unusable frames in Step 2.
4. Generate person masks in Step 3. Enable stitch seam and overexposure masks only when they match the source material.
5. Import the generated `masks/` folder into Metashape as per-image masks, then run SfM.
6. Use Step 4 with the Metashape XML/PLY result to export training images, masks, and `transforms.json`.

## COLMAP Route

1. Use Steps 1-3 as usual.
2. In Step 4, choose `COLMAP Export` to write viewpoint images and masks to `output/colmap_rig/`.
3. Enable `Run COLMAP after export` when you want the app to run Feature, Matcher, and Mapper stages.
4. After completion, pass `output/colmap_rig/` as the COLMAP project folder to COLMAP-compatible 3DGS apps.

For normal video frames or still-camera image sequences placed in `images/`, choose `Image Type: Normal` in Step 3. This keeps YOLO/SAM and overexposure masking available while disabling stitch seam masking and 360 bottom re-detection.

Stitch seam masks are useful when the seam position is stable in the equirectangular image. If FlowState stabilization, direction lock, AI stitching, or similar processing moves the seam, leave seam masking disabled unless you have verified the preview.

## Requirements

- Windows 10/11
- Python 3.12 (3.12.10 confirmed)
- CUDA-capable GPU
- CUDA Toolkit 12.8
- FFmpeg / FFprobe

Main Python packages resolved by `setup_windows.bat`:

```text
torch / torchvision / torchaudio from the CUDA 12.8 wheel index
numpy, opencv-python, Pillow, open3d, ultralytics, tqdm, PySide6, pytest
```

Both setup and update intentionally avoid fixed package pins. They resolve and verify the latest compatible versions for the selected Python environment.

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
