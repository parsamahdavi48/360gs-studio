# stechdrive-3dgs-utils

**v1.4.0**

A Windows-first integrated GUI tool for turning 360° camera video into images, masks, and camera data that are practical for 3D Gaussian Splatting (3DGS) training.

`setup_windows.bat` detects Python 3.12, installs it when needed, creates a virtual environment, and installs the required runtime packages. Day-to-day launch is handled by `run_gui.bat`, so users do not need to run Python commands manually for the normal GUI workflow.

[JP 日本語の説明](README.ja.md)

Forked from [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils).

![STechDrive 3DGS Utils GUI](images/stechdrive-3dgs-utils-gui.jpg)

## What You Can Do

### 1. 360° Video to Metashape SfM and 3DGS Training

Extract equirectangular still frames from Insta360 or similar 360° camera video, review which frames to keep, and generate masks for people, the camera operator, tripods, stitch seams, and blown-out highlights before running SfM in Metashape.

After Metashape SfM, export viewpoint images, masks, and `transforms.json` for LichtFeld Studio, Postshot, and Brush. This is the main workflow for preparing 360° video as a 3DGS training dataset.

### 2. 360° Video to COLMAP Rig Dataset

You can also skip Metashape and export a COLMAP Rig viewpoint dataset from extracted 360° frames. The GUI can optionally run COLMAP/GLOMAP so the result is ready to pass to COLMAP-compatible 3DGS tools.

### 3. Mask Preprocessing for Normal Photos or Video Frames

For DSLR, mirrorless, smartphone, or normal video image sequences, Step 3 can generate YOLO/SAM masks for people, vehicles, and other classes, plus overexposure masks. This is useful as a mask-preparation stage before sending images to SfM software.

## Highlights

- Extract SfM-friendly frames from 360° video
- Review extracted frames in single-preview or thumbnail-list mode and apply keep/drop decisions, including Windows Explorer-style thumbnail selection
- Generate masks with YOLO + SAM2.1
- Improve detection near the bottom of 360° images for camera operators, tripods, and hands
- Mask stitch seams, overexposed regions, and user-provided PNG custom masks
- Preview mask results in single-preview or thumbnail-list mode while tuning settings, with cached thumbnails for large image sets
- Reprocess the current preview image or multiple selected thumbnails with updated settings
- Convert Metashape SfM results for LichtFeld Studio, Postshot, and Brush
- Export COLMAP Rig viewpoint datasets and optionally run COLMAP/GLOMAP
- Windows setup scripts and a Japanese/English GUI

## Easy Setup

For a normal release ZIP, extract it and run:

```bat
setup_windows.bat
run_gui.bat
```

`setup_windows.bat` looks for Python 3.12 and can install it through winget when needed. It then creates `.venv`, installs packages such as PyTorch CUDA wheels, OpenCV, Pillow, Open3D, ultralytics, and PySide6, and verifies the environment.

`run_gui.bat` activates `.venv` and launches the integrated GUI. If an existing `.venv` is already healthy, setup reports that state and does not rebuild it. Use `setup_windows.bat --force` when you intentionally want to recreate the environment.

To update an existing `.venv` to the latest compatible package set, run:

```bat
update_venv.bat
```

To rebuild with the pinned known-good package set from `requirements/`, run `update_venv.bat --locked`.

YOLO/SAM2 model weights are downloaded automatically by ultralytics on first use. Release ZIP assets do not include model weights or generated scene data.

## GUI Workflow

If the scene folder path contains non-ASCII characters, an extremely long path, control characters, or `"`, the GUI stops before running. These paths are likely to fail in OpenCV or external 3DGS/SfM tools. Spaces and OneDrive paths are not blocked by themselves. Use a short ASCII working path, for example `D:\work\scene01`.

```text
360° video or images
  -> Step 1: frame extraction
  -> Step 2: frame review and keep/drop decisions
  -> Step 3: mask generation
  -> Step 4: export
      -> build 3DGS-ready outputs from Metashape SfM results
      -> export COLMAP Rig viewpoint images and optionally run COLMAP/GLOMAP
```

| Step | Purpose | Current Default |
| --- | --- | --- |
| 1. Frame Extraction | Extract equirectangular still frames from 360° video | Fixed interval + motion adjustment |
| 2. Frame Review | Review extracted frames in single/thumbnail views and apply keep/drop decisions to CSV | Review low-quality candidates and unwanted frames |
| 3. Mask Generation | Generate YOLO, stitch seam, overexposure, and custom masks | YOLO enabled, quality setting for 360° images |
| 4. Export | Export 3DGS outputs from SfM results, or export COLMAP Rig viewpoint images | Metashape Import / LichtFeld / Full / Cube6 |

## Recommended Workflow: Metashape Route

1. Prepare 360° video from an Insta360 or similar camera.
2. Extract SfM-friendly frames in Step 1.
3. Review low-quality or unnecessary frames in Step 2.
4. Generate masks for people, camera operators, tripods, and similar objects in Step 3. For 360° images, `YOLO Level 2 Quality` is the recommended starting point.
5. If the bottom-view camera operator is missed, raise `Bottom Enhance` to `High` or `Max`.
6. Enable stitch seam and overexposure masks when they match the source material.
7. Import the generated `masks/` folder into Metashape as per-image masks, then run SfM.
8. Use Step 4 with the Metashape XML/PLY result to export training images, masks, and `transforms.json`.

## COLMAP Route

1. Use Steps 1-3 in the same way as the Metashape route.
2. In Step 4, choose `COLMAP Export` to write viewpoint images and masks to `output/colmap_rig/`.
3. Enable `Run COLMAP after export` when you want the app to run Feature, Matcher, and Mapper stages.
4. After completion, pass `output/colmap_rig/` as the COLMAP project folder to COLMAP-compatible 3DGS tools.

## Mask Preprocessing for Normal Images

For normal video frames or still-camera image sequences placed in `images/`, choose `Image Type: Normal` in Step 3. This keeps YOLO/SAM and overexposure masking available while disabling stitch seam masking and 360° bottom re-detection.

Use this when you want to exclude people, vehicles, blown-out regions, or similar areas before importing images into SfM software.

## Mask Tuning Notes

- For 360° images, start with `YOLO Level 2 Quality`.
- Use `1 Standard` for faster test runs.
- If people leak through, try `3 Best` or raise `Expand` slightly.
- When you find a miss in preview, adjust settings and use `Reprocess Current` to save only that image back to `masks/`. In thumbnail mode, use `Ctrl` / `Shift` selection to reprocess multiple selected images together. Arrow keys move the visible thumbnail selection while in thumbnail mode.
- If only the bottom-view camera operator leaks through, try `Bottom Enhance` in this order: `Standard -> High -> Max`.
- `Max` is slower and can mask extra floor or ground.
- Stitch seam masks are useful when the seam position is stable in the equirectangular image. If FlowState stabilization, direction lock, AI stitching, or similar processing moves the seam, verify it in the preview before using it.

## Requirements

- Windows 10/11
- Python 3.12 (3.12.10 confirmed)
- CUDA-capable GPU
- CUDA Toolkit 12.8
- FFmpeg / FFprobe

Main Python packages resolved by `setup_windows.bat`:

```text
torch / torchvision / torchaudio from the CUDA 12.8 wheel index
numpy, opencv-python, Pillow, open3d, ultralytics, tqdm, PySide6
```

`setup_windows.bat` uses the pinned known-good package set under `requirements/` for reproducible first-time setup. `update_venv.bat` resolves the latest compatible packages by default; pass `--locked` when you want to rebuild from the pinned set instead.

## CLI Tools

The GUI wraps these CLI engines, which can also be used directly.

| Script | Purpose | Docs |
| --- | --- | --- |
| `extract_frames.py` | Extract frames from 360° video | [EN](doc/extract_frames.md) |
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
