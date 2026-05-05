# stechdrive-3dgs-utils

**v1.8.0**

A Windows-first integrated GUI tool for turning 360° camera video into images, masks, and camera data that are practical for 3D Gaussian Splatting (3DGS) training.

`setup_windows.bat` detects Python 3.12 and FFmpeg/FFprobe, installs missing system dependencies through winget when needed, creates a virtual environment, and installs the required runtime packages. Day-to-day launch is handled by `run_gui.bat`, so users do not need to run Python commands manually for the normal GUI workflow.

[JP 日本語の説明](README.ja.md)

Forked from [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils).

![STechDrive 3DGS Utils GUI](images/stechdrive-3dgs-utils-gui.jpg)

## What You Can Do

### 1. 360° Video to Metashape SfM and 3DGS Training

Extract equirectangular still frames from Insta360 or similar 360° camera video, review which frames to keep, and generate masks for people, the camera operator, tripods, sky, stitch seams, and blown-out highlights before running SfM in Metashape.

After Metashape SfM, export viewpoint images, masks, and `transforms.json` for LichtFeld Studio, Postshot, and Brush. This is the main workflow for preparing 360° video as a 3DGS training dataset.

### 2. 360° Video to COLMAP Rig Dataset

You can also skip Metashape and export a COLMAP Rig viewpoint dataset from extracted 360° frames. The GUI can optionally run COLMAP/GLOMAP so the result is ready to pass to COLMAP-compatible 3DGS tools.

### 3. Mask Preprocessing for Normal Photos or Video Frames

For DSLR, mirrorless, smartphone, or normal video image sequences, Step 3 can generate fast YOLO/SAM2.1 masks for people, vehicles, and other selectable object types, higher-accuracy SAM3.1 prompt masks for people and sky, optional Mask2Former sky masks, plus overexposure masks. This is useful as a mask-preparation stage before sending images to SfM software.

## Highlights

- Extract 360° video into still frames that are practical for SfM and 3DGS training. The GUI can thin footage for walking shots or aerial/distant scenes, and it marks frames that may need review because they are blurry, too similar, or contain a large viewpoint change.
- Review extracted frames in a large single-image view or a thumbnail list, then mark unwanted frames as keep/drop decisions. For 360° images, the 90° FOV perspective view lets you inspect details in a normal-camera-like view.
- Generate masks for people, the camera operator, tripods, hands, vehicles, sky, blown-out highlights, and stitch seams. Use YOLO/SAM2.1 when you want fast person-focused masks, or SAM3.1 when you want higher-accuracy people and sky masks plus prompt-based cleanup after generation.
- Preview mask results before saving and inspect them in the thumbnail list. When only a few frames have misses or false detections, regenerate just those frames instead of rerunning the whole image set.
- With SAM3.1, add missed targets such as tripods or subtract false detections such as signs and logos from existing masks. This reduces the amount of manual mask painting needed after the first pass.
- Mask2Former remains available as a helper option when you want to try sky masks without setting up SAM3.1.
- Use the same mask-preparation workflow for normal photo sets or normal video frame sequences, not only 360° images. This is useful before sending images to SfM software.
- Import Metashape SfM results and export viewpoint images, masks, and `transforms.json` for Postshot, Brush, and LichtFeld Studio. Output profiles and viewpoints can be chosen in the GUI.
- Skip Metashape when needed by exporting COLMAP Rig viewpoint images and masks from extracted 360° frames. The GUI can optionally continue into COLMAP/GLOMAP SfM processing.
- Prepare the Windows environment with setup scripts that handle Python, FFmpeg/FFprobe, and the main Python packages. Normal use starts from `run_gui.bat`.

## Easy Setup

For a normal release ZIP, extract it and run:

```bat
setup_windows.bat
run_gui.bat
```

The first `setup_windows.bat` run can take a while. It checks Python 3.12, FFmpeg/FFprobe, GPU-oriented Python packages, and prepares missing pieces where it can.

Python packages are installed into a virtual environment dedicated to this app, so your everyday Python environment is less likely to be affected. After setup completes, normal use is just running `run_gui.bat` to launch the GUI.

### What Setup Does Internally

`setup_windows.bat` looks for Python 3.12 and FFmpeg/FFprobe and can install missing system dependencies through winget when needed. It then creates this app's dedicated virtual environment under `.venv/`, installs packages such as PyTorch CUDA wheels, OpenCV, Pillow, Open3D, ultralytics, PySide6, and the SAM3.1 runtime, and verifies the environment.

Python packages are kept inside `.venv/`, so they are not normally installed into the system-wide Python environment or other projects. `.venv/` is an internal working directory, and you usually do not need to edit it manually.

### Updating or Rebuilding the Environment

This is usually unnecessary. To update an existing environment to the latest compatible package set, run:

```bat
update_venv.bat
```

To rebuild with the pinned known-good package set from `requirements/`, run `update_venv.bat --locked`. To recreate the environment from scratch, run `setup_windows.bat --force`.

YOLO/SAM2, Mask2Former, and SAM3.1 model weights may be downloaded on first use. Local YOLO/SAM weights can be placed under `models/ultralytics/`; local Mask2Former weights can be placed under `models/mask2former-swin-large-ade-semantic/`; SAM3.1 prompt masking uses `models/sam3.1/sam3.1_multiplex.pt`. Legacy `.pt` files in the repository root are still detected for compatibility. Release ZIP assets do not include model weights, generated scene data, user settings, or local setup logs. These third-party libraries and model weights are governed by separate license terms; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

### Mask Generation Model Guide

- Use YOLO/SAM2.1 when you want fast person-only masks.
- Use SAM3.1 when you want the highest practical accuracy for people or sky. Because it is prompt-controlled, you can add missed targets after generation or subtract false detections.
- Use Mask2Former when you want to try sky masks without setting up SAM3.1.

### SAM3.1 Prompt Masks

`setup_windows.bat` installs the SAM3.1 runtime package, but the checkpoint is not bundled because access requires your Hugging Face account and SAM License acceptance.

This app uses the official `facebook/sam3.1` `sam3.1_multiplex.pt` checkpoint. SAM3.1 is a CUDA-GPU-oriented model. Running it on an NVIDIA GPU environment is recommended.

If GPU memory runs out during SAM3.1 batch processing, completed masks remain saved. Rerun with the same settings to resume from unfinished images.

When mask accuracy is the priority, SAM3.1 is recommended over YOLO/SAM2.1. Use SAM3.1 when you want more accurate prompt-controlled masks, especially for sky masks or targeted cleanup. After generating masks once, you can select only the images that need correction and use SAM3.1 prompts to add missed regions such as `tripod`, `hand`, `selfie stick`, or `cell phone`, or subtract false detections such as `male icon`, `female icon`, `logo`, or `sign`.

1. Create or sign in to a Hugging Face account.
2. Open Meta's [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) Hugging Face repository and request access/accept the SAM License. Hugging Face gated model requests are tied to an individual user account and may require sharing your username/email with the model author.
   - Hugging Face gated models can use automatic or manual approval. If you can open the Files tab or download `sam3.1_multiplex.pt` from `facebook/sam3.1` in the browser after accepting the terms, your account already has access and you do not need to wait for an email reply. If the page shows a pending/approval-waiting state, wait for the model author approval.
3. Create a Hugging Face access token from your account settings.
   - App downloads require a `Read` token created by the same Hugging Face account that has access. Browser login state is not used by this app.
   - Copy the token value immediately after creating it. Hugging Face may not show existing token values again from the token list. If you missed the value, create a new `Read` token or use `Invalidate and refresh` to issue a new value. Refreshing invalidates the old token.
   - Treat access tokens as secrets equivalent to passwords. Do not paste them into README files, issues, chats, screenshots, or logs. `Read` permission is enough for downloading the SAM3.1 checkpoint. Prefer creating a dedicated token for SAM3.1, and delete or refresh it from Hugging Face settings when you no longer need it. This app does not save the token.
4. In Step 3, choose `SAM3.1`. If `models/sam3.1/sam3.1_multiplex.pt` is missing, the app asks for the token and downloads the checkpoint. The token is used only for that download and is not saved by this app.

You can also place the checkpoint manually at `models/sam3.1/sam3.1_multiplex.pt`.

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
| 3. Mask Generation | Generate model-based masks plus optional stitch seam, overexposure, and custom masks | YOLO/SAM2.1, High quality |
| 4. Export | Export 3DGS outputs from SfM results, or export COLMAP Rig viewpoint images | Metashape Import / LichtFeld / Full / Cube6 |

Detailed GUI docs:

| Step | Docs |
| --- | --- |
| Step 1 Frame Extraction | [EN](doc/extract_frames_gui.md) / [JP](doc/extract_frames_gui.ja.md) |
| Step 2 Frame Review | [EN](doc/review_frames_gui.md) / [JP](doc/review_frames_gui.ja.md) |
| Step 3 Mask Generation | [EN](doc/mask_tools_gui.md) / [JP](doc/mask_tools_gui.ja.md) |
| Step 4 Export | [EN](doc/cubemap_tools_gui.md) / [JP](doc/cubemap_tools_gui.ja.md) |

## Recommended Workflow: Metashape Route

1. Prepare 360° video from an Insta360 or similar camera.
2. Extract SfM-friendly frames in Step 1.
3. Review low-quality or unnecessary frames in Step 2.
4. Generate masks for people, camera operators, tripods, sky, or similar SfM-unfriendly regions in Step 3. `Quality: High` is the recommended starting point.
5. If masks still leak through, switch only the affected images to `Quality: Best` or regenerate them with SAM3.1. Mask2Former is also available when you want to try sky masks without setting up SAM3.1.
6. Enable stitch seam, overexposure, and custom masks when they match the source material.
7. Import the generated `masks/` folder into Metashape as per-image masks, then run SfM.
8. Use Step 4 with the Metashape XML/PLY result to export training images, masks, and `transforms.json`.

## COLMAP Route

1. Use Steps 1-3 in the same way as the Metashape route.
2. In Step 4, choose `COLMAP Export` to write viewpoint images and masks to `output/colmap_rig/`.
3. Enable `Run COLMAP after export` when you want the app to run Feature, Matcher, and Mapper stages.
4. After completion, pass `output/colmap_rig/` as the COLMAP project folder to COLMAP-compatible 3DGS tools.

## Mask Preprocessing for Normal Images

For normal video frames or still-camera image sequences, place them in `images/` or use the `+` icon on the Step 3 `Images Folder` row to copy them into the scene. Then choose `Image Type: Normal`. This keeps model-based masking and overexposure masking available while disabling stitch seam masking and 360° pole projection assist.

Use this when you want to exclude people, vehicles, blown-out regions, or similar areas before importing images into SfM software.

## Mask Tuning Notes

- Start with `Quality: High`.
- Use `Quality: Standard` for faster test runs.
- If people leak through, try `Quality: Best` or raise `Expand` slightly.
- `Quality: Best` prioritizes accuracy and takes longer, so it is best used to regenerate only images where misses remain.
- When you find a miss in preview, adjust settings and use `Regenerate Mask` to save only that image back to `masks/` using the current model and enabled extra masks. In thumbnail mode, use `Ctrl` / `Shift` selection to regenerate multiple selected images together. SAM3.1 can also add or subtract prompt detections against existing saved masks.
- Stitch seam masks are useful when the seam position is stable in the equirectangular image. If FlowState stabilization, direction lock, AI stitching, or similar processing moves the seam, verify it in the preview before using it.

## Requirements

- Windows 10/11
- Python 3.12 (3.12.10 confirmed)
- CUDA-capable GPU
- CUDA Toolkit 12.8
- FFmpeg / FFprobe (`setup_windows.bat` installs Gyan.FFmpeg through winget when missing)

Main Python packages resolved by `setup_windows.bat`:

```text
torch / torchvision / torchaudio from the CUDA 12.8 wheel index
numpy, opencv-python, Pillow, open3d, ultralytics, transformers, safetensors, tqdm, PySide6, sam3
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
| `sky_mask.py` | Semantic mask generation with Mask2Former ADE20K labels or SAM3.1 prompts | [EN](doc/sky_mask.md) |
| `stitch_mask.py` | Stitch seam mask generation | [EN](doc/stitch_mask.md) |
| `overexposure_mask.py` | Overexposure mask generation | [EN](doc/overexposure_mask.md) |
| `custom_mask.py` | AND-merge a user-provided PNG mask | [EN](doc/custom_mask.md) |
| `cubemap_transforms_json.py` | Convert equirectangular images to cubemap views | [EN](doc/cubemap_transforms_json.md) |
| `transforms_to_colmap.py` | Export COLMAP files from `transforms.json` | [EN](doc/transforms_to_colmap.md) |

## License

MIT License. See [LICENSE](LICENSE).

Mask generation features use third-party libraries and model weights with separate license terms. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Original code by [tetraface Inc.](https://github.com/tetraface)
Fork extensions by [stechdrive](https://github.com/stechdrive)
