# tetraface-3dgs-utils

A collection of scripts I use and develop as part of a 3D Gaussian Splatting (3DGS) workflow.

[JP 日本語の説明](README.ja.md)

## Requirements

- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) (for GPU-enabled PyTorch workflows)
- [Python 3.x](https://www.python.org/) (confirmed with 3.11.8, 3.11 recommended)
- [FFmpeg / FFprobe](https://ffmpeg.org/) (for video frame extraction workflow)
- [metashape_360_lfs.py (fork)](https://github.com/tetraface/metashape_360_lfs)
  - Bundled in this repository: `vendor/metashape_360_lfs/metashape_360_lfs.py`
  - Source record: `vendor/metashape_360_lfs/VENDOR_SOURCE.md`

### Depended python modules

- NumPy
- OpenCV
- Pillow
- PySide6 (used by GUI wrappers)
- Open3D (used by `metashape_360_lfs`)
- PyTorch 2.8.0 (with CUDA 12.8)
- ultralytics
- tqdm

Install example (CUDA 12.8 PyTorch wheel + other deps):

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy opencv-python Pillow open3d ultralytics tqdm PySide6
```

## Windows quick start

For this repository, use Python 3.11 in a virtual environment. `open3d` is not Python 3.12-only.
If Python 3.11 is missing, `setup_windows.bat` tries to install Python 3.11.8 via `winget`.

```bat
setup_windows.bat
start_extract_frames_gui.bat
```

## Summary of scripts

### `cubemap_transforms_json.py`

Convert transforms.json produced for 360° equirectangular data (by `metashape_360_lfs`) into a cubemap-friendly format usable by common 3DGS tools.<br>
See detailed documentation: [doc/cubemap_transforms_json.md](doc/cubemap_transforms_json.md).<br>
![mask example](images/yaw45.jpg)

### `stitch_mask.py`

Generate masks that exclude angular regions outside the two fisheye lenses in a 360° image. Useful when stitch seams become visible (for example in tight indoor scenes).<br>
See details: [doc/stitch_mask.md](doc/stitch_mask.md)<br>
![mask example](images/stitch_mask.png)

### `yolo_mask.py`

Detect people in 360° images and generate mask PNGs.<br>
See details: [doc/yolo_mask.md](doc/yolo_mask.md)<br>
![mask example](images/yolo_mask.png)

### `extract_frames.py`

Extract still frames from equirectangular video using FFmpeg with fixed/change-based selection and blur-aware replacement.<br>
See details: [doc/extract_frames.md](doc/extract_frames.md)

### `review_frames.py`

Review extracted frames in a lightweight GUI and edit keep/drop decisions in `selected_frames.csv`.<br>
See details: [doc/review_frames.md](doc/review_frames.md)

### `apply_frame_decisions.py`

Apply keep/drop decisions in `selected_frames.csv` and export only keep images for Metashape import.<br>
See details: [doc/apply_frame_decisions.md](doc/apply_frame_decisions.md)

### `extract_frames_gui.py`

Wrapper GUI for extraction workflow: run extraction, open review GUI, and export keep images.<br>
Includes video metadata display (fps/duration/frame count) and estimated output count for current parameters.<br>
See details: [doc/extract_frames_gui.md](doc/extract_frames_gui.md)
