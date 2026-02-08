# apply_frame_decisions.py — Export keep frames for Metashape

## Overview

`apply_frame_decisions.py` reads `selected_frames.csv` and copies only rows with `decision != drop` into a dedicated output directory (default: `metashape_images`).

This is the final step before importing still images into Metashape.

## Usage

```bash
python apply_frame_decisions.py ./scene01 --clean-output
```

Options:

- `scene_dir`: scene directory containing `selected_frames.csv` and extracted `images/`
- `--csv`: CSV filename (default: `selected_frames.csv`)
- `--output`: output folder name under scene directory (default: `metashape_images`)
- `--clean-output`: remove existing image files in output folder before exporting

## Outputs

- `scene_dir/metashape_images/*.jpg` (or `.png`)
- `scene_dir/metashape_images/selected_frames_keep.csv`
