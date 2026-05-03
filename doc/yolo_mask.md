
# yolo_mask.py — Person mask generation

## Overview

`yolo_mask.py` detects people in images using YOLO and refines masks with SAM (Segment Anything Model), producing PNG mask images. By default it is tailored for 360° panorama workflows and pays particular attention to photographer/bystander areas near the bottom of the image and pedestrians near the horizon. Use `--projection normal` for normal video frames or still-camera image sequences.

![mask example](../images/yolo_mask.png)

## Usage

```
python yolo_mask.py [images_dir] [output_dir] [--add_ext] [--level N] [--expand M] [--classes IDS] [--projection equirect|normal]
```

- `images_dir`: input image directory (default: `images`)
- `output_dir`: output mask directory (default: `masks`)
- `--add_ext`: keep the original extension and append `.png` (e.g. `hoge.jpg.png`)
- `--level N`: detection level (0–3, default: 1). Increasing the value enables higher-precision local extraction.
- `--expand M`: fixed-pixel expansion after SAM refinement (default: 2)
  - Clamped to `-16..32` for safety.
  - Negative values shrink the mask region.
- `--classes IDS`: comma-separated YOLO class ids (default: `0`, person only)
- `--projection equirect|normal`: source image projection (default: `equirect`)
  - `equirect`: equirectangular 360 images. Enables bottom-view re-detection.
  - `normal`: normal images. Disables 360-specific bottom-view re-detection.

Example:

```bash
python yolo_mask.py .\images .\masks --level 2 --expand 5 --classes 0,2,3
```

Normal images:

```bash
python yolo_mask.py .\images .\masks --projection normal --level 1
```

## Output

- PNG files. By default the input file extension is replaced with `.png`; use `--add_ext` to append `.png` instead.
- Mask convention: background = white (255), person = black (0).

## Notes

- On first run the script may download model files (.pt); this can take time. Downloaded files are stored next to the script.
- Raising `--level` increases processing time and memory usage.
- For very large panoramas or high-resolution images a CUDA-capable GPU and CUDA-enabled PyTorch are recommended.

## Reference

See the implementation: [yolo_mask.py](yolo_mask.py#L1-L400)
