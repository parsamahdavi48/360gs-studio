
# yolo_mask.py — Person mask generation

## Overview

`yolo_mask.py` detects people in images using YOLO and refines masks with SAM (Segment Anything Model), producing PNG mask images. By default it is tailored for 360° panorama workflows and pays particular attention to photographer/bystander areas near the bottom of the image and pedestrians near the horizon. Use `--projection normal` for normal video frames or still-camera image sequences.

![mask example](../images/yolo_mask.png)

## Usage

```
python yolo_mask.py [images_dir] [output_dir] [--add_ext] [--level N] [--expand M] [--classes IDS] [--projection equirect|normal] [--bottom-conf C] [--bottom-tta-rotations 1|2|4] [--bottom-model same|m|l|x] [--bottom-temporal-window N]
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
- `--bottom-conf C`: YOLO confidence threshold used only for equirectangular bottom re-detection (default: `0.3`).
- `--bottom-tta-rotations 1|2|4`: run bottom re-detection on rotated bottom-face views and merge the results (default: `1`).
- `--bottom-model same|m|l|x`: YOLO model used only for bottom re-detection. `same` reuses the model selected by `--level`; `x` is slower and may download `yolo26x.pt` on first use.
- `--bottom-temporal-window N`: after per-frame detection, merge bottom detections from neighboring frames within `N` frames. This is only applied to directory inputs and equirectangular mode.

Example:

```bash
python yolo_mask.py .\images .\masks --level 2 --expand 5 --classes 0,2,3
```

Normal images:

```bash
python yolo_mask.py .\images .\masks --projection normal --level 1
```

Harder bottom-view photographer masking:

```bash
python yolo_mask.py .\images .\masks --level 3 --bottom-conf 0.15 --bottom-tta-rotations 4 --bottom-temporal-window 2
```

Maximum bottom-only YOLO strength:

```bash
python yolo_mask.py .\images .\masks --level 3 --bottom-conf 0.10 --bottom-tta-rotations 4 --bottom-model x --bottom-temporal-window 4
```

## Output

- PNG files. By default the input file extension is replaced with `.png`; use `--add_ext` to append `.png` instead.
- Mask convention: background = white (255), person = black (0).

## Notes

- On first run the script may download model files (.pt); this can take time. Local `.pt` files next to the script are used when present, otherwise Ultralytics resolves the named model.
- Raising `--level` increases processing time and memory usage.
- Bottom TTA, temporal fill, and `--bottom-model x` increase processing time only for the equirectangular bottom re-detection path.
- For very large panoramas or high-resolution images a CUDA-capable GPU and CUDA-enabled PyTorch are recommended.

## Reference

See the implementation: [yolo_mask.py](yolo_mask.py#L1-L400)
