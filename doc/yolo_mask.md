
# yolo_mask.py — Person mask generation

## Overview

`yolo_mask.py` detects people in images using YOLO and refines masks with SAM (Segment Anything Model), producing PNG mask images. By default it is tailored for 360° panorama workflows and pays particular attention to photographer/bystander areas near the bottom of the image and pedestrians near the horizon. Use `--projection normal` for normal video frames or still-camera image sequences.

![mask example](../images/yolo_mask.png)

## Usage

```
python yolo_mask.py [images_dir] [output_dir] [--add_ext] [--quality standard|high|best] [--expand M] [--classes IDS] [--projection equirect|normal] [--bottom-conf C] [--bottom-tta-rotations 1|2|4] [--bottom-model same|m|l|x] [--bottom-filter] [--profile-json PATH]
```

- `images_dir`: input image directory (default: `images`)
- `output_dir`: output mask directory (default: `masks`)
- `--add_ext`: keep the original extension and append `.png` (e.g. `hoge.jpg.png`)
- `--quality standard|high|best`: shared input-view recipe (default: `high`).
  - `standard`: direct full-image inference. For 360° images, also runs a light bottom-pole pass.
  - `high`: adds person-oriented tiles and, for 360° images, top/bottom projection assist.
  - `best`: uses denser tiles and stronger bottom-pole settings for difficult 360° masks.
- `--expand M`: fixed-pixel expansion after SAM refinement (default: 2)
  - Clamped to `-16..32` for safety.
  - Negative values shrink the mask region.
- `--classes IDS`: comma-separated YOLO class ids (default: `0`, person only)
- `--projection equirect|normal`: source image projection (default: `equirect`)
  - `equirect`: equirectangular 360 images. Enables bottom-view re-detection.
  - `normal`: normal images. Disables 360-specific bottom-view re-detection.
- `--bottom-conf C`: override the quality-selected YOLO confidence threshold used only for equirectangular bottom re-detection.
- `--bottom-tta-rotations 1|2|4`: override the quality-selected bottom-view rotation count.
- `--bottom-model same|m|l|x`: YOLO model used only for bottom re-detection. `same` reuses the model selected by `--quality`; `x` is slower and may download `yolo26x.pt` on first use.
- `--bottom-filter`: remove unreliable bottom-view mask components before merging into the final panorama mask.
- `--profile-json PATH`: write timing and detection metrics as JSON. Normal runs are unchanged when this option is omitted.

Example:

```bash
python yolo_mask.py .\images .\masks --quality high --expand 5 --classes 0,2,3
```

Normal images:

```bash
python yolo_mask.py .\images .\masks --projection normal --quality standard
```

Harder bottom-view photographer masking:

```bash
python yolo_mask.py .\images .\masks --quality best
```

Maximum bottom-only YOLO strength:

```bash
python yolo_mask.py .\images .\masks --quality best --bottom-model x
```

Benchmark a fixed dataset:

```bash
python scripts/benchmark_yolo_mask.py --dataset D:\3DGS\test --output-root D:\3DGS\test\benchmarks --label baseline --repeat 3
```

Compare a later candidate run against the baseline masks:

```bash
python scripts/benchmark_yolo_mask.py --dataset D:\3DGS\test --output-root D:\3DGS\test\benchmarks --label candidate --compare-label baseline --repeat 3 --overwrite
```

## Output

- PNG files. By default the input file extension is replaced with `.png`; use `--add_ext` to append `.png` instead.
- Mask convention: background = white (255), person = black (0).

## Notes

- On first run the script may download model files (.pt); this can take time. Local `.pt` files under `models/ultralytics/` are used first. Legacy `.pt` files next to the script are still used as a fallback; otherwise Ultralytics resolves the named model.
- The YOLO/SAM feature uses third-party libraries and model weights with separate license terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
- Raising `--quality` increases processing time and memory usage.
- Bottom TTA, filtering, and `--bottom-model x` increase processing time only for the equirectangular bottom re-detection path.
- For very large panoramas or high-resolution images a CUDA-capable GPU and CUDA-enabled PyTorch are recommended.

## Reference

See the implementation: [yolo_mask.py](yolo_mask.py#L1-L400)
