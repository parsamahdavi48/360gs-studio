# sky_mask.py — Sky mask generation

## Overview

`sky_mask.py` detects sky regions with Mask2Former ADE20K semantic segmentation
and writes PNG masks using the project convention: white = keep, black =
exclude. Existing masks are AND-merged, so sky masking can be combined with
YOLO/SAM, stitch seam, overexposure, and custom masks.

For equirectangular 360° images, the default `hybrid` mode combines direct
equirectangular inference with a top projection view to reduce pole distortion.

## Usage

```bash
python sky_mask.py [images_dir_or_file] [masks_dir] [--projection equirect|normal] [--mode direct|top|hybrid] [--inference-size N] [--expand PX] [--min-score S] [--min-area-ratio R] [--no-top-connected]
```

- `images_dir_or_file`: source image directory or one source image.
- `masks_dir`: output mask directory. Existing masks are AND-merged.
- `--projection equirect|normal`: source image projection (default: `equirect`).
- `--mode direct|top|hybrid`: sky detection mode (default: `hybrid`).
- `--inference-size N`: Mask2Former input size, 384-2048 (default: `768`).
- `--view-size N`: top projection face size for equirect mode; `0` means auto.
- `--expand PX`: expand sky exclusion by pixels. Negative values shrink it.
- `--min-score S`: optional sky-class score floor. `0` disables this filter.
- `--min-area-ratio R`: remove small sky candidates by image-area ratio.
- `--no-top-connected`: keep sky components that do not touch the top edge.
- `--model-dir PATH`: local Mask2Former model directory override.
- `--device auto|cpu|cuda`: inference device (default: `auto`).

Example:

```bash
python sky_mask.py .\images .\masks --projection equirect --mode hybrid --inference-size 768
```

Conservative re-run:

```bash
python sky_mask.py .\images .\masks --min-score 0.8 --expand -2
```

## Model Files

The default local model directory is:

```text
models/mask2former-swin-large-ade-semantic/
  config.json
  preprocessor_config.json
  model.safetensors
```

If the local directory is missing, Transformers may resolve
`facebook/mask2former-swin-large-ade-semantic` from Hugging Face.

## Notes

- The output mask has sky = black (0), non-sky = white (255).
- `hybrid` is intended for 360° panoramas. For normal images, it falls back to direct inference.
- `top-connected` filtering is enabled by default to reduce sky-like false positives.
- Sky masking uses third-party model weights and dataset-derived checkpoints with separate terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
