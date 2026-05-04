# sky_mask.py — Sky mask generation

## Overview

`sky_mask.py` detects sky regions with a selectable backend and writes PNG masks
using the project convention: white = keep, black = exclude. Existing masks are
AND-merged by default, so sky masking can be combined with YOLO/SAM, stitch
seam, overexposure, and custom masks.

The default backend is Mask2Former ADE20K semantic segmentation. An
experimental SAM3.1 backend is also available when the user provides the Meta
SAM3.1 checkpoint locally.

For equirectangular 360° images, the default `hybrid` mode combines direct
equirectangular inference with a top projection view to reduce pole distortion.

## Usage

```bash
python sky_mask.py [images_dir_or_file] [masks_dir] [--backend mask2former|sam31] [--projection equirect|normal] [--mode direct|top|hybrid] [--inference-size N] [--expand PX] [--min-score S] [--min-area-ratio R] [--no-top-connected] [--replace]
```

- `images_dir_or_file`: source image directory or one source image.
- `masks_dir`: output mask directory. Existing masks are AND-merged.
- `--backend mask2former|sam31`: segmentation backend (default: `mask2former`).
- `--projection equirect|normal`: source image projection (default: `equirect`).
- `--mode direct|top|hybrid`: sky detection mode (default: `hybrid`).
- `--inference-size N`: backend input size, 384-2048 (default: `768`; SAM3.1 currently requires `1008`).
- `--view-size N`: top projection face size for equirect mode; `0` means auto.
- `--expand PX`: expand sky exclusion by pixels. Negative values shrink it.
- `--min-score S`: optional score floor. For SAM3.1, `0` uses its default `0.5` text-prompt confidence threshold.
- `--min-area-ratio R`: remove small sky candidates by image-area ratio.
- `--no-top-connected`: keep sky components that do not touch the top edge.
- `--model-dir PATH`: local model directory or SAM3.1 checkpoint override.
- `--sam-prompt TEXT`: text prompt for the SAM3.1 backend (default: `sky`).
- `--device auto|cpu|cuda`: inference device (default: `auto`).
- `--replace`: ignore existing masks and write sky-only masks.

Example:

```bash
python sky_mask.py .\images .\masks --projection equirect --mode hybrid --inference-size 768
```

Conservative re-run:

```bash
python sky_mask.py .\images .\masks --min-score 0.8 --expand -2
```

## Model Files

The default Mask2Former local model directory is:

```text
models/mask2former-swin-large-ade-semantic/
  config.json
  preprocessor_config.json
  model.safetensors
```

If the local directory is missing, Transformers may resolve
`facebook/mask2former-swin-large-ade-semantic` from Hugging Face.

The experimental SAM3.1 backend expects the user-provided checkpoint here:

```text
models/sam3.1/
  sam3.1_multiplex.pt
  config.json
  LICENSE
  README.md
```

SAM3.1 support requires Meta's `sam3` Python package in the active venv. The
current implementation uses the package image API with the local SAM3.1
checkpoint for comparison, so the loader may print checkpoint-key warnings.
Because this package is not part of this project's default setup, this backend
is meant for local comparison until the dependency, Windows runtime, and license
path is stable enough for normal users.

For local testing, install the package separately. The `sam3` package currently
declares `numpy<2`, while this project uses NumPy 2.x, so install the required
runtime dependencies first and then install `sam3` itself with `--no-deps`:

```bat
.\.venv\Scripts\python.exe -m pip install timm ftfy==6.1.1 iopath regex einops triton-windows pycocotools
.\.venv\Scripts\python.exe -m pip install --no-deps git+https://github.com/facebookresearch/sam3.git@847e1a3b15115a04c87c0760297f044f0555d970
```

This intentionally leaves `sam3`'s declared `numpy<2` requirement unresolved;
`pip check` will report that metadata conflict in a SAM3.1 test venv.

## Notes

- The output mask has sky = black (0), non-sky = white (255).
- `hybrid` is intended for 360° panoramas. For normal images, it falls back to direct inference.
- `top-connected` filtering is enabled by default to reduce sky-like false positives.
- Sky masking uses third-party model weights and dataset-derived checkpoints with separate terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
