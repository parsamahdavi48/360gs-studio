# sky_mask.py — Semantic prompt/class mask generation

## Overview

`sky_mask.py` writes PNG masks with the project convention: white = keep,
black = exclude. Existing masks are AND-merged by default.

Despite the historical filename, the script now supports more than sky:

- `Mask2Former` ADE20K class masks via `--labels`.
- local `SAM3.1` prompt masks via one or more `--sam-prompt` values.

For equirectangular 360° images, projection-assist modes can combine direct
equirectangular inference with top and/or bottom cube-pole views. This helps
with distorted sky near the top pole and distorted people/tripods near the
bottom pole.

## Usage

```bash
python sky_mask.py [images_dir_or_file] [masks_dir] [--backend mask2former|sam31] [--projection equirect|normal] [--quality standard|high|best] [--labels LABELS] [--sam-prompt TEXT] [--subtract-sam-prompt TEXT] [--merge-mode replace|add|subtract] [--inference-size N] [--expand PX] [--min-score S] [--min-area-ratio R] [--top-connected] [--replace] [--safe-batch]
```

- `images_dir_or_file`: source image directory or one source image.
- `masks_dir`: output mask directory. Existing masks are AND-merged by default.
- `--backend mask2former|sam31`: segmentation backend (default: `mask2former`).
- `--projection equirect|normal`: source image projection (default: `equirect`).
- `--quality standard|high|best`: shared input-view recipe (default: `high`).
  - For 360° images, `high` adds top/bottom projection assist and target-oriented tiles.
  - For normal images, quality uses direct inference plus whole-image tiling; 360° pole projection is skipped.
  - `--mode direct|top|bottom|hybrid|full` remains available as a low-level override.
- `--labels LABELS`: comma-separated Mask2Former ADE20K label names or ids (default: `sky`).
- `--sam-prompt TEXT`: text prompt for SAM3.1. Can be passed multiple times; masks are OR-merged.
- `--subtract-sam-prompt TEXT`: SAM3.1 prompt to subtract from detected prompt masks. Can be passed multiple times.
- `--merge-mode replace|add|subtract`: how to apply detected regions to existing masks. `add` blackens detected regions in existing masks, `subtract` turns detected regions white, and `replace` writes only the current detection result.
- `--inference-size N`: backend input size, 384-2048 (default: `768`; SAM3.1 currently uses `1008` in the GUI).
- `--view-size N`: cube-pole projection face size for equirect mode; `0` means auto.
- `--expand PX`: expand detected exclusion regions by pixels. Negative values shrink them.
- `--min-score S`: optional Mask2Former score floor from `0.00` to `1.00`; `0` disables the filter.
- `--min-area-ratio R`: remove small sky candidates by image-area ratio. This applies only to sky masks.
- `--top-connected`: keep only sky components touching the top edge. This applies only to sky masks and is off by default.
- `--model-dir PATH`: local model directory or SAM3.1 checkpoint override.
- `--device auto|cpu|cuda`: inference device (default: `auto`).
- `--replace`: compatibility shortcut for `--merge-mode replace`.
- `--safe-batch`: run SAM3.1 directory batches with saved progress and unfinished-image resume. The GUI uses this automatically for SAM3.1 batch generation.

Examples:

```bash
python sky_mask.py .\images .\masks --projection equirect --quality high --labels sky,person --inference-size 768
```

SAM3.1 sky/person prompts:

```bash
python sky_mask.py .\images .\masks --backend sam31 --quality high --inference-size 1008 --sam-prompt sky --sam-prompt person --replace
```

SAM3.1 add-only correction on existing masks:

```bash
python sky_mask.py .\images .\masks --backend sam31 --quality best --inference-size 1008 --sam-prompt tripod --merge-mode add
```

SAM3.1 subtract correction on existing masks:

```bash
python sky_mask.py .\images .\masks --backend sam31 --quality best --inference-size 1008 --sam-prompt "male icon" --sam-prompt "female icon" --merge-mode subtract
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

The SAM3.1 backend requires this checkpoint:

```text
models/sam3.1/
  sam3.1_multiplex.pt
```

SAM3.1 support requires Meta's `sam3` Python package in the active venv. The
standard `setup_windows.bat` environment installs that runtime package. In the
GUI, choosing SAM3.1 can download `sam3.1_multiplex.pt` after Hugging Face
access approval and SAM License acceptance. CLI users should place the
checkpoint in the location above or pass it with `--model-dir`. Optional
README or LICENSE files may also be placed under `models/sam3.1/`, but the
script only requires the checkpoint file.

## Notes

- Detected target regions become black (0); all other pixels remain white (255).
- Mask2Former resolves multiple ADE20K labels in one inference and merges them.
- SAM3.1 runs one prompt at a time and OR-merges the prompt masks.
- SAM3.1 subtract prompts are detected with the same input-view recipe, then removed from the positive prompt mask before merge-mode is applied.
- With `--safe-batch`, completed masks remain saved if GPU memory runs out. Rerun with the same settings to resume unfinished images.
- Small-area and top-connected filtering are applied only to sky labels/prompts. Person, tripod, and custom prompt masks are not removed by these sky post-filters.
- This feature uses third-party model weights and dataset-derived checkpoints with separate terms. See [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).
