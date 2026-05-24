# custom_mask.py — Static PNG Mask Merge

`custom_mask.py` AND-merges one user-provided PNG mask into the masks generated
for a scene.

Mask convention is the same as the rest of this project:

- white (`255`) = keep / usable pixels
- black (`0`) = exclude / masked pixels

## Usage

```bash
python -m core.custom_mask ./scene01/images ./scene01/masks ./custom_mask.png
```

Arguments:

| Argument | Description |
| --- | --- |
| `images_dir` | Source image folder. Subfolders are scanned recursively. |
| `masks_dir` | Mask output folder. Existing masks are read from and written back to this folder. |
| `custom_mask` | PNG mask applied to every source image with matching dimensions. |

Option:

| Option | Description |
| --- | --- |
| `--replace` | Ignore existing masks and write custom-mask-only outputs. |

Supported source image extensions are `.jpg`, `.jpeg`, `.png`, `.tif`, and
`.tiff`.

## Input Mask Rules

- The custom mask must be PNG.
- 8-bit and 16-bit grayscale, RGB, and RGBA PNGs are accepted.
- RGB/RGBA inputs are converted to grayscale. Alpha is ignored.
- 8-bit values `>=128` become white; lower values become black.
- 16-bit values `>=32768` become white; lower values become black.
- The custom mask is not resized. It is applied only to source images with the
  same width and height.

## Output Behavior

- Output mask paths mirror the source image subfolder under `masks_dir`.
- Output filenames use the source image stem with `.png`, for example
  `images/a/frame_0001.jpg` -> `masks/a/frame_0001.png`.
- If a matching mask already exists, the script writes
  `existing_mask AND custom_mask`.
- If no matching mask exists, or `--replace` is used, the custom mask itself becomes the output mask.
- Images with size mismatches are skipped.
- If source images exist but none match the custom mask size, the command exits
  with an error.

In the integrated GUI, the custom mask step runs after the selected model-based
mask, stitch seam, and overexposure masks, so it acts as the final AND-merge
pass.
