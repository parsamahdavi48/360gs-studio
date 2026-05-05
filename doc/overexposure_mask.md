# overexposure_mask.py - Overexposure Mask Generation

## Overview

`overexposure_mask.py` detects blown-out pixels in source images and merges the result into mask PNGs.

The mask convention is the same as the rest of this repository:

- white (`255`) = usable pixels
- black (`0`) = excluded pixels

Pixels whose RGB channels are all above the threshold are treated as overexposed. The detected region can be dilated to cover fringe artifacts, then AND-merged with an existing mask. If no mask exists, a new white mask with black overexposed regions is created.

## Usage

Merge overexposure into existing masks:

```bash
python overexposure_mask.py ./scene01/images ./scene01/masks
```

Use a slightly lower threshold and wider dilation:

```bash
python overexposure_mask.py ./scene01/images ./scene01/masks --threshold 250 --dilate 2
```

Write overexposure-only masks, ignoring existing masks:

```bash
python overexposure_mask.py ./scene01/images ./scene01/masks --replace
```

## Options

- `images_dir`: source image directory
- `masks_dir`: mask output directory. Existing masks are AND-merged unless `--replace` is set
- `--threshold`: 8-bit-equivalent RGB threshold for overexposure detection. Default `254`, valid range `1-254`
- `--dilate`: dilation radius in pixels. Default `1`, `0` disables dilation
- `--workers`: number of parallel workers. Default is CPU count
- `--replace`: ignore existing masks and write overexposure-only masks

## Notes

- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`
- For RGB/RGBA images, all three color channels must exceed the threshold.
- For grayscale images, the single channel is compared against the threshold.
- For 16-bit integer images, the 8-bit threshold is scaled to the image bit depth.
- Existing masks with different dimensions are resized with nearest-neighbor sampling before merging.
- Output mask names are `<image_stem>.png`.

