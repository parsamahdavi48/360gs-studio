
# stitch_mask.py — Stitching exclusion mask generation

## Overview

`stitch_mask.py` removes stitching regions (the seam areas near the front/back lens edges) from masks used for 360° panoramas. The script can generate a single base mask or apply stitching-region masks on top of existing person-removal masks.

![mask example](../images/stitch_mask.png)

Applying the stitching mask is useful when seams become noticeable (for example in tight indoor scenes where nearby surfaces make stitch seams visible). Masking out seam areas helps exclude them from alignment or blending steps and can improve final quality.

## Usage

```
python stitch_mask.py [-h] [--single w h] [--boundary-width DEG] [--fov FOV] [--workers WORKERS] [input_dir] [output_dir]
```

- **`input_dir`**: directory containing input mask PNGs (if omitted the script searches for `masks`)
- **`output_dir`**: output directory (defaults to the input directory)
- **`--single w h`**: generate a single base mask at the specified resolution and save it as `single_mask.png` in the input directory
- **`--boundary-width`**: total stitch seam band to exclude in degrees. Default is `5.0`, equivalent to the legacy `--fov 175`.
- **`--fov`**: legacy fisheye effective FOV in degrees. If specified, it takes precedence over `--boundary-width`.
- **`--workers`**: number of parallel worker processes (default: number of CPU cores)

## Examples

If masks produced by `yolo_mask.py` are already in `masks/` (use default):

```bash
python stitch_mask.py
```

Specify only the input folder:

```bash
python stitch_mask.py input_masks
```

Specify both input and output folders:

```bash
python stitch_mask.py input_masks output_masks
```

Create a single base mask at 7680×3840 and save it into the current directory:

```bash
python stitch_mask.py . --single 7680 3840
```

Use a 10° seam band and 8 workers:

```bash
python stitch_mask.py input_masks output_masks --boundary-width 10 --workers 8
```

## Notes

- When exporting MP4 from tools such as Insta360 Studio or DJI Studio, make sure to disable geometric corrections (stabilization, roll/pitch/yaw adjustments, stitching corrections). If those corrections are applied, the original lens directions are lost and the mask may not align correctly.
- In a large space with sufficient distance from the surroundings, the seams are barely noticeable, so applying a mask may actually result in a decrease in image quality. Use it appropriately by reducing the boundary width or disabling the stitch mask.

## Reference

See the implementation: [stitch_mask.py](stitch_mask.py#L1-L400)
