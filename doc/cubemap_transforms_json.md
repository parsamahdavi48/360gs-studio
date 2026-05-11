# cubemap_transforms_json.py : transforms.json converter from equirectangular to cubemap

This script converts `transforms.json` for **360° equirectangular image** by [metashape_360_lfs](https://github.com/tetraface/metashape_360_lfs) into **cubemap-based images**.

That is, the following conversions are possible:

Metashape (Standard/Professional) > xml/pointcloud > transforms.json > cubemap > 3DGS software ([Jawset Postshot](https://www.jawset.com/), [Brush](https://github.com/ArthurBrussee/brush), [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio), etc...)

[JP 日本語の説明](cubemap_transforms_json.ja.md)

## Directory structure

### Input directory example

```
input_dir/
├─ metashape.xml
├─ metashape.ply
├─ transforms.json
├─ pointcloud.ply (optional)
├─ images/
│ ├─ image_000.jpg (or .png)
│ └─ image_001.jpg
│ └─ ...
└─ masks/ # (optional)
  ├─ image_000.png (or .jpg.png, .png.png)
  └─ image_001.png
  └─ ...
```

| File | Description |
|------|-------------|
|metashape.ply|Expoted in Metashape [File > Export > Export Point Cloud]|
|metashape.xml|Expoted in Metashape [File > Export > Export Cameras]|
|transforms.json|Converted by metashape_360_lfs|
|pointcloud.ply|Converted by metashape_360_lfs (optional)|

### Output directory example

```
output_dir/
├─ transforms.json
├─ images/
│ ├─ image_000_nx.jpg (or .png)
│ ├─ image_000_ny.jpg
│ ├─ image_000_nz.jpg
│ ├─ image_000_px.jpg
│ ├─ image_000_py.jpg
│ ├─ image_000_pz.jpg
│ ├─ image_001_nx.jpg
│ └─ ...
└─ masks/
  ├─ image_000_nx.png
  └─ ...
```


## Usage

### Basic usage

Convert transforms.json and images in the current directory: (also convert if masks directory exists)
```
python metashape_360_lfs.py --images images --xml metashape.xml --output .
python cubemap_transforms_json.py .
```

### Detailed

With specifying output directory:
```
python cubemap_transforms_json.py . ./cubic
```

With options:

```
python cubemap_transforms_json.py . ./cubic \
  --yaw 45 \
  --stitch 2.5 \
  --fov 90
```

With custom view list (name/yaw/pitch):

```bash
python cubemap_transforms_json.py . ./cubic --views-json views_config.json --fov 90
```

Specifying `--yaw 45 --stitch DEGREE` will prevent the stitching area between two fisheye images from crossing the center of the cubemap image. These options are effective for Insta360 and OSMO 360 images **without any image correction** like camera tilt and stitching.

The following images illustrate how each face of the cubemap and the boundary between two fisheye images occupy a portion of the equirectangular image.

![Example: --yaw 0](../images/yaw0.jpg)<br>
*--yaw 0*

![Example: --yaw 45](../images/yaw45.jpg)<br>
*--yaw 45*

![Example: --yaw 45 --stitch 2.5 --fov 91.5](../images/yaw45_s2_5_f91_5.jpg)<br>
*--yaw 45 --stitch 2.5 --fov 91.5*

### For Brush

By default, coordinate axis transformation suitable for Postshot is performed. For Brush, specify `--brush`.

```
python metashape_360_lfs.py --images images --xml metashape.xml --output .
python cubemap_transforms_json.py . ./cubic --brush
```

### For LichtFeld Studio

For LichtFeld Studio, specify `--no_transform`. This writes cubemap
`transforms.json` as `PINHOLE`. In the GUI workflow, Step 4 then applies the
LichtFeld final orientation correction to `transforms.json` and `pointcloud.ply`
together.

```
python metashape_360_lfs.py --images images --xml metashape.xml \
  --ply metashape.ply --output .
python cubemap_transforms_json.py . ./cubic --no_transform
```

### Options

|Option|Argument|Description|
|------|----|-----------|
|--json|filename|transforms.json with a different filename (default='transforms.json')|
|--image-dir|directory name|Input image directory used when reading `file_path` entries from transforms.json (default=`<input_dir>`)|
|--mask_dir|directory name|Input mask images directory (default='<input_dir>/masks')|
|--mask_from_alpha|(no)|Extract masks from alpha channel in images|
|--invert_masks|(no)|Invert output masks (black/white polarity)|
|--yaw|degrees|Shift the horizontal angle (default=45.0 degrees)|
|--stitch|degrees|Angle to avoid stitching areas (default=0.0 degrees)|
|--fov|degrees|Field of view for cubemap faces (default=90.0 degrees)|
|--output_scale|ratio|Output face size ratio to input height (default=0.5, `1.0` for full size)|
|--views-json|path|Use custom views list JSON (`[{name,yaw,pitch,enabled}]` or `{\"views\":[...]}`)|
|--no_bottom|(no)|Output without a bottom face of cube-map.|
|--no_top|(no)|Output without a top face of cube-map.|
|--no_image|(no)|Disable image conversion. Only transforms.json will be converted.|
|--skip-images|(no)|Do not write viewpoint images; update masks and/or transforms.json only.|
|--skip-masks|(no)|Do not write masks; update viewpoint images and/or transforms.json only.|
|--image-only|(no)|Export viewpoint images and masks from `<input_dir>/images` and `<input_dir>/masks` without reading `transforms.json`.|
|--colmap-rig|(no)|Write `--image-only` output as a COLMAP Rig dataset under `<output_dir>/colmap_rig/` and create `rig_config.json`. Per-frame yaw rotation is forced to 0 degrees.|
|--colmap-rig-name|name|COLMAP rig name (default=`rig1`).|
|--no_transform|(no)|Disable coordinate axis conversion.|
|--brush|(no)|Convert coordinates for Brush.|
|--duplicate|(no)|Allow duplicated image files by merging chunks.|
|--yaw-offset-per-frame|degrees|Per-frame cubemap yaw rotation step (default=30.0). Each unique input image gets `yaw = frame_index * step (mod 360)`. Diversifies sampling angles to reduce 3DGS face-boundary artifacts. Set to `0` to disable.|
|--output-format|auto/jpg/png/tiff/webp|Output image format (default=auto, preserves input format).|
|--output-bit-depth|8/source|Output image bit depth (default=8). `8` down-converts images for compatibility; `source` preserves PNG/TIFF source bit depth. Mask outputs are always 8-bit PNG.|
|--jpg-quality|1-100|JPEG / WebP quality (default=95).|
|--workers|auto/N|Image conversion worker processes (default=auto). Auto caps workers by CPU count and estimated memory use.|
|--remap-cache-limit|auto/N|Per-worker yaw remap table cache limit (default=auto). Auto keeps the cache bounded by available memory.|

### Per-frame yaw rotation

By default, each unique input frame gets a different cubemap yaw offset (`frame_index * 30°` mod 360°), so the cubemap face boundaries fall on different scene directions across frames. This diversifies the sampling and reduces 3DGS training instabilities caused by repeated boundary artifacts at the same world locations. Set `--yaw-offset-per-frame 0` to restore the legacy fixed-yaw behavior.

The unique offsets cycle through `{0°, 30°, 60°, ..., 330°}` (12 values) at the default step. Internally, worker remap tables are cached per `(yaw_offset, view)`, bounded by `--remap-cache-limit`, so memory does not grow with the number of frames.

### Bit depth and alpha channel

The cubemap converter now uses OpenCV throughout for I/O and supports:

- 8-bit image output by default for broad 3DGS tool compatibility
- source 8/16-bit preservation for PNG/TIFF only when `--output-bit-depth source` is specified
- RGBA alpha channel through the remap (split-and-recombine to avoid color/alpha bleeding)
- Output format conversion (png ↔ tiff ↔ webp ↔ jpg) via `--output-format`
- converted masks are always 8-bit single-channel PNG

JPEG and WebP are 8-bit only and have no alpha; the writer will downconvert and drop alpha automatically when targeting them.

## How to import into 3DGS software

Import the following files in each software:

### Postshot / Brush

- metashape.ply (exported in `Metashape`)
- transforms.json (in the output directory)
- images (in the output directory)
- masks (in the output directory: optional)

### LichtFeld Studio

- pointcloud.ply (converted by `metashape_360_lfs`)
- transforms.json (in the output directory)
- images (in the output directory)
- masks (in the output directory: optional)

The final `transforms.json` camera poses and `pointcloud.ply` points are a
matched set. Do not apply an extra orientation correction to only one of them.
