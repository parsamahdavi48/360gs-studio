# realityscan_to_transforms.py : RealityScan CSV to NeRF transforms

[JP 日本語の説明](realityscan_to_transforms.ja.md)

This script converts a RealityScan registration CSV plus an optional RealityScan
PLY into a NeRF-style dataset:

```text
transforms.json
pointcloud.ply
images/   (referenced, not copied)
masks/    (referenced when matching masks exist)
```

It is meant for the workflow where RealityScan is used to realign a mixed image
set, for example cubemap faces plus normal perspective images, and the result
needs to be loaded by a NeRF/3DGS tool without rebuilding a COLMAP folder.

## Example

Write a LichtFeld-oriented JSON and PLY next to the RealityScan export without
copying `images/` or `masks/`:

```powershell
python -m core.realityscan_to_transforms `
  D:\3DGS\sakume\output\realityscan\rs_sakume.csv `
  D:\3DGS\sakume\output\realityscan `
  --ply D:\3DGS\sakume\output\realityscan\rs_sakume.ply `
  --json-name transforms_lfs.json `
  --pointcloud-name pointcloud_lfs.ply `
  --target-profile lichtfeld
```

Use `--json-name transforms.json` and `--pointcloud-name pointcloud.ply` when the
output folder is dedicated to this converted dataset.

## Mixed Cameras

All CSV rows are converted by default. The script does not filter to cubemap
faces.

- Frames with zero distortion are written as frame-level `PINHOLE` cameras.
- Frames with RealityScan distortion values are written as frame-level `OPENCV`
  cameras with `k1` through `k4` and `p1` / `p2`.
- `w`, `h`, `fl_x`, `fl_y`, `cx`, and `cy` are written per frame, so cubemap
  and normal images can have different resolutions and intrinsics.

If `<csv folder>/transforms.json` exists, it is used only as a metadata source
for matching frames. Camera poses always come from the RealityScan CSV.

For `transforms.json`, the top-level camera block is chosen from the most common
camera group so that tools which ignore per-frame intrinsics get the dominant
camera setup instead of whichever frame appears first in the CSV.

## LichtFeld Profile

`--target-profile lichtfeld` is the default. It applies the same final scene
orientation used by this toolkit's Metashape-to-LichtFeld cubemap output, and
applies the matching LichtFeld file-space transform to `pointcloud.ply`.
LichtFeld applies its own point-cloud axis conversion when loading
`transforms.json`, so the PLY transform is intentionally not the same matrix as
the JSON camera transform.

Use `--target-profile realityscan` only when you want to keep RealityScan's CSV
coordinate frame.

Current LichtFeld builds read the camera model, image size, focal length,
principal point, and distortion from the top level of `transforms.json`; they do
not use those fields per frame. That means a single mixed-intrinsic JSON can
preserve all rows for tools that support per-frame intrinsics, but LichtFeld's
JSON path will train it as one dominant pinhole camera group. For fully correct
LichtFeld training with cubemap and distorted normal images together, use a
COLMAP-style export that carries one camera record per image/camera group. In
this toolkit, use `realityscan_to_lfs_colmap.py` for that route.

## Notes

The script references existing image files by default. `--image-path-mode` can
write paths as `images/...`, relative paths from the output folder, or absolute
paths.

When a matching mask exists in `masks/<image stem>.png`, the frame gets a
`mask_path`. RealityScan layer masks next to images are not required for this
NeRF-style output.
