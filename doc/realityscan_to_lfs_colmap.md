# realityscan_to_lfs_colmap.py : RealityScan CSV/PLY to LichtFeld COLMAP

[JP 日本語の説明](realityscan_to_lfs_colmap.ja.md)

This script creates a LichtFeld Studio COLMAP text dataset directly from
RealityScan registration CSV and PLY exports.

It is intended for the mixed-camera RealityScan workflow where `transforms.json`
is not sufficient because LichtFeld's NeRF JSON loader reads one top-level
camera model, while COLMAP can store multiple camera records.

## Output Layout

The default output directory is a dedicated dataset folder,
`output/realityscan/lfs_colmap/`. The script links the existing image and mask
folders instead of copying them, then writes COLMAP sparse files:

```text
output/realityscan/
├─ images/              existing source images
├─ masks/               existing source masks
├─ transforms.json      optional, not used by this route
└─ lfs_colmap/
   ├─ images/           link to ../images
   ├─ masks/            link to ../masks, when masks exist
   └─ sparse/
      └─ 0/
         ├─ cameras.txt
         ├─ images.txt
         ├─ points3D.txt
         └─ points3D.ply
```

Load `output/realityscan/lfs_colmap/` as the dataset folder in LichtFeld.
`images.txt` keeps the original image names and extensions, so JPG/PNG mixtures
are preserved.

## Example

```powershell
python realityscan_to_lfs_colmap.py `
  D:\3DGS\sakume\output\realityscan\rs_sakume.csv `
  --ply D:\3DGS\sakume\output\realityscan\rs_sakume.ply
```

To create a dataset that does not require LichtFeld's runtime `Undistort`
option, pre-undistort only the distorted RealityScan rows:

```powershell
python realityscan_to_lfs_colmap.py `
  D:\3DGS\sakume\output\realityscan\rs_sakume.csv `
  --ply D:\3DGS\sakume\output\realityscan\rs_sakume.ply `
  --pre-undistort-distorted-images
```

When this option is used, the default output root is
`output/realityscan/lfs_colmap_undistorted/`. Distorted images and matching
masks are remapped into that dataset, while already-pinhole cubemap images are
hard-linked when possible. The corresponding distorted COLMAP cameras are
written as `PINHOLE` cameras with the undistorted intrinsics.

## Coordinate Handling

Camera poses are written as COLMAP/OpenCV `images.txt` poses. Because the
RealityScan CSV camera centers and RealityScan PLY are in the same raw world
frame, both are rotated to the LichtFeld COLMAP world by default.

The point cloud is written to `sparse/0/points3D.ply` with X-axis +90 degrees by
default. This matches the observed RealityScan-to-LichtFeld COLMAP workflow.
Camera poses use the same X-axis +90 degrees by default so the camera centers
and `points3D.ply` stay in the same world frame. Use
`--camera-rotation-x-deg 0` only when importing camera poses that are already in
the target COLMAP world.

`points3D.txt` is written as an empty text point file so the dataset remains a
valid COLMAP text folder. LichtFeld prefers `points3D.ply` when it exists.

## Masks

LichtFeld searches the dataset root for `masks/`, `mask/`, `segmentation/`, or
`dynamic_masks/`. The dedicated `lfs_colmap/masks/` folder is linked to the
existing `output/realityscan/masks/` directory when it exists. Mask dimensions
must match the corresponding images.

With `--pre-undistort-distorted-images`, masks for distorted images are remapped
with nearest-neighbor sampling and thresholded back to white=keep, black=exclude.
Masks for already-pinhole images are hard-linked or copied unchanged.

## Notes

Local LichtFeld source shows the COLMAP and Blender/NeRF loaders have the same
priority. A folder that contains both `sparse/0/` and `transforms.json` is
therefore ambiguous. The script refuses that layout by default; use the dedicated
`lfs_colmap/` root, or pass `--allow-mixed-loader-root` only when you deliberately
want to rely on that mixed layout.
