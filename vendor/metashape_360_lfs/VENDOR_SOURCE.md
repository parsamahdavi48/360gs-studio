# Vendored Source Record

This repository vendors `metashape_360_lfs.py` for local/offline workflow setup.

The vendored file is taken from the tetraface fork of `metashape_360_lfs`, not
directly from the original gradeeterna repository. This fork is used because it
is the variant adopted by `tetraface/tetraface-3dgs-utils` for the LichtFeld
Studio-oriented Metashape workflow.

`metashape_360_lfs.py` itself is a LichtFeld-focused converter derived from the
nerfstudio Metashape conversion path. The important behavior for this app is the
LichtFeld coordinate handling for camera transforms and point clouds, not a
generic Metashape export replacement.

- Local path: `vendor/metashape_360_lfs/metashape_360_lfs.py`
- Upstream repository: https://github.com/tetraface/metashape_360_lfs
- Original repository: https://github.com/gradeeterna/metashape_360_lfs
- Upstream file URL: https://raw.githubusercontent.com/tetraface/metashape_360_lfs/main/metashape_360_lfs.py
- Pinned source revision at retrieval: `71434c86922a55c318d14b9d30ccdd2f4264898a` (`refs/heads/main`)
- Retrieved on (UTC): `2026-02-08`
- SHA256 (vendored file): `ed769f6a09504feb687eb092d86e22ec3dfa64cbb97ed6d7b8d2955c0a82ec61`

Local divergence:

- The LichtFeld camera-side 180° Y pre-compensation is retained because current
  LichtFeld builds still apply an internal 180° Y rotation while importing
  transform datasets. The source PLY path does not receive that same
  camera-local rotation.
- The vendored converter does not apply that 180° Y pre-compensation to PLY.
  Downstream Step 4 applies its final orientation correction to `transforms.json`
  and `pointcloud.ply` together so the final dataset remains a matched set.
