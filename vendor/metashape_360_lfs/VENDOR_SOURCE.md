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
- SHA256 (vendored file): `1506d9f55189d7710947eb104ed370ef6fca47f1b5ee7190f6d86e6b12ff0b71`

Local divergence:

- The old LichtFeld camera-only 180° Y pre-compensation was removed for the
  current StechDrive/LichtFeld workflow. Current LichtFeld builds expect the
  external `transforms.json` camera poses and `pointcloud.ply` points to arrive
  in the same NeRF/Blender-style input world, then normalize camera axes and
  point-cloud basis consistently during dataset import. Downstream Step 4 no
  longer applies a separate LichtFeld final-orientation correction.
