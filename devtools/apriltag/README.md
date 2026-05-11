# AprilTag Development Tools

This directory contains development-only tools for validating the AprilTag scale
workflow. These tools are not part of the normal user-facing GUI flow.

## AprilTag Scene Viewer

`scripts/dev_apriltag_scene_viewer.py` opens the AprilTag synthetic validation
viewer. The viewer exists to validate the scale detector and the Cube6 coordinate
pipeline, not to act as a general SfM or COLMAP viewer.

Primary workflow:

1. Load a scene folder that has already been exported to Cube6.
2. Inspect `transforms.json`, optional `pointcloud.ply`, Cube6 images, and camera
   face directions.
3. Place a synthetic AprilTag in SfM space.
4. Project the tag into selected Cube6 images.
5. Run the existing AprilTag scale detector against the generated images.
6. Compare the detected scale with the expected synthetic scale.

## Run

From this directory, double-click:

```bat
start_scene_viewer.bat
```

Or run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\dev_apriltag_scene_viewer.py
```

To open a scene directly:

```powershell
.\.venv\Scripts\python.exe scripts\dev_apriltag_scene_viewer.py --case D:\path\to\scene
```

The `--case` path can be either:

- a scene folder containing `output/transforms.json`
- an AprilTag dev case folder containing `case.json`

## Expected Inputs

For normal scene loading, the expected layout is:

```text
scene/
  output/
    transforms.json
    images/
      ...
    pointcloud.ply        optional
```

`transforms.json` must currently use `PINHOLE` or `SIMPLE_PINHOLE` camera data.
`pointcloud.ply` is optional; without it the viewer can still show cameras and
tag projections, but not the point cloud.

## Scope

Validated and intended use:

- LichtFeld Cube6 output from this repository's Step 4 workflow
- AprilTag synthetic placement and injection checks
- scale-estimation regression checks
- visual inspection of Cube6 image rays, JSON face rays, and tag projection

Not intended as:

- a general COLMAP sparse model viewer
- a general SphereSfM result viewer
- a replacement for the normal application GUI
- a production UI for end users

COLMAP and SphereSfM raw sparse outputs need an adapter before this viewer can
load them. SphereSfM Cube6 projection output can be tested when it produces
compatible `PINHOLE`/`SIMPLE_PINHOLE` `transforms.json` and image files.

## Output

Synthetic validation runs are written under the loaded case or scene validation
run directory. For scene folders, this is under:

```text
scene/output/apriltag_scale_validation/
```

Each run writes generated images, a filtered `transforms.json`, and a validation
report JSON.

## Regression Tests

The related tests are mainly in:

```text
tests/test_apriltag_scene_viewer.py
tests/test_apriltag_world_debug_view.py
tests/test_apriltag_devtools.py
```

The optional local comparison fixture under `_compare/apriltag_test/cases/current`
is intentionally not tracked. Tests that need it skip when the local fixture is
not present.
