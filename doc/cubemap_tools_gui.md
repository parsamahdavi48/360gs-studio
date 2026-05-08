# Step 4 Export GUI

Step 4 converts the 360° images and masks prepared in Steps 1-3, Metashape SfM results, or SphereSfM results created by this step into training data that 3DGS applications can load.

In the common workflow, you select the camera XML exported from Metashape, optionally select a point-cloud PLY, and choose whether the output is for Postshot, Brush, or LichtFeld Studio. If you are not using Metashape, you can export COLMAP Rig cubemap images or run SphereSfM directly on equirectangular images and convert the result into 3DGS-ready data. When needed, the same Step 4 can continue into LichtFeld Studio or Postshot CLI training from the `Training` tab.

## Launch

```bat
run_gui.bat --scene .\scene01
```

Then open `Step 4: Export` in the workflow sidebar.

The `?` help icon at the right edge of the center-panel header opens this step's GitHub documentation. It opens the Japanese or English page to match the active UI language.

## First Choice

When you open Step 4, first decide which route you are on.

SfM, or Structure from Motion, estimates camera positions and a sparse point cloud from differences between multiple images. 3DGS tools use those camera positions to read the training data. The Metashape route converts an SfM result created in Metashape; the SphereSfM route and the COLMAP route with COLMAP enabled can run SfM from this app.

| Goal | Route | Main settings |
| --- | --- | --- |
| Use Metashape SfM results in Postshot / Brush / LichtFeld | `Metashape` | `Output Preset`, `Output Shape`, `Camera XML`, `Point Cloud PLY` |
| Create direct LichtFeld 3DGUT data | `Metashape` | `Output Preset: LichtFeld Studio`, `Output Shape`, `Point Cloud PLY` |
| Skip Metashape and continue from extracted 360° images to COLMAP/GLOMAP | `COLMAP` | `COLMAP Run Settings`, `Cubemap` |
| Skip Metashape and run spherical SfM directly on extracted equirectangular images | `SphereSfM` | `SphereSfM COLMAP Executable`, `Run Scope`, `Matcher`, `SfM Quality`, `Output Shape` |
| Rebuild only images, only masks, or metadata for an existing export | Same route as before | `Cubemap`, `Image Size` |

`Scene Directory` is the scene folder used by Steps 1-3. It usually contains `images/` and `masks/`. In the Metashape route, Step 4 combines that scene folder with the XML/PLY exported from Metashape. In the SphereSfM route, Step 4 runs SfM and conversion from these `images/` and `masks/`.

The left navigation always shows the Step 4 sub-stages `SfM`, `Cube`, and `Train`. Inside Step 4, the route buttons `Metashape`, `COLMAP`, and `SphereSfM` live in the `SfM` tab. Output preset, image/mask toggles, Cube6, yaw, and image size live in `Cubemap`; external CLI training settings live in `Training`.

## Metashape Route

If Metashape has already aligned the 360° images, use this flow.

1. Set the route to `Metashape`.
2. Check `Camera XML`. If a Metashape-exported XML exists at the scene root, Step 4 fills it automatically.
3. Check `Point Cloud PLY` when exporting for LichtFeld Studio or when you want to package a point cloud. If a Metashape-exported PLY exists at the scene root, Step 4 fills it automatically.
4. Choose the downstream app in `Output Preset`.
5. Choose whether to convert to cubemap images or create `3DGUT (LichtFeld)` in `Output Shape`.
6. If creating cubemap images, review Cube6, yaw, and image size in the `Cubemap` tab.
7. Run the export.

In the Metashape route, Step 4 first runs the bundled `vendor/metashape_360_lfs/metashape_360_lfs.py` converter to create `transforms.json` from the Metashape XML. When `Output Shape` is `Convert to Projection Views`, Step 4 then runs `cubemap_transforms_json.py` to create cubemap images and masks.

### Auto-Detecting Camera XML And Point Cloud PLY

When the scene folder is set, Step 4 automatically looks for Metashape input candidates. If the chosen candidate is wrong, use the browse button on that row to select the correct file manually.

- `Camera XML`: checks `metashape.xml`, `cameras.xml`, then the first `.xml` file at the scene root.
- `Point Cloud PLY`: checks `metashape.ply`, `sparse.ply`, then the first `.ply` file at the scene root.

`pointcloud.ply` is the output filename this app writes for LichtFeld, so it is excluded from automatic Metashape input candidates.

## Output Preset

`Output Preset` chooses the downstream 3DGS application.

| Preset | Use when |
| --- | --- |
| `Postshot` | Creating cubemap data for Postshot |
| `Brush` | Creating cubemap data for Brush |
| `LichtFeld Studio` | Creating cubemap data or 3DGUT data for LichtFeld |
| `Custom` | Manually adjusting coordinate transforms or PLY handling |

Usually, choose the name of the app you will load the dataset into. If advanced settings change the coordinate transform, PLY usage, or Metashape import options away from the preset defaults, the preset switches to `Custom`.

## Output Shape

`Output Shape` controls how equirectangular images become training data. It is used by both the Metashape route and the SphereSfM route.

### Convert to Projection Views

This is the normal path. Step 4 converts the equirectangular images into cubemap images, then writes images, masks, and `transforms.json` under `output/`. Cube6 is the standard preset, but you can adjust the exported directions in the `Cubemap` tab when needed.

This output is easier to use in Postshot, Brush, and LichtFeld Studio because it behaves like a normal pinhole-camera dataset. When training this output in LichtFeld, you normally do not enable GUT or Undistort.

### 3DGUT (LichtFeld)

This mode creates data for LichtFeld Studio's 3DGUT training path. It places the source equirectangular `images/` and `masks/` under `output/` with hard links where possible, and does not create converted cubemap images or converted masks.

This mode writes these files under `output/` for the LichtFeld 3DGUT dataset:

- `transforms.json`
- `pointcloud.ply`

It also saves `_stechdrive/export_settings.json` as this app's settings record.

In the Metashape route, `Output Preset: LichtFeld Studio` and a point-cloud PLY are required. In the SphereSfM route, Step 4 creates `pointcloud.ply` from the SfM result. While `3DGUT (LichtFeld)` is active, projection-view controls, image/mask output toggles, and COLMAP text-model export are disabled. The finished dataset is still `<scene>/output/`.

## Using Cubemap Data And 3DGUT In LichtFeld

To prepare both cubemap data and direct 3DGUT data for LichtFeld, export twice from the same Metashape result.

### Cubemap Version

1. Set the route to `Metashape`.
2. Set `Output Preset` to `LichtFeld Studio`.
3. Set `Output Shape` to `Convert to Projection Views`.
4. Check `Point Cloud PLY`. If it was not filled automatically, or if the candidate is wrong, select it manually.
5. In `Cubemap`, start with `Cube6`, yaw 45°, and the image size you want to test.
6. Run the export.

The output is normally `<scene>/output/`. Load that `output/` folder in LichtFeld.

### 3DGUT Version

1. Set the route to `Metashape`.
2. Set `Output Preset` to `LichtFeld Studio`.
3. Set `Output Shape` to `3DGUT (LichtFeld)`.
4. Check `Point Cloud PLY`. If it was not filled automatically, or if the candidate is wrong, select it manually.
5. Run the export.

This output links the existing `<scene>/images/` and `<scene>/masks/` into `<scene>/output/images/` and `<scene>/output/masks/`, then writes `<scene>/output/transforms.json` and `<scene>/output/pointcloud.ply`. In LichtFeld, use `<scene>/output/` as the dataset and enable GUT during training.

## Projection View Settings

`Convert to Projection Views` exports cubemap images from the 360-degree source. The standard `Cube6` preset creates front, back, left, right, up, and down views. In the `Cubemap` tab, adjust direction count, vertical rows, and which views are written when needed.

### Start With Cube6

`Cube6` is the usual starting point. It writes the front, back, left, right, up, and down directions, which is a practical baseline for Postshot, Brush, and LichtFeld Studio.

The default 45° `Yaw Offset` is recommended. For unstabilized dual-fisheye 360° footage, stitch seams often sit around the 25% and 75% horizontal positions of the equirectangular image. A 45° offset keeps those seams away from the center of the cubemap faces.

### Custom Grid

Use `Custom Grid` when six directions are not enough, when you want fewer top/bottom views, or when you want upward/downward oblique views.

- `Yaw Slots`: horizontal slot count, from 4 to 8.
- `Pitch Rows`: vertical rows, from `-90..90`, up to 5 rows.
- Each checkbox: whether that view is exported.

More enabled views means more output files and longer processing. The GUI warns above 24 enabled views and blocks execution above 40.

### Image Size

`Image Size` controls the resolution of each projection view.

| Setting | Use when |
| --- | --- |
| `Full` | Final-quality checks. Heavier, but preserves more detail |
| `Normal` | Standard first pass. Matches the center angular resolution of a 90° view to the source image |
| `Half` | Lightweight tests. Faster, but training can look softer |

If VRAM or processing time is tight, use `Normal` or `Half` to verify the workflow first, then use `Full` for final output.

## Rebuilding Only Images Or Masks

The `Output` checkboxes control whether images and/or masks are written.

| Goal | Setting |
| --- | --- |
| Rebuild both images and masks | `Images` ON, `Masks` ON |
| Rebuild masks only | `Images` OFF, `Masks` ON |
| Rebuild images only | `Images` ON, `Masks` OFF |
| Update camera metadata only | `Images` OFF, `Masks` OFF |

After adjusting masks, turning `Images` off avoids reconverting existing cubemap images. `3DGUT (LichtFeld)` references source images and masks directly, so these output toggles are not used in that mode.

## Training Tab

The `Training` tab can launch an external training CLI after Step 4 export or SfM conversion finishes. Choose the training application at the top, then enable `Start training after export`.

| Training app | Use when |
| --- | --- |
| `LichtFeld Studio` | Pass the dataset, output folder, and generated config JSON to LichtFeld Studio CLI |
| `Postshot` | Pass images and, when available, a COLMAP/SphereSfM sparse model to Postshot CLI and create a `.psht` project |
| `Custom` | Launch any CLI with template-based arguments |

Normally, leave `Dataset` on the automatic value. Cubemap conversion uses `<scene>/output/`, 3DGUT uses `<scene>/`, and the COLMAP route uses `<scene>/output/colmap_rig/`. The default `Training Output` is `<scene>/output/`, so the portable dataset and training result stay together.

For LichtFeld Studio, the visible controls mirror the parameters LichtFeld keeps at the top of its own training panel: `Strategy`, `Iterations`, `Max Gaussians`, output PLY name, `SH Degree`, `Tile Mode`, `Steps Scaler`, mask toggles, PPISP conditionals, and background `Mode` / `Color`. The output PLY name defaults to the scene folder name and is passed as LichtFeld's `--output-name`. Less common dataset, optimizer, refinement, loss, initialization, MRNF/IGS+, sparsity, and save/eval settings are under `Advanced Training Parameters`; sections appear only when the matching strategy or top-level checkbox makes them relevant. With `Steps Scaler` on `Auto`, Step 4 counts the emitted training images and applies the same 300-image baseline scaling LichtFeld Studio uses when loading a dataset in its GUI. At runtime, Step 4 writes `_stechdrive/training/lichtfeld_config.json` and passes that JSON plus dataset-only CLI options to LichtFeld.

For Postshot, the main controls cover the project file, model `Profile`, automatic or fixed `kSteps`, maximum image size, optional mask import, image selection, and `Camera Poses`. `Camera Poses` defaults to `Import`, so the CLI receives the generated images, `transforms.json`, and the raw Metashape PLY when available; switch to `Estimate` only when Postshot should estimate poses and use `Pose Quality`. Postshot mask wording is polarity-driven in this GUI: use `Exclude black / use white (background)` for the app's normal white=usable, black=excluded masks, and `Exclude white / use black (occluders)` only when white marks temporary occluders to ignore. Less common GPU, profile-specific model limits, anti-aliasing, sky/training-context flags, Crop/ROI boxes, and optional PLY/SPZ export are grouped under `Postshot Advanced Parameters`.

SphereSfM `SfM Only` does not create a training dataset, so it cannot be combined with automatic training. To continue into training, use `SfM + Convert`, or use `Convert Existing SfM` after a sparse model already exists.

## COLMAP Route

If you want to skip Metashape and continue from extracted 360° images to COLMAP/GLOMAP, set the route to `COLMAP`.

1. Confirm that `Scene Directory` contains `images/` and, when needed, `masks/`.
2. Set the route to `COLMAP`.
3. Choose view count, yaw, and image size in `Cubemap`.
4. Enable `Run COLMAP after export` when you want COLMAP/GLOMAP to estimate camera positions and a sparse point cloud. This continues into SfM after writing the cubemap images, so it can take a long time depending on frame count.
5. Choose `Matcher` and `Mapper`. Start with `Sequential` and `Global` in most video workflows.
6. Run the export.

The COLMAP route writes a COLMAP Rig dataset under `output/colmap_rig/`, including cubemap images, masks, and `rig_config.json`. If `Run COLMAP after export` is enabled, the GUI then runs Feature, Rig setup, Matcher, and Mapper to create a COLMAP/GLOMAP SfM result.

Per-frame yaw rotation is always forced to 0 for COLMAP Rig export because changing yaw per frame breaks the fixed-rig assumption.

## SphereSfM Route

The SphereSfM route runs SfM directly on extracted equirectangular images as spherical cameras, without using Metashape. It has two goals:

- create a direct LichtFeld 3DGUT dataset
- create cubemap data that is easier to use in Postshot, Brush, or LichtFeld

This route requires SphereSfM's `colmap.exe`, not standard COLMAP. The app does not bundle SphereSfM itself, so select the executable you prepared from a [json87/SphereSfM](https://github.com/json87/spheresfm) release or local build.

On RTX 50-series GPUs, the Windows binary distributed on GitHub can stop during CUDA SIFT. RTX 50-series GPUs need CUDA code built for the newer `sm_120` architecture; if the distributed binary was built without that target, CUDA fails with `no kernel image is available for execution on the device`. For RTX 50-series systems, build SphereSfM locally with a CUDA/CMake environment that supports RTX 50-series GPUs and set `CMAKE_CUDA_ARCHITECTURES=120`, then select that `colmap.exe`.

1. Confirm that `Scene Directory` contains `images/` and, when used, `masks/`.
2. Set the route to `SphereSfM`.
3. Set `SphereSfM COLMAP Executable` to the `colmap.exe` from a SphereSfM release or build.
4. Usually keep `Use masks/` enabled. Step 3 masks use white=keep and black=exclude; the GUI converts them to COLMAP's `image.jpg.png` naming.
5. Set `Run Scope`. `SfM + Convert` is the normal route, `SfM Only` rebuilds just the sparse model, and `Convert Existing SfM` reuses an existing `<scene>/output/spheresfm/sparse/` model.
6. Use `Sequential` matcher for video frames. Use `Spatial` only when you provide a POS file.
7. Start with `SfM Quality: Standard`; use `Fast` for trials or large frame sets and `Quality` when registration coverage is weak.
8. In the `Cubemap` tab, choose `Output Shape`.
9. Run the export.

`SfM Only` creates only the SfM result under `<scene>/output/spheresfm/sparse/`. It does not yet create a dataset for a 3DGS app. `Convert Existing SfM` reuses that sparse result when you only want to rebuild the 3DGUT/cubemap output.

At the start of a SphereSfM run, the GUI automatically copies one source image into `<scene>/output/spheresfm/preflight/` and runs a small GPU SIFT check before the full database is populated. If the selected binary cannot run CUDA SIFT on the current GPU, the run stops there and shows the phase log and likely cause.

When `Output Shape` is `3DGUT (LichtFeld)`, Step 4 uses the existing `<scene>/images/` and `<scene>/masks/`, places them under `<scene>/output/images/` and `<scene>/output/masks/` with hard links where possible, then writes `<scene>/output/transforms.json` and `<scene>/output/pointcloud.ply`. The LichtFeld 3DGUT dataset is `<scene>/output/`. If existing 3DGUT dataset files are present in `output/`, the GUI asks before replacing them.

When `Output Shape` is `Convert to Projection Views`, `<scene>/output/` is the cubemap dataset to load in the downstream app. The `Cubemap` tab's projection controls and image/mask output toggles are active, and Step 4 writes `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, and `<scene>/output/pointcloud.ply`, matching the Metashape route.

The SphereSfM project root at `<scene>/output/spheresfm/` contains working files such as `preflight/`, `database.db`, `masks_colmap/`, `sparse/`, `equirect/`, `logs/`, and `stechdrive_spheresfm_project.json`. This folder is for SfM reuse and log inspection; the portable training dataset is `output/` for both 3DGUT and cubemap output.

After the run, use `View Result in COLMAP GUI` to inspect registered camera poses and sparse points. If your SphereSfM build has no Qt GUI support, only this viewer button is unavailable; SfM and conversion output are separate from that viewer.

## Outputs

| Route | Main outputs |
| --- | --- |
| Metashape + cubemap conversion (`Convert to Projection Views`) | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json` |
| Metashape + `3DGUT (LichtFeld)` | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/pointcloud.ply` |
| COLMAP | `<scene>/output/colmap_rig/images/`, `<scene>/output/colmap_rig/masks/`, `<scene>/output/colmap_rig/rig_config.json` |
| COLMAP with SfM enabled | The COLMAP/GLOMAP SfM result in addition to the files above |
| SphereSfM + `3DGUT (LichtFeld)` | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/pointcloud.ply` |
| SphereSfM + cubemap conversion (`Convert to Projection Views`) | Load `<scene>/output/` in the downstream app. It contains `images/`, `masks/`, `transforms.json`, and `pointcloud.ply` |
| Training enabled | The selected route output plus training results in `<scene>/output/`; LichtFeld also writes `_stechdrive/training/lichtfeld_config.json` |

Step 4 saves this app's settings record to `<scene>/_stechdrive/export_settings.json`. Cubemap routes also save the view layout to `<scene>/_stechdrive/views_config.json`. These files are for reopening and reproducing the export in this app, not the dataset files you pass to 3DGS apps.

With the `LichtFeld Studio` profile, Step 4 applies the same final orientation correction to `transforms.json` and `pointcloud.ply` so +X / +Z / up directions match the Metashape scene in LichtFeld.

## Common Decisions

- Use `Convert to Projection Views` for Postshot and Brush.
- For normal LichtFeld training, start with `LichtFeld Studio` + `Convert to Projection Views`.
- To use LichtFeld GUT, choose `LichtFeld Studio` + `3DGUT (LichtFeld)`.
- When training cubemap data in LichtFeld, GUT and Undistort are usually unnecessary.
- When training `3DGUT (LichtFeld)` data, enable GUT in LichtFeld.
- If stitch seams are not visible, keep stitch masks off or use a narrow seam mask first. Keeping yaw at 45° is usually fine because it does not discard pixels.
- In the Metashape route, if the selected profile requires a `Point Cloud PLY` and it is missing, Step 4 stops before running.
