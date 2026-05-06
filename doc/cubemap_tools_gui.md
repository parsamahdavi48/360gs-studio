# Step 4 Export GUI

Step 4 converts the 360° images and masks prepared in Steps 1-3, or the result of a Metashape SfM project, into training data that 3DGS applications can load.

In the common workflow, you select the camera XML exported from Metashape, optionally select a point-cloud PLY, and choose whether the output is for Postshot, Brush, or LichtFeld Studio. For LichtFeld Studio, Step 4 can also create a `3DGUT (LichtFeld)` dataset that keeps the original equirectangular images and masks in place.

## Launch

```bat
run_gui.bat --scene .\scene01
```

Then open `Step 4: Export` in the workflow sidebar.

## First Choice

When you open Step 4, first decide which route you are on.

| Goal | `Select:` | Main settings |
| --- | --- | --- |
| Use Metashape SfM results in Postshot / Brush / LichtFeld | `Metashape` | `Output Preset`, `Output Shape`, `Camera XML`, `Point Cloud PLY` |
| Compare cubemap/projection-view training against LichtFeld 3DGUT | `Metashape` | `Output Preset: LichtFeld Studio`, `Output Shape` |
| Skip Metashape and continue from extracted 360° images to COLMAP/GLOMAP | `COLMAP` | `COLMAP Run Settings`, `Projection Views` |
| Skip Metashape and run spherical SfM directly on extracted equirectangular images | `SphereSfM` | `SphereSfM COLMAP Executable`, `Run Scope`, `Matcher`, `SfM Quality`, `Output Shape` |
| Rebuild only images, only masks, or metadata for an existing export | Same route as before | `Output`, `Projection Views`, `Image Size` |

`Scene Directory` is the scene folder used by Steps 1-3. It usually contains `images/` and `masks/`. In the Metashape route, Step 4 combines that scene folder with the XML/PLY exported from Metashape to create 3DGS-ready data.

## Metashape Route

If Metashape has already aligned the 360° images, use this flow.

1. Set `Select:` to `Metashape`.
2. Set `Camera XML` to the camera XML exported from Metashape.
3. Set `Point Cloud PLY` when exporting for LichtFeld Studio or when you want to package a point cloud.
4. Choose the downstream app in `Output Preset`.
5. Choose whether to convert to projection views or create `3DGUT (LichtFeld)` in `Output Shape`.
6. If converting to projection views, review the view preset, yaw, and image size in the `Projection Views` tab.
7. Run the export.

In the Metashape route, Step 4 first runs the bundled `vendor/metashape_360_lfs/metashape_360_lfs.py` converter to create `transforms.json` from the Metashape XML. When `Output Shape` is `Convert to Projection Views`, Step 4 then runs `cubemap_transforms_json.py` to create viewpoint images and masks.

## Output Preset

`Output Preset` chooses the downstream 3DGS application.

| Preset | Use when |
| --- | --- |
| `Postshot` | Creating projection-view data for Postshot |
| `Brush` | Creating projection-view data for Brush |
| `LichtFeld Studio` | Creating projection-view data or direct 3DGUT data for LichtFeld |
| `Custom` | Manually adjusting coordinate transforms or PLY handling |

Usually, choose the name of the app you will load the dataset into. If advanced settings change the coordinate transform, PLY usage, or Metashape import options away from the preset defaults, the preset switches to `Custom`.

## Output Shape

`Output Shape` controls how the Metashape equirectangular images become training data.

### Convert to Projection Views

This is the normal path. Step 4 converts the equirectangular images into cubemap/projection-view images, then writes images, masks, and `transforms.json` under `output/`.

This output is easier to use in Postshot, Brush, and LichtFeld Studio because it behaves like a normal pinhole-camera dataset. When training this output in LichtFeld, you normally do not enable GUT or Undistort.

### 3DGUT (LichtFeld)

This mode is for testing LichtFeld Studio's GUT training path. It keeps the Metashape source equirectangular `images/` and `masks/` in place, and does not create converted viewpoint images or converted masks.

This mode writes these files at the scene root:

- `transforms.json`
- `pointcloud.ply`
- `stechdrive_export_settings.json`

`3DGUT (LichtFeld)` requires `Output Preset: LichtFeld Studio` and PLY usage. While it is active, the `Projection Views` tab, image/mask output toggles, and COLMAP text-model export are disabled.

## Comparing Outputs In LichtFeld

To compare projection-view training against direct 3DGUT training in LichtFeld, export twice from the same Metashape result.

### Cubemap / Projection-View Version

1. Set `Select:` to `Metashape`.
2. Set `Output Preset` to `LichtFeld Studio`.
3. Set `Output Shape` to `Convert to Projection Views`.
4. Select the `Point Cloud PLY`.
5. In `Projection Views`, start with `Cube6`, yaw 45°, and the image size you want to test.
6. Run the export.

The output is normally `<scene>/output/`. Load that `output/` folder in LichtFeld.

### Direct 3DGUT Version

1. Set `Select:` to `Metashape`.
2. Set `Output Preset` to `LichtFeld Studio`.
3. Set `Output Shape` to `3DGUT (LichtFeld)`.
4. Select the `Point Cloud PLY`.
5. Run the export.

The output is written at the scene root. Load the scene folder in LichtFeld and enable GUT during training.

For a fair comparison, keep the Metashape XML, PLY, `images/`, `masks/`, and LichtFeld training settings as close as possible. If differences are subtle, inspect nearby objects, thin lines, text, stitch-seam regions, and floor/wall continuity.

## Projection View Settings

When `Output Shape` is `Convert to Projection Views`, use the `Projection Views` tab to choose which directions to export.

### Start With Cube6

`Cube6` is the usual starting point. It creates six faces including top and bottom, which is a practical baseline for Postshot, Brush, and LichtFeld Studio.

The default 45° `Yaw Offset` is recommended. For unstabilized dual-fisheye 360° footage, stitch seams often sit around the 25% and 75% horizontal positions of the equirectangular image. A 45° offset keeps those seams away from the center of the cubemap faces.

### Custom Grid

Use `Custom Grid` when you want to add only specific directions, reduce top/bottom views, or add upward/downward oblique views.

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

If VRAM or processing time is tight, use `Normal` or `Half` to verify the workflow first, then try `Full` for the final comparison.

## Rebuilding Only Images Or Masks

The `Output` checkboxes control whether images and/or masks are written.

| Goal | Setting |
| --- | --- |
| Rebuild both images and masks | `Images` ON, `Masks` ON |
| Rebuild masks only | `Images` OFF, `Masks` ON |
| Rebuild images only | `Images` ON, `Masks` OFF |
| Update camera metadata only | `Images` OFF, `Masks` OFF |

After adjusting masks, turning `Images` off avoids reconverting existing viewpoint images. `3DGUT (LichtFeld)` references source images and masks directly, so these output toggles are not used in that mode.

## COLMAP Route

If you want to skip Metashape and continue from extracted 360° images to COLMAP/GLOMAP, set `Select:` to `COLMAP`.

1. Confirm that `Scene Directory` contains `images/` and, when needed, `masks/`.
2. Set `Select:` to `COLMAP`.
3. Choose view count, yaw, and image size in `Projection Views`.
4. Enable `Run COLMAP after export` only when you want to start SfM immediately.
5. Choose `Matcher` and `Mapper`. Start with `Sequential` and `Global` in most video workflows.
6. Run the export.

The COLMAP route writes a COLMAP Rig dataset under `output/colmap_rig/`, including viewpoint images, masks, and `rig_config.json`. If `Run COLMAP after export` is enabled, the GUI then runs Feature, Rig setup, Matcher, and Mapper.

Per-frame yaw rotation is always forced to 0 for COLMAP Rig export because changing yaw per frame breaks the fixed-rig assumption.

## SphereSfM Route

If you have SphereSfM's COLMAP build, you can run SfM directly on extracted equirectangular images and then export either a LichtFeld 3DGUT dataset or projected viewpoint data from the same sparse model.

1. Confirm that `Scene Directory` contains `images/` and, when used, `masks/`.
2. Set `Select:` to `SphereSfM`.
3. Set `SphereSfM COLMAP Executable` to the `colmap.exe` from a SphereSfM release or build.
4. Usually keep `Use masks/` enabled. Step 3 masks use white=keep and black=exclude; the GUI converts them to COLMAP's `image.jpg.png` naming.
5. Set `Run Scope`. `SfM + Convert` is the normal route, `SfM Only` rebuilds just the sparse model, and `Convert Existing SfM` reuses an existing `<scene>/output/spheresfm/sparse/` model.
6. Use `Sequential` matcher for video frames. Use `Spatial` only when you provide a POS file.
7. Start with `SfM Quality: Standard`; use `Fast` for trials or large frame sets and `Quality` when registration coverage is weak.
8. In `Conversion`, choose `Output Shape`.
9. Run the export.

At the start of a SphereSfM run, the GUI automatically copies one source image into `<scene>/output/spheresfm/preflight/` and runs a small GPU SIFT check before the full database is populated. If the selected binary cannot run CUDA SIFT on the current GPU, the run stops there and the phase log points to the cause.

When `Output Shape` is `3DGUT (LichtFeld)`, Step 4 writes `<scene>/transforms.json` and `<scene>/pointcloud.ply`. Frame paths reference the scene-level `images/` folder. If an existing `transforms.json` or `pointcloud.ply` is present, the GUI asks before replacing it.

When `Output Shape` is `Convert to Projection Views`, the `Projection Views` tab and image/mask output toggles are active. Step 4 first writes an equirectangular intermediate under `<scene>/output/spheresfm/equirect/`, then writes `images/`, `masks/`, and `transforms.json` directly under `<scene>/output/`, matching the Metashape route.

The SphereSfM project root at `<scene>/output/spheresfm/` contains working files such as `preflight/`, `database.db`, `masks_colmap/`, `sparse/`, `equirect/`, and `stechdrive_spheresfm_project.json`.

After the run, use `View Result in COLMAP GUI` to inspect registered camera poses and sparse points.

## Outputs

| Route | Main outputs |
| --- | --- |
| Metashape + Convert to Projection Views | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/stechdrive_export_settings.json` |
| Metashape + `3DGUT (LichtFeld)` | `<scene>/transforms.json`, `<scene>/pointcloud.ply`, `<scene>/stechdrive_export_settings.json` |
| COLMAP | `<scene>/output/colmap_rig/images/`, `<scene>/output/colmap_rig/masks/`, `<scene>/output/colmap_rig/rig_config.json` |
| COLMAP with SfM enabled | The COLMAP/GLOMAP SfM result in addition to the files above |
| SphereSfM + `3DGUT (LichtFeld)` | `<scene>/transforms.json`, `<scene>/pointcloud.ply`, plus the SphereSfM project files under `<scene>/output/spheresfm/` |
| SphereSfM + Convert to Projection Views | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, plus the SphereSfM project files under `<scene>/output/spheresfm/` |

With the `LichtFeld Studio` profile, Step 4 applies the same final orientation correction to `transforms.json` and `pointcloud.ply` so +X / +Z / up directions match the Metashape scene in LichtFeld.

## Common Decisions

- Use `Convert to Projection Views` for Postshot and Brush.
- For normal LichtFeld training, start with `LichtFeld Studio` + `Convert to Projection Views`.
- To test LichtFeld GUT, use `LichtFeld Studio` + `3DGUT (LichtFeld)`.
- When training projection-view data in LichtFeld, GUT and Undistort are usually unnecessary.
- When training `3DGUT (LichtFeld)` data, enable GUT in LichtFeld.
- If stitch seams are not visible, keep stitch masks off or use a narrow seam mask first. Keeping yaw at 45° is usually fine because it does not discard pixels.
- If a required PLY is missing for the selected profile, Step 4 stops before running.
