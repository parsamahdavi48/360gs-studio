# Step 5 Training GUI

Step 5 launches LichtFeld Studio, Postshot, or a custom CLI with a 3DGS dataset that Step 4 has already created. It does not run image conversion or SfM. Use `Step 4: Convert` to create the dataset, then use `Step 5: Training` to launch a training application with that dataset.

## First Choice

When you open Step 5, first decide which app you want to run and which dataset you want to test.

| Goal | Training app | Main settings to check |
| --- | --- | --- |
| Train normal cubemap data in LichtFeld | `LichtFeld Studio` | `Dataset`, `GUT` off, output PLY name, `Strategy`, `Iterations` |
| Try 3DGUT data in LichtFeld | `LichtFeld Studio` | `Dataset`, `GUT` on, confirm `pointcloud.ply` exists |
| Create a Postshot project | `Postshot` | `Dataset`, `Camera Poses`, project name, `Profile` |
| Run any training CLI | `Other... > Custom` | `Executable`, `Argument Template`, `Dataset`, `Training Output` |

Normally, leave `Dataset` on the automatic value. Metashape and SphereSfM conversion results use `<scene>/output/`; the COLMAP route uses `<scene>/output/colmap_rig/`. `Training Output` defaults to `<scene>/output/`. Keeping the dataset and training result under `output/` makes the scene easier to move later.

## Basic Flow

1. Create the dataset in Step 4.
2. Open `Step 5: Training`.
3. Confirm that `Dataset` and `Training Output` point to the intended folders.
4. Choose `LichtFeld Studio`, `Postshot`, or `Other... > Custom`.
5. If the executable cannot be found automatically, select the installed exe.
6. Review the app-specific settings on the right.
7. Press `Launch`.

Before running, Step 5 checks that the selected training mode matches the dataset shape. For example, LichtFeld `GUT` expects 3DGUT data, while normal LichtFeld and Postshot expect projected cubemap data.

## Layout

Step 5 uses a wider center panel and is arranged as two columns.

| Area | Contents |
| --- | --- |
| Left | Training app, Headless, executable, dataset, training output |
| Right | LichtFeld / Postshot / Custom settings |

The right side separates common settings from advanced parameters. Start with the visible settings, run once, then open the advanced sections only when you have a reason to compare a specific option.

## Common Settings

### Training App

`LichtFeld Studio` and `Postshot` are available as primary choices. `Other...` opens secondary choices; currently it contains `Custom`.

### Executable

When this field is empty, the GUI tries the default executable name or a known install location. If it is not found, select the executable yourself.

| App | Example |
| --- | --- |
| LichtFeld Studio | `LichtFeld-Studio.exe` |
| Postshot | `postshot-cli.exe` |
| Custom | The CLI executable you want to run |

### Dataset

This is the dataset folder passed to the training app. It is normally set from the current Step 4 route and output shape.

| Step 4 result | Step 5 dataset |
| --- | --- |
| Metashape + projected cubemap | `<scene>/output/` |
| Metashape + 3DGUT | `<scene>/output/` |
| SphereSfM + projected cubemap | `<scene>/output/` |
| SphereSfM + 3DGUT | `<scene>/output/` |
| COLMAP Rig | `<scene>/output/colmap_rig/` |

You can choose another folder manually. If you do, make sure it contains the files required by the selected training app: `images/`, optional `masks/`, camera poses, and point cloud data when needed.

### Training Output

This is where training results or project files are written. The default is `<scene>/output/`.

If the final LichtFeld PLY, Postshot `.psht`, or optional Postshot PLY/SPZ export already exists, Step 5 stops before running so the result is not overwritten. Change the output name or output folder, then run again.

## LichtFeld Studio

For LichtFeld Studio, Step 5 starts training with the selected dataset, output folder, and training settings.

### Main Settings

| Setting | How to use it |
| --- | --- |
| `Strategy` | Start with the default MRNF. Try MCMC or IGS+ only when you need to compare them. |
| `Iterations` | Training step count. Keep the default for a first run, then adjust for comparisons. |
| `Max Gaussians` | Upper limit for gaussian count. It affects quality, VRAM, and speed. |
| `Output PLY Name` | File name LichtFeld writes. It defaults to the scene folder name. |
| `SH Degree` | Usually 3. Lower it only for lighter tests. |
| `Tile Mode` | Adjust for VRAM and speed. |
| `Steps Scaler` | `Auto` computes the multiplier from the dataset image count. |
| `GUT` | Turn this on only for 3DGUT datasets. |

### Training Cubemap Data

This is data created in Step 4 with `Convert to Projection Views`. `Dataset` is normally `<scene>/output/`.

- Keep `GUT` off.
- Usually keep `Undistort` off too.
- If using masks, choose the LichtFeld mask mode that matches the result you want.

### Training 3DGUT Data

This is data created in Step 4 with `3DGUT (LichtFeld)`. `Dataset` is normally `<scene>/output/`.

- Turn `GUT` on.
- `<scene>/output/pointcloud.ply` is required.
- Create the 3DGUT dataset from the Metashape or SphereSfM route in Step 4 before launching Step 5.

### Steps Scaler

With `Steps Scaler` set to `Auto`, Step 5 counts images in `Dataset/images/` and uses the same 300-image baseline adjustment that LichtFeld Studio applies when loading a dataset in its GUI. Use a fixed value only when you intentionally want to compare different schedule scaling.

### Advanced Parameters

Less common Dataset, Optimizer, Refinement, Loss, Initialization, MRNF/IGS+, Sparsity, and Save/Eval options are grouped under `Advanced Training Parameters`. Only the settings relevant to the selected strategy or top-level checkbox are shown.

## Postshot

For Postshot, Step 5 creates a `.psht` project from the selected images and camera poses.

### Main Settings

| Setting | How to use it |
| --- | --- |
| `Project Name` | Output `.psht` filename. It defaults to the scene folder name. |
| `Profile` | Start with `Splat3` for normal scenes. |
| `kSteps` | `Auto` keeps Postshot's automatic step count. Turn it off only to force a value. |
| `Max Image Size` | Long-edge limit for images passed to Postshot. `0` means no limit. |
| `Camera Poses` | Use `Import` for existing poses, or `Estimate` when Postshot should estimate them. |
| `Import Masks` | Turn on when you want to pass the dataset `masks/` folder to Postshot. |

### Camera Poses

`Import` passes camera poses created by Step 4 into Postshot.

| Step 4 route | What Import passes |
| --- | --- |
| COLMAP | COLMAP sparse model |
| SphereSfM | SphereSfM sparse model |
| Metashape | `transforms.json` and the available Metashape point-cloud PLY |

If `Import` is selected and no camera poses are available, Step 5 stops before running. Either run SfM/conversion in Step 4 first, or switch to `Estimate` so Postshot estimates poses.

### Masks

This app's masks are white=used and black=excluded. Use `Exclude black / use white (background)` for normal masks. Use `Exclude white / use black (occluders)` only when white marks temporary occluders to ignore.

### Advanced Parameters

GPU, profile-specific model limits, anti-aliasing, sky model, training-context storage, Crop/ROI, and optional PLY/SPZ export are under `Postshot Advanced Parameters`. Start with defaults and change only the settings you are deliberately comparing.

## Custom

`Other... > Custom` runs any CLI. Select the executable, then build the command arguments with `Argument Template`.

Available placeholders:

| Placeholder | Value |
| --- | --- |
| `{dataset}` | Dataset folder |
| `{images}` | Image folder inside the dataset |
| `{masks}` | Mask folder inside the dataset, or an empty string if missing |
| `{sparse}` | Detected COLMAP/SphereSfM sparse model, or an empty string if missing |
| `{output}` | Training output folder |

Example:

```text
--data {dataset} --out {output}
```

## When To Return To Step 4

Step 5 does not create datasets. In these cases, go back to Step 4 and run conversion first.

| Situation | What to do |
| --- | --- |
| `Dataset` has no `images/` or `transforms.json` | Turn on `Cube` in Step 4 and run conversion. |
| SphereSfM was run with `SfM` on and `Cube` off | Use `SfM` off / `Cube` on in Step 4 to convert from the existing sparse model. |
| LichtFeld `GUT` is on but `pointcloud.ply` is missing | Create 3DGUT data in Step 4. |
| Postshot `Camera Poses: Import` has no poses | Run SfM/conversion in Step 4, or switch to `Estimate`. |

## Outputs

| Training app | Main outputs |
| --- | --- |
| LichtFeld Studio | Final PLY in `Training Output`. PPISP can add related files. |
| Postshot | `.psht` project in `Training Output`. Optional PLY/SPZ export is also available. |
| Custom | Whatever the specified CLI writes. |

Step 5 launches external apps, so final quality depends on the Step 4 dataset shape, training-app settings, step count, and mask usage. When comparing settings with the same `output/` dataset, change the output name so each result remains available.

## Common Decisions

- For a first run after Step 4, use LichtFeld with `GUT` off, or Postshot with `Camera Poses: Import`.
- To test LichtFeld 3DGUT, create `3DGUT (LichtFeld)` data in Step 4, then turn on `GUT` in Step 5.
- Use `Camera Poses: Estimate` only when you want Postshot to estimate poses.
- COLMAP route data is projected cubemap data. Use the Metashape or SphereSfM route for 3DGUT comparisons.
- To keep existing results, change the LichtFeld output PLY name, Postshot project name, or `Training Output`.
