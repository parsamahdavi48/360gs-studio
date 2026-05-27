# Step 6 Training GUI

Step 6 launches training apps that provide a compatible CLI with the 3DGS dataset created in Step 5. With a compatible [LichtFeld Studio](https://lichtfeld.io/) or [Postshot](https://www.jawset.com/) version, you can start repeat runs or headless training from the GUI.

When you want to inspect quality and tune model settings inside the training app, you can skip Step 6 and open the Step 5 output dataset directly in [LichtFeld Studio](https://lichtfeld.io/), [Postshot](https://www.jawset.com/), [Brush](https://github.com/ArthurBrussee/brush), or another 3DGS app. Step 6 does not run image conversion or SfM; dataset creation is handled by `Step 5: Dataset`.

## Using the Dataset in Training Apps

The Step 5 output is the dataset intended for downstream 3DGS apps. Open the matching dataset folder directly in the training app when you want to inspect results and tune settings in that app's GUI. Use Step 6 when the settings are ready for repeat runs or when you want training to run headlessly through a CLI.

| Path | Best for |
| --- | --- |
| Open the dataset in the training app | First checks, visual tuning, app-specific training settings |
| Launch from Step 6 | Repeatable CLI runs and headless training |

Step 6 CLI launch targets a LichtFeld Studio v0.5.2-compatible CLI and the Postshot v1.0/v1.1 Release Build CLI. If you are not using CLI training, the Step 5 output dataset remains ready to open directly in each training app.

## First Choice

When you open Step 6, first decide which app you want to run and which dataset you want to test.

| Goal | Training app | Main settings to check |
| --- | --- | --- |
| Train normal cubemap data in LichtFeld | `LichtFeld Studio` | `Dataset`, `GUT` off, output PLY name, `Strategy`, `Iterations` |
| Try ERP 360° / GUT data in LichtFeld | `LichtFeld Studio` | `Dataset`, `GUT` on, confirm `pointcloud.ply` exists |
| Create a Postshot project | `Postshot` | `Dataset`, `Camera Poses`, project name, `Profile` |

Normally, leave `Dataset` on the automatic value. The app uses the latest registered dataset artifact when one exists, such as a Metashape, RealityScan, SphereSfM, or COLMAP dataset under `<scene>/output/`. `Training Output` defaults to `<scene>/output/`. Keeping datasets and training results under `output/` makes the scene easier to move later.

## Basic Flow

1. Create the dataset in Step 5.
2. If you want to tune settings in the training app GUI, open the Step 5 output dataset directly there.
3. If you want repeatable CLI launch or headless training, open `Step 6: Training`.
4. Confirm that `Dataset` and `Training Output` point to the intended folders.
5. Choose `LichtFeld Studio` or `Postshot`.
6. If the executable cannot be found automatically, select the installed exe.
7. Review the app-specific settings on the right.
8. Press `Launch`.

Before running, Step 6 checks that the selected training mode matches the dataset shape. For example, LichtFeld `GUT` expects ERP 360° GUT data, while normal LichtFeld and Postshot expect PINHOLE cubemap data.

## Layout

Step 6 uses a wider center panel and is arranged as two columns.

| Area | Contents |
| --- | --- |
| Left | Training app, Headless, executable, dataset, training output |
| Right | LichtFeld / Postshot settings |

The right side separates common settings from advanced parameters. Start with the visible settings, run once, then open the advanced sections only when you have a reason to compare a specific option.

## Common Settings

### Training App

`LichtFeld Studio` and `Postshot` are available. See [LichtFeld Studio](https://lichtfeld.io/) and [Postshot](https://www.jawset.com/) for the supported apps. If you want to build an arbitrary CLI command, run that CLI directly instead of using this screen.

### Executable

When this field is empty, the GUI tries the default executable name or a known install location. If it is not found, select the executable yourself.

| App | Example |
| --- | --- |
| LichtFeld Studio | `LichtFeld-Studio.exe` |
| Postshot | `postshot-cli.exe` |

### Dataset

This is the dataset folder passed to the training app. It is normally set from the latest registered Step 5 dataset, or from the current Step 5 route and output shape when no artifact record exists yet.

| Step 5 result | Step 6 dataset |
| --- | --- |
| Metashape + PINHOLE cubemap | `<scene>/output/metashape_cubemap/` |
| Metashape + ERP 360° / GUT | `<scene>/output/metashape_3dgut/` |
| SphereSfM + PINHOLE cubemap | `<scene>/output/spheresfm_cubemap/` |
| SphereSfM + ERP 360° / GUT | `<scene>/output/spheresfm_3dgut/` |
| COLMAP Rig | `<scene>/output/colmap_rig/` |
| RealityScan -> COLMAP Dataset | `<scene>/output/realityscan/lfs_colmap/` |
| Metashape -> COLMAP Dataset | `<scene>/output/metashape_colmap/` |

You can choose another folder manually. If you do, make sure it contains the files required by the selected training app: `images/`, optional `masks/`, camera poses, and point cloud data when needed.

### Training Output

This is where training results or project files are written. The default is `<scene>/output/`.

If the final LichtFeld PLY, Postshot `.psht`, or optional Postshot PLY/SPZ export already exists, Step 6 stops before running so the result is not overwritten. Change the output name or output folder, then run again.

## LichtFeld Studio

For LichtFeld Studio, Step 6 starts training with the selected dataset, output folder, and training settings.

For CLI launch from Step 6, use a LichtFeld Studio v0.5.2-compatible CLI as the baseline. If you want to review training settings inside LichtFeld Studio, open the cubemap or ERP 360° / GUT dataset created in Step 5 directly in LichtFeld Studio.

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
| `GUT` | Turn this on only for GUT datasets that use ERP 360° images directly. |

### Training Cubemap Data

This is data created in Step 5 with image type `PINHOLE`. `Dataset` is normally `<scene>/output/metashape_cubemap/` for the Metashape route or `<scene>/output/spheresfm_cubemap/` for the SphereSfM route.

`Metashape -> COLMAP Dataset` and `RealityScan -> COLMAP Dataset` are also PINHOLE-style dataset routes for LichtFeld. For the RealityScan route, use `<scene>/output/realityscan/lfs_colmap/` with `GUT` off.

- Keep `GUT` off.
- Usually keep `Undistort` off too.
- If using masks, choose the LichtFeld mask mode that matches the result you want.

### Training ERP 360° / GUT Data

This is data created in Step 5 with image type `ERP 360°`. `Dataset` is normally `<scene>/output/metashape_3dgut/` for the Metashape route or `<scene>/output/spheresfm_3dgut/` for the SphereSfM route.

- Turn `GUT` on.
- `pointcloud.ply` inside the selected dataset is required.
- Create the ERP 360° / GUT dataset from the Metashape or SphereSfM route in Step 5 before launching Step 6.

### Steps Scaler

With `Steps Scaler` set to `Auto`, Step 6 counts images in `Dataset/images/` and uses the same 300-image baseline adjustment that LichtFeld Studio applies when loading a dataset in its GUI. Use a fixed value only when you intentionally want to compare different schedule scaling.

### Advanced Parameters

Less common Dataset, Optimizer, Refinement, Loss, Initialization, MRNF/IGS+, Sparsity, and Save/Eval options are grouped under `Advanced Training Parameters`. Only the settings relevant to the selected strategy or top-level checkbox are shown.

## Postshot

For Postshot, Step 6 creates a `.psht` project from the selected images and camera poses.

For CLI launch from Step 6, use a Postshot v1.0/v1.1 Release Build CLI as the baseline. If you want to review settings inside Postshot, open the Step 5 images, camera poses, and optional masks directly in Postshot.

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

`Import` passes camera poses created by Step 5 into Postshot.

| Step 5 route | What Import passes |
| --- | --- |
| COLMAP | COLMAP sparse model |
| SphereSfM | SphereSfM sparse model |
| Metashape | The selected profile's transforms JSON and point-cloud PLY, such as `transforms_postshot.json` and `pointcloud_postshot.ply` |

If `Import` is selected and no camera poses are available, Step 6 stops before running. Either run SfM/conversion in Step 5 first, or switch to `Estimate` so Postshot estimates poses.

### Masks

This app's masks are white=used and black=excluded. Use `Exclude black / use white (background)` for normal masks. Use `Exclude white / use black (occluders)` only when white marks temporary occluders to ignore.

### Advanced Parameters

GPU, profile-specific model limits, anti-aliasing, sky model, training-context storage, Crop/ROI, and optional PLY/SPZ export are under `Postshot Advanced Parameters`. Start with defaults and change only the settings you are deliberately comparing.

Postshot v1.1.0 adds Photometric Compensation in the Postshot GUI for exposure, white balance, and vignetting variation across images. As of Postshot v1.1.0, the `postshot-cli.exe train --help` output does not expose a matching CLI option, so enable that setting inside Postshot when you need it.

## When To Return To Step 5

Step 6 does not create datasets. Whether you open the dataset manually or launch through CLI, go back to Step 5 and run conversion first in these cases.

| Situation | What to do |
| --- | --- |
| `Dataset` has no images or camera data | Choose the matching card in Step 5 and create a dataset for the training app. |
| You only have a SphereSfM SfM result, not a training dataset | Choose `SphereSfM -> NeRF Dataset (JSON/PLY)` in Step 5 and export PINHOLE or ERP 360° data. |
| LichtFeld `GUT` is on but `pointcloud.ply` is missing | Create ERP 360° / GUT data in Step 5. |
| Postshot `Camera Poses: Import` has no poses | Run SfM/conversion in Step 5, or switch to `Estimate`. |

## Outputs

| Training app | Main outputs |
| --- | --- |
| LichtFeld Studio | Final PLY in `Training Output`. PPISP can add related files. |
| Postshot | `.psht` project in `Training Output`. Optional PLY/SPZ export is also available. |

Final quality depends on the Step 5 dataset shape, training-app settings, step count, and mask usage. When comparing settings with the same dataset, change the output name so each result remains available.

## Common Decisions

- For the first check after Step 5, open the dataset directly in the training app when you want to inspect quality and settings.
- For CLI runs, start with LichtFeld `GUT` off, or Postshot with `Camera Poses: Import`.
- To test LichtFeld GUT, choose image type `ERP 360°` in Step 5, then turn on `GUT` in Step 6.
- Use `Camera Poses: Estimate` only when you want Postshot to estimate poses.
- COLMAP route data is PINHOLE cubemap data. Use the Metashape or SphereSfM route for ERP 360° / GUT comparisons.
- To keep existing results, change the LichtFeld output PLY name, Postshot project name, or `Training Output`.
