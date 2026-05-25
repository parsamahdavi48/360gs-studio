# Step 4 SfM / Step 5 Dataset GUI

Step 4 is where you choose how camera poses and sparse points will be prepared for the training dataset. If you already ran SfM in Metashape, RealityScan, COLMAP, or SphereSfM, there is usually nothing else to run in this step. Open a route card only when you want this app to run COLMAP/SphereSfM or when you want to create RealityScan realignment data from a Metashape result.

Step 5 converts SfM results into datasets that training apps can read. Use it to create NeRF-style JSON/PLY datasets from Metashape or SphereSfM, COLMAP-format datasets from Metashape or RealityScan, or to apply AprilTag scale to an existing dataset.

## First Decision

Start by deciding whether camera poses already exist.

| Situation | What to do |
| --- | --- |
| Metashape SfM is already done | In Step 4, choose `Use Existing SfM Result`. In Step 5, choose a Metashape dataset card |
| RealityScan realignment is already done | In Step 4, choose `Use Existing SfM Result`. In Step 5, choose `RealityScan -> COLMAP Dataset` |
| You already have a COLMAP images/masks/sparse dataset | In Step 4, choose `Use Existing SfM Result`. For COLMAP-compatible training apps, continue directly to training |
| You want this app to run COLMAP | In Step 4, choose `Run COLMAP SfM` |
| You want this app to run SphereSfM | In Step 4, choose `Run SphereSfM` |
| You want to realign a Metashape result in RealityScan | In Step 4, choose `Metashape -> RealityScan Data` |
| You want to inspect an existing result | In Step 4, choose `Inspect SfM Result` |

## Step 4: SfM Route Cards

### Use Existing SfM Result

Choose this when Metashape, RealityScan, COLMAP, SphereSfM, or another tool has already produced camera poses and sparse points. This route means "nothing to run here"; continue to Step 5 and choose the dataset format you need.

For Metashape results, export cameras as Agisoft XML and sparse points as Stanford PLY. Saving both files in the scene folder is recommended because it keeps the SfM result together with the scene; if they are elsewhere, select the XML and PLY manually in the Step 5 card.

### Run COLMAP SfM

Choose this when you want this app to run COLMAP or GLOMAP without using Metashape.

- 360° images are expanded into cubemap rigs
- normal images and normal video frames remain normal cameras
- mixed sources can be processed in one COLMAP dataset
- output goes to `output/colmap_rig/`

For video-like input, start with `Sequential` matching. For a smaller unordered photo set, consider `Exhaustive`.

For normal images, start with automatic camera estimation. Only edit normal-camera settings when you need to explicitly control focal length or distortion model.

### Run SphereSfM

Choose this when you want to run SfM on equirectangular 360° images as spherical cameras. Treat SphereSfM as a route for same-resolution ERP 360° images only. Use COLMAP or Metashape when you need mixed normal images or multiple ERP resolutions.

SphereSfM requires SphereSfM's own `colmap.exe`, not standard COLMAP. On RTX 50-series GPUs, some distributed binaries can fail during CUDA SIFT; in that case, select a build made for the RTX 50-series CUDA architecture.

The SfM working folder is `output/spheresfm/`. Create JSON/PLY or cubemap datasets from that result in Step 5 with `SphereSfM -> NeRF Dataset (JSON/PLY)`.

### Metashape -> RealityScan Data

Choose this when you have a Metashape camera XML and want to import cubemap images plus XMP camera data into RealityScan. This is useful when you want RealityScan to realign the scene, include extra image sources already registered in Steps 1-3, or export RealityScan CSV/PLY for downstream tools.

The output is `output/realityscan/`. Images present in the Metashape XML are written to `images/` as cubemap images with XMP sidecars. Registered scene images missing from the XML are copied or hard-linked to `extra_images/` as unposed extra inputs. When a matching mask exists, it is also written as a RealityScan layer such as `image.jpg.mask.png`. In RealityScan, add `images/` first and run Align until sparse points are generated, then add `extra_images/` and run Align again. After exporting CSV and PLY from RealityScan, use Step 5 `RealityScan -> COLMAP Dataset` when LichtFeld needs a COLMAP-format Dataset.

The staged RealityScan import is intentional. Aligning the cubemap images first creates a stable component from the Metashape result; adding normal images after that usually avoids more wrong placements and small disconnected components than importing all images at once.

### Inspect SfM Result

Open a read-only viewer for point clouds, camera positions, selected camera images, and matching masks. Use it to confirm that camera poses, point clouds, image links, and masks line up before training.

## Step 5: Dataset Cards

### Metashape -> NeRF Dataset (JSON/PLY)

Create a NeRF/3DGS-style dataset using `transforms.json` and `pointcloud.ply` from a Metashape camera XML and point-cloud PLY.

| Choice | Use when |
| --- | --- |
| `PINHOLE` | You want cubemap-style perspective images for Postshot, Brush, LichtFeld, or similar tools |
| `ERP 360°` | You want LichtFeld GUT to train directly on equirectangular images |

Start with `PINHOLE` unless you specifically need LichtFeld GUT. If the Metashape result contains normal images or multiple ERP resolutions, direct ERP 360° output is not safe; use `PINHOLE`. For LichtFeld, mixed Metashape results with multiple camera settings are safer through `Metashape -> COLMAP Dataset` because LichtFeld's JSON/PLY import does not handle per-frame camera intrinsics.

Typical outputs:

| Output | Folder |
| --- | --- |
| PINHOLE cubemap | `output/metashape_cubemap/` |
| ERP 360° / GUT | `output/metashape_3dgut/` |

### Metashape -> COLMAP Dataset

Create a COLMAP-format dataset with `images/`, `masks/`, and `sparse/0/` from Metashape camera XML and point-cloud PLY. Choose this for COLMAP-input training apps or when Metashape aligned mixed 360° and normal-camera sources.

- ERP 360° cameras are expanded to the selected view set
- PINHOLE normal images are referenced without cubemap conversion
- distorted normal images are undistorted to PINHOLE, and matching masks are transformed the same way
- output goes to `output/metashape_colmap/`

For LichtFeld with mixed Metashape sources, this is the safer route.

### RealityScan -> COLMAP Dataset

Create a LichtFeld-readable COLMAP dataset from RealityScan Internal/External CSV and a PLY exported in the same coordinate state.

Normally, use it when CSV, PLY, `images/`, and `masks/` are already under `output/realityscan/`. If `extra_images/` and `extra_masks/` exist, additional images listed in the CSV are also gathered into the output `images/` and `masks/` folders. Output goes to `output/realityscan/lfs_colmap/`.

Before exporting from RealityScan, confirm the component you want to train from. The CSV should contain the cameras you expect; images that are not in the exported camera CSV are kept out of the COLMAP poses even if their files exist.

Turn on `Undistort to PINHOLE` only when RealityScan includes normal-camera images with distortion and LichtFeld refuses to train on them. Cubemap-derived PINHOLE images are linked or copied into the output, while only distorted normal images are converted. Invalid image regions introduced by undistortion are also reflected in the masks.

### SphereSfM -> NeRF Dataset (JSON/PLY)

Create a JSON/PLY dataset from the SphereSfM sparse model created in Step 4, or from another selected SphereSfM sparse model.

Use same-resolution ERP 360° input for SphereSfM. The output can be ERP 360° data for LichtFeld GUT or PINHOLE cubemap data for Postshot, Brush, or LichtFeld.

### Scale Adjustment

Use printed AprilTags to correct the metric scale of an existing dataset. Tags must be printed and fixed in the scene before capture.

1. Create a cubemap or COLMAP-style dataset in Step 5.
2. Open `Scale Adjustment`, then confirm the target dataset, tag family, tag IDs, and printed tag size.
3. Run estimation and review the detected observations and scale value.
4. Apply the scale only when the result looks reasonable.

Scale application multiplies the target dataset's camera positions and point cloud by the same factor. A backup is created before files are changed.

## Output Choices

### Image Type

`PINHOLE` expands ERP 360° images into normal perspective views. Use it first for normal Postshot, Brush, and LichtFeld training.

`ERP 360°` keeps equirectangular images for LichtFeld GUT. It is available only for the LichtFeld preset.

### View Set

`Cubemap` exports the standard six front/back/left/right/up/down views. Use it when unsure.

`Custom Grid` adds or removes view directions. More views mean more output images, longer conversion time, and more training images.

### Image Size

Start with the default. Use a smaller size for fast tests and a larger size when you need final-detail output. Also consider source resolution and training-app VRAM use.

## Scene Preview

Open `Inspect SfM Result` from Step 4 to inspect point clouds, camera positions, images, and masks together.

The viewer can load output datasets, Metashape XML/PLY, COLMAP sparse models, and SphereSfM sparse models found in the scene. Selecting a camera shows its image and matching mask.

## Common Decisions

- If Metashape SfM is already done, choose `Use Existing SfM Result` in Step 4.
- For normal Postshot / Brush / LichtFeld training, start with `PINHOLE` + `Cubemap`.
- For LichtFeld GUT, choose `ERP 360°` and enable GUT in the training app.
- If a Metashape result mixes normal images or multiple ERP resolutions, use `Metashape -> COLMAP Dataset` for LichtFeld.
- If RealityScan output is going to Postshot, CSV/PLY may be enough. If LichtFeld needs a Dataset folder, use `RealityScan -> COLMAP Dataset`.
- For the Metashape -> RealityScan -> LichtFeld route, keep Metashape XML/PLY and RealityScan CSV/PLY with the scene, then train from `output/realityscan/lfs_colmap/`.
- Treat SphereSfM as same-resolution ERP 360° only. Use COLMAP or Metashape for mixed sources.
- To rebuild only images or masks, reopen the same dataset card, check the output settings, and run it again.

## Continue to Step 6

After Step 5 creates a dataset, load it directly in LichtFeld Studio, Postshot, Brush, or another training app. Use Step 6 only when you want to launch a compatible CLI for repeat runs or headless training.

See [Step 6 Training GUI](training_gui.md) for Step 6 details.
