# Scene Import

`Import Scene` registers an existing folder's `images/`, `masks/`, and `output/` as a scene managed by this app. It records the assets as external imports, not as video extraction or conversion runs created by the app.

## Behavior

1. Press `Import Scene` in the header.
2. Select the folder to import.
3. The app rescans the current folder contents and rebuilds external-import metadata.

No confirmation dialog is shown. Image files, mask files, and `output/` assets are not deleted.

## Reimport

Running `Import Scene` again for the same folder is a full re-registration, not an append. External-import metadata from the previous import is replaced with the current folder contents.

Main replaced metadata:

- `_stechdrive/frames/selected_frames.csv`
- External-import records in `_stechdrive/sources/image_sets.json`
- External-import mask records under `_stechdrive/masks/`
- `_stechdrive/step4/export_settings.json`
- External-import records in `_stechdrive/step4/dataset_runs.json`

The previous management files are backed up under `_stechdrive/imports/backups/`.

## Validation

Import checks mask pairing, image sizes, `output/transforms.json` references, 4x4 transform matrices, and `pointcloud.ply` presence when relevant. Warnings do not block import; they are written to `_stechdrive/imports/scene_imports.json` and the bottom log panel.

To keep large cubemap datasets responsive, image-size validation under `output/images/` uses sampled validation by default. `transforms.json` image references are checked against the image paths already found during folder scanning.

## Cancellation

Scene import can be canceled while it is running. If cancellation happens during scan, validation, or metadata build, the import stops without replacing management metadata.

After the final apply step starts, the app prioritizes metadata consistency and finishes the apply step. In that case, avoiding a partially written `_stechdrive` state takes precedence over canceling immediately.
