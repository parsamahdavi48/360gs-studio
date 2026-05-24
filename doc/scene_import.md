# Scene Import

`Import Scene` lets the GUI continue working with a folder that already contains images, masks, or output datasets.

It is not meant to re-run extraction or conversion. Use it when you want existing scene assets to appear again in the review, mask, SfM, dataset, and training screens.

## When To Use It

| Situation | How to use it |
| --- | --- |
| You moved a scene from another PC or drive | Import the folder and continue work in this GUI. |
| The folder already has `images/` or `masks/` | Register the images and masks as the current scene assets. |
| The folder already has datasets under `output/` | Register them as candidates for Step 5 and Step 6. |
| You manually added images or masks | Import again so the GUI reflects the current folder contents. |

For normal new work, add sources in Step 1. Use scene import only when you are resuming or registering an existing folder.

## Before Importing

- Put source images under the scene `images/` folder.
- Put masks under `masks/` when available. White means used, black means excluded.
- Keep Step 5 datasets under `output/` when you want the GUI to find them.
- If you renamed images, make sure masks and dataset references still match.

Import does not delete asset files. It scans the current folder and rebuilds the GUI's registration for that scene.

## Steps

1. Open the three-line menu next to `Scene Folder`.
2. Choose `Import Scene...`.
3. Select the scene folder.
4. After import, check the registered assets in Step 2, Step 3, Step 4, Step 5, and Step 6.

`Open Scene Folder...` only opens the folder in Explorer. Use `Import Scene...` when you want existing assets to be registered in the GUI.

## What To Check After Import

| Screen | Check |
| --- | --- |
| Step 2 Review | Images from `images/` appear as review items |
| Step 3 Mask | Existing masks are detected, and you can decide whether to generate more |
| Step 4 SfM | Choose whether to use an existing SfM result or run SfM from this app |
| Step 5 Dataset | Use an existing dataset under `output/` or create a new one |
| Step 6 Training | The dataset selected for the training app is the one you expect |

If warnings appear, first check image-mask pairing, image references inside datasets, and whether the required point cloud file exists. A warning does not always mean the scene is unusable; it marks something to review before the next step.

## Reimport

Importing the same folder again updates the GUI registration to match the current folder contents.

| Goal | Reimport? |
| --- | --- |
| You manually added images or masks | Yes, reimport to refresh the registration. |
| You replaced a Step 5 output dataset by hand | Yes, reimport to refresh dataset candidates. |
| You only want to change a GUI selection | No. Change the selection in the relevant step. |

Reimporting does not delete files in `images/`, `masks/`, or `output/`.

## Canceling

Scene import can be canceled while it is running. Large image sets or large datasets can take time to scan.

When canceled, the partial registration is discarded and the previous GUI state is kept.
