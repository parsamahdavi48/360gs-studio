# Step 2 Frame Review GUI

Step 2 is where you review the frames extracted in Step 1 and keep only the images that should continue to Step 3 mask generation and then Metashape or SphereSfM. It shows labels such as `Added`, `Drop`, and `Review` from Step 1 analysis, and lets you manually change keep/drop decisions.

When you press `Apply`, Step 2 removes dropped frames from `images/` and finalizes `_stechdrive/frames/selected_frames.csv` to keep-only rows. Kept filenames are preserved by default; before masks or Step 4 outputs exist, you can also enable `Renumber kept images` to rename the kept files in CSV order. The resulting `images/` folder becomes the input for Step 3 and for Metashape or SphereSfM SfM.

## First Things To Check

| Goal | Where to look |
| --- | --- |
| Scan the extracted set quickly | `Thumbnail List` |
| Inspect blur or fine detail | `Single Preview` |
| View a 360° image like a normal camera | `90° Perspective Preview` |
| Review only frames flagged by Step 1 | `Next Review Target` / `Previous Review Target` |
| Change whether a frame is kept or dropped | Flag keep/drop button |
| Rename kept images to a clean sequence before masks/Step 4 | `Renumber kept images` |
| Apply decisions to the actual `images/` folder | `Apply` |

The efficient workflow is to scan the thumbnail list first, then jump through only the review targets and dropped frames that need attention.

## Basic Flow

1. Open Step 2 after Step 1 finishes.
2. Use the thumbnail list to check overall density and obvious blur.
3. Use `Next Review Target` to move through frames flagged by Step 1.
4. Use single preview for detailed checks. For 360° images, switch to the 90° perspective preview when detail is easier to judge that way.
5. Mark useful frames as Keep and unwanted frames as Drop.
6. If needed, enable `Renumber kept images` before masks or Step 4 outputs exist.
7. Press `Apply` when the decisions are ready.
8. Continue to Step 3.

Keep/drop changes are written to `_stechdrive/frames/selected_frames.csv` immediately. Dropped image files are not removed from `images/` until you press `Apply`.

## Review Views

### Thumbnail List

Use this to scan many frames, check capture density, and find repeated or blurry runs. Selection follows Windows Explorer behavior: click, Shift-click, and Ctrl-click.

When multiple thumbnails are selected, keep/drop and reset actions apply to the selection. If nothing is selected, they apply to the current frame.

### Single Preview

Use this for detail checks such as blur, unwanted people, stitch seams, and thin visual features. `Left` / `Right` moves through frames.

### 90° Perspective Preview

For 360° equirectangular images, this switches the preview to a square 90° FOV perspective view. Drag to look around and use the mouse wheel for 2D zoom on the displayed image.

## Advisory Labels

Step 2 labels are not a separate quality score. They explain why a frame is kept, dropped, added, or marked for review.

| Label | How to read it |
| --- | --- |
| `OK: kept` | Normal kept frame |
| `Quick: extracted at the specified interval` | From quick extraction; no analysis label |
| `Added: viewpoint change` | Added before the fixed cadence because viewpoint change was useful |
| `Added: preserved spacing` | Kept as a safety frame so the gap does not become too large |
| `Review: possible blur` | Kept, but blur may be present |
| `Review: low texture` | Kept, but SfM features may be weak |
| `Review: weak feature tracking` | Kept, but pair tracking may be weak |
| `Drop: similar frame` | Planned drop because it is too similar to the previous kept frame |
| `Drop: possible blur` | Planned drop because blur may weaken SfM |
| `Drop: manually excluded` | Manually switched to Drop |

Red labels are dropped frames, yellow labels are kept frames that need review, blue labels are added frames, purple labels are quick-extract frames, and green labels are normal OK frames. Thumbnail view uses the same category colors in the bottom ribbon.

## Keep/Drop Decisions

When unsure, judge whether the frame will help the later SfM step estimate camera positions.

- Drop frames with strong blur or weak visible features.
- Drop redundant runs where many frames show almost the same camera position.
- Keep frames with useful viewpoint change, especially around nearby objects.
- For 360° images, use perspective preview to inspect floors, walls, columns, text, and thin lines.
- It is acceptable to keep borderline frames. You can return later if masks or SfM results show a problem.

## Apply

`Apply` writes the current keep/drop decisions into the image folder.

Before applying changes, the GUI saves a review backup under `_stechdrive/review/backups/<review_id>/`. This backup contains `selected_frames.before.csv`, `selected_frames.after.csv` after success, and copies of the dropped image files that existed at apply time. It also writes a CSV backup under `_stechdrive/frames/backups/`.

Apply does the following:

- deletes drop-marked images from `images/`
- preserves filenames for kept images by default
- optionally renumbers kept images to `images/frame_000001.ext`, `frame_000002.ext`, ... in CSV order
- rewrites `_stechdrive/frames/selected_frames.csv` to keep-only rows
- writes `_stechdrive/frames/backups/selected_frames.before_finalize.csv` and `_stechdrive/frames/selected_frames_keep.csv`
- records the GUI review run and review backup under `_stechdrive/review/`
- updates frame/source metadata paths when kept images are renumbered
- refreshes Step 2 after success

### Renumber Kept Images

Use this only as a cleanup step before Step 3 mask generation and before Step 4 export. It renames kept files in the current CSV order and preserves each file's original extension.

The option is disabled if downstream outputs already exist, including `masks/`, mask metadata, `output/`, or `_stechdrive/step4/`. Those assets may already refer to the old image filenames.

Deleted drop images can be restored manually from the review backup if needed. The CSV backup is always created.

## Common Decisions

- If there are no changes to write into the image folder, no extra action is needed. You can continue to Step 3.
- If many frames are `Drop: similar frame`, you do not need to inspect every one when the remaining coverage is sufficient.
- Always inspect `Review: possible blur`. If it looks acceptable, keep it.
- To rerun extraction, return to Step 1 and use `Reset and Overwrite`.

## Shortcuts

- `Left` / `Right`: previous / next frame in single-preview mode
- `F` / `Shift+F`: next / previous review target
- `Space`: toggle keep/drop
- `0`: reset preview zoom
