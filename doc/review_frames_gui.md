# Step 2 Frame Review GUI

Step 2 is where you review images extracted or registered in Step 1 and keep only the images that should continue to Step 3 mask generation and downstream SfM. It shows labels such as `Added`, `Drop`, and `Review` from Step 1 analysis, plus `External` for images registered through scene import, and lets you manually change keep/drop decisions.

When you press `Apply`, Step 2 removes dropped frames from `images/`. Kept filenames are preserved by default; before masks or Step 5 dataset outputs exist, you can also enable `Renumber kept images` to rename the kept files into a clean sequence. The resulting `images/` folder becomes the input for Step 3 and for Metashape, COLMAP, SphereSfM, or another SfM route.

## First Things To Check

| Goal | Where to look |
| --- | --- |
| Scan the extracted set quickly | `Thumbnail List` |
| Inspect blur or fine detail | `Single Preview` |
| View a 360° image like a normal camera | `90° Perspective Preview` |
| Review only planned drops and quality warnings | Thumbnail filter / `Next Review Target` / `Previous Review Target` |
| Usable-looking images are marked as blur | `Blur Detection`: `Standard` / `Low sensitivity` |
| Change whether a frame is kept or dropped | Flag keep/drop button |
| Rename kept images to a clean sequence before masks/Step 5 dataset output | `Renumber kept images` |
| Apply decisions to the actual `images/` folder | `Apply` |

The efficient workflow is to scan the thumbnail list first, then use the thumbnail filter for `Drops Only`, `Review Only`, or `Drops + Review`. Turn on `Include added frames` when you also want the review-target buttons to audit frames added for viewpoint change, blur replacement, or spacing.

## Basic Flow

1. Open Step 2 after Step 1 finishes.
2. Use the thumbnail list to check overall density and obvious blur.
3. Use the thumbnail filter or `Next Review Target` to move through planned drops and quality-warning frames.
4. If images that look usable are marked as blur, set `Blur Detection` to `Low sensitivity`. You can switch back to `Standard` while reviewing.
5. Use single preview for detailed checks. For 360° images, switch to the 90° perspective preview when detail is easier to judge that way.
6. Mark useful frames as Keep and unwanted frames as Drop.
7. If needed, enable `Renumber kept images` before masks or Step 5 dataset outputs exist.
8. Press `Apply` when the decisions are ready.
9. Continue to Step 3.

Keep/drop decisions are kept as you work. Dropped image files are not removed from `images/` until you press `Apply`.

## Review Views

### Thumbnail List

Use this to scan many frames, check capture density, and find repeated or blurry runs. Selection follows Windows Explorer behavior: click, Shift-click, and Ctrl-click.

When multiple thumbnails are selected, keep/drop and reset actions apply to the selection. If nothing is selected, they apply to the current frame.

The header thumbnail filter can show only dropped frames, only review warnings, or both. Keep/drop changes still write to the same CSV while filtered; choose `All Frames` to return to the full list.

### Single Preview

Use this for detail checks such as blur, unwanted people, stitch seams, and thin visual features. `Left` / `Right` moves through frames.

### 90° Perspective Preview

For 360° equirectangular images, this switches the preview to a square 90° FOV perspective view. Drag to look around and use the mouse wheel for 2D zoom on the displayed image.

For normal images or normal video frames, this button is disabled. If you move from a 360° image in perspective preview to a normal image, Step 2 automatically returns to single preview.

## Advisory Labels

Step 2 labels are not a separate quality score. They explain why a frame is kept, dropped, added, or marked for review.

| Label | How to read it |
| --- | --- |
| `OK: kept` | Normal kept frame |
| `Quick: extracted at the specified interval` | From quick extraction; no analysis label |
| `Added: viewpoint change` | Added before the fixed cadence because viewpoint change was useful |
| `Added: blur replacement` | Added as a more usable nearby candidate for a frame that may have been blurred |
| `Added: preserved spacing` | Kept as a safety frame so the gap does not become too large |
| `Review: possible blur` | Kept, but blur may be present and deserves manual inspection |
| `Review: low texture` | Kept, but SfM features may be weak |
| `Review: weak feature tracking` | Kept, but pair tracking may be weak |
| `Drop: similar frame` | Planned drop because it is too similar to the previous kept frame |
| `Drop: blur` | Planned drop because blur may weaken SfM |
| `Drop: manually excluded` | Manually switched to Drop |
| `External: imported image` | Image registered through scene import or external image add |

Thumbnail view shows a short label and category color in the bottom ribbon of each image. Dropped, review, added, quick-extract, external-import, and normal kept images can be distinguished by color.

## Blur Detection

`Standard` marks frames that may be visually blurred as review targets. Borderline frames stay kept, so you can decide whether they are still useful for SfM.

Use `Low sensitivity` when outdoor footage, white walls, or low-detail regions create too many blur warnings even though the images look usable. It still catches obvious blur, but reduces warnings caused by low texture or exposure changes.

`Drop: blur` means the app plans to remove the frame automatically. `Review: possible blur` means the frame is still kept and needs a human decision. Keep review frames when they still help SfM coverage. Switch them to Drop only when the visible blur is strong or a nearby frame is clearly better.

The setting can be switched while reviewing. Manual keep/drop changes you already made are preserved.

## Keep/Drop Decisions

When unsure, judge whether the frame will help the later SfM step estimate camera positions.

- Drop frames with strong blur or weak visible features.
- Drop redundant runs where many frames show almost the same camera position.
- Keep frames with useful viewpoint change, especially around nearby objects.
- For 360° images, use perspective preview to inspect floors, walls, columns, text, and thin lines.
- It is acceptable to keep borderline frames. You can return later if masks or SfM results show a problem.

## Apply

`Apply` writes the current keep/drop decisions into the image folder.

Before applying changes, the GUI creates a review backup so removed images can be restored manually if needed.

Apply does the following:

- removes dropped images from `images/`
- keeps accepted image filenames unchanged by default
- when `Renumber kept images` is on, renames accepted images into a sequence starting at `frame_000001`
- keeps the original image format when renumbering
- refreshes Step 2 after success

### Renumber Kept Images

Use this only as a cleanup step before Step 3 mask generation and before Step 5 dataset export. It renames kept files in the current order and preserves the image format.

This option is unavailable once masks or Step 5 conversion outputs already exist. That prevents existing masks or converted datasets from breaking because they still refer to the old image names.

Removed images can be restored manually from the review backup if needed.

## Common Decisions

- If there are no changes to write into the image folder, no extra action is needed. You can continue to Step 3.
- If many frames are `Drop: similar frame`, you do not need to inspect every one when the remaining coverage is sufficient.
- Always inspect `Review: possible blur`. If they look acceptable, keep them.
- To rerun extraction, return to Step 1 and use `Reset and Overwrite`.

## Shortcuts

- `Left` / `Right`: previous / next frame in single-preview mode
- `F` / `Shift+F`: next / previous issue target. Turn on `Include added frames` to include added-frame labels too.
- `Space`: toggle keep/drop
- `0`: reset preview zoom
