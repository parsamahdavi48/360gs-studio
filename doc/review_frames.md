# review_frames.py — Visual review of extracted frames

## Overview

`review_frames.py` opens a PySide6 GUI to inspect extracted frames and edit keep/drop decisions in `selected_frames.csv`.

Requirements:

- `PySide6`

## Usage

```bash
python review_frames.py ./scene01
```

If CSV filename is different:

```bash
python review_frames.py ./scene01 --csv selected_frames.csv
```

## Keyboard shortcuts

- `Left` / `Right`: previous / next frame in single-preview mode. In thumbnail-list mode, the thumbnail list keeps focus and arrow keys move the visible thumbnail selection.
- `F` / `Shift+F`: next / previous problem frame (`status != ok`)
- `Space`: toggle `keep` / `drop`
- `Q`: quit

## Behavior

- Reads `output_file` from each CSV row and displays the image.
- Supports single-image preview and thumbnail-list preview using the header icons.
- In single-image preview, the projection button switches 360° equirectangular images to a square 90° FOV perspective view. Drag the preview to look around; the mouse wheel still performs 2D zoom on the displayed image.
- Thumbnail mode shows keep/drop state with green/red framing.
- Thumbnail selection follows Windows Explorer-style multi-selection (`Ctrl` click, `Ctrl+Shift` click, and `Shift` click).
- Switching to thumbnail mode focuses the thumbnail list, so arrow-key navigation changes the selected thumbnail instead of silently changing the hidden single-preview frame.
- In thumbnail mode, the keep/drop flag button applies to the selected thumbnails; the reset button restores selected frames to the state loaded from the CSV.
- Double-clicking a thumbnail switches back to single-image preview on that frame.
- Supports problem-frame navigation.
- Saves the updated `decision` column back to the same CSV immediately when decisions change.
- Other columns (scores, status, indices) are preserved.
