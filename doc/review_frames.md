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

- `Left` / `Right`: previous / next frame
- `F` / `Shift+F`: next / previous problem frame (`status != ok`)
- `Space`: toggle `keep` / `drop`
- `S`: save CSV
- `Q`: quit

## Behavior

- Reads `output_file` from each CSV row and displays the image.
- Supports direct jump by sequence number (`Jump Seq`) and problem-frame navigation.
- Saves the updated `decision` column back to the same CSV.
- Other columns (scores, status, indices) are preserved.
