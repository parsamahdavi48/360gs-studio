# transforms_to_colmap.py — Convert transforms.json to COLMAP Text

[English] | [日本語](transforms_to_colmap.ja.md)

Converts the cubemap `transforms.json` produced by [cubemap_transforms_json.py](cubemap_transforms_json.md) into the standard **COLMAP text format** (`cameras.txt` + `images.txt` + `points3D.txt`). This unlocks downstream 3DGS implementations that natively consume COLMAP datasets:

- [PostShot](https://www.jawset.com/) (via COLMAP import)
- [Brush](https://github.com/ArthurBrussee/brush)
- [Inria gaussian-splatting (official)](https://github.com/graphdeco-inria/gaussian-splatting)
- [nerfstudio](https://docs.nerf.studio/)
- Any tool that ingests COLMAP text

## Pipeline placement

```
Metashape SfM
  → GUI Metashape preprocessing route
    → transforms.json (EQUIRECTANGULAR)
      → cubemap_transforms_json.py
        → transforms.json (SIMPLE_PINHOLE / cubemap views)
          → transforms_to_colmap.py   ← THIS SCRIPT
            → cameras.txt + images.txt + points3D.txt
```

The script does **not** perform any additional coordinate transform. The input `transforms.json` is expected to be in the convention chosen by the cubemap conversion profile (Postshot / Brush / LichtFeld). Choose the profile in `cubemap_transforms_json.py` to match your downstream tool.

## Usage

```bash
python transforms_to_colmap.py ./cubic
# Output: ./cubic/colmap/{cameras.txt, images.txt, points3D.txt}
```

With explicit output directory and PLY:

```bash
python transforms_to_colmap.py ./cubic ./cubic/colmap \
    --json transforms.json \
    --ply ./cubic/pointcloud.ply
```

## Options

| Option | Argument | Description |
|---|---|---|
| `input_dir` | path | Directory containing `transforms.json` (output of cubemap_transforms_json.py) |
| `output_dir` | path | Output directory (default=`<input_dir>/colmap`) |
| `--json` | filename | Input JSON filename (default=`transforms.json`) |
| `--ply` | path | Optional PLY for `points3D.txt` and a copied `points3D.ply` |
| `--image-prefix` | string | Prefix to strip from `file_path` entries when writing image names (default=`images/`) |

## Output

| File | Contents |
|---|---|
| `cameras.txt` | Single shared intrinsic (`SIMPLE_PINHOLE` if `fx==fy`, else `PINHOLE`). Camera ID = 1 |
| `images.txt` | One entry per cubemap face image with quaternion (qw, qx, qy, qz) and translation (tx, ty, tz) in COLMAP world-to-camera convention. POINTS2D is empty |
| `points3D.txt` | One entry per PLY point. Track info is empty (no SfM track linkage available from the converted JSON) |
| `points3D.ply` | A copy of the input PLY (preserved as-is for tools that read PLY directly) |

## Coordinate conventions

The script converts each frame's 4×4 camera-to-world matrix from `transforms.json` to COLMAP's world-to-camera convention:

```
R_w2c = R_c2w.T
t_w2c = -R_w2c @ t_c2w
```

Quaternion is normalized and written with `qw ≥ 0` (positive hemisphere) per COLMAP convention.

## PLY support

The script reads PLY via the following fallback chain:

1. **open3d** (best: full ASCII / binary support, color preservation)
2. **plyfile** (good: most PLY variants)
3. **Internal ASCII fallback** (basic: ASCII PLY only, no binary)

If neither `open3d` nor `plyfile` is installed and the PLY is binary, an error is raised with installation instructions. Install one of:

```bash
pip install open3d   # recommended
# or
pip install plyfile
```

## Example: full pipeline for PostShot

```bash
# 1. Prepare an equirectangular transforms.json with the GUI Metashape route.

# 2. Cubemap conversion (Postshot profile = default)
python cubemap_transforms_json.py . ./cubic

# 3. COLMAP text export
python transforms_to_colmap.py ./cubic

# Result: ./cubic/colmap/ ready for PostShot/Brush/gaussian-splatting
```
