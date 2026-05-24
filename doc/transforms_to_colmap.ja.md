# transforms_to_colmap.py — transforms.json を COLMAP テキストに変換

[English](transforms_to_colmap.md) | [日本語]

[cubemap_transforms_json.py](cubemap_transforms_json.ja.md) が出力する cubemap 用 `transforms.json` を、標準的な **COLMAP テキスト形式** (`cameras.txt` + `images.txt` + `points3D.txt`) に変換します。これにより COLMAP データセットを直接受け付ける 3DGS 実装に渡せます：

- [PostShot](https://www.jawset.com/)（COLMAP インポート経由）
- [Brush](https://github.com/ArthurBrussee/brush)
- [Inria gaussian-splatting (公式)](https://github.com/graphdeco-inria/gaussian-splatting)
- [nerfstudio](https://docs.nerf.studio/)
- COLMAP テキストを食う任意のツール

## パイプライン上の位置

```
Metashape SfM
  → metashape_360_lfs.py
    → transforms.json (EQUIRECTANGULAR)
      → cubemap_transforms_json.py
        → transforms.json (SIMPLE_PINHOLE / cubemap views)
          → transforms_to_colmap.py   ← 本スクリプト
            → cameras.txt + images.txt + points3D.txt
```

本スクリプトは**追加の座標変換は行いません**。入力 `transforms.json` は cubemap 変換時に選択した profile (Postshot / Brush / LichtFeld) の規約のままです。下流ツールに合わせて `cubemap_transforms_json.py` の profile を選択してください。

## 使い方

```bash
python transforms_to_colmap.py ./cubic
# 出力: ./cubic/colmap/{cameras.txt, images.txt, points3D.txt}
```

出力先と PLY を明示する場合：

```bash
python transforms_to_colmap.py ./cubic ./cubic/colmap \
    --json transforms.json \
    --ply ./cubic/pointcloud.ply
```

## オプション

| オプション | 引数 | 説明 |
|---|---|---|
| `input_dir` | パス | `transforms.json`（cubemap_transforms_json.py の出力）が置かれたディレクトリ |
| `output_dir` | パス | 出力ディレクトリ (default=`<input_dir>/colmap`) |
| `--json` | ファイル名 | 入力 JSON 名 (default=`transforms.json`) |
| `--ply` | パス | 任意の PLY。`points3D.txt` と `points3D.ply` の元データに使用 |
| `--image-prefix` | 文字列 | `file_path` から取り除く先頭プレフィックス (default=`images/`) |

## 出力

| ファイル | 内容 |
|---|---|
| `cameras.txt` | 共有イントリンシクス 1 個（`fx==fy` なら `SIMPLE_PINHOLE`、異なれば `PINHOLE`）。CAMERA_ID = 1 |
| `images.txt` | 各 cubemap 面画像のエントリ。COLMAP 規約の world-to-camera クォータニオン (qw, qx, qy, qz) と平行移動 (tx, ty, tz)。POINTS2D は空 |
| `points3D.txt` | PLY 各点のエントリ。track 情報は空（変換後 JSON からは SfM track 紐付けは取れないため） |
| `points3D.ply` | 入力 PLY のコピー（PLY を直接読むツール向け） |

## 座標規約

各フレームの 4×4 camera-to-world 行列を、COLMAP の world-to-camera 規約へ変換：

```
R_w2c = R_c2w.T
t_w2c = -R_w2c @ t_c2w
```

クォータニオンは正規化され、COLMAP 規約に従って `qw ≥ 0`（正半球）で書き出されます。

## PLY 対応

PLY は以下の優先順で読み込まれます：

1. **open3d**（最強：ASCII / バイナリ両対応、色情報も保持）
2. **plyfile**（良好：ほとんどの PLY バリアントに対応）
3. **内蔵 ASCII フォールバック**（最低限：ASCII PLY のみ、バイナリ非対応）

`open3d` も `plyfile` もインストールされておらず、入力がバイナリ PLY の場合はエラーになります。どちらかをインストールしてください：

```bash
pip install open3d   # 推奨
# または
pip install plyfile
```

## 例: PostShot 向けフルパイプライン

```bash
# 1. Equirectangular SfM 結果 → NeRF系 JSON
python metashape_360_lfs.py \
    --images images --xml metashape.xml --output .

# 2. Cubemap 変換 (Postshot profile = デフォルト)
python cubemap_transforms_json.py . ./cubic

# 3. COLMAP テキスト出力
python transforms_to_colmap.py ./cubic

# 結果: ./cubic/colmap/ が PostShot/Brush/gaussian-splatting 向けに用意された状態
```
