# cubemap_tools_gui.py — キューブマップ変換ラッパーGUI

`cubemap_tools_gui.py` は `cubemap_transforms_json.py` をプレビュー付きで実行する PySide6 GUI です。

## 目的

このGUIは次の用途向けです。
- `FOV=90°` 固定で運用したい
- 複数ピッチ行（例: `-30,0,30`）を使いたい
- ピッチごとに6視点スロットのON/OFFを選びたい
- エクイレクタングラー画像上で切り出し範囲を確認したい
- 既存マスクを半透明で重ねて確認したい

## 起動

```bash
python cubemap_tools_gui.py --scene-dir ./scene01
```

Windows推奨:

```bat
start_cubemap_tools_gui.bat
```

## 主な入力項目

- `Scene Directory`:
  - `transforms.json` と `images/` を含む作業フォルダ。
- `Output Directory`:
  - 出力先。既定は `<scene>/cubic`。
- `Transforms JSON`:
  - 入力JSONファイル名。既定は `transforms.json`。
- `Target Profile`:
  - 連携先ツール向けのプリセットです。
  - `Postshot / Brush`: `--no_transform` をOFF、前処理`--ply`をOFFにします。
  - `LichtFeld Studio`: `--no_transform` をON、前処理`--ply`をONにします。
  - `Custom (manual)`: 上記2項目を手動で編集できます。
  - プリセット時は不一致防止のため、該当チェックボックスはロックされます。
- `Preprocess`:
  - 有効時、キューブマップ変換の前に同梱の
    `vendor/metashape_360_lfs/metashape_360_lfs.py` を実行します。
- `MS Images Dir`:
  - `metashape_360_lfs.py --images` に渡す画像フォルダ。
- `MS XML`:
  - `metashape_360_lfs.py --xml` に渡すMetashape XML。
- `MS PLY (optional)`:
  - `MS PLY Usage` が有効なときに使うPLYパス。
- `MS PLY Usage`:
  - 前処理で `--ply` を渡すかどうかを制御します。
- `MS Scale`:
  - `metashape_360_lfs.py --scale` の値（正の値が必要）。
- `MS Options`:
  - `Disable rotation fix (--no-fix-rotation)` を前処理に渡します。
- `Mask Directory`:
  - 変換時のマスク入力先。プレビュー合成にも使用。
- `View Mode`:
  - `Custom Pitch/Yaw`: 既存のピッチ行 + YAWスロット方式。
  - `Cube6 (4 sides + top/bottom)`: 6面キューブ固定方式（FOV 90）。
- `Yaw Offset (deg)`:
  - スロット角度の基準値。
- `Yaw Slots`:
  - 各ピッチ行のYAWスロット数（`4..8`）。
  - 各スロット角度は `offset + slot*(360 / yaw_slots)`。
- `Pitch Rows (deg CSV)`:
  - ピッチ一覧。例: `-30,0,30`。
  - 最大9行まで。
- `Cube6 Options`:
  - `Drop Top (+90deg)`: 上面を無効化。
  - `Drop Bottom (-90deg)`: 下面を無効化。
- `FOV`:
  - このGUIでは `90.0` 固定。
- `Preview Image`:
  - プレビュー対象のエクイレクタングラー画像。
  - `Auto` でシーン画像の先頭を自動選択します。
  - `Reload` でシーン画像一覧を再スキャンします。
- `Preview Timeline`:
  - シーン画像（優先: `images/`、無ければシーン直下）をスライダーで切り替えます。
  - フレームごとの視点ON/OFF確認に便利です。
- `Mask Overlay (%)`:
  - マスク重ね表示の不透明度。
- `Preview Mask Image`:
  - プレビュー合成に使うマスク画像を手動指定できます。
  - 空欄なら `Mask Directory` から対応ファイルを自動探索します。

## 視点選択

- `Apply Pitch Rows` で各ピッチ行に `Yaw Slots` 分のスロットが生成されます。
- 各チェックボックスで出力対象をON/OFFできます。
- 典型例:
  - pitch `0`: その行の全スロットON
  - pitch `+/-30`: 必要なスロットだけON

## 実行オプション

- `Extract mask from alpha (--mask_from_alpha)`
- `Transforms only (--no_image)`
- `No axis transform (--no_transform)`
- `Allow duplicate (--duplicate)`
- `Invert masks (--invert_masks)`

## 実行時の挙動

- 実行時に `<output_dir>/views_config.json` を生成します。
- `Preprocess` が有効な場合は先に
  `vendor/metashape_360_lfs/metashape_360_lfs.py --images ... --xml ... --output <scene_dir> [...]`
  を実行します。
- その後
  `cubemap_transforms_json.py --fov 90 --views-json <そのファイル>` を呼び出します。
- OFFのスロットは `enabled=false` として保存され、変換時に無視されます。
- `Postshot / Brush` プロファイルでは、点群とカメラの不一致を避けるため、前処理`--ply`が既定でOFFになります。
- マスク反転は自動では行われません。必要な場合のみ `Invert masks (--invert_masks)` を有効にしてください。
- 変換後、GUIはPLYを `<output_dir>` に同梱し、`<output_dir>/transforms.json` の `ply_file_path` を更新します。
  - `Postshot / Brush`: MetashapeのPLY（例: `metashape.ply` / `sparse.ply`）をコピー
  - `LichtFeld Studio`: `pointcloud.ply` をコピー
- 選択プロファイルで必要なPLYが見つからない場合は、実行前にエラーで停止します。

## 注意

- 1視点もONになっていない場合は実行できません。
- 有効視点数が24を超えると警告を表示します。
- 有効視点数が40を超えると実行できません。
- プレビューは確認用で、最終出力は `views_config.json` の内容に従います。
