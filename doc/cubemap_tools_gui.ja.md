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
  - `images/` と必要に応じて `transforms.json` を含む作業フォルダ。
- `Output Directory`:
  - 出力先。既定は `<scene>/output`。
- `書き出し方式`:
  - `Metashapeインポート`: Metashape SfM結果から、3DGS向けの視点画像、マスク、`transforms.json` を書き出します。
  - `COLMAP書き出し`: Metashape結果を使わず、抽出済みの `images/` と `masks/` からCOLMAP SfM向けの視点画像とマスクを書き出します。
- `出力プロファイル`:
  - `Metashapeインポート` 選択時に使う、連携先3DGSツール向けのプリセットです。
  - `Postshot / Brush`: 対象アプリ向けの座標プリセットを適用し、シーン内のPLYを直接同梱します。
  - `LichtFeld Studio`: Metashapeの点群PLYを `pointcloud.ply` として取り込み、LichtFeld向けのカメラ情報を作成します。
  - 詳細設定で座標変換、PLY使用、Metashapeインポート詳細をプリセット値から変更すると、プロファイル表示は `カスタム` に切り替わります。
- `Metashapeインポート設定`:
  - `Metashapeインポート` 方式では、キューブマップ変換の前に同梱の
    `vendor/metashape_360_lfs/metashape_360_lfs.py` を実行します。
  - `画像フォルダ`: シーンフォルダ内の `images/` 固定。
  - `カメラXML`: MetashapeからエクスポートしたカメラポーズXML。`--xml` に渡されます。
  - `点群PLY`: Metashapeからエクスポートした点群PLY。LichtFeldでは自動的に使用します。
  - `COLMAP形式モデルを追加出力`: `output/transforms.json` とPLYから `output/colmap/` に `cameras.txt` / `images.txt` / `points3D.txt` を追加生成します。COLMAPで再SfMするための画像書き出しではありません。
  - `詳細設定`: `--scale`、`--ply` の使用有無、`--no-fix-rotation` を指定できます。
- `書き出し対象`:
  - 常時表示されます。`画像` / `マスク` を個別にON/OFFできます。マスクだけ作り直した場合は `画像` をOFF、`マスク` をONにします。
- `視点書き出し設定`:
  - ビューモード、ヨーオフセット、画像サイズ、フレーム別ヨー回転、出力フォーマット、ビット深度、マスク反転など、各方式で共通する視点画像の書き出し設定です。
- マスク:
  - 変換とプレビューは、シーンフォルダ内の `masks/` から対応ファイルを自動的に使用します。
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
- `Cube6`:
  - 6面すべてを書き出します。上面・下面もカメラポーズ固定の有効な観測として扱います。
- `FOV`:
  - このGUIでは `90.0` 固定。
- `Image Size`:
  - `Full (Quality)`: 出力面サイズ = 入力画像高さ x `1.0`。最終品質向け。
  - `Normal`: 出力面サイズ = 入力画像高さ x `2 / pi`（約 `0.637`）。90度画像中央部の角度解像度を元画像に近づけます。
  - `Half (Light)`: 出力面サイズ = 入力画像高さ x `0.5`。軽量ですが、学習後に柔らかく見えやすい設定です。
- `Preview`:
  - シーンフォルダ内のエクイレクタングラー画像を自動的に使います。
  - シーン画像（優先: `images/`、無ければシーン直下）をスライダーで切り替えます。
  - フレームごとの視点ON/OFF確認に便利です。
  - マスク重ね表示は `masks/` の対応ファイルを使い、不透明度スライダーで表示量を調整します。

## 視点選択

- `Apply Pitch Rows` で各ピッチ行に `Yaw Slots` 分のスロットが生成されます。
- 各チェックボックスで出力対象をON/OFFできます。
- 典型例:
  - pitch `0`: その行の全スロットON
  - pitch `+/-30`: 必要なスロットだけON

## 実行オプション

- `Invert masks (--invert_masks)`
  - 通常はOFF。出力先アプリで逆極性が必要な場合だけON。
- `書き出し対象`
  - `画像` OFFで `--skip-images`、`マスク` OFFで `--skip-masks` を追加します。
  - 両方OFFの場合は画像とマスクを再変換せず、カメラ情報だけ更新します。既存の `output/` 内ファイルは保持されます。

## ワークフロータブ

- `Cubemap`:
  - 既存の変換ワークフロー（`Metashapeインポート` 方式では `metashape_360_lfs.py` + `cubemap_transforms_json.py`）。
- `COLMAP Rig SfM`:
  - COLMAP リグデータセットの出力と、必要なら COLMAP SfM ステージ実行。
- `RealityScan Rig XMP`:
  - RealityScan 読み込み用の自己完結パッケージを出力します。
  - `RS Output Root`: パッケージ出力先ルート。
  - `Pose Prior`: XMP姿勢Prior（`Draft/Exact/Locked`）。
  - `Calibration Prior`: XMP内部パラメータPrior（既定推奨は `Fixed`）。
  - `Focal35mm Override`: 焦点距離の上書き（空欄時は `FOV=90` から自動換算で `18mm`）。
  - `Pose Transform`: RS出力スクリプトに `--no_transform` を渡します。
  - `Mask Export`: RSパッケージ用マスク反転の有無。

## 実行時の挙動

- 実行時に `<output_dir>/views_config.json` を生成します。
- `Metashapeインポート` 方式では先に
  `vendor/metashape_360_lfs/metashape_360_lfs.py --images ... --xml ... --output <scene_dir> [...]`
  を実行します。
- その後
  `cubemap_transforms_json.py --fov 90 --output_scale <1.0|0.6366|0.5> --views-json <そのファイル>` を呼び出します。
- `画像` / `マスク` の書き出し対象がOFFの場合は、それぞれ `--skip-images` / `--skip-masks` を追加します。
- 通常変換が成功すると、`<output_dir>/stechdrive_export_settings.json` にターゲットプロファイル、画像サイズ、ビュー設定、`views_config.json` のスナップショット、フレーム別ヨー回転、出力形式などを保存します。
- OFFのスロットは `enabled=false` として保存され、変換時に無視されます。
- `LichtFeld Studio` プロファイルでは、点群PLYのインポートが自動的に有効になります。
- マスクは通常「黒=除外領域」で変換されます。Postshot はアプリ側の Mask Mode で扱いを選べるため、GUI側では自動反転しません。
- 変換後、GUIはPLYを `<output_dir>` に同梱し、`<output_dir>/transforms.json` の `ply_file_path` を更新します。
  - `Postshot / Brush`: MetashapeのPLY（例: `metashape.ply` / `sparse.ply`）をコピー
  - `LichtFeld Studio`: `pointcloud.ply` をコピー
- `LichtFeld Studio` プロファイルでは、最終出力の `transforms.json` と `pointcloud.ply` に同じ向き補正を適用し、LichtFeld上でMetashapeと同じ +X / +Z / 上下方向になるようにします。
- 選択プロファイルで必要なPLYが見つからない場合は、実行前にエラーで停止します。
- `RealityScan Rig XMP` タブでは:
  - `<rs_output_root>/views_config.json` を生成します。
  - `realityscan_rig_export.py` を実行します。
  - RealityScan 読み込み用ファイルは `<rs_output_root>/inputs` にまとめて出力されます。
    - `<image_name>`
    - `<image_stem>.xmp`
    - `<image_name>.mask.png`（マスクがある場合）

## 注意

- 1視点もONになっていない場合は実行できません。
- 有効視点数が24を超えると警告を表示します。
- 有効視点数が40を超えると実行できません。
- プレビューは確認用で、最終出力は `views_config.json` の内容に従います。
