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
  - 出力先。既定は `<scene>/output`。
- `Target Profile`:
  - 連携先ツール向けのプリセットです。
  - `Postshot / Brush`: 対象アプリ向けの座標プリセットを適用し、シーン内のPLYを直接同梱します。
  - `LichtFeld Studio`: Metashapeの点群PLYを `pointcloud.ply` として取り込み、LichtFeld向けのカメラ情報を作成します。
- `Metashapeインポート設定`:
  - 有効時、キューブマップ変換の前に同梱の
    `vendor/metashape_360_lfs/metashape_360_lfs.py` を実行します。
  - `画像フォルダ`: シーンフォルダ内の `images/` 固定。
  - `カメラXML`: MetashapeからエクスポートしたカメラポーズXML。`--xml` に渡されます。
  - `点群PLY`: Metashapeからエクスポートした点群PLY。LichtFeldでは自動的に使用します。
  - `詳細設定`: 特殊な座標補正用のスケール係数と `--no-fix-rotation`。
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
- `Cube6 Options`:
  - `Drop Top (+90deg)`: 上面を無効化。
  - `Drop Bottom (-90deg)`: 下面を無効化。
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
- `画像とマスク変換なし (--no_image)`
  - キューブマップ画像とマスクを再変換せず、`transforms.json` だけ更新します。`output/` 内の既存ファイルは保持されます。
  - 通常変換時に保存された `output/stechdrive_export_settings.json` は上書きしません。

## ワークフロータブ

- `Cubemap`:
  - 既存の変換ワークフロー（必要に応じて `metashape_360_lfs.py` 前処理 + `cubemap_transforms_json.py`）。
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
- `Metashapeインポート設定` が有効な場合は先に
  `vendor/metashape_360_lfs/metashape_360_lfs.py --images ... --xml ... --output <scene_dir> [...]`
  を実行します。
- その後
  `cubemap_transforms_json.py --fov 90 --output_scale <1.0|0.6366|0.5> --views-json <そのファイル>` を呼び出します。
- `画像とマスク変換なし` が有効な場合は `--no_image` を追加し、`<output_dir>` 内の既存ファイルをリセットしません。
- 通常変換が成功すると、`<output_dir>/stechdrive_export_settings.json` にターゲットプロファイル、画像サイズ、ビュー設定、`views_config.json` のスナップショット、フレーム別ヨー回転、出力形式などを保存します。
- OFFのスロットは `enabled=false` として保存され、変換時に無視されます。
- `LichtFeld Studio` プロファイルでは、点群PLYのインポートが自動的に有効になります。
- マスクは通常「黒=除外領域」で変換されます。Postshot はアプリ側の Mask Mode で扱いを選べるため、GUI側では自動反転しません。
- 変換後、GUIはPLYを `<output_dir>` に同梱し、`<output_dir>/transforms.json` の `ply_file_path` を更新します。
  - `Postshot / Brush`: MetashapeのPLY（例: `metashape.ply` / `sparse.ply`）をコピー
  - `LichtFeld Studio`: `pointcloud.ply` をコピー
- 選択プロファイルで必要なPLYが見つからない場合は、実行前にエラーで停止します。
- `RealityScan Rig XMP` タブでは:
  - `<rs_output_root>/views_config.json` を生成します。
  - `Metashapeインポート設定` 有効時は、RS出力の前にMetashape取り込み処理を実行します。
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
