# Step 4 書き出しGUI — 投影視点 / 3DGUT / COLMAP Rig書き出し

統合GUIの Step 4 は、Metashape SfM結果や抽出済み360°画像から、3DGS向けの
投影視点データ、LichtFeld 3DGUT直接データ、COLMAP Rigデータを作る
PySide6 GUIです。投影視点とCOLMAP Rig書き出しでは `cubemap_transforms_json.py` を使い、
`3DGUT (LichtFeld)` ではMetashapeインポートだけを実行してキューブマップ変換を行いません。

## 目的

Step 4 は次の用途向けです。
- `FOV=90°` 固定で運用したい
- 複数Pitch行（例: `-30,0,30`）を使いたい
- Pitchごとに6視点スロットのON/OFFを選びたい
- エクイレクタングラー画像上で切り出し範囲を確認したい
- 既存マスクの重ね表示をオン/オフして確認したい
- LichtFeld 3DGUT比較用に、Metashapeで使ったエクイレクタングラー画像とマスクをそのまま使いたい

## 起動

```bat
run_gui.bat --scene .\scene01
```

起動後、ワークフロー左側の `Step 4: 書き出し` を開きます。

## 主な入力項目

- `Scene Directory`:
  - `images/` と必要に応じて `transforms.json` を含む作業フォルダ。
- `Output Directory`:
  - 投影視点とCOLMAP Rigの出力先。既定は `<scene>/output`。`3DGUT (LichtFeld)` ではシーン直下に `transforms.json` と `pointcloud.ply` を作成します。
- `選択:`:
  - `Metashape`: Metashape SfM結果から、3DGS向けの視点画像、マスク、`transforms.json` を書き出します。
  - `COLMAP`: 抽出済みの `images/` と `masks/` から、COLMAP Rig形式の視点画像、マスク、`rig_config.json` を `output/colmap_rig/` に書き出します。
- `COLMAP実行設定`:
  - `COLMAP` 選択時に、設定タブの `変換設定` として表示されます。
  - `書き出し後にCOLMAPを実行`: 視点画像の書き出し後に `feature_extractor` → `rig_configurator` → matcher → mapper を実行します。重い処理なので必要な時だけONにします。
  - `COLMAP実行ファイル`: 使用する `colmap.exe` を指定します。空欄ならPATH上の `colmap.exe` を使います。
  - `Matcher`: `Sequential` は高速で動画の連番フレーム向けです。`Exhaustive` は全ペアを照合するため精度が出る場合がありますが、枚数が増えると数十時間規模になることがあります。
  - `Mapper`: `Global` はCOLMAP 4.0以降に統合されたGLOMAP系のグローバルSfMで、既定推奨です。`Incremental` は従来の `colmap mapper`、`GLOMAP` は外部 `glomap.exe` 用です。
- `出力プリセット`:
  - `Metashape` 選択時に使う、連携先3DGSツール向けのプリセットです。
  - `Postshot / Brush`: 対象アプリ向けの座標プリセットを適用し、シーン内のPLYを直接同梱します。
  - `LichtFeld Studio`: Metashapeの点群PLYを `pointcloud.ply` として取り込み、LichtFeld向けのカメラ情報を作成します。
  - 詳細設定で座標変換、PLY使用、Metashapeインポート詳細をプリセット値から変更すると、プリセット表示は `カスタム` に切り替わります。
- `出力形状`:
  - `投影視点に変換`: 従来通り、下の `投影視点` タブの設定でキューブマップ/視点画像とマスクを `output/` に書き出します。
  - `3DGUT (LichtFeld)`: LichtFeldの3DGUT比較用に、Metashapeで使ったエクイレクタングラーの `images/` と `masks/` をそのまま使います。視点画像と変換マスクは作らず、シーン直下の `transforms.json` と `pointcloud.ply` を更新します。
  - この直接モードでは `LichtFeld Studio` 向けの座標設定とPLY使用が必要になり、`投影視点` タブと視点画像/マスク出力のON/OFFは無効になります。
- `Metashapeインポート設定`:
  - `Metashape` 選択時に、設定タブの `変換設定` として表示されます。
  - `Metashape` 方式では、キューブマップ変換の前に同梱の
    `vendor/metashape_360_lfs/metashape_360_lfs.py` を実行します。
  - `画像フォルダ`: シーンフォルダ内の `images/` 固定。
  - `カメラXML`: MetashapeからエクスポートしたカメラポーズXML。`--xml` に渡されます。
  - `点群PLY`: Metashapeからエクスポートした点群PLY。LichtFeldでは自動的に使用します。
  - `COLMAP形式モデルを追加出力`: `output/transforms.json` とPLYから `output/colmap/` に `cameras.txt` / `images.txt` / `points3D.txt` を追加生成します。COLMAPで再SfMするための画像書き出しではありません。
  - `詳細設定`: `--scale`、`--ply` の使用有無、`--no-fix-rotation` を指定できます。
- `出力`:
  - 常時表示されます。`画像` / `マスク` を個別にON/OFFできます。マスクだけ作り直した場合は `画像` をOFF、`マスク` をONにします。
- `投影視点`:
  - 設定タブの `投影視点` に常時表示されます。
  - ビュープリセット、Yawオフセット、画像サイズ、フレーム別Yaw回転、出力フォーマット、ビット深度、マスク反転など、各方式で共通する視点画像の書き出し設定です。
  - `出力形状` が `3DGUT (LichtFeld)` の場合は使用しません。
- マスク:
  - 変換とプレビューは、シーンフォルダ内の `masks/` から対応ファイルを自動的に使用します。
- `プリセット`:
  - `Cube6`: 4列 x 3行グリッド上の6面プリセットです。既定ではYawスロット数 `4`、Pitch行 `-90,0,90`、pitch `0` の4スロットと上下の `S3` をONにします。
  - `Custom Grid`: Pitch行 + Yawスロットを自由に編集する方式です。Cube6グリッドを編集すると自動的にカスタム扱いになります。
- `Yaw Offset (deg)`:
  - Yawスロット角度の基準値。
- `Yaw Slots`:
  - 各Pitch行のYawスロット数（`4..8`）。出力視点の上部にある `-` / `+` で列数を増減します。
  - 増減しても既存列のON/OFFとPitch値は維持され、追加列はONで作られます。
  - 各スロット角度は `offset + slot*(360 / yaw_slots)`。
- `Pitch Rows`:
  - Pitch一覧。`Cube6` の標準は `-90,0,90`、`Custom Grid` の標準は `-45,0,45`。
  - 範囲は `-90..90`、最大5行まで。Pitch行左側の `+` で行を追加し、各Pitch行左端の削除ボタンで行を削除します。
  - 行の増減時は残る行のPitch値とON/OFFを行順で維持します。
- `Cube6`:
  - 6面すべてを書き出します。上面・下面もカメラポーズ固定の有効な観測として扱います。
- `FOV`:
  - このGUIでは `90.0` 固定。
- `Image Size`:
  - `Full`: 出力面サイズ = 入力画像高さ x `1.0`。最終品質向け。
  - `Normal`: 出力面サイズ = 入力画像高さ x `2 / pi`（約 `0.637`）。90度画像中央部の角度解像度を元画像に近づけます。
  - `Half`: 出力面サイズ = 入力画像高さ x `0.5`。軽量ですが、学習後に柔らかく見えやすい設定です。
- `Preview`:
  - シーンフォルダ内のエクイレクタングラー画像を自動的に使います。
  - シーン画像（優先: `images/`、無ければシーン直下）をスライダーで切り替えます。
  - フレームごとの視点ON/OFF確認に便利です。
  - マスク重ね表示は `masks/` の対応ファイルを使い、マスク表示ボタンでオン/オフします。
  - パース表示ボタンで、FOV90°の正方形パース表示へ切り替えられます。パース表示中はドラッグで視線方向を回し、ホイールは表示中画像の2Dズームとして動作します。マスク重ね表示と視点枠線もパース表示に反映されます。視点グリッドをホバーすると、その視点方向にパース表示を合わせます。パース表示は利用可能な環境ではOpenGL/GPUで高速化し、使えない環境ではCPU描画へフォールバックします。

## 視点選択

- 各チェックボックスで出力対象をON/OFFできます。
- 出力視点は常時表示され、上部のボタンで全選択/全解除とYaw列追加/削除、Pitch行左側のボタンでPitch行追加ができます。
- Cube6の既定状態では、上面・下面は `S3` に対応します。既定Yawオフセット45度では `S3=-45°` です。
- 典型例:
  - pitch `0`: その行の全スロットON
  - pitch `+/-30`: 必要なスロットだけON

## 実行オプション

- `Invert masks (--invert_masks)`
  - 通常はOFF。出力先アプリで逆極性が必要な場合だけON。
- `出力`
  - `画像` OFFで `--skip-images`、`マスク` OFFで `--skip-masks` を追加します。
  - 両方OFFの場合は画像とマスクを再変換せず、カメラ情報だけ更新します。既存の `output/` 内ファイルは保持されます。

## 実行時の挙動

- 実行時に `<output_dir>/views_config.json` を生成します。
- `Metashape` 方式では先に
  `vendor/metashape_360_lfs/metashape_360_lfs.py --images ... --xml ... --output <scene_dir> [...]`
  を実行します。
- その後
  `cubemap_transforms_json.py --fov 90 --output_scale <1.0|0.6366|0.5> --views-json <そのファイル>` を呼び出します。
- `出力形状` が `3DGUT (LichtFeld)` の場合は、上記のMetashapeインポートだけを実行し、`cubemap_transforms_json.py` は呼び出しません。
  - 出力: `<scene_dir>/transforms.json`
  - 点群: `<scene_dir>/pointcloud.ply`
  - 画像: `<scene_dir>/images/` をそのまま参照
  - マスク: `<scene_dir>/masks/` をそのまま参照
  - 実行設定は `<scene_dir>/stechdrive_export_settings.json` に保存されます。
- `COLMAP` 方式では、`cubemap_transforms_json.py --image-only --colmap-rig --yaw-offset-per-frame 0 ...` を呼び出します。
  - 画像: `<output_dir>/colmap_rig/images/rig1/camXX/frame_00001.<ext>`
  - マスク: `<output_dir>/colmap_rig/masks/rig1/camXX/frame_00001.<ext>.png`
  - リグ設定: `<output_dir>/colmap_rig/rig_config.json`
  - フレーム別Yaw回転は固定リグ前提を崩すため、COLMAP Rig書き出しでは常に0度です。
- `書き出し後にCOLMAPを実行` がONの場合は、続けて `feature_extractor`、`rig_configurator`、matcher、mapperを実行します。
- `画像` / `マスク` の書き出し対象がOFFの場合は、それぞれ `--skip-images` / `--skip-masks` を追加します。
- 通常変換が成功すると、`<output_dir>/stechdrive_export_settings.json` にターゲットプロファイル、画像サイズ、ビュー設定、`views_config.json` のスナップショット、フレーム別Yaw回転、出力形式などを保存します。
- OFFのスロットは `enabled=false` として保存され、変換時に無視されます。
- `LichtFeld Studio` プロファイルでは、点群PLYのインポートが自動的に有効になります。
- マスクは通常「黒=除外領域」で変換されます。Postshot はアプリ側の Mask Mode で扱いを選べるため、GUI側では自動反転しません。
- 変換後、GUIはPLYを `<output_dir>` に同梱し、`<output_dir>/transforms.json` の `ply_file_path` を更新します。
  - `Postshot / Brush`: MetashapeのPLY（例: `metashape.ply` / `sparse.ply`）をコピー
  - `LichtFeld Studio`: `pointcloud.ply` をコピー
- `LichtFeld Studio` プロファイルでは、最終出力の `transforms.json` と `pointcloud.ply` に同じ向き補正を適用し、LichtFeld上でMetashapeと同じ +X / +Z / 上下方向になるようにします。
- 選択プロファイルで必要なPLYが見つからない場合は、実行前にエラーで停止します。

## 注意

- 1視点もONになっていない場合は実行できません。
- 有効視点数が24を超えると警告を表示します。
- 有効視点数が40を超えると実行できません。
- プレビューは確認用で、最終出力は `views_config.json` の内容に従います。
