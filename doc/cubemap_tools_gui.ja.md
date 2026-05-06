# Step 4 書き出しGUI

Step 4 は、Step 1-3で用意した360°画像とマスク、またはMetashapeでSfMした結果を、3DGSアプリが読み込める学習データに変換する画面です。

多くの場合は、Metashapeから書き出したカメラXMLと、必要に応じて点群PLYを指定し、Postshot / Brush / LichtFeld Studio のどれに渡すかを選びます。LichtFeld Studioでは、通常の投影視点データに加えて、エクイレクタングラー画像をそのまま使う `3DGUT (LichtFeld)` 用データも作れます。

## 起動

```bat
run_gui.bat --scene .\scene01
```

起動後、ワークフロー左側の `Step 4: 書き出し` を開きます。

## まず決めること

Step 4を開いたら、最初に「自分はどのルートか」を決めます。

| やりたいこと | `選択:` | 主に使う設定 |
| --- | --- | --- |
| MetashapeでSfM済みの結果をPostshot / Brush / LichtFeldへ渡したい | `Metashape` | `出力プリセット`, `出力形状`, `カメラXML`, `点群PLY` |
| LichtFeldでキューブマップ版と3DGUT版を比較したい | `Metashape` | `出力プリセット: LichtFeld Studio`, `出力形状` |
| Metashapeを使わず、抽出済み360°画像からCOLMAP/GLOMAPへ進みたい | `COLMAP` | `COLMAP実行設定`, `投影視点` |
| すでに作った出力の画像やマスクだけ作り直したい | 元のルート | `出力`, `投影視点`, `画像サイズ` |

`Scene Directory` は、Step 1-3で使っているシーンフォルダです。通常は `images/` と `masks/` が入っています。Metashapeルートでは、そこにMetashapeから書き出したXML/PLYを指定して、3DGS向けのデータを作ります。

## Metashapeルートの基本操作

Metashapeで360°画像をアライメント済みなら、基本はこの流れです。

1. `選択:` を `Metashape` にします。
2. `カメラXML` にMetashapeからエクスポートしたカメラXMLを指定します。
3. LichtFeld Studio向け、または点群も同梱したい場合は `点群PLY` を指定します。
4. `出力プリセット` で渡し先を選びます。
5. `出力形状` で、投影視点に変換するか、`3DGUT (LichtFeld)` にするかを選びます。
6. 投影視点に変換する場合は、`投影視点` タブでビュー、Yaw、画像サイズを確認します。
7. `実行` します。

Metashapeルートでは、最初に同梱の `vendor/metashape_360_lfs/metashape_360_lfs.py` を実行して、Metashape XMLから `transforms.json` を作ります。`出力形状` が `投影視点に変換` の場合は、そのあと `cubemap_transforms_json.py` で視点画像とマスクを作ります。

## 出力プリセットの選び方

`出力プリセット` は、書き出したデータをどの3DGSアプリに渡すかの選択です。

| プリセット | 使う場面 |
| --- | --- |
| `Postshot` | Postshot向けに投影視点データを作る |
| `Brush` | Brush向けに投影視点データを作る |
| `LichtFeld Studio` | LichtFeld向けに投影視点データ、または3DGUT直接データを作る |
| `カスタム` | 座標変換やPLY使用を手動で調整する |

通常は渡し先のアプリ名をそのまま選びます。詳細設定で座標変換、PLY使用、Metashapeインポート設定をプリセット値から変えると、自動的に `カスタム` 扱いになります。

## 出力形状の選び方

`出力形状` は、Metashapeのエクイレクタングラー画像をどう学習データにするかの選択です。

### 投影視点に変換

通常はこちらを使います。エクイレクタングラー画像をキューブマップ/視点画像に変換し、`output/` に画像、マスク、`transforms.json` を作ります。

この出力はPostshot / Brush / LichtFeld Studioで扱いやすく、通常のピンホールカメラに近いデータになります。LichtFeldでこのデータを学習するときは、基本的にGUTやUndistortは使いません。

### 3DGUT (LichtFeld)

LichtFeld StudioでGUT学習を試すための比較用モードです。Metashapeで使ったエクイレクタングラーの `images/` と `masks/` をそのまま使い、視点画像や変換マスクは作りません。

このモードでは、シーン直下に次のファイルを作ります。

- `transforms.json`
- `pointcloud.ply`
- `stechdrive_export_settings.json`

`3DGUT (LichtFeld)` では `出力プリセット: LichtFeld Studio` とPLY使用が必要です。選択中は `投影視点` タブ、画像/マスク出力のON/OFF、COLMAP形式モデル追加出力は無効になります。

## LichtFeldで比較したい場合

LichtFeldで「投影視点に変換したデータ」と「3DGUTでエクイレクタングラーを直接読むデータ」を比べたい場合は、同じMetashape結果から2回書き出します。

### キューブマップ/投影視点版

1. `選択:` を `Metashape` にします。
2. `出力プリセット` を `LichtFeld Studio` にします。
3. `出力形状` を `投影視点に変換` にします。
4. `点群PLY` を指定します。
5. `投影視点` で `Cube6`、Yaw 45°、必要な `画像サイズ` を選びます。
6. 実行します。

出力先は通常 `<scene>/output/` です。LichtFeldにはこの `output/` を読み込ませます。

### 3DGUT直接版

1. `選択:` を `Metashape` にします。
2. `出力プリセット` を `LichtFeld Studio` にします。
3. `出力形状` を `3DGUT (LichtFeld)` にします。
4. `点群PLY` を指定します。
5. 実行します。

出力先はシーンフォルダ直下です。LichtFeldにはシーンフォルダを読み込ませ、学習時にGUTを有効にします。

比較するときは、Metashape XML、PLY、`images/`、`masks/`、LichtFeld側の学習設定をできるだけ揃えてください。違いが見えにくい場合は、近距離物体、細い線、文字、スティッチ境界付近、床や壁のつながりを見ると判断しやすくなります。

## 投影視点の調整

`出力形状` が `投影視点に変換` の場合、`投影視点` タブでどの方向の画像を作るかを調整します。

### まずはCube6

通常は `Cube6` から始めます。上下を含む6面を作る設定で、Postshot / Brush / LichtFeld Studio向けの標準的な比較に使いやすいです。

`Yaw Offset` は既定の45°が推奨です。補正なしで書き出した2眼360°カメラでは、スティッチ境界がエクイレクタングラー画像の横25%/75%付近に来ることが多く、45°ずらすと境界がキューブ面の中心を横切りにくくなります。

### Custom Grid

必要な方向だけ増やしたい、上下を減らしたい、斜め上/斜め下の観測を足したい場合は `Custom Grid` を使います。

- `Yaw Slots`: 水平方向の分割数です。4から8まで増減できます。
- `Pitch Rows`: 上下方向の行です。`-90..90` の範囲で最大5行まで使えます。
- 各チェックボックス: その視点を書き出すかどうかです。

有効視点数が増えるほど出力枚数と処理時間が増えます。24視点を超えると警告、40視点を超えると実行不可になります。

### 画像サイズ

`画像サイズ` は投影視点1枚の解像度です。

| 設定 | 使いどころ |
| --- | --- |
| `Full` | 最終品質確認。重いが細部が残りやすい |
| `Normal` | まず試す標準設定。90°視点中央部の角度解像度を元画像に近づける |
| `Half` | 軽量テスト。速いが柔らかく見えやすい |

VRAMや処理時間が厳しい場合は、まず `Normal` または `Half` で流れを確認し、最後に `Full` を試すのが現実的です。

## 画像やマスクだけ作り直したい場合

`出力` の `画像` / `マスク` チェックで、どちらを書き出すかを選べます。

| やりたいこと | 設定 |
| --- | --- |
| 画像とマスクを両方作り直す | `画像` ON, `マスク` ON |
| マスクだけ作り直す | `画像` OFF, `マスク` ON |
| 画像だけ作り直す | `画像` ON, `マスク` OFF |
| カメラ情報だけ更新する | `画像` OFF, `マスク` OFF |

マスクだけ調整したあとに再出力する場合は、`画像` をOFFにすると既存の視点画像を再変換せずに済みます。`3DGUT (LichtFeld)` では元画像と元マスクをそのまま使うため、この出力ON/OFFは使いません。

## COLMAPルート

Metashapeを使わず、抽出済みの360°画像からCOLMAP/GLOMAPへ進みたい場合は `選択:` を `COLMAP` にします。

1. `Scene Directory` に `images/` と必要なら `masks/` があることを確認します。
2. `選択:` を `COLMAP` にします。
3. `投影視点` で視点数、Yaw、画像サイズを決めます。
4. `書き出し後にCOLMAPを実行` を必要な場合だけONにします。
5. `Matcher` と `Mapper` を選びます。通常は `Sequential` と `Global` から始めます。
6. 実行します。

COLMAPルートでは、`output/colmap_rig/` にCOLMAP Rig形式の視点画像、マスク、`rig_config.json` を作ります。`書き出し後にCOLMAPを実行` がONの場合は、続けてFeature、Rig設定、Matcher、Mapperまで実行します。

フレーム別Yaw回転は、固定リグの前提を崩すためCOLMAP Rig書き出しでは常に0度です。

## 実行後にできるもの

| ルート | 主な出力 |
| --- | --- |
| Metashape + 投影視点に変換 | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/stechdrive_export_settings.json` |
| Metashape + `3DGUT (LichtFeld)` | `<scene>/transforms.json`, `<scene>/pointcloud.ply`, `<scene>/stechdrive_export_settings.json` |
| COLMAP | `<scene>/output/colmap_rig/images/`, `<scene>/output/colmap_rig/masks/`, `<scene>/output/colmap_rig/rig_config.json` |
| COLMAP実行あり | 上記に加えて、COLMAP/GLOMAPのSfM結果 |

`LichtFeld Studio` プロファイルでは、最終出力の `transforms.json` と `pointcloud.ply` に同じ向き補正を適用し、LichtFeld上でMetashapeと同じ +X / +Z / 上下方向になるようにします。

## よくある判断

- Postshot / Brushへ渡すなら、まず `投影視点に変換` を使います。
- LichtFeldで通常学習するなら、まず `LichtFeld Studio` + `投影視点に変換` を使います。
- LichtFeldでGUTを試すなら、`LichtFeld Studio` + `3DGUT (LichtFeld)` を使います。
- キューブマップ/投影視点データをLichtFeldで学習するときは、基本的にGUTもUndistortも不要です。
- `3DGUT (LichtFeld)` で学習するときは、LichtFeld側でGUTを有効にします。
- スティッチが目立たない素材では、スティッチマスクはOFFまたは細めから試します。Yaw 45°は画素を捨てない対策なので、通常は維持して構いません。
- `点群PLY` が必要なプロファイルで見つからない場合は、実行前にエラーで止まります。
