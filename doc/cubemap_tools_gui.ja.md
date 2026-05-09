# Step 4 変換GUI

Step 4 は、Step 1-3で用意した360°画像とマスク、MetashapeでSfMした結果、またはSphereSfMで作るSfM結果を、3DGSアプリが読み込めるトレーニングデータに変換する画面です。

多くの場合は、Metashapeから書き出したカメラXMLと、必要に応じて点群PLYを指定し、Postshot / Brush / LichtFeld Studio のどれに渡すかを選びます。Metashapeを使わない場合は、COLMAP Rig用のキューブマップ画像を書き出すか、SphereSfMでエクイレクタングラー画像を直接SfMしてから3DGS向けデータへ変換できます。学習アプリの起動はStep 5で、Step 4が作成済みのデータセットを読み込んで実行します。

## まず決めること

Step 4を開いたら、最初に「自分はどのルートか」を決めます。

SfMは、複数画像の見え方の差からカメラ位置と疎な点群を推定する処理です。3DGSでは、このカメラ位置を使ってトレーニングデータを読み込みます。MetashapeルートはMetashapeで作ったSfM結果を変換し、SphereSfMルートとCOLMAP実行ありのCOLMAPルートはこのアプリからSfM処理まで進めます。

| やりたいこと | ルート | 主に使う設定 |
| --- | --- | --- |
| MetashapeでSfM済みの結果をPostshot / Brush / LichtFeldへ渡したい | `Metashape` | `出力プリセット`, `出力形状`, `カメラXML`, `点群PLY` |
| LichtFeldで3DGUT用データを作りたい | `Metashape` | `出力プリセット: LichtFeld Studio`, `出力形状`, `点群PLY` |
| Metashapeを使わず、抽出済み360°画像からCOLMAP/GLOMAPへ進みたい | `COLMAP` | `COLMAP実行設定`, `Cubemap` |
| Metashapeを使わず、抽出済みエクイレクタングラー画像を直接SfMしたい | `SphereSfM` | `SphereSfM COLMAP実行ファイル`, `SfM入力`, `Matcher`, `SfM品質`, `出力形状` |
| すでに作った出力の画像やマスクだけ作り直したい | 元のルート | `Cubemap`, `画像サイズ` |

`シーンフォルダ` は、Step 1-3で使っている作業フォルダです。通常は `images/` と `masks/` が入っています。MetashapeルートではMetashapeから書き出したXML/PLYを組み合わせます。SphereSfMルートでは、この `images/` と `masks/` からSfMと変換を実行します。

左ナビゲーションには、Step 4のサブ工程として `SfM` と `Cube` が表示されます。各行の左アイコンまたは文字部分で、今回その工程を対象にするかを切り替えます。右アイコンは `準備済み`、`未準備`、`スキップ` の状態表示です。`!` の警告アイコンだけはクリックすると該当する設定タブへ移動しますが、チェックマークは状態表示だけでタブ移動はしません。Metashapeルートでは、`Cube` がMetashapeのカメラポーズを必要とするときだけ `SfM` が自動でONになります。次に何をすればよいかは各アイコンのツールチップで確認できます。Step 4内では、ルートボタン `Metashape`、`COLMAP`、`SphereSfM` と、Cubeが参照するXML/PLYや既存sparseモデルは `SfM` タブにあります。出力プリセット、画像/マスクのON/OFF、Cube6、Yaw、画像サイズは `Cubemap` にあります。

## Metashapeルートの基本操作

Metashapeで360°画像をアライメント済みなら、基本はこの流れです。

1. ルートを `Metashape` にします。
2. `カメラXML` を確認します。シーンフォルダ直下のXMLは内容を読み取り、Metashapeのカメラ情報として判定できるものを自動入力します。
3. LichtFeld Studio向け、または点群も同梱したい場合は `点群PLY` を確認します。シーンフォルダ直下のPLY候補が1つなら入力欄に入り、自動で使用対象になります。候補が複数ある場合は、使うPLYを手動で選びます。
4. `出力プリセット` で渡し先を選びます。
5. `出力形状` で、キューブマップ画像へ変換するか、`3DGUT (LichtFeld)` にするかを選びます。
6. キューブマップ画像を作る場合は、`Cubemap` タブでCube6、Yaw、画像サイズを確認します。
7. `実行` します。

Metashapeルートでは、最初に同梱の `vendor/metashape_360_lfs/metashape_360_lfs.py` を実行して、Metashape XMLから `transforms.json` を作ります。`出力形状` が `投影視点に変換` の場合は、そのあと `cubemap_transforms_json.py` でキューブマップ画像とマスクを作ります。

### カメラXMLと点群PLYの自動検出

Step 4は、シーンフォルダが設定された時点でMetashape用の入力候補を自動で探します。XMLはファイル名ではなく内容を読み、Metashapeの `document` / `chunk` / `sensors` / `cameras` / 4x4 `transform` があり、可能なら `images/` の画像名と照合できるものを自動入力します。

- `カメラXML`: シーンフォルダ直下の `.xml` を解析します。複数の有効XMLが同点で曖昧な場合は自動選択せず、候補をツールチップに表示します。
- `点群PLY`: シーンフォルダ直下の `.ply` を候補にします。1つだけなら入力欄へ表示し、自動で使用対象にします。複数ある場合は自動選択せず、候補をツールチップに表示します。

`pointcloud.ply` はこのアプリがLichtFeld用に作る出力ファイル名なので、Metashapeからエクスポートした入力PLYの候補からは除外します。Metashapeから出した元XML/PLYは `output/` の外に置いてください。

## 出力プリセットの選び方

`出力プリセット` は、書き出したデータをどの3DGSアプリに渡すかの選択です。

| プリセット | 使う場面 |
| --- | --- |
| `Postshot` | Postshot向けにキューブマップデータを作る |
| `Brush` | Brush向けにキューブマップデータを作る |
| `LichtFeld Studio` | LichtFeld向けにキューブマップデータ、または3DGUTデータを作る |
| `カスタム` | 座標変換やPLY使用を手動で調整する |

通常は渡し先のアプリ名をそのまま選びます。詳細設定で座標変換、PLY使用、Metashapeインポート設定をプリセット値から変えると、自動的に `カスタム` 扱いになります。

## 出力形状の選び方

`出力形状` は、エクイレクタングラー画像をどうトレーニングデータにするかの選択です。MetashapeルートとSphereSfMルートのどちらでも使います。

### 投影視点に変換

通常はこちらを使います。エクイレクタングラー画像をキューブマップ画像に変換し、`output/` に画像、マスク、`transforms.json` を作ります。Cube6が標準ですが、必要に応じて `Cubemap` タブで書き出す向きを調整できます。

この出力はPostshot / Brush / LichtFeld Studioで扱いやすく、通常のピンホールカメラに近いデータになります。LichtFeldでこのデータをトレーニングするときは、基本的にGUTやUndistortは使いません。

### 3DGUT (LichtFeld)

LichtFeld Studioで3DGUTトレーニングに使うデータを作るモードです。SfMに使ったエクイレクタングラーの `images/` と `masks/` を可能ならハードリンクで `output/` 配下に配置し、キューブマップ画像や変換マスクは作りません。

このモードでは、LichtFeldへ読み込ませるために `output/` へ次のファイルを作ります。

- `transforms.json`
- `pointcloud.ply`

あわせて、このアプリ用の設定記録として `_stechdrive/step4/export_settings.json` を保存します。

Metashapeルートでは `出力プリセット: LichtFeld Studio` と点群PLYの指定が必要です。SphereSfMルートではSfM結果から `pointcloud.ply` を作ります。`3DGUT (LichtFeld)` の選択中は投影視点の調整、画像/マスク出力のON/OFF、COLMAP形式モデル追加出力は無効になります。完成データセットはこの場合も `<scene>/output/` です。

## LichtFeldでキューブマップ版と3DGUT版を使い分ける場合

LichtFeldで「キューブマップデータ」と「3DGUTでエクイレクタングラーを直接読むデータ」の両方を用意したい場合は、同じMetashape結果から2回書き出します。

### キューブマップ版

1. ルートを `Metashape` にします。
2. `出力プリセット` を `LichtFeld Studio` にします。
3. `出力形状` を `投影視点に変換` にします。
4. `点群PLY` を確認します。1つだけの候補が自動入力されていればそのまま使われます。空欄、または候補が違う場合は手動で指定します。
5. `Cubemap` で `Cube6`、Yaw 45°、必要な `画像サイズ` を選びます。
6. 実行します。

出力先は通常 `<scene>/output/` です。LichtFeldにはこの `output/` を読み込ませます。

### 3DGUT版

1. ルートを `Metashape` にします。
2. `出力プリセット` を `LichtFeld Studio` にします。
3. `出力形状` を `3DGUT (LichtFeld)` にします。
4. `点群PLY` を確認します。1つだけの候補が自動入力されていればそのまま使われます。空欄、または候補が違う場合は手動で指定します。
5. 実行します。

この出力では、既存の `<scene>/images/` と `<scene>/masks/` を `<scene>/output/images/` と `<scene>/output/masks/` に配置し、`<scene>/output/transforms.json` と `<scene>/output/pointcloud.ply` を新しく作ります。LichtFeldでは `<scene>/output/` をデータセットとして指定し、トレーニング時にGUTを有効にします。

## 投影視点の調整

`投影視点に変換` は、360度画像からキューブマップ画像を書き出す出力です。標準の `Cube6` では前後左右上下の6方向を作ります。`Cubemap` タブでは、必要に応じて方向数や上下方向の行、書き出す視点のON/OFFを調整できます。

### まずはCube6

通常は `Cube6` から始めます。前後左右上下の6方向を書き出す設定で、Postshot / Brush / LichtFeld Studio向けの標準的な出力として使いやすいです。

`Yaw Offset` は既定の45°が推奨です。補正なしで書き出した2眼360°カメラでは、スティッチ境界がエクイレクタングラー画像の横25%/75%付近に来ることが多く、45°ずらすと境界がキューブ面の中心を横切りにくくなります。

### Custom Grid

6方向では足りない向きを追加したい、上下方向を減らしたい、斜め上/斜め下の視点を足したい場合は `Custom Grid` を使います。

- `Yaw Slots`: 水平方向の分割数です。4から8まで増減できます。
- `Pitch Rows`: 上下方向の行です。`-90..90` の範囲で最大5行まで使えます。
- 各チェックボックス: その視点を書き出すかどうかです。

有効視点数が増えるほど出力枚数と処理時間が増えます。24視点を超えると警告、40視点を超えると実行不可になります。

### 画像サイズ

`画像サイズ` はキューブマップ画像1枚の解像度です。

| 設定 | 使いどころ |
| --- | --- |
| `Full` | 最終品質確認。重いが細部が残りやすい |
| `Normal` | まず試す標準設定。90°視点中央部の角度解像度を元画像に近づける |
| `Half` | 軽量テスト。速い代わりに細部の解像感が落ちる |

VRAMや処理時間が厳しい場合は、まず `Normal` または `Half` で流れを確認し、最終出力では `Full` を使うのが現実的です。

## 画像やマスクだけ作り直したい場合

`出力` の `画像` / `マスク` チェックで、どちらを書き出すかを選べます。

| やりたいこと | 設定 |
| --- | --- |
| 画像とマスクを両方作り直す | `画像` ON, `マスク` ON |
| マスクだけ作り直す | `画像` OFF, `マスク` ON |
| 画像だけ作り直す | `画像` ON, `マスク` OFF |
| カメラ情報だけ更新する | `画像` OFF, `マスク` OFF |

マスクだけ調整したあとに再出力する場合は、`画像` をOFFにすると既存のキューブマップ画像を再変換せずに済みます。`3DGUT (LichtFeld)` では元画像と元マスクをそのまま使うため、この出力ON/OFFは使いません。

## Step 5へ進む

Step 4は、3DGSアプリへ渡すデータセットを作る工程です。LichtFeld StudioやPostshotで使う場合は、Step 4でデータセットを作成してから `Step 5: 学習` を開きます。

Step 5の操作は [Step 5 学習GUI](training_gui.ja.md) を参照してください。

## COLMAPルート

Metashapeを使わず、抽出済みの360°画像からCOLMAP/GLOMAPへ進みたい場合はルートを `COLMAP` にします。

1. `シーンフォルダ` に `images/` と必要なら `masks/` があることを確認します。
2. ルートを `COLMAP` にします。
3. `Cubemap` で視点数、Yaw、画像サイズを決めます。
4. COLMAP/GLOMAPでカメラ位置と疎な点群まで推定したい場合は、左サブ工程の `SfM` をONにします。COLMAP SfMには視点画像が必要なため、`SfM` をONにすると `Cube` もONになります。`Cube` をOFFにすると `SfM` もOFFになります。
5. `Matcher` と `Mapper` を選びます。通常は `Sequential` と `Global` から始めます。
6. 実行します。

COLMAPルートでは、`output/colmap_rig/` にCOLMAP Rig形式のキューブマップ画像、マスク、`rig_config.json` を作ります。左サブ工程の `SfM` がONの場合は、続けてFeature、Rig設定、Matcher、Mapperを実行し、COLMAP/GLOMAPのSfM結果も作ります。`Cube` ON / `SfM` OFF は、COLMAP Rig形式の視点画像だけを書き出します。

COLMAPルートは投影視点のCOLMAP Rigデータ専用です。3DGUT用のエクイレクタングラー出力はMetashapeルートまたはSphereSfMルートで作成します。既存のCOLMAP sparseモデルをトレーニングに渡したい場合は、`SfM` タブの `SfM入力` で選択できます。

フレーム別Yaw回転は、固定リグの前提を崩すためCOLMAP Rig書き出しでは常に0度です。

## SphereSfMルート

SphereSfMルートは、Metashapeを使わずに、抽出済みエクイレクタングラー画像をそのまま球面カメラとしてSfMするルートです。目的は2つあります。

- LichtFeld 3DGUTで使う直接データセットを作る
- Postshot / Brush / LichtFeldで扱いやすいキューブマップデータを作る

通常のCOLMAPではなく、SphereSfM版の `colmap.exe` が必要です。アプリにはSphereSfM本体やバイナリは含まれないため、[json87/SphereSfM](https://github.com/json87/spheresfm) のリリースまたはローカルビルドで用意した実行ファイルを指定します。

RTX 50系GPUでは、GitHubで配布されているSphereSfMのWindowsバイナリはCUDA SIFTで停止することがあります。RTX 50系は新しいCUDAアーキテクチャ `sm_120` 向けの実行コードが必要ですが、配布版バイナリがそれを含まずにビルドされていると、`no kernel image is available for execution on the device` で失敗します。RTX 50系で使う場合は、SphereSfMをRTX 50系対応のCUDA/CMake環境で `CMAKE_CUDA_ARCHITECTURES=120` を指定して自前ビルドし、その `colmap.exe` を指定してください。

1. `シーンフォルダ` に `images/` と、使用する場合は `masks/` があることを確認します。
2. ルートを `SphereSfM` にします。
3. `SphereSfM COLMAP実行ファイル` に、SphereSfM配布版またはビルド済みの `colmap.exe` を指定します。
4. `masks/ を使用` は通常ONにします。Step 3の白=使用、黒=除外マスクをCOLMAPの `image.jpg.png` 命名へ変換して使います。
5. 左サブ工程で今回実行する範囲を選びます。通常は `SfM` と `Cube` をON、SfMだけ作り直す場合は `Cube` をOFF、既存sparseから変換だけやり直す場合は `SfM` をOFFにします。既存sparseを明示したい場合は `SfM` タブの `SfM入力` で選びます。
6. `Matcher` は動画フレームなら `Sequential` から始めます。POSファイルがある場合だけ `Spatial` を使います。
7. `SfM品質` はまず `標準`、試行や大量フレームでは `軽量`、登録が弱い場合は `クオリティ` を試します。
8. `Cubemap` タブで `出力形状` を選びます。
9. 実行します。

`SfM` ON / `Cube` OFF は、`<scene>/output/spheresfm/sparse/` のSfM結果だけを作ります。3DGSアプリへ渡すデータセットはまだ作られません。`SfM` OFF / `Cube` ON は、`SfM入力` で選んだ既存sparse結果を再利用して、3DGUT/キューブマップの出力だけを作り直すときに使います。

SphereSfM実行の開始時に、GUIは元画像を1枚だけ `<scene>/output/spheresfm/preflight/` へコピーし、フルのdatabaseを作る前に小さなGPU SIFT確認を自動実行します。選択したバイナリが現在のGPUでCUDA SIFTを実行できない場合はそこで停止し、ログへのリンクと原因の候補を表示します。

`出力形状` が `3DGUT (LichtFeld)` の場合は、既存の `<scene>/images/` と `<scene>/masks/` を可能ならハードリンクで `<scene>/output/images/` と `<scene>/output/masks/` に配置し、`<scene>/output/transforms.json` と `<scene>/output/pointcloud.ply` を作ります。LichtFeldで使う3DGUTデータセットは `<scene>/output/` です。`output/` 内に既存の3DGUTデータセットファイルがある場合は、上書き前に確認します。

`出力形状` が `投影視点に変換` の場合は、`<scene>/output/` が下流アプリへ渡すキューブマップデータセットになります。`Cubemap` タブの投影視点設定と画像/マスク出力のON/OFFを使い、Metashapeルートと同じ形で `<scene>/output/images/`、`<scene>/output/masks/`、`<scene>/output/transforms.json`、`<scene>/output/pointcloud.ply` を作ります。

SphereSfMプロジェクト `<scene>/output/spheresfm/` には、作業用の `preflight/`、`database.db`、`masks_colmap/`、`sparse/`、`equirect/`、`logs/`、`stechdrive_spheresfm_project.json` を作ります。ここはSfMの再利用やログ確認用で、3DGUTでもキューブマップでも実際に持ち出すデータセットは `output/` です。

実行後は `結果をCOLMAP GUIで表示` で、登録されたカメラ位置と疎点群を確認できます。GUIなしでビルドされたSphereSfM版COLMAPではこの表示だけ使えませんが、SfMや変換結果の出力自体とは別です。

## 実行後にできるもの

| ルート | 主な出力 |
| --- | --- |
| Metashape + キューブマップ変換 (`投影視点に変換`) | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json` |
| Metashape + `3DGUT (LichtFeld)` | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/pointcloud.ply` |
| COLMAP | `<scene>/output/colmap_rig/images/`, `<scene>/output/colmap_rig/masks/`, `<scene>/output/colmap_rig/rig_config.json` |
| COLMAP実行あり | 上記に加えて、COLMAP/GLOMAPのSfM結果 |
| SphereSfM + `3DGUT (LichtFeld)` | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/pointcloud.ply` |
| SphereSfM + キューブマップ変換 (`投影視点に変換`) | 下流アプリへ渡すのは `<scene>/output/`。`images/`, `masks/`, `transforms.json`, `pointcloud.ply` を作ります |

Step 4では `<scene>/output/` を、他PCへコピーしたり3DGSアプリへ直接読み込ませたりする現在のポータブルな正本データセットとして扱います。アプリ内で再現するための状態は別に保存します。Step 4の設定記録は `<scene>/_stechdrive/step4/export_settings.json` に保存し、キューブマップを書き出すルートでは視点構成も `<scene>/_stechdrive/step4/views_config.json` に保存します。これらはこのアプリで再開・再現するためのメタデータで、3DGSアプリへ渡すデータセット本体ではありません。

`LichtFeld Studio` プロファイルでは、最終出力の `transforms.json` と `pointcloud.ply` に同じ向き補正を適用し、LichtFeld上でMetashapeと同じ +X / +Z / 上下方向になるようにします。

## よくある判断

- Postshot / Brushへ渡すなら、まず `投影視点に変換` を使います。
- LichtFeldで通常トレーニングするなら、まず `LichtFeld Studio` + `投影視点に変換` を使います。
- LichtFeldでGUTを使うなら、`LichtFeld Studio` + `3DGUT (LichtFeld)` を使います。
- キューブマップデータをLichtFeldでトレーニングするときは、基本的にGUTもUndistortも不要です。
- `3DGUT (LichtFeld)` でトレーニングするときは、LichtFeld側でGUTを有効にします。
- スティッチが目立たない素材では、スティッチマスクはOFFまたは細めから試します。Yaw 45°は画素を捨てない対策なので、通常は維持して構いません。
- Metashapeルートで `点群PLY` が必要なプロファイルなのに使用可能なPLYが選択されていない場合は、実行前にエラーで止まります。
