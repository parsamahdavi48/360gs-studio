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
| Metashapeを使わず、抽出済み360°画像からCOLMAPへ進みたい | `COLMAP` | `COLMAP実行設定`, `Cubemap` |
| Metashapeを使わず、抽出済みエクイレクタングラー画像を直接SfMしたい | `SphereSfM` | `SphereSfM COLMAP実行ファイル`, `SfM入力`, `Matcher`, `SfM品質`, `出力形状` |

作業を再開する場合は、先にヘッダーで対象のシーンフォルダを読み込みます。そのうえで、使うルートと出力形状を選びます。

左ナビゲーションの `SfM` はカメラ位置を用意する工程、`Cube` は3DGSアプリへ渡すデータセットを書き出す工程です。Metashape結果を変換するだけなら、通常は `Cube` を実行します。COLMAPやSphereSfMでカメラ位置の推定から行う場合は `SfM` も実行します。既存のSfM結果から出力だけ作り直す場合は `SfM` をOFF、`Cube` をONにします。警告アイコンが出ている場合は、その行から不足している設定へ移動して確認します。

## Metashapeルートの基本操作

Metashapeで360°画像をアライメント済みなら、基本はこの流れです。

1. ルートを `Metashape` にします。
2. `カメラXML` を確認します。候補が自動入力されていない場合や別のXMLを使いたい場合は、Metashapeから書き出したカメラXMLを手動で指定します。
3. LichtFeld Studio向け、または点群も同梱したい場合は `点群PLY` を確認します。候補が1つだけなら自動入力されます。候補が複数ある場合は、使うPLYを手動で選びます。
4. `出力プリセット` で渡し先を選びます。
5. `出力形状` で、キューブマップ画像へ変換するか、`3DGUT (LichtFeld)` にするかを選びます。
6. キューブマップ画像を作る場合は、`Cubemap` タブでCube6、Yaw、画像サイズを確認します。
7. `実行` します。

Metashapeルートでは、Metashapeのカメラ情報を3DGS用のカメラデータに変換します。`出力形状` が `投影視点に変換` の場合は、あわせてキューブマップ画像とマスクを書き出します。

### カメラXMLと点群PLYの選び方

Metashapeルートでは、Metashapeから書き出したカメラXMLを指定します。シーンフォルダに使えそうなXMLが1つだけ見つかった場合は自動入力されますが、候補が複数ある場合や想定と違う場合は手動で選びます。

- `カメラXML`: Metashapeでアライメントしたカメラを書き出したXMLを指定します。
- `点群PLY`: LichtFeld向けや点群同梱が必要な場合に、Metashapeから書き出したPLYを指定します。候補が複数ある場合は、使いたい点群を手動で選びます。

Metashapeから書き出した元のXML/PLYは、Step 4の出力フォルダとは別の場所に置いておくと選びやすくなります。

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

LichtFeld Studioで3DGUTトレーニングに使うデータを作るモードです。SfMに使ったエクイレクタングラーの `images/` と `masks/` を `output/` 配下に配置し、キューブマップ画像や変換マスクは作りません。

このモードでは、LichtFeldへ読み込ませるために `output/` へ次のファイルを作ります。

- `transforms.json`
- `pointcloud.ply`

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

この出力では、既存の `<scene>/images/` と `<scene>/masks/` を使い、`<scene>/output/transforms.json` と `<scene>/output/pointcloud.ply` を新しく作ります。LichtFeldでは `<scene>/output/` をデータセットとして指定し、トレーニング時にGUTを有効にします。

## 投影視点の調整

`投影視点に変換` は、360度画像からキューブマップ画像を書き出す出力です。標準の `Cube6` では前後左右上下の6方向を作ります。`Cubemap` タブでは、必要に応じて方向数や上下方向の行、書き出す視点のON/OFFを調整できます。

### まずはCube6

通常は `Cube6` から始めます。前後左右上下の6方向を書き出す設定で、Postshot / Brush / LichtFeld Studio向けの標準的な出力として使いやすいです。生成ファイル名の面サフィックスは、通常のCubemap軸名に合わせて `px` / `nx` / `py` / `ny` / `pz` / `nz` を使います。

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
| `Normal` | 既定の標準設定。90°視点中央部の角度解像度を元画像に近づける |
| `Half` | 軽量テスト。速い代わりに細部の解像感が落ちる |

通常は既定の `Normal` から始め、VRAMや処理時間が厳しい場合は `Half`、細部を優先したい最終確認では `Full` を選びます。

## 画像やマスクだけ作り直したい場合

`出力` の `画像` / `マスク` チェックで、どちらを書き出すかを選べます。

| やりたいこと | 設定 |
| --- | --- |
| 画像とマスクを両方作り直す | `画像` ON, `マスク` ON |
| マスクだけ作り直す | `画像` OFF, `マスク` ON |
| 画像だけ作り直す | `画像` ON, `マスク` OFF |
| カメラ情報だけ更新する | `画像` OFF, `マスク` OFF |

マスクだけ調整したあとに再出力する場合は、`画像` をOFFにすると既存のキューブマップ画像を再変換せずに済みます。`3DGUT (LichtFeld)` では元画像と元マスクをそのまま使うため、この出力ON/OFFは使いません。

## AprilTagでスケールを推定する

AprilTagを使う場合は、撮影前にタグを印刷して現場へ配置しておきます。`スケール` タブの折りたたみ `タグPDF` から、タグファミリ、タグID、実寸、用紙サイズを指定して印刷用PDFを作成できます。印刷時は100%/実寸で出力してください。プリンタ側で拡大縮小されると、推定に使うタグ実寸が変わります。

複数地点にタグを置く場合は、場所ごとに別のタグIDを使います。同じIDを複数の場所に置くと、どのタグを見たのか区別できず、推定が壊れます。タグは撮影中に動かない場所へ固定し、できるだけ複数のフレームから見えるように配置してください。

撮影後の手順は次の通りです。

1. 通常どおりStep 4でCubemap出力を作成します。スケール推定には投影済みの `output/transforms.json` と `output/` 配下の画像が必要です。3DGUT向けのエクイレクタングラー出力のままでは推定できません。
2. `スケール` タブを開きます。
3. 印刷したタグの実寸、タグファミリ、撮影に使ったタグIDを入力します。別のタグを印刷していない限り、既定の `tag36h11 / ID 7` を使います。`変換プリセット` は通常 `自動` のままにします。別の場所から持ち込んだCube6で推定が崩れる時だけ、変換時に選んだ `LichtFeld`、`Postshot`、`Brush` などを選びます。
4. `推定` を押します。処理中は下部ログと進捗バーに検出状況が表示されます。この時点ではファイルを書き換えず、推定scaleと観測数などの統計だけを表示します。
5. 結果が妥当な場合だけ `Scaleへ反映` を押します。現在のファイルを `output/apriltag_scale_backup_日時/` にバックアップし、`output/transforms.json` のカメラ位置と、存在する場合は `output/pointcloud.ply` の点群座標へ同じscaleを掛けます。

## Step 5へ進む

Step 4は、3DGSアプリへ渡すデータセットを作る工程です。LichtFeld StudioやPostshotで使う場合は、Step 4でデータセットを作成してから `Step 5: 学習` を開きます。

Step 5の操作は [Step 5 学習GUI](training_gui.ja.md) を参照してください。

## COLMAPルート

Metashapeを使わず、抽出済みの360°画像からCOLMAPへ進みたい場合はルートを `COLMAP` にします。

1. `シーンフォルダ` に `images/` と必要なら `masks/` があることを確認します。
2. ルートを `COLMAP` にします。
3. `Cubemap` で視点数、Yaw、画像サイズを決めます。
4. COLMAPでカメラ位置と疎な点群まで推定したい場合は、左サブ工程の `SfM` をONにします。COLMAP SfMには視点画像が必要なため、`SfM` をONにすると `Cube` もONになります。`Cube` をOFFにすると `SfM` もOFFになります。
5. `Matcher` と `Mapper` を選びます。通常は `Sequential` と `Global` から始めます。
6. 実行します。

COLMAPルートでは、`output/colmap_rig/` にCOLMAP Rig形式のキューブマップ画像、マスク、`rig_config.json` を作ります。左サブ工程の `SfM` がONの場合は、続けてCOLMAPでカメラ位置と疎な点群も推定します。`Cube` ON / `SfM` OFF は、COLMAP Rig形式の視点画像だけを書き出します。

COLMAPルートは投影視点のCOLMAP Rigデータ専用です。3DGUT用のエクイレクタングラー出力はMetashapeルートまたはSphereSfMルートで作成します。既存のCOLMAP結果をトレーニングに渡したい場合は、`SfM` タブの `SfM入力` で選択できます。

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
4. `masks/ を使用` は通常ONにします。Step 3で作った白=使用、黒=除外のマスクを、SphereSfMで使える形にしてSfMへ渡します。
5. 左サブ工程で今回実行する範囲を選びます。通常は `SfM` と `Cube` をON、SfMだけ作り直す場合は `Cube` をOFF、既存のSfM結果から変換だけやり直す場合は `SfM` をOFFにします。使うSfM結果を明示したい場合は `SfM` タブの `SfM入力` で選びます。
6. `Matcher` は動画フレームなら `Sequential` から始めます。POSファイルがある場合だけ `Spatial` を使います。
7. `SfM品質` はまず `標準`、試行や大量フレームでは `軽量`、登録が弱い場合は `クオリティ` を試します。
8. `Cubemap` タブで `出力形状` を選びます。
9. 実行します。

`SfM` ON / `Cube` OFF は、カメラ位置推定だけを実行します。3DGSアプリへ渡すデータセットはまだ作られません。`SfM` OFF / `Cube` ON は、`SfM入力` で選んだ既存結果を再利用して、3DGUT/キューブマップの出力だけを作り直すときに使います。

SphereSfM実行の開始時には、本処理の前に小さなGPU SIFT確認を行います。選択したSphereSfMが現在のGPUでCUDA SIFTを実行できない場合はそこで停止し、ログへのリンクと原因の候補を表示します。

`出力形状` が `3DGUT (LichtFeld)` の場合は、既存の `<scene>/images/` と `<scene>/masks/` を使い、`<scene>/output/transforms.json` と `<scene>/output/pointcloud.ply` を作ります。LichtFeldで使う3DGUTデータセットは `<scene>/output/` です。`output/` 内に既存の3DGUTデータセットファイルがある場合は、上書き前に確認します。

`出力形状` が `投影視点に変換` の場合は、`<scene>/output/` が下流アプリへ渡すキューブマップデータセットになります。`Cubemap` タブの投影視点設定と画像/マスク出力のON/OFFを使い、Metashapeルートと同じ形で `<scene>/output/images/`、`<scene>/output/masks/`、`<scene>/output/transforms.json`、`<scene>/output/pointcloud.ply` を作ります。

SphereSfMの作業ファイルとログは `<scene>/output/spheresfm/` にまとまります。3DGUTでもキューブマップでも、実際に下流アプリへ渡すデータセットは `output/` です。

実行後は `結果をCOLMAP GUIで表示` で、登録されたカメラ位置と疎点群を確認できます。GUIなしでビルドされたSphereSfM版COLMAPではこの表示だけ使えませんが、SfMや変換結果の出力自体とは別です。

## 実行後にできるもの

| ルート | 主な出力 |
| --- | --- |
| Metashape + キューブマップ変換 (`投影視点に変換`) | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json` |
| Metashape + `3DGUT (LichtFeld)` | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/pointcloud.ply` |
| COLMAP | `<scene>/output/colmap_rig/images/`, `<scene>/output/colmap_rig/masks/`, `<scene>/output/colmap_rig/rig_config.json` |
| COLMAP実行あり | 上記に加えて、COLMAPのSfM結果 |
| SphereSfM + `3DGUT (LichtFeld)` | `<scene>/output/images/`, `<scene>/output/masks/`, `<scene>/output/transforms.json`, `<scene>/output/pointcloud.ply` |
| SphereSfM + キューブマップ変換 (`投影視点に変換`) | 下流アプリへ渡すのは `<scene>/output/`。`images/`, `masks/`, `transforms.json`, `pointcloud.ply` を作ります |

Step 4では `<scene>/output/` を、他PCへコピーしたり3DGSアプリへ直接読み込ませたりする現在のデータセットとして扱います。3DGSアプリへ渡す場合は、基本的にこの `output/` を指定します。

`LichtFeld Studio` プロファイルでは、最終出力の `transforms.json` と `pointcloud.ply` に同じ向き補正を適用し、LichtFeld上でMetashapeと同じ +X / +Z / 上下方向になるようにします。

## よくある判断

- Postshot / Brushへ渡すなら、まず `投影視点に変換` を使います。
- LichtFeldで通常トレーニングするなら、まず `LichtFeld Studio` + `投影視点に変換` を使います。
- LichtFeldでGUTを使うなら、`LichtFeld Studio` + `3DGUT (LichtFeld)` を使います。
- キューブマップデータをLichtFeldでトレーニングするときは、基本的にGUTもUndistortも不要です。
- `3DGUT (LichtFeld)` でトレーニングするときは、LichtFeld側でGUTを有効にします。
- スティッチが目立たない素材では、スティッチマスクはOFFまたは細めから試します。Yaw 45°は画素を捨てない対策なので、通常は維持して構いません。
- Metashapeルートで `点群PLY` が必要なプロファイルなのに使用可能なPLYが選択されていない場合は、実行前にエラーで止まります。
