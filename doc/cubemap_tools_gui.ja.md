# Step 4 SfM / Step 5 データセットGUI

Step 4は、学習データセットの元になるカメラポーズと疎点群をどう用意するかを選ぶ画面です。すでにMetashape、RealityScan、COLMAP、SphereSfMなどでSfM済みなら、ここで追加作業をする必要はありません。これからこのアプリでCOLMAPやSphereSfMを実行したい場合、またはMetashape結果からRealityScan再アライン用データを作りたい場合だけ、対応するカードを開きます。

Step 5は、SfM結果を学習アプリで読み込めるデータセットへ変換する画面です。MetashapeやSphereSfMの結果からNeRF系JSON/PLYを作る、MetashapeやRealityScanの結果からCOLMAP形式データセットを作る、AprilTagでスケールを反映する、といった作業をここで行います。

## 関連ツール

| ツール | このアプリでの用途 |
| --- | --- |
| [COLMAP](https://github.com/colmap/colmap) | Step 4のCOLMAPルートとCOLMAP形式データセット出力 |
| [SphereSfM](https://github.com/json87/SphereSfM) | Step 4の球面画像SfMルート |
| [LichtFeld Studio](https://lichtfeld.io/) | LichtFeldプリセット、GUT出力、RealityScanからCOLMAPデータセットへの変換先 |
| [Postshot](https://www.jawset.com/) | PostshotプリセットとStep 6のCLI起動 |
| [Brush](https://github.com/ArthurBrussee/brush) | オープンソースのGaussian Splatting学習向けCubemap出力 |

## まず決めること

最初に決めるのは、「カメラポーズはもうあるか」です。

| 状態 | 進み方 |
| --- | --- |
| MetashapeでSfM済み | Step 4は `既存のSfM結果を使う`。Step 5でMetashape系カードを選ぶ |
| RealityScanで再アライン済み | Step 4は `既存のSfM結果を使う`。Step 5で `RealityScan → COLMAPデータセット` を選ぶ |
| COLMAPのimages/masks/sparseがすでにある | Step 4は `既存のSfM結果を使う`。COLMAP対応アプリならそのまま学習へ進む |
| このアプリからCOLMAPでSfMしたい | Step 4で `COLMAPでSfMを実行` |
| このアプリからSphereSfMでSfMしたい | Step 4で `SphereSfMでSfMを実行` |
| Metashape結果をRealityScanで再アラインしたい | Step 4で `Metashape → RealityScan用データ作成` |
| 作成済み結果を確認したい | Step 4で `SfM結果を確認` |

## Step 4: SfMカードの選び方

### 既存のSfM結果を使う

Metashape、RealityScan、COLMAP、SphereSfMなどで、すでにカメラポーズと疎点群を作ってある場合に選びます。このカードは「何もしないで次へ進む」ための選択肢です。次のStep 5で、その結果をどのデータセット形式へ変換するかを選びます。

Metashape結果を使う場合は、カメラをAgisoft XML、疎点群をStanford PLYとしてエクスポートします。どちらもシーンフォルダに保存しておくと、SfM結果をシーンと一緒に管理しやすくなります。別の場所に保存した場合は、Step 5のカードでXMLとPLYを手動選択してください。

### COLMAPでSfMを実行

Metashapeを使わず、このアプリから[COLMAP](https://github.com/colmap/colmap)またはGLOMAPでSfMしたい場合に選びます。

- 360°画像はCubemap Rigへ展開します
- 通常画像や通常動画フレームは通常カメラとして扱います
- 混在ソースを一つのCOLMAPデータセットとして処理できます
- 出力は `output/colmap_rig/` です

通常は、動画順に撮った素材なら `Sequential` から始めます。写真枚数が少なく、順序より全体照合を優先したい場合は `Exhaustive` を検討します。

通常画像のカメラモデルは基本的に自動推定から始めます。焦点距離や歪みモデルを明示したい場合だけ、通常画像カメラ設定で対象グループを選んで調整します。

### SphereSfMでSfMを実行

エクイレクタングラー360°画像を球面カメラとしてSfMしたい場合に選びます。[SphereSfM](https://github.com/json87/SphereSfM)は同一解像度のERP 360°画像だけを入力にするのが安全です。通常画像や複数解像度ERPを混ぜたい場合はCOLMAPまたはMetashapeを使ってください。

通常のCOLMAPではなく、SphereSfM版の `colmap.exe` が必要です。RTX 50系GPUでは配布バイナリがCUDA SIFTで止まる場合があるため、その場合はRTX 50系に対応したビルドを指定します。

出力されるSfM作業フォルダは `output/spheresfm/` です。学習アプリへ渡すJSON/PLYやCubemapデータは、Step 5で `SphereSfM → NeRFデータセット(JSON/PLY)` を実行して作ります。

### Metashape → RealityScan用データ作成

Metashapeで作ったカメラXMLから、RealityScanへ読み込ませるCubemap画像とXMPを作ります。RealityScanで再アラインしたい、ステップ1から3で登録済みの別ソース画像も一緒に投入したい、RealityScanのCSV/PLYを書き出して後段へ渡したい場合に使います。

出力は `output/realityscan/` です。Metashape XMLにある画像はCubemap画像とXMPとして `images/_geometry/` に書き出され、マスクは `images/_mask/` に入ります。XMLにない登録済み画像は姿勢なしの追加画像として `extra_images/_geometry/` へ可能ならハードリンク、必要ならコピーされ、対応マスクは `extra_images/_mask/` に入ります。RealityScanでは先に `images/` を追加してAlignし、点群まで生成してから `extra_images/` を追加して再度Alignします。その後、CSVとPLYを書き出し、LichtFeld用COLMAPデータセットが必要な場合はStep 5の `RealityScan → COLMAPデータセット` を使います。

RealityScanへ段階的に投入するのは意図した使い方です。先にCubemap画像だけでMetashape由来の安定したコンポーネントを作り、その後で通常画像を追加するほうが、最初から全画像をまとめて入れるより誤配置や小さな別コンポーネントが起きにくくなります。

### SfM結果を確認

点群、カメラ位置、選択カメラ画像、対応マスクを読み取り専用ビューで確認します。SfM結果が意図した位置関係になっているか、画像とマスクの対応が壊れていないかを見るためのカードです。

## Step 5: データセットカードの選び方

### Metashape → NeRFデータセット(JSON/PLY)

MetashapeのカメラXMLと点群PLYから、NeRF/3DGS系データセットを作ります。PINHOLE出力では `transforms_postshot.json` と `pointcloud_postshot.ply` のように、出力プリセット別のJSON/PLYを書き出します。

| 選択 | 使う場面 |
| --- | --- |
| `PINHOLE` | 360°画像をCubemapへ展開して、Postshot / Brush / LichtFeldなどで扱いやすいデータにする |
| `ERP 360°` | LichtFeldでGUTを使い、エクイレクタングラー画像を直接使う |

通常は `PINHOLE` から始めます。通常画像や複数解像度ERPが混在するMetashape結果では、ERP 360°のまま安全に出力できないため、`PINHOLE` を使ってください。[LichtFeld Studio](https://lichtfeld.io/)のJSON/PLY読み込みはフレームごとのカメラ内部パラメータを扱えないため、複数カメラ設定の混在結果では `Metashape → COLMAPデータセット` のほうが安全です。

出力先は主に次の通りです。

| 出力 | フォルダ |
| --- | --- |
| PINHOLE Cubemap | `output/metashape_cubemap/` |
| ERP 360° / GUT | `output/metashape_3dgut/` |

### Metashape → COLMAPデータセット

MetashapeのカメラXMLと点群PLYから、`images/`, `masks/`, `sparse/0/` を持つCOLMAP形式データセットを作ります。COLMAP入力に対応した学習ソフトへ渡したい場合、またはMetashapeで360°画像と通常画像を混在SfMした結果を安全に使いたい場合に選びます。

- ERP 360°カメラは選択した視点セットへ展開します
- PINHOLEの通常画像はCubemap化せず参照します
- 歪み係数を持つ通常画像はPINHOLEへ補正し、対応マスクも同じ変換をかけます
- 出力は `output/metashape_colmap/` です

LichtFeldでMetashape混在結果を使う場合は、このルートが安全です。

### RealityScan → COLMAPデータセット

RealityScanのRegistrationから書き出したInternal/External CSVと、同じ座標状態で書き出したPLYから、LichtFeldでDatasetとして開けるCOLMAPデータセットを作ります。

通常は `output/realityscan/` 配下にCSV、PLY、`images/`、必要に応じて `extra_images/` がある状態で使います。各 `_geometry` layerから画像を、対応する `_mask` layerからマスクを読み取り、CSVに載っている画像を出力先の `images/` と `masks/` に統合します。出力先は `output/realityscan/lfs_colmap/` です。

RealityScanからエクスポートする前に、学習に使うコンポーネントを確認してください。COLMAP側のカメラ姿勢として使われるのはCSVに含まれるカメラだけです。画像ファイルが存在していても、CSVにない画像には姿勢は付きません。

`レンズ補正してPINHOLE化` は、RealityScanで通常画像も混ぜてアラインし、LichtFeldが歪みつきカメラを受け付けず止まる場合に使います。Cubemap由来のPINHOLE画像は出力先へリンクまたはコピーし、歪み係数を持つ通常画像だけを補正します。補正で生じる無効領域はマスクにも反映されます。

### SphereSfM → NeRFデータセット(JSON/PLY)

Step 4で作ったSphereSfM sparse、または別途指定したSphereSfM sparseから、JSON/PLYデータセットを作ります。

SphereSfM入力は同一解像度のERP 360°画像に限定するのが安全です。出力は、LichtFeldでGUTを使うERP 360°データ、またはPostshot / Brush / LichtFeldで扱いやすいPINHOLE Cubemapデータから選びます。

### スケール調整

AprilTagの実寸から、作成済みデータセットのスケールを補正します。撮影前にタグを印刷し、現場に固定しておく必要があります。

1. Step 5でCubemapまたはCOLMAP系データセットを作ります。
2. `スケール調整` を開き、対象データセット、タグファミリ、タグID、実寸を確認します。
3. `推定` を実行し、検出数や推定値を確認します。
4. 結果が妥当な場合だけ反映します。

スケール反映は対象データセットのカメラ位置と点群に同じ倍率をかけます。反映前にはバックアップを作ります。

## 出力設定の選び方

### 画像タイプ

`PINHOLE` は、ERP 360°画像をCubemapなどの通常視点画像へ展開する出力です。Postshot、Brush、LichtFeldの通常学習ではまずこれを使います。

`ERP 360°` は、LichtFeldでGUTを使ってエクイレクタングラー画像を直接学習する場合だけ選びます。LichtFeld以外のプリセットでは選べません。

### 視点セット

`Cubemap` は前後左右上下の6方向を出力する標準設定です。迷ったらこれを使います。

`Custom Grid` は、6方向では足りない場合に視点方向を増やす設定です。視点数を増やすほど出力枚数、処理時間、学習時の画像数が増えます。

### 画像サイズ

通常は既定値から始めます。軽く試す場合は小さめ、最終品質を見たい場合は大きめを選びます。元画像の情報量や学習アプリのVRAM消費も合わせて判断します。

## シーンプレビュー

Step 4の `SfM結果を確認` から、点群、カメラ位置、画像、マスクを同じ画面で確認できます。

プレビューでは、出力済みデータセット、Metashape XML/PLY、COLMAP sparse、SphereSfM sparseなどを候補として選べます。カメラをクリックすると、そのカメラ画像と対応マスクを確認できます。

## よくある判断

- MetashapeでSfM済みなら、Step 4は `既存のSfM結果を使う` で十分です。
- Postshot / Brush / LichtFeldの通常学習へ渡すなら、まず `PINHOLE` + `Cubemap` を使います。
- LichtFeldでGUTを試すなら、`ERP 360°` を選び、学習時にGUTをONにします。
- Metashape結果に通常画像や複数解像度ERPが混ざるなら、LichtFeld向けには `Metashape → COLMAPデータセット` が安全です。
- RealityScanからPostshotへ渡すだけならCSV/PLYで足りる場合があります。LichtFeldでDatasetとして読みたい場合は `RealityScan → COLMAPデータセット` を使います。
- Metashape → RealityScan → LichtFeldのルートでは、Metashape XML/PLYとRealityScan CSV/PLYをシーンと一緒に管理し、最終的に `output/realityscan/lfs_colmap/` を学習に使います。
- SphereSfMは同一解像度ERP 360°専用と考えてください。混在ソースはCOLMAPまたはMetashapeを使います。
- 画像やマスクだけを作り直したい場合は、同じカードを開き、出力設定を確認して再実行します。

## Step 6へ進む

Step 5でデータセットを作ったら、LichtFeld Studio、Postshot、Brushなどの学習アプリへ直接読み込ませます。対応CLIで再実行やヘッドレス学習をしたい場合だけ、Step 6を使います。

Step 6の操作は [Step 6 学習GUI](training_gui.ja.md) を参照してください。
