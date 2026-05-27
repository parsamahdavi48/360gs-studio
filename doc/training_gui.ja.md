# Step 6 学習GUI

Step 6は、Step 5で作成した3DGS用データセットを使って、対応CLIを持つ学習アプリを起動する画面です。対応バージョンの[LichtFeld Studio](https://lichtfeld.io/)、[Postshot](https://www.jawset.com/)、[Brush](https://github.com/ArthurBrussee/brush)、[gsplat](https://github.com/nerfstudio-project/gsplat)を使うと、同じ設定の再実行やヘッドレス学習をGUIから始められます。

学習アプリ側で画質やモデル設定を確認しながら進めたい場合は、Step 6を使わず、Step 5の出力データセットを[LichtFeld Studio](https://lichtfeld.io/)、[Postshot](https://www.jawset.com/)、[Brush](https://github.com/ArthurBrussee/brush)などで直接読み込んで学習できます。画像変換やSfMはStep 6では行いません。データセットを作る作業は `Step 5: データセット` です。

## 学習アプリで使う

Step 5の出力は、下流の3DGSアプリに渡すためのデータセットです。まず学習アプリのGUIで結果を見ながら調整したい場合は、該当するデータセットフォルダを直接読み込みます。設定を固めたあとに同じ条件で回したい場合や、画面を開かずに学習だけ走らせたい場合は、Step 6のCLI起動を使います。

| 進め方 | 向いている場面 |
| --- | --- |
| 学習アプリで直接読み込む | 初回確認、見た目を確認しながらの調整、学習アプリ固有の設定を細かく使う場合 |
| Step 6からCLI起動する | 対応CLIで同じ条件を再実行したい場合、ヘッドレスで学習を走らせたい場合 |

Step 6のCLI実行は、LichtFeld Studio v0.5.2互換CLI、Postshot v1.0/v1.1 Release BuildのCLI、Brush CLI、gsplat `examples/simple_trainer.py` を目安にしています。CLIを使わない場合でも、Step 5の出力データセットは各学習アプリで直接使えます。

## まず決めること

Step 6を開いたら、最初に「どのアプリで、どのデータを試すか」を決めます。

| やりたいこと | 実行アプリ | 主に確認する設定 |
| --- | --- | --- |
| LichtFeldで通常のCubemapデータを学習したい | `LichtFeld Studio` | `入力データ`, `GUT` OFF, `出力PLY名`, `Strategy`, `Iterations` |
| LichtFeldでERP 360° / GUTデータを試したい | `LichtFeld Studio` | `入力データ`, `GUT` ON, `pointcloud.ply` があること |
| Postshotでプロジェクトを作りたい | `Postshot` | `入力データ`, `Camera Poses`, `プロジェクト名`, `Profile` |
| Brushをヘッドレスで実行したい | `Brush` | `入力データ`, `出力PLY名`, `Iterations`, `Max Resolution` |
| gsplatでCOLMAPデータを学習したい | `gsplat` | `COLMAP形式の入力データ`, `simple_trainer.py`, `Strategy`, `Max Steps`, `3DGUT` |

`入力データ` は通常、自動設定のままで使います。登録済みのデータセット成果物があれば、Metashape、RealityScan、SphereSfM、COLMAPなどの最新データセットを `<scene>/output/` 配下から使います。`出力先` は既定で `<scene>/output/` です。データセットと学習結果を同じ `output/` 配下に置くことで、後から持ち出しやすくしています。

## 基本操作

1. Step 5でデータセットを作成します。
2. 学習アプリのGUIで調整したい場合は、Step 5の出力データセットを直接読み込みます。
3. CLIで再実行またはヘッドレス実行したい場合は、`Step 6: 学習` を開きます。
4. `入力データ` と `出力先` が意図したフォルダになっているか確認します。
5. `LichtFeld Studio`、`Postshot`、`Brush`、`gsplat` のカードを選びます。
6. `実行ファイル` が空欄で自動検出できない場合は、インストール先のexeを指定します。
7. 右側のアプリ別設定を確認します。
8. `起動` を押します。

Step 6は、選択した学習方式とデータセットの形が合っているかを実行前に確認します。たとえばLichtFeldの `GUT` はERP 360°画像を使うGUT用データ、通常のLichtFeldとPostshotはPINHOLEのCubemapデータを前提にします。

## 画面構成

Step 6では中央パネルを広く使うため、左右2カラムに整理しています。

| 場所 | 内容 |
| --- | --- |
| 左側 | 実行アプリ、ヘッドレス実行、実行ファイル、入力データ、出力先 |
| 右側 | 選択した学習アプリごとの設定 |

右側の詳細パラメーターは、普段触る項目と、必要なときだけ開く詳細項目に分けています。まずは折りたたまれていない項目だけで実行し、結果を見てから詳細を調整する想定です。

## 共通設定

### 実行アプリ

`LichtFeld Studio`、`Postshot`、`Brush`、`gsplat` を選べます。任意CLIのコマンドを組み立てたい場合は、この画面ではなく各CLIを直接実行してください。

### 実行ファイル

空欄の場合は既定名や既知の場所を探します。見つからない場合は、次のような実行ファイルを指定します。

| アプリ | 例 |
| --- | --- |
| LichtFeld Studio | `LichtFeld-Studio.exe` |
| Postshot | `postshot-cli.exe` |
| Brush | `brush.exe` |
| gsplat | `python.exe` と `examples/simple_trainer.py` |

### 入力データ

学習アプリに渡すデータセットフォルダです。通常はStep 5で登録された最新データセットを使い、まだ成果物記録がない場合は現在のStep 5ルートと出力形状から自動設定されます。

| Step 5の結果 | Step 6の入力データ |
| --- | --- |
| Metashape + PINHOLE Cubemap | `<scene>/output/metashape_cubemap/` |
| Metashape + ERP 360° / GUT | `<scene>/output/metashape_3dgut/` |
| SphereSfM + PINHOLE Cubemap | `<scene>/output/spheresfm_cubemap/` |
| SphereSfM + ERP 360° / GUT | `<scene>/output/spheresfm_3dgut/` |
| COLMAP Rig | `<scene>/output/colmap_rig/` |
| RealityScan -> COLMAPデータセット | `<scene>/output/realityscan/lfs_colmap/` |
| Metashape -> COLMAPデータセット | `<scene>/output/metashape_colmap/` |

手動で別フォルダを指定することもできます。その場合は、`images/`、必要なら `masks/`、カメラポーズ、点群など、選んだ学習アプリが必要とするファイルがそのフォルダにあることを確認します。

### 出力先

学習結果やPostshotプロジェクトを書き込むフォルダです。既定では `<scene>/output/` です。

LichtFeldの最終PLY、Postshotの `.psht`、Brushの最終PLY、gsplatの結果フォルダ、Postshotの任意書き出しPLY/SPZが既に存在する場合、Step 6は上書きを避けるため実行前に止まります。出力名または出力先を変えてから再実行します。

## LichtFeld Studio

LichtFeld Studioでは、データセット、出力先、学習設定を指定して学習を開始します。

Step 6からCLI起動する場合は、v0.5.2互換のLichtFeld Studio CLIを目安にします。学習設定をLichtFeld Studio側で確認しながら進めたい場合は、Step 5で作成したCubemapデータまたはERP 360° / GUTデータをLichtFeld Studioで直接読み込んでください。

### まず確認する項目

| 設定 | 使い方 |
| --- | --- |
| `Strategy` | まずは既定のMRNFから始めます。必要に応じてMCMCやIGS+を試します。 |
| `Iterations` | 学習ステップ数です。初回は既定値のまま試し、比較時に増減します。 |
| `Max Gaussians` | ガウシアン数の上限です。品質とVRAM/速度のバランスに効きます。 |
| `出力PLY名` | LichtFeldが保存するPLY名です。既定ではシーンフォルダ名になります。 |
| `SH Degree` | 通常は3です。軽量化したい場合だけ下げます。 |
| `Tile Mode` | VRAMや速度に合わせて調整します。 |
| `Steps Scaler` | `Auto` ではデータセット画像数から自動計算します。 |
| `GUT` | ERP 360°画像を直接使うGUT用データのときだけONにします。 |

### Cubemapデータを学習する場合

Step 5で画像タイプ `PINHOLE` を選んだデータです。`入力データ` は通常、Metashapeルートでは `<scene>/output/metashape_cubemap/`、SphereSfMルートでは `<scene>/output/spheresfm_cubemap/` です。

`Metashape → COLMAPデータセット` と `RealityScan → COLMAPデータセット` も、LichtFeldではPINHOLE系データセットとして扱います。RealityScanルートでは `<scene>/output/realityscan/lfs_colmap/` を指定し、`GUT` はOFFで使います。

- `GUT` はOFFにします。
- `Undistort` も通常はOFFです。
- マスクを使う場合は、LichtFeld側のマスクモードを結果に合わせて選びます。

### ERP 360° / GUTデータを学習する場合

Step 5で画像タイプ `ERP 360°` を選んだデータです。`入力データ` は通常、Metashapeルートでは `<scene>/output/metashape_3dgut/`、SphereSfMルートでは `<scene>/output/spheresfm_3dgut/` です。

- `GUT` をONにします。
- 選択中の入力データ内に `pointcloud.ply` が必要です。
- Step 5でMetashapeまたはSphereSfMからERP 360° / GUT用データを作成しておきます。

### Steps Scaler

`Steps Scaler` を `Auto` にすると、Step 6が `入力データ/images/` の画像数を数え、LichtFeld StudioのGUIがデータセット読み込み時に行う300枚基準の調整と同じ考え方で倍率を決めます。手動で比較したい場合だけ固定値にします。

### 詳細パラメーター

頻繁には触らないDataset、Optimizer、Refinement、Loss、Initialization、MRNF/IGS+、Sparsity、Save/Eval系の項目は `Advanced Training Parameters` にまとめています。ストラテジーや上部チェックに関係する項目だけ表示されるため、必要になった項目から開いて調整します。

## Postshot

Postshotでは、画像とカメラポーズから `.psht` プロジェクトを作成します。

Step 6からCLI起動する場合は、v1.0/v1.1 Release BuildのPostshot CLIを目安にします。Postshot側で設定を確認しながら進めたい場合は、Step 5の画像、カメラポーズ、必要なマスクをPostshotで直接読み込んでください。

### まず確認する項目

| 設定 | 使い方 |
| --- | --- |
| `プロジェクト名` | 出力する `.psht` のファイル名です。既定ではシーンフォルダ名になります。 |
| `Profile` | 通常は `Splat3` から始めます。 |
| `kSteps` | `Auto` ではPostshot側の自動計算を使います。固定したい場合だけOFFにします。 |
| `最大画像サイズ` | Postshotに渡す画像の長辺上限です。0は制限なしです。 |
| `Camera Poses` | 既存ポーズを使うなら `Import`、Postshotに推定させるなら `Estimate` です。 |
| `マスクを読み込む` | `masks/` をPostshotへ渡す場合にONにします。 |

### Camera Poses

`Import` は、Step 5で作ったカメラポーズをPostshotに渡す設定です。

| Step 5のルート | Importで渡すもの |
| --- | --- |
| COLMAP | COLMAP sparseモデル |
| SphereSfM | SphereSfM sparseモデル |
| Metashape | `transforms_postshot.json` と `pointcloud_postshot.ply` など、選択中プリセットのカメラJSONと点群PLY |

カメラポーズが見つからない状態で `Import` のまま起動すると、Step 6は実行前に止まります。先にStep 5でSfMまたは変換を実行するか、Postshot側に推定させるため `Estimate` に切り替えます。

### マスク

このアプリのマスクは白=使用、黒=除外です。通常は `黒を除外・白を使用 (background)` を使います。白い領域を一時的な遮蔽物として除外したい場合だけ `白を除外・黒を使用 (occluders)` を使います。

### 詳細パラメーター

GPU、プロファイル依存のモデル上限、Anti-Aliasing、Sky Model、継続学習データ、Crop/ROI、PLY/SPZ書き出しは `Postshot詳細パラメーター` にあります。まずは既定値で実行し、比較や再実行の目的がある項目だけ変更します。

Postshot v1.1.0では、露出、ホワイトバランス、周辺減光のばらつきを補正するPhotometric CompensationがPostshot GUIに追加されています。ただしPostshot v1.1.0時点の `postshot-cli.exe train --help` には対応するCLIオプションが出ていないため、この設定が必要な場合はPostshot側のGUIで有効にしてください。

## Brush

Brushでは、NeRF `transforms.json` 系またはCOLMAP形式のデータセットをCLIに渡して学習し、PLYを書き出します。OSSのBrush CLIをViewerなしで実行できます。

| 設定 | 使い方 |
| --- | --- |
| `出力PLY名` | `{iter}` を含めると最終ステップ番号に置換されます。 |
| `Iterations` | Brushの学習ステップ数です。 |
| `Export Every` | PLYを書き出す間隔です。 |
| `Max Resolution` | Brushへ読み込む画像の長辺上限です。 |
| `Render Mode` | 通常はAuto。Mip比較をしたい場合だけ切り替えます。 |

詳細パラメーターでは、Refine間隔、Gaussian上限、評価分割、画像や点群の間引きを指定できます。

## gsplat

gsplatでは、`python examples/simple_trainer.py` を起動します。入力データは `images/` と `sparse/0/` を含むCOLMAP形式データセットが必要です。MetashapeやRealityScanから使う場合は、Step 5でCOLMAPデータセットを作成してから選びます。

PINHOLE画像の標準学習に加えて、魚眼・広角などのレンズ歪み補正前画像向けに3DGUTを使えます。LichtFeld用の `ERP 360° / GUT` データをそのまま渡す設定ではありません。

| 設定 | 使い方 |
| --- | --- |
| `simple_trainer.py` | gsplatリポジトリ内の `examples/simple_trainer.py` を指定します。 |
| `結果フォルダ名` | `ckpts/`, `stats/`, `ply/` などを保存するフォルダ名です。 |
| `Strategy` | 通常はDefault。MCMCや3DGUTを試す場合はMCMCを使います。 |
| `Max Steps` | 学習ステップ数です。 |
| `Data Factor` | 画像縮小率です。1は縮小なしです。 |
| `3DGUT` | レンズ歪み補正前画像向けの3DGUT経路を使います。MCMC Strategyで実行します。 |

## Step 5へ戻るべき状態

Step 6はデータセットを作りません。直接読み込みでもCLI起動でも、次の状態では先にStep 5へ戻って変換を実行します。

| 状態 | 対処 |
| --- | --- |
| `入力データ` に画像やカメラ情報がない | Step 5で目的のカードを選び、学習アプリに渡すデータセットを作成します。 |
| SphereSfMのSfM結果だけがあり、学習用データセットがない | Step 5で `SphereSfM → NeRFデータセット(JSON/PLY)` を選び、PINHOLEまたはERP 360°で出力します。 |
| LichtFeldの `GUT` をONにしたが `pointcloud.ply` がない | Step 5でERP 360° / GUT用データを作ります。 |
| Postshotの `Camera Poses: Import` でポーズがない | Step 5でSfM/変換を実行するか、`Estimate` に切り替えます。 |

## 実行後にできるもの

| 実行アプリ | 主な出力 |
| --- | --- |
| LichtFeld Studio | `出力先` に最終PLY。PPISP使用時は関連ファイルも出力されます。 |
| Postshot | `出力先` に `.psht` / `.ply` / `.spz`。 |
| Brush | `出力先` に `.ply`。 |
| gsplat | 結果フォルダ内の `ply/` に `.ply`。 |

最終品質の差はStep 5で作ったデータ形状、学習アプリ側の設定、学習ステップ数、マスクの使い方に左右されます。同じデータセットから設定だけ変えて試す場合は、出力名を変えて結果を残しておくと比較しやすくなります。

## よくある判断

- Step 5直後にまず見た目や設定を確認するなら、学習アプリ側でデータセットを直接読み込みます。
- CLIで回すなら、LichtFeldは `GUT` OFF、Postshotは `Camera Poses: Import` から始めます。
- LichtFeldでGUTを試すなら、Step 5で画像タイプ `ERP 360°` を選んでから、Step 6で `GUT` をONにします。
- Postshotにポーズを推定させたい場合だけ `Camera Poses: Estimate` を使います。
- COLMAPルートのデータはPINHOLEのCubemap用です。ERP 360° / GUT比較はMetashapeまたはSphereSfMルートで作ります。
- 既存結果を残したい場合は、LichtFeldの出力PLY名、Postshotのプロジェクト名、または `出力先` を変えます。
