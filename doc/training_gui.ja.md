# Step 6 学習GUI

Step 6は、Step 5で作成した3DGS用データセットを使って、対応CLIを持つ学習アプリを起動する画面です。対応バージョンのLichtFeld StudioやPostshotを使うと、同じ設定の再実行やヘッドレス学習をGUIから始められます。

学習アプリ側で画質やモデル設定を確認しながら進めたい場合は、Step 6を使わず、Step 5の出力データセットをLichtFeld Studio、Postshot、Brushなどで直接読み込んで学習できます。画像変換やSfMはStep 6では行いません。データセットを作る作業は `Step 5: データセット` です。

## 学習アプリで使う

Step 5の出力は、下流の3DGSアプリに渡すためのデータセットです。まず学習アプリのGUIで結果を見ながら調整したい場合は、該当するデータセットフォルダを直接読み込みます。設定を固めたあとに同じ条件で回したい場合や、画面を開かずに学習だけ走らせたい場合は、Step 6のCLI起動を使います。

| 進め方 | 向いている場面 |
| --- | --- |
| 学習アプリで直接読み込む | 初回確認、見た目を確認しながらの調整、学習アプリ固有の設定を細かく使う場合 |
| Step 6からCLI起動する | 対応CLIで同じ条件を再実行したい場合、ヘッドレスで学習を走らせたい場合 |

Step 6のCLI実行は、LichtFeld Studio v0.5.2互換CLIとPostshot v1.0/v1.1 Release BuildのCLIを目安にしています。CLIを使わない場合でも、Step 5の出力データセットは各学習アプリで直接使えます。

## まず決めること

Step 6を開いたら、最初に「どのアプリで、どのデータを試すか」を決めます。

| やりたいこと | 実行アプリ | 主に確認する設定 |
| --- | --- | --- |
| LichtFeldで通常のCubemapデータを学習したい | `LichtFeld Studio` | `入力データ`, `GUT` OFF, `出力PLY名`, `Strategy`, `Iterations` |
| LichtFeldで3DGUTデータを試したい | `LichtFeld Studio` | `入力データ`, `GUT` ON, `pointcloud.ply` があること |
| Postshotでプロジェクトを作りたい | `Postshot` | `入力データ`, `Camera Poses`, `プロジェクト名`, `Profile` |
| 任意の学習CLIを起動したい | `その他... > Custom` | `実行ファイル`, `引数テンプレート`, `入力データ`, `出力先` |

`入力データ` は通常、自動設定のままで使います。Metashape / SphereSfMの変換結果は `<scene>/output/` 配下のStep 5データセットフォルダ、COLMAPルートは `<scene>/output/colmap_rig/` が基準になります。`出力先` は既定で `<scene>/output/` です。データセットと学習結果を同じ `output/` 配下に置くことで、後から持ち出しやすくしています。

## 基本操作

1. Step 5でデータセットを作成します。
2. 学習アプリのGUIで調整したい場合は、Step 5の出力データセットを直接読み込みます。
3. CLIで再実行またはヘッドレス実行したい場合は、`Step 6: 学習` を開きます。
4. `入力データ` と `出力先` が意図したフォルダになっているか確認します。
5. `LichtFeld Studio`、`Postshot`、または `その他... > Custom` を選びます。
6. `実行ファイル` が空欄で自動検出できない場合は、インストール先のexeを指定します。
7. 右側のアプリ別設定を確認します。
8. `起動` を押します。

Step 6は、選択した学習方式とデータセットの形が合っているかを実行前に確認します。たとえばLichtFeldの `GUT` は3DGUT用データ、通常のLichtFeldとPostshotは投影Cubemapデータを前提にします。

## 画面構成

Step 6では中央パネルを広く使うため、左右2カラムに整理しています。

| 場所 | 内容 |
| --- | --- |
| 左側 | 実行アプリ、ヘッドレス実行、実行ファイル、入力データ、出力先 |
| 右側 | LichtFeld / Postshot / Customごとの設定 |

右側の詳細パラメーターは、普段触る項目と、必要なときだけ開く詳細項目に分けています。まずは折りたたまれていない項目だけで実行し、結果を見てから詳細を調整する想定です。

## 共通設定

### 実行アプリ

`LichtFeld Studio` と `Postshot` は主要候補として直接選べます。`その他...` は追加候補を開くメニューで、現在は `Custom` を選べます。

### 実行ファイル

空欄の場合は既定名や既知の場所を探します。見つからない場合は、次のような実行ファイルを指定します。

| アプリ | 例 |
| --- | --- |
| LichtFeld Studio | `LichtFeld-Studio.exe` |
| Postshot | `postshot-cli.exe` |
| Custom | 起動したいCLIのexe |

### 入力データ

学習アプリに渡すデータセットフォルダです。通常はStep 5の現在ルートと出力形状から自動設定されます。

| Step 5の結果 | Step 6の入力データ |
| --- | --- |
| Metashape + 投影Cubemap | `<scene>/output/metashape_cubemap/` |
| Metashape + 3DGUT | `<scene>/output/metashape_3dgut/` |
| SphereSfM + 投影Cubemap | `<scene>/output/spheresfm_cubemap/` |
| SphereSfM + 3DGUT | `<scene>/output/spheresfm_3dgut/` |
| COLMAP Rig | `<scene>/output/colmap_rig/` |

手動で別フォルダを指定することもできます。その場合は、`images/`、必要なら `masks/`、カメラポーズ、点群など、選んだ学習アプリが必要とするファイルがそのフォルダにあることを確認します。

### 出力先

学習結果やPostshotプロジェクトを書き込むフォルダです。既定では `<scene>/output/` です。

LichtFeldの最終PLY、Postshotの `.psht`、Postshotの任意書き出しPLY/SPZが既に存在する場合、Step 6は上書きを避けるため実行前に止まります。出力名または出力先を変えてから再実行します。

## LichtFeld Studio

LichtFeld Studioでは、データセット、出力先、学習設定を指定して学習を開始します。

Step 6からCLI起動する場合は、v0.5.2互換のLichtFeld Studio CLIを目安にします。学習設定をLichtFeld Studio側で確認しながら進めたい場合は、Step 5で作成したCubemapデータまたは3DGUTデータをLichtFeld Studioで直接読み込んでください。

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
| `GUT` | 3DGUTデータを使うときだけONにします。 |

### Cubemapデータを学習する場合

Step 5で `投影視点に変換` を使ったデータです。`入力データ` は通常、Metashapeルートでは `<scene>/output/metashape_cubemap/`、SphereSfMルートでは `<scene>/output/spheresfm_cubemap/` です。

- `GUT` はOFFにします。
- `Undistort` も通常はOFFです。
- マスクを使う場合は、LichtFeld側のマスクモードを結果に合わせて選びます。

### 3DGUTデータを学習する場合

Step 5で `3DGUT (LichtFeld)` を使ったデータです。`入力データ` は通常、Metashapeルートでは `<scene>/output/metashape_3dgut/`、SphereSfMルートでは `<scene>/output/spheresfm_3dgut/` です。

- `GUT` をONにします。
- 選択中の入力データ内に `pointcloud.ply` が必要です。
- Step 5でMetashapeまたはSphereSfMから3DGUT用データを作成しておきます。

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
| Metashape | `transforms.json` と、利用可能なMetashape点群PLY |

カメラポーズが見つからない状態で `Import` のまま起動すると、Step 6は実行前に止まります。先にStep 5でSfMまたは変換を実行するか、Postshot側に推定させるため `Estimate` に切り替えます。

### マスク

このアプリのマスクは白=使用、黒=除外です。通常は `黒を除外・白を使用 (background)` を使います。白い領域を一時的な遮蔽物として除外したい場合だけ `白を除外・黒を使用 (occluders)` を使います。

### 詳細パラメーター

GPU、プロファイル依存のモデル上限、Anti-Aliasing、Sky Model、継続学習データ、Crop/ROI、PLY/SPZ書き出しは `Postshot詳細パラメーター` にあります。まずは既定値で実行し、比較や再実行の目的がある項目だけ変更します。

Postshot v1.1.0では、露出、ホワイトバランス、周辺減光のばらつきを補正するPhotometric CompensationがPostshot GUIに追加されています。ただしPostshot v1.1.0時点の `postshot-cli.exe train --help` には対応するCLIオプションが出ていないため、この設定が必要な場合はPostshot側のGUIで有効にしてください。

## Custom

`その他... > Custom` は任意のCLIを起動するための設定です。`実行ファイル` にCLIのexeを指定し、`引数テンプレート` で渡す引数を組み立てます。

使えるプレースホルダーは次の通りです。

| プレースホルダー | 展開される値 |
| --- | --- |
| `{dataset}` | 入力データフォルダ |
| `{images}` | 入力データ内の画像フォルダ |
| `{masks}` | 入力データ内のマスクフォルダ。ない場合は空文字 |
| `{sparse}` | 検出されたCOLMAP/SphereSfM sparseモデル。ない場合は空文字 |
| `{output}` | 出力先フォルダ |

例:

```text
--data {dataset} --out {output}
```

## Step 5へ戻るべき状態

Step 6はデータセットを作りません。直接読み込みでもCLI起動でも、次の状態では先にStep 5へ戻って変換を実行します。

| 状態 | 対処 |
| --- | --- |
| `入力データ` に `images/` や `transforms.json` がない | Step 5で `Cube` をONにして変換します。 |
| SphereSfMで `SfM` ON / `Cube` OFF だけ実行した | Step 5で `SfM` OFF / `Cube` ON にして既存sparseから変換します。 |
| LichtFeldの `GUT` をONにしたが `pointcloud.ply` がない | Step 5で3DGUT用データを作ります。 |
| Postshotの `Camera Poses: Import` でポーズがない | Step 5でSfM/変換を実行するか、`Estimate` に切り替えます。 |

## 実行後にできるもの

| 実行アプリ | 主な出力 |
| --- | --- |
| LichtFeld Studio | `出力先` に最終PLY。PPISP使用時は関連ファイルも出力されます。 |
| Postshot | `出力先` に `.psht` プロジェクト。任意でPLY/SPZも書き出せます。 |
| Custom | 指定したCLIの出力。 |

最終品質の差はStep 5で作ったデータ形状、学習アプリ側の設定、学習ステップ数、マスクの使い方に左右されます。同じデータセットから設定だけ変えて試す場合は、出力名を変えて結果を残しておくと比較しやすくなります。

## よくある判断

- Step 5直後にまず見た目や設定を確認するなら、学習アプリ側でデータセットを直接読み込みます。
- CLIで回すなら、LichtFeldは `GUT` OFF、Postshotは `Camera Poses: Import` から始めます。
- LichtFeld 3DGUTを試すなら、Step 5で `3DGUT (LichtFeld)` を作ってから、Step 6で `GUT` をONにします。
- Postshotにポーズを推定させたい場合だけ `Camera Poses: Estimate` を使います。
- COLMAPルートのデータは投影Cubemap用です。3DGUT比較はMetashapeまたはSphereSfMルートで作ります。
- 既存結果を残したい場合は、LichtFeldの出力PLY名、Postshotのプロジェクト名、または `出力先` を変えます。
