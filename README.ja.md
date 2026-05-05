# stechdrive-3dgs-utils

**v1.7.0**

360°カメラの動画から、3D Gaussian Splatting (3DGS) のトレーニングに使いやすい画像・マスク・カメラデータを作るためのWindows向け統合GUIツールです。

`setup_windows.bat` がPython 3.12とFFmpeg/FFprobeの検出、必要に応じたwinget経由のインストール、仮想環境の作成、依存パッケージ導入まで行います。起動も `run_gui.bat` から行えるため、普段の作業ではPythonコマンドを直接打たずに使えます。

[EN English](README.md)

Fork元: [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils)

![STechDrive 3DGS Utils GUI](images/stechdrive-3dgs-utils-gui.jpg)

## このアプリでできること

### 1. 360°動画からMetashape SfM、3DGSトレーニングへ

Insta360などの360°カメラで撮影した動画から、SfM向けのエクイレクタングラー静止画を抽出します。抽出後はフレームを確認し、人物、撮影者、三脚、空、スティッチ境界、白飛びをマスクしてからMetashapeに渡せます。

MetashapeでSfMした結果は、LichtFeld Studio / Postshot / Brush 向けの視点画像、マスク、`transforms.json` に変換できます。360°動画を3DGSトレーニング用データセットにするためのメインワークフローです。

### 2. 360°動画からCOLMAP Rigデータセットへ

Metashapeを使わず、抽出済みの360°画像からCOLMAP Rig形式の視点画像セットを書き出すこともできます。必要に応じてGUIからCOLMAP/GLOMAPまで実行し、3DGSソフトに渡せるSfM済みデータを作成できます。

### 3. 通常の静止画・動画向けのマスク前処理

デジタル一眼、スマホ、通常動画の連番画像に対しても、YOLO/SAMによる人物・車両、Mask2Former/SAM3.1による空などのマスクや白飛びマスクを作成できます。SfMソフトに読み込む前のマスク生成ツールとして使えます。

## 主な特徴

- 360°動画からSfM向けフレームを抽出。ペア解析による間隔最適化、シーン距離に応じた自動閾値、冗長・変化大・弱追跡・ブレ・低テクスチャ候補のStep 2確認に対応
- 抽出フレームを単一プレビュー/サムネイル一覧で確認し、Windows Explorer式のサムネイル選択で採用/除外を整理。360°画像は利用可能な環境ではOpenGLで高速化したFOV90°のパース表示でも確認できます
- YOLO + SAM2.1、Mask2Former ADE20Kクラス、SAM3.1プロンプトによる人物・空などのマスク生成
- 360°画像の下部に写りやすい撮影者、三脚、手元の検出強化
- スティッチ境界、白飛び領域、ユーザー指定PNGカスタムマスクの合成
- 大量画像でも使いやすいキャッシュ付きの単一プレビュー/サムネイル一覧で、マスク結果を確認しながら調整。マスク重ね表示は360°表示/OpenGL高速化パース表示の両方に反映されます
- マスク漏れのあるフレームだけを選択し、設定を変えて再生成。SAM3.1では既存マスクに対して、プロンプトで狙った対象を加算したり誤検出だけを減算できます
- Metashape SfM結果をLichtFeld Studio / Postshot / Brush向けに変換
- COLMAP Rig形式の視点画像セットを書き出し、COLMAP/GLOMAP実行まで対応
- Windows向けセットアップスクリプトと日本語GUI

## かんたん導入

通常はリリースZIPを展開し、次の2つを順番に実行します。

```bat
setup_windows.bat
run_gui.bat
```

`setup_windows.bat` はPython 3.12とFFmpeg/FFprobeを探し、必要な場合はwinget経由で不足しているシステム依存を導入します。その後、`.venv` を作成し、PyTorch CUDA wheel、OpenCV、Pillow、Open3D、ultralytics、PySide6、SAM3.1実行用パッケージなどをインストールして検証します。

`run_gui.bat` は `.venv` を有効化して統合GUIを起動します。既存の `.venv` が正常ならセットアップは再構築せず、その状態を表示して終了します。意図的に作り直す場合は `setup_windows.bat --force` を使います。

既存環境を互換する最新パッケージへ更新する場合:

```bat
update_venv.bat
```

`requirements/` の固定済み既知良好セットで作り直す場合は `update_venv.bat --locked` を使います。

YOLO/SAM2、Mask2Former、SAM3.1のモデルファイルは初回利用時にダウンロードされる場合があります。ローカルのYOLO/SAM重みは `models/ultralytics/`、Mask2Former重みは `models/mask2former-swin-large-ade-semantic/`、SAM3.1プロンプトマスクは `models/sam3.1/sam3.1_multiplex.pt` を使います。互換性のため、リポジトリ直下の `.pt` も引き続き検出します。リリースZIPにはモデル重み、生成データ、ユーザー設定、ローカルセットアップログは含めていません。これらの第三者ライブラリおよびモデル重みには別ライセンスが適用されます。詳細は [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照してください。

### マスク生成モデルの使い分け

- 人物だけを高速にマスクしたい場合は YOLO/SAM2.1 が向いています。
- 人物や空をできるだけ高精度にマスクしたい場合は SAM3.1 を推奨します。プロンプトで対象を指定できるため、生成後に漏れた対象だけを加算したり、誤検出だけを減算できます。
- Mask2Former は、SAM3.1を導入していない環境でも空マスクを手軽に試したい場合の選択肢です。

### SAM3.1プロンプトマスク

`setup_windows.bat` はSAM3.1の実行用パッケージを入れますが、checkpointは同梱しません。checkpointの取得にはユーザー自身のHugging FaceアカウントとSAM Licenseへの同意が必要なためです。

このアプリでは、公式 `facebook/sam3.1` の `sam3.1_multiplex.pt` を使います。SAM3.1はCUDA対応GPU向けモデルです。NVIDIA製GPU環境での実行を推奨します。

SAM3.1の一括処理中にGPUメモリ不足が発生した場合でも、完了済みのマスクは保存されます。同じ設定で再実行すると、未処理の画像から再開します。

マスク精度を優先する場合は、YOLO/SAM2.1よりもSAM3.1の利用を推奨します。SAM3.1は、空マスクや狙った対象だけの補正など、プロンプトで制御したいマスク生成に向いています。一度マスクを生成したあと、漏れがある画像だけを選択し、`tripod`、`hand`、`selfie stick`、`cell phone` などを加算したり、`male icon`、`female icon`、`logo`、`sign` などの誤検出を減算したりできます。

1. Hugging Faceアカウントを作成、またはログインします。
2. Metaの [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) Hugging Faceリポジトリを開き、アクセス申請とSAM Licenseへの同意を行います。Hugging Faceのgated model申請は個人アカウント単位で、ユーザー名やメールアドレスがモデル提供者へ共有される場合があります。
   - Hugging FaceのGated Modelには自動承認と手動承認があります。`facebook/sam3.1` で同意後にFilesタブや `sam3.1_multiplex.pt` をブラウザから開ける/ダウンロードできる場合、そのアカウントでは承認済みです。メールなどの返答を待つ必要はありません。承認待ち表示の場合は、モデル提供者側の承認を待つ必要があります。
3. Hugging Faceのアカウント設定からアクセストークンを作成します。
   - アプリからのダウンロードには、承認済みの同じHugging Faceアカウントで作成した `Read` トークンを使ってください。ブラウザでログインしていても、このアプリはブラウザのログイン状態を使いません。
   - トークンは作成直後に表示される値を必ずコピーしてください。Hugging Faceのトークン一覧では、既存トークンの値を後から再表示・コピーできない場合があります。コピーし忘れた場合は、新しい `Read` トークンを作成するか、既存トークンを `Invalidate and refresh` して新しい値を発行します。`Invalidate and refresh` すると古いトークンは無効になります。
   - アクセストークンはパスワード相当の秘密情報として扱ってください。README、Issue、チャット、スクリーンショット、実行ログなどに貼らないでください。SAM3.1 checkpointのダウンロード用途では `Read` 権限で足ります。可能ならSAM3.1用の専用トークンを作り、不要になったらHugging Faceの設定画面で削除またはrefreshしてください。このアプリは入力されたトークンを保存しません。
4. Step 3で `SAM3.1` を選びます。`models/sam3.1/sam3.1_multiplex.pt` が無い場合、アプリがトークン入力を求めてcheckpointをダウンロードします。このアプリはトークンを保存しません。

checkpointを手動で `models/sam3.1/sam3.1_multiplex.pt` に置くこともできます。

## GUIワークフロー

シーンフォルダのパスに日本語などの非ASCII文字、極端に長いパス、制御文字や `"` が含まれる場合、GUIは実行前に停止します。OpenCVや外部3DGS/SfMツールで失敗しやすいためです。空白やOneDrive配下であることだけでは停止しません。英数字だけの短い作業パス（例: `D:\work\scene01`）を使ってください。

```text
360°動画または画像
  -> Step 1: フレーム抽出
  -> Step 2: フレーム確認・採用/除外
  -> Step 3: マスク生成
  -> Step 4: 書き出し
      -> Metashape SfM結果から3DGS向けデータを作成
      -> COLMAP Rig視点画像を書き出し、必要に応じてCOLMAP/GLOMAPを実行
```

| Step | 内容 | 主なデフォルト |
| --- | --- | --- |
| 1. フレーム抽出 | 360°動画からエクイレクタングラー静止画を抽出 | 固定間隔 + 変化補正 |
| 2. フレーム確認 | 抽出フレームを単一/サムネイル表示で確認し、採用/除外をCSVに反映 | 低品質候補や不要フレームの確認に対応 |
| 3. マスク生成 | 人物、スティッチ境界、白飛び、空、カスタムマスクを生成 | YOLO/SAM2.1、高品質設定 |
| 4. 書き出し | SfM結果からの3DGS出力、またはCOLMAP Rig視点画像を書き出し | Metashapeインポート / LichtFeld / Full / Cube6 |

## 推奨ワークフロー: Metashapeルート

1. Insta360などの360°動画を用意します。
2. Step 1でSfM向けフレームを抽出します。
3. Step 2で低品質候補や不要フレームを確認して除外します。
4. Step 3で人物・撮影者・三脚・空など、SfMに使いたくない領域のマスクを生成します。`品質: 高品質` が推奨開始点です。
5. マスク漏れが残る場合は、該当画像だけ `品質: 最高` に上げるか、Mask2Former/SAM3.1に切り替えて再生成します。
6. 必要に応じてスティッチ境界マスク、白飛びマスク、カスタムマスクも有効にします。
7. 生成された `masks/` フォルダをMetashapeにマスクとして読み込み、SfMを実行します。
8. Step 4でMetashapeのXML/PLYを使い、3DGSトレーニング用の画像、マスク、`transforms.json` を出力します。

## COLMAPルート

1. Step 1からStep 3まではMetashapeルートと同じです。
2. Step 4で `COLMAP書き出し` を選び、視点画像とマスクを `output/colmap_rig/` に書き出します。
3. 必要に応じて `書き出し後にCOLMAPを実行` をONにし、Feature、Matcher、Mapperまで実行します。
4. 完了後は `output/colmap_rig/` をCOLMAPプロジェクトとして、COLMAP対応の3DGSアプリに渡します。

## 通常画像・通常動画のマスク前処理

通常動画やデジタル一眼・スマホの連番画像は、`images/` に直接置くか、Step 3 の `画像フォルダ` 行にある `+` アイコンからシーンへコピーします。そのうえで `画像タイプ: 通常` を選びます。このモードではスティッチ境界と360°専用の極投影補助を使わず、モデルによるマスク生成や白飛びマスクを使えます。

人物、車両、白飛びなどをSfM前に除外したい場合の前処理として使えます。

## マスク調整のポイント

- `品質: 高品質` から始めます。
- 処理速度を優先する確認用では `品質: 標準` を使います。
- 人物が漏れる場合は `品質: 最高`、または `拡張` を少し上げます。
- `品質: 最高` は精度を優先するぶん処理時間が増えます。最初から全画像に使うより、漏れが残った画像だけ上げて再生成する使い方が現実的です。
- プレビューで漏れを見つけた場合は、設定を調整して `マスク再生成` を使うと、その1枚だけ現在ONのマスク処理で `masks/` に保存し直せます。サムネイル一覧では `Ctrl` / `Shift` 選択した複数枚をまとめて再生成できます。SAM3.1では既存マスクへプロンプト検出結果を加算/減算して補正できます。
- スティッチ境界マスクは、エクイレクタングラー画像上でスティッチ位置が固定されている素材向けです。FlowState手ブレ補正、方向ロック、AIスティッチ等で境界位置が動く場合は、プレビューで確認してから使ってください。

## 動作環境

- Windows 10/11
- Python 3.12 (3.12.10で確認)
- CUDA対応GPU
- CUDA Toolkit 12.8
- FFmpeg / FFprobe (`setup_windows.bat` が未検出時にwinget経由で Gyan.FFmpeg を導入)

`setup_windows.bat` で解決される主なPythonパッケージ:

```text
torch / torchvision / torchaudio from the CUDA 12.8 wheel index
numpy, opencv-python, Pillow, open3d, ultralytics, transformers, safetensors, tqdm, PySide6, sam3
```

`setup_windows.bat` は `requirements/` 以下の固定済み既知良好セットを使い、初回セットアップの再現性を優先します。`update_venv.bat` はデフォルトで互換する最新パッケージを解決し、固定セットで作り直したい場合だけ `--locked` を渡します。

## CLIツール

GUIは以下のCLIエンジンを呼び出しています。必要なら単体でも実行できます。

| スクリプト | 内容 | ドキュメント |
| --- | --- | --- |
| `extract_frames.py` | 360°動画からフレーム抽出 | [EN](doc/extract_frames.md) |
| `apply_frame_decisions.py` | CSVの採用/除外判定を反映 | [JP](doc/apply_frame_decisions.md) |
| `review_frames.py` | フレーム確認GUI | [JP](doc/review_frames.md) |
| `yolo_mask.py` | YOLO+SAM2.1 マスク生成 | [JP](doc/yolo_mask.ja.md) |
| `sky_mask.py` | Mask2Former ADE20KラベルまたはSAM3.1プロンプトによるセマンティックマスク生成 | [JP](doc/sky_mask.ja.md) |
| `stitch_mask.py` | スティッチ境界マスク生成 | [JP](doc/stitch_mask.ja.md) |
| `overexposure_mask.py` | 白飛びマスク生成 | - |
| `custom_mask.py` | ユーザー指定PNGマスクをAND合成 | [JP](doc/custom_mask.ja.md) |
| `cubemap_transforms_json.py` | エクイレクタングラーからキューブマップへ変換 | [JP](doc/cubemap_transforms_json.ja.md) |
| `transforms_to_colmap.py` | `transforms.json` からCOLMAP形式を書き出し | [JP](doc/transforms_to_colmap.ja.md) |

## ライセンス

MIT License。詳細は [LICENSE](LICENSE) を参照してください。

マスク生成機能では、別ライセンスの第三者ライブラリおよびモデル重みを使用します。詳細は [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照してください。

Original code by [tetraface Inc.](https://github.com/tetraface)
Fork extensions by [stechdrive](https://github.com/stechdrive)
