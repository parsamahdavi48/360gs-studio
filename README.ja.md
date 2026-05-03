# stechdrive-3dgs-utils

**v1.0.0**

360度動画や連番画像から、SfM と 3D Gaussian Splatting (3DGS) 学習へ渡すための画像・マスク・カメラ情報を作るデスクトップツールです。Windows上で `setup_windows.bat` と `run_gui.bat` だけで使える、日英対応の統合GUI中心のワークフローとして整備しています。

[EN English](README.md)

Fork元: [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils)

![STechDrive 3DGS Utils GUI](images/stechdrive-3dgs-utils-gui.jpg)

## 何をするツールか

```text
360度動画
  -> フレーム抽出
  -> フレーム確認・採用/除外
  -> マスク生成
  -> 書き出し
      -> Metashape SfM結果から3DGS向けデータを作成
      -> COLMAP Rig視点画像を書き出し、必要に応じてCOLMAP/GLOMAPを実行
```

このリポの主目的は、エクイレクタングラー動画をSfM/3DGS向けに扱いやすいデータセットへ整えることです。高精度ルートでは、エクイレクタングラー画像をMetashapeでSfMし、その結果を3DGS用の視点画像・マスク・`transforms.json` へ変換します。もう一つのルートとして、抽出済みエクイレクタングラー画像からCOLMAP Rig形式の視点画像を書き出し、必要に応じてCOLMAP/GLOMAPを実行できます。マスクはSfM前に生成し、人物、車両、スティッチ境界、白飛び領域を特徴点マッチングから除外するために使います。

## クイックスタート

```bat
setup_windows.bat
run_gui.bat
```

`setup_windows.bat` は Python 3.12 の検出、必要に応じた winget インストール、検証済み `.venv` の作成を行います。既存の `.venv` が正常なら、その状態を表示して再構築せず終了します。意図的に作り直す場合は `setup_windows.bat --force` を使います。パッケージのバージョンはセットアップ時点で解決し、PyTorch は CUDA 12.8 wheel index から導入します。

セットアップウィンドウは最後にキー入力待ちになり、サマリーを読んでから閉じられます。既存のターミナルから実行する場合は `setup_windows.bat --no-pause` を使えます。

`run_gui.bat` はvenvを有効化して統合GUIを起動します。

既存の `.venv` を互換する最新パッケージへ更新する場合:

```bat
update_venv.bat
```

`update_venv.bat` はインストール済み、または winget で入手可能な Python 候補を新しい順に調べます。まず対象 Python ABI に対する pip dry-run 互換チェックを行い、成立しそうな候補だけ必要に応じて winget で Python を入れます。その後、一時venvで `pip check` と import/CUDAスモークテストが通った最初の候補を `.venv` として採用します。

更新ウィンドウは最後にキー入力待ちになり、サマリーを読んでから閉じられます。既存のターミナルから実行する場合は `update_venv.bat --no-pause` を使えます。

## GUIワークフロー

| Step | 内容 | 主な現在のデフォルト |
| --- | --- | --- |
| 1. フレーム抽出 | 360度動画からエクイレクタングラー静止画を抽出 | 固定間隔 + 変化補正 |
| 2. フレーム確認 | 抽出フレームを確認し、採用/除外をCSVに反映 | 代表置換・低品質フレームの確認に対応 |
| 3. マスク生成 | YOLO検出、スティッチ境界、白飛びマスクを生成 | YOLO検出ON、人物検出が標準 |
| 4. 書き出し | SfM結果からの3DGS出力、またはCOLMAP Rig視点画像を書き出し | Metashapeインポート / LichtFeld / Full / Cube6 |

### 現在のGUI方針

- 左の縦タブでStep 1からStep 4を切り替えます。
- 各Stepの実行/キャンセルはウィンドウ下部に統一しています。
- 中央左の設定ペインは固定幅で、横スクロールせず、必要な時だけ縦スクロールします。
- 長い設定は折りたたみ式です。YOLOの80クラス一覧、スティッチ/白飛び設定、キューブマップの詳細ビュー設定は普段閉じた状態で使えます。
- Step 4のMetashapeルートは `LichtFeld`、`Full (1.0x)`、`Cube6 (4面+上下)` が標準です。
- Step 4では `選択ビュー` と `出力画像` を表示します。`出力画像` は入力画像数と有効ビュー数から決まる書き出し枚数です。
- Step 4のCOLMAPルートでは `output/colmap_rig/` へCOLMAP Rigデータセットを書き出し、ユーザー指定のCOLMAP/GLOMAP実行ファイルでSfMまで実行できます。

## 推奨ワークフロー: Metashapeルート

1. Insta360等の360度動画を用意します。
2. Step 1でフレームを抽出します。
3. Step 2で低品質候補や不要フレームを確認して除外します。
4. Step 3で人物マスクを生成します。必要に応じてスティッチ境界、白飛びマスクも有効にします。
5. 生成された `masks/` フォルダをMetashapeにマスクとして読み込み、SfMを実行します。
6. Step 4でMetashapeのXML/PLYを使い、3DGSトレーニング用の画像、マスク、`transforms.json` を出力します。

## COLMAPルート

1. Step 1からStep 3までは同じです。
2. Step 4で `COLMAP書き出し` を選び、視点画像とマスクを `output/colmap_rig/` に書き出します。
3. 必要に応じて `書き出し後にCOLMAPを実行` をONにし、Feature、Matcher、Mapperまで実行します。
4. 完了後は `output/colmap_rig/` をCOLMAPプロジェクトとして、COLMAP対応の3DGSアプリに渡します。

通常動画やデジタル一眼の連番画像を `images/` に置いた場合は、Step 3で `画像タイプ: 通常` を選ぶことで、スティッチ境界と360底面再検出を使わずにYOLO/SAMや白飛びマスクだけを生成できます。

スティッチ境界マスクは、エクイレクタングラー画像上でスティッチ位置が固定されている素材向けです。FlowState手ブレ補正、方向ロック、AIスティッチ等で境界位置が動く場合は、無理に有効化しない方が安全です。

## 動作環境

- Windows 10/11
- Python 3.12 (3.12.10で確認)
- CUDA対応GPU
- CUDA Toolkit 12.8
- FFmpeg / FFprobe

`setup_windows.bat` で解決される主なPythonパッケージ:

```text
torch / torchvision / torchaudio from the CUDA 12.8 wheel index
numpy, opencv-python, Pillow, open3d, ultralytics, tqdm, PySide6
```

setup/update とも、これらのパッケージバージョンは固定せず、選択した Python 環境で最新互換バージョンを解決し、検証に通った環境だけを採用します。

YOLO/SAM2のモデルファイルは初回利用時にultralyticsが自動ダウンロードします。

## CLIツール

GUIは以下のCLIエンジンを呼び出しています。必要なら単体でも実行できます。

| スクリプト | 内容 | ドキュメント |
| --- | --- | --- |
| `extract_frames.py` | 360度動画からフレーム抽出 | [JP](doc/extract_frames.md) |
| `apply_frame_decisions.py` | CSVの採用/除外判定を反映 | [JP](doc/apply_frame_decisions.md) |
| `review_frames.py` | フレーム確認GUI | [JP](doc/review_frames.md) |
| `yolo_mask.py` | YOLO+SAM2.1 マスク生成 | [JP](doc/yolo_mask.ja.md) |
| `stitch_mask.py` | スティッチ境界マスク生成 | [JP](doc/stitch_mask.ja.md) |
| `overexposure_mask.py` | 白飛びマスク生成 | - |
| `cubemap_transforms_json.py` | エクイレクタングラーからキューブマップへ変換 | [JP](doc/cubemap_transforms_json.ja.md) |
| `transforms_to_colmap.py` | `transforms.json` からCOLMAP形式を書き出し | [JP](doc/transforms_to_colmap.ja.md) |

## ライセンス

MIT License。詳細は [LICENSE](LICENSE) を参照してください。

Original code by [tetraface Inc.](https://github.com/tetraface)
Fork extensions by [stechdrive](https://github.com/stechdrive)
