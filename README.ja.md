# stechdrive-3dgs-utils

**v0.3.0**

360度動画から、Metashape SfM と 3D Gaussian Splatting (3DGS) 学習へ渡すための画像・マスク・カメラ情報を作るデスクトップツールです。Windows上で `setup_windows.bat` と `start_gui.bat` だけで使える、日本語GUI中心のワークフローとして整備しています。

[EN English](README.md)

Fork元: [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils)

## 何をするツールか

```text
360度動画
  -> フレーム抽出
  -> フレーム確認・採用/除外
  -> マスク生成
  -> Metashape SfM
  -> キューブマップ変換
  -> 3DGSトレーニング
```

このリポの主目的は、エクイレクタングラー動画をそのままMetashapeでSfMし、SfM結果を3DGS用のパース画像・マスク・`transforms.json` へ変換することです。マスクはMetashape SfMの前に生成し、人物、車両、スティッチ境界、白飛び領域を特徴点マッチングから除外するために使います。

## クイックスタート

```bat
setup_windows.bat
start_gui.bat
```

`setup_windows.bat` は Python 3.11 の検出、必要に応じた winget インストール、venv作成、PyTorch CUDA 12.8 と依存パッケージの導入を行います。

`start_gui.bat` はvenvを有効化して統合GUIを起動します。

## GUIワークフロー

| Step | 内容 | 主な現在のデフォルト |
| --- | --- | --- |
| 1. フレーム抽出 | 360度動画からエクイレクタングラー静止画を抽出 | 固定間隔または変化量ベース |
| 2. フレーム確認 | 抽出フレームを確認し、採用/除外をCSVに反映 | ブラーワースト順の確認に対応 |
| 3. マスク生成 | YOLO検出、スティッチ境界、白飛びマスクを生成 | YOLO検出ON、人物検出が標準 |
| 4. キューブマップ変換 | Metashape結果を3DGS用パース画像とJSONに変換 | LichtFeld / Full / Cube6 |

### 現在のGUI方針

- 左の縦タブでStep 1からStep 4を切り替えます。
- 各Stepの実行/キャンセルはウィンドウ下部に統一しています。
- 中央左の設定ペインは固定幅で、横スクロールせず、必要な時だけ縦スクロールします。
- 長い設定は折りたたみ式です。YOLOの80クラス一覧、スティッチ/白飛び設定、キューブマップの詳細ビュー設定は普段閉じた状態で使えます。
- Step 4の標準出力は `LichtFeld`、`Full (1.0x)`、`Cube6 (4面+上下)` です。
- Step 4では `選択ビュー` と `出力画像` を表示します。`出力画像` は入力画像数と有効ビュー数から決まる書き出し枚数です。

## 推奨ワークフロー

1. Insta360等の360度動画を用意します。
2. Step 1でフレームを抽出します。
3. Step 2でブレや不要フレームを除外します。
4. Step 3で人物マスクを生成します。必要に応じてスティッチ境界、白飛びマスクも有効にします。
5. 生成された `masks/` フォルダをMetashapeにper-imageマスクとして読み込み、SfMを実行します。
6. Step 4でMetashapeのXML/PLYを使い、3DGSトレーニング用の画像、マスク、`transforms.json` を出力します。

スティッチ境界マスクは、エクイレクタングラー画像上でスティッチ位置が固定されている素材向けです。FlowState手ブレ補正、方向ロック、AIスティッチ等で境界位置が動く場合は、無理に有効化しない方が安全です。

## 動作環境

- Windows 10/11
- Python 3.11 (3.11.8で確認)
- CUDA対応GPU
- CUDA Toolkit 12.8
- FFmpeg / FFprobe

`setup_windows.bat` で導入される主なPythonパッケージ:

```text
torch==2.8.0 (CUDA 12.8), torchvision, torchaudio
numpy, opencv-python, Pillow, open3d, ultralytics, tqdm, PySide6
```

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
