# stechdrive-3dgs-utils

360度動画から3Dガウシアンスプラッティング (3DGS) 学習用アセットを作成するためのスタジオツールキット。日本語GUIで統合ワークフローを提供します。

[tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils) からフォークし、統合ワークフローGUI、追加マスク機能、マルチフレームワーク出力対応を拡張しています。

[EN English](README.md)

## ワークフロー概要

```
360度動画  ──  フレーム抽出  ──  レビュー+選別  ──  マスク生成
                                                          │
                                                Metashape SfM (手動、マスクと一緒に)
                                                          │
                                                キューブマップ変換  ──  3DGSトレーニング
                                                (Postshot / Brush /
                                                 LichtFeld Studio)
```

マスクは Metashape SfM の**前**に生成し、Metashape にインポートして人物・車両等の動体、スティッチ継ぎ目、白飛び領域を特徴点マッチングから除外することで SfM 精度が大きく向上し、結果として 3DGS の品質も上がります。

| ステップ | 内容 |
|----------|------|
| **1. フレーム抽出** | 360度動画からエクイレクタングラー静止画を切り出し。固定間隔 (推奨: 0.8-1秒) または変化量ベース |
| **2. フレーム確認** | ブラーワースト順ナビでブレ画像を確認、閾値一括dropで効率的に選別 |
| **3. マスク生成** | YOLO+SAM2.1 人物検出、スティッチ継ぎ目マスク、白飛び (過露出) マスク。**生成された `masks/` フォルダを Metashape の per-image マスクとしてインポートしてから SfM を実行** |
| **4. キューブマップ変換** | Metashape SfM 完了後、エクイレクタングラーの結果 (XML + PLY) をキューブマップビューに変換し、Postshot / Brush / LichtFeld Studio 用の transforms.json を出力。マスクもキューブマップ面に伝搬 |

## クイックスタート (Windows)

```bat
setup_windows.bat
start_gui.bat
```

`setup_windows.bat` は Python 3.11 (未インストールなら winget で自動導入)、venv作成、PyTorch (CUDA 12.8) + 全依存パッケージをインストールします。

`start_gui.bat` で統合GUIアプリ (ダークモダンテーマ、日本語UI) が起動します。

## 動作環境

- Windows 10/11
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)
- [Python 3.11](https://www.python.org/) (3.11.8で確認済み)
- [FFmpeg / FFprobe](https://ffmpeg.org/)
- CUDA対応GPU (YOLO/SAM2, PyTorch用)

### Python依存パッケージ

`setup_windows.bat` で自動インストール:

```
torch==2.8.0 (CUDA 12.8), torchvision, torchaudio
numpy, opencv-python, Pillow, open3d, ultralytics, tqdm, PySide6
```

### MLモデルファイル

YOLO/SAM2のモデルは初回実行時に ultralytics が自動ダウンロードします:
- `yolo26m.pt` / `yolo26l.pt` (YOLO v26)
- `sam2.1_l.pt` (SAM 2.1 Large)

## CLIツール

全CLIエンジンはGUIなしで単体利用可能です:

| スクリプト | 内容 | ドキュメント |
|-----------|------|-------------|
| `extract_frames.py` | 360動画からフレーム抽出 (変化/固定選択 + ブラー置換) | [JP](doc/extract_frames.md) |
| `apply_frame_decisions.py` | CSVのkeep/drop判定を適用 | [JP](doc/apply_frame_decisions.md) |
| `review_frames.py` | フレームレビューGUI (ブラーワースト順ナビ付き) | [JP](doc/review_frames.md) |
| `yolo_mask.py` | YOLO+SAM2.1 人物検出マスク (360度画像対応) | [JP](doc/yolo_mask.ja.md) |
| `stitch_mask.py` | デュアル魚眼カメラのスティッチ境界マスク | [JP](doc/stitch_mask.ja.md) |
| `overexposure_mask.py` | 白飛びピクセル検出とマスク合成 | - |
| `cubemap_transforms_json.py` | エクイレクタングラー → キューブマップ変換 | [JP](doc/cubemap_transforms_json.ja.md) |

## Fork元からの変更点

このフォークで追加・変更した機能:
- **統合GUI** (`gui/`) ダークモダンテーマ + 日本語UI (PySide6)
- **4ステップワークフロー** を1つのタブウィンドウに統合
- **白飛びマスク** 検出 (`overexposure_mask.py`)
- **ブラーワースト順ナビ** + 閾値一括dropをレビューGUIに追加
- **Brushプロファイル** 対応 (`--brush` 座標変換)
- **削除**: COLMAP rig export, RealityScan rig export (MetashapeベースのSfMワークフローに特化)

upstreamの変更は [tetraface/tetraface-3dgs-utils](https://github.com/tetraface/tetraface-3dgs-utils) から定期的に取り込みます。

## ライセンス

MIT License。[LICENSE](LICENSE) を参照。

オリジナルコード: [tetraface Inc.](https://github.com/tetraface)
フォーク拡張: [stechdrive](https://github.com/stechdrive)
