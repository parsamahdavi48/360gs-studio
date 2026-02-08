# tetraface-3dgs-utils

3Dガウシアンスプラッティング(3DGS)用のワークフローとして自作しながら使っているスクリプト集です。

## 必要項目

以下のソフト・モジュールをインストールしてください。すべてのスクリプトで共通です。

- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) (他のバージョンは不可)
- [Python 3.x](https://www.python.org/) (3.11.8で確認、3.11推奨)
- [FFmpeg / FFprobe](https://ffmpeg.org/) (動画から静止画切り出しで使用)
- [metashape_360_lfs.py (フォーク版)](https://github.com/tetraface/metashape_360_lfs) 
  - このリポジトリ内に同梱: `vendor/metashape_360_lfs/metashape_360_lfs.py`
  - 配布元の記録: `vendor/metashape_360_lfs/VENDOR_SOURCE.md`

### 依存pythonモジュール

- NumPy
- OpenCV
- Pillow
- PySide6 (GUIラッパーで使用)
- Open3D (metashape_360_lfs内で使用)
- PyTorch 2.8.0 (with CUDA 12.8)
- ultralytics
- tqdm

インストール例:
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy opencv-python Pillow open3d ultralytics tqdm PySide6
```

## Windows クイックスタート

このリポでは Python 3.11 の venv 運用を推奨します。`open3d` は Python 3.12 必須ではありません。
Python 3.11 が無い場合、`setup_windows.bat` は `winget` で Python 3.11.8 の導入を試みます。

```bat
setup_windows.bat
start_extract_frames_gui.bat
```

## 各スクリプトの概要

### cubemap_transforms_json.py

Metashapeが出力する**360度画像用**xmlファイルからtransforms.jsonに変換したものを元に、さらにキューブマップ用に変換し、一般的な3DGSソフトで入力できるようにします。<br>
[→詳細を見る](doc/cubemap_transforms_json.ja.md)<br>
![mask example](images/yaw45.jpg)

### stitch_mask.py

360度画像内の２つの魚眼画像の指定角度外にマスクを生成します。レンズ間のつなぎ目付近で被写体との距離が近くて、スチッティング領域が目立つ場合に有効です。<br>
[→詳細を見る](doc/stitch_mask.ja.md)<br>
![マスク例](images/stitch_mask.png)


### yolo_mask.py

360度画像内の人物を検知してマスクを生成します。<br>
[→詳細を見る](doc/yolo_mask.ja.md)<br>
![マスク例](images/yolo_mask.png)

### extract_frames.py

FFmpegでエクイレクタングラー動画から静止画を切り出し、固定間隔または変化量ベース選択＋ブラー差し替えを行います。<br>
[→詳細を見る](doc/extract_frames.md)<br>

### review_frames.py

抽出した静止画を軽量GUIで確認し、`selected_frames.csv` の keep/drop 判定を編集します。<br>
[→詳細を見る](doc/review_frames.md)<br>

### apply_frame_decisions.py

`selected_frames.csv` の keep/drop 判定を反映し、keep のみを Metashape 読み込み用に出力します。<br>
[→詳細を見る](doc/apply_frame_decisions.md)<br>

### extract_frames_gui.py

抽出ワークフロー用ラッパーGUIです。抽出実行、レビューGUI起動、keep画像の確定出力まで行えます。<br>
[→詳細を見る](doc/extract_frames_gui.md)<br>
