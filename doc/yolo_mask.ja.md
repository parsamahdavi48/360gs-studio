# yolo_mask.py — 人物マスク生成

## 概要
`yolo_mask.py` は、YOLO（人物検出）と SAM（Segment Anything Model）を組み合わせ、画像中の人物領域を検出してマスク画像（PNG）を生成するスクリプトです。既定では360°パノラマ画像向けに、底面付近の撮影者や水平線付近の通行人を重点的に検出します。普通の動画フレームやデジタル一眼の連番画像では `--projection normal` を指定します。

![マスク例](../images/yolo_mask.png)

## 使い方
```
python -m core.yolo_mask [images_dir] [output_dir] [--add_ext] [--quality standard|high|best] [--expand M] [--classes IDS] [--projection equirect|normal] [--bottom-conf C] [--bottom-tta-rotations 1|2|4] [--bottom-model same|m|l|x] [--bottom-filter] [--profile-json PATH]
```

- `images_dir`: 入力画像ディレクトリ（省略時: `images`）
- `output_dir`: 出力マスク保存先（省略時: `masks`）
- `--add_ext`: 元の拡張子を残してさらに `.png` を追加（出力例: `hoge.jpg.png`）
- `--quality standard|high|best`: 入力ビューと投影補助の品質レシピ（デフォルト: `high`）。
  - `standard`: 全体直処理中心。360°画像では軽い下部投影も実行します。
  - `high`: 人物向けタイルと、360°画像向けの上部/下部投影補助を追加します。
  - `best`: より細かいタイルと強い下部補助を使います。
- `--expand M`: SAM後の検出領域を広げる固定ピクセル数（デフォルト=0）
  - 安全のため `-16〜32` にクランプされます。
  - 負値を指定するとマスク領域を収縮します。
- `--classes IDS`: YOLOクラスIDのカンマ区切り指定（デフォルト: `0` = personのみ）
- `--projection equirect|normal`: 入力画像の種類（デフォルト: `equirect`）
  - `equirect`: 360°エクイレクタングラー画像。底面再検出を行います。
  - `normal`: 通常画像。360°専用の底面再検出を行いません。
- `--bottom-conf C`: 360底面再検出だけに使うYOLO信頼度しきい値を、品質レシピの値から上書きします。
- `--bottom-tta-rotations 1|2|4`: 品質レシピで選ばれる底面画像の回転回数を上書きします。
- `--bottom-model same|m|l|x`: 底面再検出だけに使うYOLOモデル。`same` は `--quality` で選ばれたモデルを再利用します。`x` は重く、初回に `yolo26x.pt` のダウンロードが発生する場合があります。
- `--bottom-filter`: 信頼しにくい底面マスク成分を除外してから最終マスクに合成します。
- `--profile-json PATH`: 処理時間と検出数の内訳をJSONに出力します。指定しない通常実行の動作は変わりません。

例:

```
python -m core.yolo_mask .\images .\masks --quality high --expand 5 --classes 0,2,3
```

通常画像:

```
python -m core.yolo_mask .\images .\masks --projection normal --quality standard
```

真上から見た撮影者など、底面検出が難しい場合:

```
python -m core.yolo_mask .\images .\masks --quality best
```

底面だけYOLO Xまで使う最大設定:

```
python -m core.yolo_mask .\images .\masks --quality best --bottom-model x
```

固定データセットでベンチマーク:

```
python scripts/benchmark_yolo_mask.py --dataset D:\3DGS\test --output-root D:\3DGS\test\benchmarks --label baseline --repeat 3
```

改善後の出力をbaselineと比較:

```
python scripts/benchmark_yolo_mask.py --dataset D:\3DGS\test --output-root D:\3DGS\test\benchmarks --label candidate --compare-label baseline --repeat 3 --overwrite
```

## 出力について
- 出力は PNG 形式。デフォルトでは入力ファイルの拡張子を `.png` に置換して保存（`--add_ext` を使うと元名に `.png` を追加）。
- マスクは「背景=白 (255)、人物=黒 (0)」になるよう出力されます。

## 注意点
- 初回実行時に学習モデルファイルが自動でダウンロードされる場合があり、時間がかかります。ローカルの `.pt` を使う場合は `models/ultralytics/` に置いてください。未配置の名前付きモデルはUltralytics側の解決に任せます。
- YOLO/SAM機能では、別ライセンスの第三者ライブラリおよびモデル重みを使用します。詳細は [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) を参照してください。
- `--quality` を上げると処理時間とメモリ使用量が増加します。
- 底面TTA、候補フィルタ、`--bottom-model x` は、360底面再検出部分だけの処理時間を増やします。
- 大きなパノラマや高解像度画像では GPU（CUDA）対応の環境が推奨されます。CUDA対応PyTorchをインストールしてください。
