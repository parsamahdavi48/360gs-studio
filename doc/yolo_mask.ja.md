# yolo_mask.py — 人物マスク生成

## 概要
`yolo_mask.py` は、YOLO（人物検出）と SAM（Segment Anything Model）を組み合わせ、画像中の人物領域を検出してマスク画像（PNG）を生成するスクリプトです。360°パノラマ画像への処理に特化しており、底面付近の撮影者や、水平線付近の通行人を重点的に検出することができます。

![マスク例](../images/yolo_mask.png)

## 使い方
```
python yolo_mask.py [images_dir] [output_dir] [--add_ext] [--level N] [--expand M] [--classes IDS]
```

- `images_dir`: 入力画像ディレクトリ（省略時: `images`）
- `output_dir`: 出力マスク保存先（省略時: `masks`）
- `--add_ext`: 元の拡張子を残してさらに `.png` を追加（出力例: `hoge.jpg.png`）
- `--level N`: 検出レベル（0〜3、デフォルト=1）。値を上げると局所領域での高精度抽出が有効になります。
- `--expand M`: 検出領域を広げるピクセル数（デフォルト=2）
- `--classes IDS`: YOLOクラスIDのカンマ区切り指定（デフォルト: `0` = personのみ）

例:

```
python yolo_mask.py .\images .\masks --level 2 --expand 5 --classes 0,2,3
```

## 出力について
- 出力は PNG 形式。デフォルトでは入力ファイルの拡張子を `.png` に置換して保存（`--add_ext` を使うと元名に `.png` を追加）。
- マスクは「背景=白 (255)、人物=黒 (0)」になるよう出力されます。

## 注意点
- 初回実行時に学習モデルファイルが自動でダウンロードされるため、時間がかかります。ダウンロードされたファイル(.pt)はスクリプトと同じディレクトリに配置されます。
- `--level` を上げると処理時間とメモリ使用量が増加します。
- 大きなパノラマや高解像度画像では GPU（CUDA）対応の環境が推奨されます。CUDA対応PyTorchをインストールしてください。
