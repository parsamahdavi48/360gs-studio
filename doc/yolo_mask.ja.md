# yolo_mask.py — 人物マスク生成

## 概要
`yolo_mask.py` は、YOLO（人物検出）と SAM（Segment Anything Model）を組み合わせ、画像中の人物領域を検出してマスク画像（PNG）を生成するスクリプトです。既定では360°パノラマ画像向けに、底面付近の撮影者や水平線付近の通行人を重点的に検出します。普通の動画フレームやデジタル一眼の連番画像では `--projection normal` を指定します。

![マスク例](../images/yolo_mask.png)

## 使い方
```
python yolo_mask.py [images_dir] [output_dir] [--add_ext] [--level N] [--expand M] [--classes IDS] [--projection equirect|normal] [--bottom-conf C] [--bottom-tta-rotations 1|2|4] [--bottom-model same|m|l|x] [--bottom-filter] [--bottom-temporal-window N] [--bottom-temporal-min-votes N]
```

- `images_dir`: 入力画像ディレクトリ（省略時: `images`）
- `output_dir`: 出力マスク保存先（省略時: `masks`）
- `--add_ext`: 元の拡張子を残してさらに `.png` を追加（出力例: `hoge.jpg.png`）
- `--level N`: 検出レベル（0〜3、デフォルト=1）。値を上げると局所領域での高精度抽出が有効になります。
- `--expand M`: SAM後の検出領域を広げる固定ピクセル数（デフォルト=2）
  - 安全のため `-16〜32` にクランプされます。
  - 負値を指定するとマスク領域を収縮します。
- `--classes IDS`: YOLOクラスIDのカンマ区切り指定（デフォルト: `0` = personのみ）
- `--projection equirect|normal`: 入力画像の種類（デフォルト: `equirect`）
  - `equirect`: 360°エクイレクタングラー画像。底面再検出を行います。
  - `normal`: 通常画像。360°専用の底面再検出を行いません。
- `--bottom-conf C`: 360底面再検出だけに使うYOLO信頼度しきい値（デフォルト: `0.3`）。
- `--bottom-tta-rotations 1|2|4`: 底面画像を回転して複数回検出し、結果を合成します（デフォルト: `1`）。
- `--bottom-model same|m|l|x`: 底面再検出だけに使うYOLOモデル。`same` は `--level` で選ばれたモデルを再利用します。`x` は重く、初回に `yolo26x.pt` のダウンロードが発生する場合があります。
- `--bottom-filter`: 信頼しにくい底面マスク成分を除外してから最終マスクに合成します。
- `--bottom-temporal-window N`: 各フレームの検出後、前後 `N` フレーム以内の底面検出結果を合成して補完します。ディレクトリ入力かつ `equirect` のときだけ有効です。
- `--bottom-temporal-min-votes N`: 時系列補完で画素を書き込む前に、近傍フレーム内で最低 `N` 回の底面検出を要求します（デフォルト: `1`）。
  - CLI専用の詳細オプションです。間隔を空けて抽出したフレームではフレーム間の動きを位置合わせしないため、前フレームのシルエットが残ることがあります。GUIプリセットでは使いません。

例:

```
python yolo_mask.py .\images .\masks --level 2 --expand 5 --classes 0,2,3
```

通常画像:

```
python yolo_mask.py .\images .\masks --projection normal --level 1
```

真上から見た撮影者など、底面検出が難しい場合:

```
python yolo_mask.py .\images .\masks --level 3 --bottom-conf 0.15 --bottom-tta-rotations 4 --bottom-filter
```

底面だけYOLO Xまで使う最大設定:

```
python yolo_mask.py .\images .\masks --level 3 --bottom-conf 0.10 --bottom-tta-rotations 4 --bottom-model x --bottom-filter
```

## 出力について
- 出力は PNG 形式。デフォルトでは入力ファイルの拡張子を `.png` に置換して保存（`--add_ext` を使うと元名に `.png` を追加）。
- マスクは「背景=白 (255)、人物=黒 (0)」になるよう出力されます。

## 注意点
- 初回実行時に学習モデルファイルが自動でダウンロードされるため、時間がかかります。スクリプトと同じディレクトリにある `.pt` は優先して使い、未配置の名前付きモデルはUltralytics側の解決に任せます。
- `--level` を上げると処理時間とメモリ使用量が増加します。
- 底面TTA、候補フィルタ、時系列補完、`--bottom-model x` は、360底面再検出部分だけの処理時間を増やします。
- GUIの下部検出強化プリセットは底面TTAと候補フィルタを使い、時系列補完は使いません。
- 大きなパノラマや高解像度画像では GPU（CUDA）対応の環境が推奨されます。CUDA対応PyTorchをインストールしてください。
