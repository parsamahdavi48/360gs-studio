# overexposure_mask.py - 白飛びマスク生成

## 概要

`overexposure_mask.py` は、元画像の白飛び画素を検出し、マスクPNGへ合成します。

このリポジトリのマスク規約は次の通りです。

- 白 (`255`) = 使用する画素
- 黒 (`0`) = 除外する画素

RGB全チャンネルがしきい値を超えた画素を白飛びとして扱います。検出領域は必要に応じて膨張し、既存マスクがあればAND合成します。既存マスクがない場合は、白地に黒の白飛び領域を持つマスクを新規作成します。

## 使い方

既存マスクへ白飛び領域を合成:

```bash
python -m core.overexposure_mask ./scene01/images ./scene01/masks
```

しきい値を少し下げ、膨張幅を広げる:

```bash
python -m core.overexposure_mask ./scene01/images ./scene01/masks --threshold 250 --dilate 2
```

既存マスクを無視し、白飛びだけのマスクを書き出す:

```bash
python -m core.overexposure_mask ./scene01/images ./scene01/masks --replace
```

## オプション

- `images_dir`: 入力画像フォルダ
- `masks_dir`: マスク出力フォルダ。`--replace` を指定しない限り、既存マスクへAND合成する
- `--threshold`: 白飛び判定のRGBしきい値。8bit相当値で指定。既定値は `254`、有効範囲は `1-254`
- `--dilate`: 検出領域を広げる半径ピクセル。既定値は `1`、`0` で無効
- `--workers`: 並列ワーカー数。既定値はCPU数
- `--replace`: 既存マスクを無視し、白飛びだけのマスクを書き出す

## 補足

- 対応画像拡張子は `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` です。
- RGB/RGBA画像では、3つの色チャンネルすべてがしきい値を超えた画素だけを白飛びとして扱います。
- グレースケール画像では、1チャンネルの値をしきい値と比較します。
- 16bit整数画像では、8bit相当のしきい値を画像のビット深度へスケールして比較します。
- 既存マスクと白飛びマスクのサイズが違う場合、既存マスクをnearest-neighborでリサイズしてから合成します。
- 出力マスク名は `<image_stem>.png` です。
