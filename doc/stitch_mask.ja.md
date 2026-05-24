# stitch_mask.py — スティッチング除外マスク生成

## 概要

[`core/stitch_mask.py`](../core/stitch_mask.py) は、360度パノラマ向けマスクからスティッチング領域（前後レンズ辺縁にあたり、２つのレンズ間のつなぎ目となる部分）を除去します。

![マスク例](../images/stitch_mask.png)<br>

このスクリプトでは単一のマスク画像を生成するだけでなく、人物除去用などのマスクが既に作成されている場合にスティッチング領域を追記することができます。

狭い室内など、周囲との距離が近い空間で撮影したときにスティッチング領域の継ぎ目が目立つ場合、マスクを適用して継ぎ目をアライメントやスプラット化処理から除外することで、品質の向上が見込めます。


## 使い方
```
python -m core.stitch_mask [-h] [--single w h] [--boundary-width DEG] [--fov FOV] [--workers WORKERS] [input_dir] [output_dir]
```

- **`input_dir`**: 入力マスクが入ったディレクトリ（省略時は `masks` を探す）
- **`output_dir`**: 出力先ディレクトリ（省略時は入力ディレクトリと同じ）
- **`--single w h`**: 指定解像度でベースマスクを1枚生成して入力ディレクトリへ保存します（ファイル名:`single_mask.png`）
- **`--boundary-width`**: 除外するスティッチ境界帯の合計幅（度）。デフォルトは `5.0` で、従来の `--fov 175` と同等です。
- **`--fov`**: 後方互換用。フィッシュアイの有効FOV（度）。指定した場合は `--boundary-width` より優先されます。
- **`--workers`**: 並列ワーカー数（デフォルトはCPUコア数）

## 使用例

yolo_mask.pyなどで生成されたマスク画像が既に `masks` フォルダにある場合（引数を省略して `masks` を使用）:

```
python -m core.stitch_mask
```

マスク用フォルダを指定する場合：

```
python -m core.stitch_mask input_masks
```

入力フォルダと出力フォルダを指定する例:

```
python -m core.stitch_mask input_masks output_masks
```

指定解像度の単一マスクを作る例（幅7680, 高さ3840）:

```
python -m core.stitch_mask . --single 7680 3840
```

境界マスク幅を10度に広げて処理し、ワーカー数も指定:

```
python -m core.stitch_mask input_masks output_masks --boundary-width 10 --workers 8
```



## 注意点

- Insta360 StudioやDJI Studioでmp4動画を出力する際に、手ブレや傾き、スティッチングなど幾何的な補正をすべてオフにしていることが前提となります。補正を行うと、元のレンズの向きが失われるため、マスクが適切に適用されなくなります。
- 広い空間で周囲との距離が十分にある場合は継ぎ目がほぼ目立たないため、マスクを適用するとかえって画質が落ちることも考えられます。シーンに応じて境界マスク幅を小さくしたり、またはマスクを使用しないなど、適切にご利用ください。
