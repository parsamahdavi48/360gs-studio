# sky_mask.py — 空マスク生成

## 概要

`sky_mask.py` は、Mask2Former ADE20K のsemantic segmentationで空領域を検出し、
このプロジェクトのマスク規約に合わせたPNGマスクを出力します。白=採用、黒=除外です。
既存マスクがある場合はAND合成するため、YOLO/SAM、スティッチ境界、白飛び、
カスタムマスクと組み合わせて使えます。

360°エクイレクタングラー画像では、既定の `hybrid` モードで直処理と上部投影ビューを合成し、
極付近の歪みによる検出漏れを減らします。

## 使い方

```bash
python sky_mask.py [images_dir_or_file] [masks_dir] [--projection equirect|normal] [--mode direct|top|hybrid] [--inference-size N] [--expand PX] [--min-score S] [--min-area-ratio R] [--no-top-connected]
```

- `images_dir_or_file`: 入力画像フォルダ、または1枚の入力画像。
- `masks_dir`: 出力マスクフォルダ。既存マスクがあればAND合成します。
- `--projection equirect|normal`: 入力画像の種類（既定: `equirect`）。
- `--mode direct|top|hybrid`: 空検出方式（既定: `hybrid`）。
- `--inference-size N`: Mask2Former入力サイズ。384〜2048（既定: `768`）。
- `--view-size N`: 360°上部投影ビューのサイズ。`0` は自動。
- `--expand PX`: 空の除外領域をピクセル単位で拡張。負値で収縮。
- `--min-score S`: 空クラスの最小スコア。`0` で無効。
- `--min-area-ratio R`: 小さな空候補を画像面積比で除去。
- `--no-top-connected`: 上端につながらない空候補も残します。
- `--model-dir PATH`: ローカルMask2Formerモデルディレクトリを明示。
- `--device auto|cpu|cuda`: 推論デバイス（既定: `auto`）。

例:

```bash
python sky_mask.py .\images .\masks --projection equirect --mode hybrid --inference-size 768
```

保守的に再処理する場合:

```bash
python sky_mask.py .\images .\masks --min-score 0.8 --expand -2
```

## モデルファイル

既定のローカルモデル配置は次の通りです。

```text
models/mask2former-swin-large-ade-semantic/
  config.json
  preprocessor_config.json
  model.safetensors
```

このローカルディレクトリがない場合、Transformersが
`facebook/mask2former-swin-large-ade-semantic` をHugging Faceから解決する場合があります。

## 注意点

- 出力マスクは、空=黒 (0)、空以外=白 (255) です。
- `hybrid` は360°パノラマ向けです。通常画像では直処理にfallbackします。
- 誤検出を減らすため、上端につながる空だけを残すフィルタが既定で有効です。
- 空マスク機能は第三者モデル重みおよびデータセット由来checkpointを使います。別ライセンス/利用条件については [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) を参照してください。
