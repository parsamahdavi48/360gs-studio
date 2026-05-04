# sky_mask.py — 空マスク生成

## 概要

`sky_mask.py` は、選択したbackendで空領域を検出し、このプロジェクトのマスク規約に
合わせたPNGマスクを出力します。白=採用、黒=除外です。既存マスクがある場合は既定で
AND合成するため、YOLO/SAM、スティッチ境界、白飛び、カスタムマスクと組み合わせて使えます。

既定backendは Mask2Former ADE20K semantic segmentation です。ユーザーがMeta SAM3.1
checkpointをローカルに配置した場合は、実験backendとしてSAM3.1も使えます。

360°エクイレクタングラー画像では、既定の `hybrid` モードで直処理と上部投影ビューを合成し、
極付近の歪みによる検出漏れを減らします。

## 使い方

```bash
python sky_mask.py [images_dir_or_file] [masks_dir] [--backend mask2former|sam31] [--projection equirect|normal] [--mode direct|top|hybrid] [--inference-size N] [--expand PX] [--min-score S] [--min-area-ratio R] [--no-top-connected] [--replace]
```

- `images_dir_or_file`: 入力画像フォルダ、または1枚の入力画像。
- `masks_dir`: 出力マスクフォルダ。既存マスクがあればAND合成します。
- `--backend mask2former|sam31`: 空検出backend（既定: `mask2former`）。
- `--projection equirect|normal`: 入力画像の種類（既定: `equirect`）。
- `--mode direct|top|hybrid`: 空検出方式（既定: `hybrid`）。
- `--inference-size N`: backend入力サイズ。384〜2048（既定: `768`、SAM3.1では現在`1008`が必須）。
- `--view-size N`: 360°上部投影ビューのサイズ。`0` は自動。
- `--expand PX`: 空の除外領域をピクセル単位で拡張。負値で収縮。
- `--min-score S`: 最小スコア。SAM3.1では`0`なら既定のテキストプロンプト信頼度`0.5`を使います。
- `--min-area-ratio R`: 小さな空候補を画像面積比で除去。
- `--no-top-connected`: 上端につながらない空候補も残します。
- `--model-dir PATH`: ローカルモデルディレクトリ、またはSAM3.1 checkpointを明示。
- `--sam-prompt TEXT`: SAM3.1 backendに渡すテキストプロンプト（既定: `sky`）。
- `--device auto|cpu|cuda`: 推論デバイス（既定: `auto`）。
- `--replace`: 既存マスクを無視して空マスクだけを書き込みます。

例:

```bash
python sky_mask.py .\images .\masks --projection equirect --mode hybrid --inference-size 768
```

保守的に再処理する場合:

```bash
python sky_mask.py .\images .\masks --min-score 0.8 --expand -2
```

## モデルファイル

既定のMask2Formerローカルモデル配置は次の通りです。

```text
models/mask2former-swin-large-ade-semantic/
  config.json
  preprocessor_config.json
  model.safetensors
```

このローカルディレクトリがない場合、Transformersが
`facebook/mask2former-swin-large-ade-semantic` をHugging Faceから解決する場合があります。

実験的なSAM3.1 backendは、ユーザーが用意したcheckpointを次の場所に置いた場合に使います。

```text
models/sam3.1/
  sam3.1_multiplex.pt
  config.json
  LICENSE
  README.md
```

SAM3.1を動かすには、アクティブなvenvにMetaの`sam3` Python packageが必要です。
現在の実装は、このpackageの画像APIにローカルSAM3.1 checkpointを渡して比較する形なので、
checkpoint keyの警告がログに出る場合があります。このpackageは現在の通常セットアップには
含めていないため、依存、Windows実行環境、ライセンス経路が安定するまではローカル比較用backendとして扱います。

ローカルで検証する場合は、別途インストールします。`sam3` packageは現在`numpy<2`を
宣言していますが、このプロジェクトはNumPy 2.xを使うため、必要な実行時依存を先に入れてから
`sam3`本体を`--no-deps`で入れます。

```bat
.\.venv\Scripts\python.exe -m pip install timm ftfy==6.1.1 iopath regex einops triton-windows pycocotools
.\.venv\Scripts\python.exe -m pip install --no-deps git+https://github.com/facebookresearch/sam3.git@847e1a3b15115a04c87c0760297f044f0555d970
```

この手順では、`sam3`が宣言している`numpy<2`要求はあえて未解決のままになります。
SAM3.1検証用venvでは`pip check`がこのmetadata conflictを報告します。

## 注意点

- 出力マスクは、空=黒 (0)、空以外=白 (255) です。
- `hybrid` は360°パノラマ向けです。通常画像では直処理にfallbackします。
- 誤検出を減らすため、上端につながる空だけを残すフィルタが既定で有効です。
- 空マスク機能は第三者モデル重みおよびデータセット由来checkpointを使います。別ライセンス/利用条件については [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) を参照してください。
