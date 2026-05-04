# sky_mask.py — セマンティック/プロンプトマスク生成

## 概要

`sky_mask.py` は、このプロジェクトの規約に合わせたPNGマスクを出力します。白=採用、黒=除外です。既存マスクがある場合は既定でAND合成します。

ファイル名は歴史的に `sky_mask.py` のままですが、現在は空以外も扱えます。

- `Mask2Former`: ADE20Kクラスを `--labels` で指定します。
- `SAM3.1`: 1つ以上の `--sam-prompt` を指定します。

360°エクイレクタングラー画像では、直処理に加えて上部/下部の投影ビューを合成できます。上部の空、下部の撮影者や三脚など、極付近で歪む対象の検出補助に使います。

## 使い方

```bash
python sky_mask.py [images_dir_or_file] [masks_dir] [--backend mask2former|sam31] [--projection equirect|normal] [--mode direct|top|bottom|hybrid|full] [--labels LABELS] [--sam-prompt TEXT] [--inference-size N] [--expand PX] [--min-score S] [--min-area-ratio R] [--no-top-connected] [--replace]
```

- `images_dir_or_file`: 入力画像フォルダ、または1枚の入力画像。
- `masks_dir`: 出力マスクフォルダ。既存マスクがあればAND合成します。
- `--backend mask2former|sam31`: セグメンテーションbackend（既定: `mask2former`）。
- `--projection equirect|normal`: 入力画像の種類（既定: `equirect`）。
- `--mode direct|top|bottom|hybrid|full`: 投影補助方式。
  - `full`: 直処理 + 上部投影 + 下部投影。
  - `hybrid`: 直処理 + 上部投影。
  - 通常画像では直処理にfallbackします。
- `--labels LABELS`: Mask2FormerのADE20Kラベル名またはIDのカンマ区切り（既定: `sky`）。
- `--sam-prompt TEXT`: SAM3.1に渡す英語プロンプト。複数回指定でき、結果はOR合成されます。
- `--inference-size N`: backend入力サイズ。384〜2048（既定: `768`、GUIのSAM3.1は現在 `1008`）。
- `--view-size N`: 360°投影ビューのサイズ。`0` は自動。
- `--expand PX`: 検出領域をピクセル単位で拡張。負値で収縮。
- `--min-score S`: Mask2Formerのスコアしきい値。
- `--min-area-ratio R`: 小さな候補を画像面積比で除去。
- `--no-top-connected`: 上端につながらない候補も残します。人物など空以外も対象にする場合は指定します。
- `--model-dir PATH`: ローカルモデルディレクトリ、またはSAM3.1 checkpointを明示。
- `--device auto|cpu|cuda`: 推論デバイス（既定: `auto`）。
- `--replace`: 既存マスクを無視して、このスクリプトの結果だけを書き込みます。

例:

```bash
python sky_mask.py .\images .\masks --projection equirect --mode full --labels sky,person --inference-size 768
```

SAM3.1で空と人物をテストする場合:

```bash
python sky_mask.py .\images .\masks --backend sam31 --mode full --inference-size 1008 --sam-prompt sky --sam-prompt person --no-top-connected --replace
```

## モデルファイル

既定のMask2Formerローカルモデル配置は次の通りです。

```text
models/mask2former-swin-large-ade-semantic/
  config.json
  preprocessor_config.json
  model.safetensors
```

このローカルディレクトリがない場合、Transformersが `facebook/mask2former-swin-large-ade-semantic` をHugging Faceから解決する場合があります。

SAM3.1 backendは、ユーザーが用意したcheckpointを次の場所に置いた場合に使います。

```text
models/sam3.1/
  sam3.1_multiplex.pt
  config.json
  LICENSE
  README.md
```

SAM3.1を動かすには、アクティブなvenvにMetaの `sam3` Python packageが必要です。現在の実装は、このpackageの画像APIにローカルSAM3.1 checkpointを渡します。

ローカルで検証する場合は、別途インストールします。`sam3` packageは現在 `numpy<2` を宣言していますが、このプロジェクトはNumPy 2.xを使うため、必要な実行時依存を先に入れてから `sam3` 本体を `--no-deps` で入れます。

```bat
.\.venv\Scripts\python.exe -m pip install timm ftfy==6.1.1 iopath regex einops triton-windows pycocotools
.\.venv\Scripts\python.exe -m pip install --no-deps git+https://github.com/facebookresearch/sam3.git@847e1a3b15115a04c87c0760297f044f0555d970
```

この手順では、`sam3` が宣言している `numpy<2` 要求はあえて未解決のままになります。SAM3.1検証用venvでは `pip check` がこのmetadata conflictを報告します。

## 注意点

- 検出対象は黒 (0)、それ以外は白 (255) です。
- Mask2Formerは複数ADE20Kラベルを1回の推論で解決し、最終マスクへ統合します。
- SAM3.1は1プロンプトずつ実行し、結果をOR合成します。
- `上端接続` フィルタは空向けです。人物、三脚、カスタムプロンプトも対象にする場合はOFFにします。
- この機能は第三者モデル重みおよびデータセット由来checkpointを使います。別ライセンス/利用条件については [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) を参照してください。
