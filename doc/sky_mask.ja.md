# sky_mask.py — セマンティック/プロンプトマスク生成

## 概要

`sky_mask.py` は、このプロジェクトの規約に合わせたPNGマスクを出力します。白=採用、黒=除外です。既存マスクがある場合は既定でAND合成します。

ファイル名は歴史的に `sky_mask.py` のままですが、現在は空以外も扱えます。

- `Mask2Former`: ADE20Kクラスを `--labels` で指定します。
- `SAM3.1`: 1つ以上の `--sam-prompt` を指定します。

360°エクイレクタングラー画像では、直処理に加えて上部/下部の投影ビューを合成できます。上部の空、下部の撮影者や三脚など、極付近で歪む対象の検出補助に使います。

## 使い方

```bash
python sky_mask.py [images_dir_or_file] [masks_dir] [--backend mask2former|sam31] [--projection equirect|normal] [--quality standard|high|best] [--labels LABELS] [--sam-prompt TEXT] [--subtract-sam-prompt TEXT] [--merge-mode replace|add|subtract] [--inference-size N] [--expand PX] [--min-score S] [--min-area-ratio R] [--top-connected] [--replace]
```

- `images_dir_or_file`: 入力画像フォルダ、または1枚の入力画像。
- `masks_dir`: 出力マスクフォルダ。既定では既存マスクへAND合成します。
- `--backend mask2former|sam31`: セグメンテーションbackend（既定: `mask2former`）。
- `--projection equirect|normal`: 入力画像の種類（既定: `equirect`）。
- `--quality standard|high|best`: モデルへ渡す入力素材レシピ（既定: `high`）。
  - 360°画像では `high` 以上で上部/下部投影補助と対象向けタイルを使います。
  - 通常画像では全体直処理と全体タイルを使い、360°専用の極投影は使いません。
  - `--mode direct|top|bottom|hybrid|full` は低レベルの比較用overrideとして残っています。
- `--labels LABELS`: Mask2FormerのADE20Kラベル名またはIDのカンマ区切り（既定: `sky`）。
- `--sam-prompt TEXT`: SAM3.1に渡す英語プロンプト。複数回指定でき、結果はOR合成されます。
- `--subtract-sam-prompt TEXT`: SAM3.1の検出結果から差し引く英語プロンプト。複数回指定できます。
- `--merge-mode replace|add|subtract`: 検出領域を既存マスクへどう適用するか。`add` は検出領域を黒で追加し、`subtract` は検出領域を白に戻し、`replace` は今回の検出結果だけを書き込みます。
- `--inference-size N`: backend入力サイズ。384〜2048（既定: `768`、GUIのSAM3.1は現在 `1008`）。
- `--view-size N`: 360°投影ビューのサイズ。`0` は自動。
- `--expand PX`: 検出領域をピクセル単位で拡張。負値で収縮。
- `--min-score S`: Mask2Formerのスコアしきい値。`0.00〜1.00`、`0`で無効です。
- `--min-area-ratio R`: 小さな空候補を画像面積比で除去。空マスクだけに適用します。
- `--top-connected`: 画像上端に接している空だけ残します。空マスクだけに適用し、既定はOFFです。
- `--model-dir PATH`: ローカルモデルディレクトリ、またはSAM3.1 checkpointを明示。
- `--device auto|cpu|cuda`: 推論デバイス（既定: `auto`）。
- `--replace`: 互換用の `--merge-mode replace` ショートカットです。

例:

```bash
python sky_mask.py .\images .\masks --projection equirect --quality high --labels sky,person --inference-size 768
```

SAM3.1で空と人物を指定する場合:

```bash
python sky_mask.py .\images .\masks --backend sam31 --quality high --inference-size 1008 --sam-prompt sky --sam-prompt person --replace
```

SAM3.1で既存マスクへ足す補正:

```bash
python sky_mask.py .\images .\masks --backend sam31 --quality best --inference-size 1008 --sam-prompt tripod --merge-mode add
```

SAM3.1で既存マスクから引く補正:

```bash
python sky_mask.py .\images .\masks --backend sam31 --quality best --inference-size 1008 --sam-prompt pictogram --merge-mode subtract
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

SAM3.1 backendは、次の場所のcheckpointを使います。

```text
models/sam3.1/
  sam3.1_multiplex.pt
  config.json
  LICENSE
  README.md
```

SAM3.1を動かすには、アクティブなvenvにMetaの `sam3` Python packageが必要です。標準の `setup_windows.bat` 環境ではこの実行用パッケージを導入します。GUIでは、Hugging Faceで利用申請とSAM Licenseへの同意が完了していれば、SAM3.1選択時に `sam3.1_multiplex.pt` をダウンロードできます。CLIで使う場合は、上記の場所にcheckpointを置くか `--model-dir` で指定してください。

## 注意点

- 検出対象は黒 (0)、それ以外は白 (255) です。
- Mask2Formerは複数ADE20Kラベルを1回の推論で解決し、最終マスクへ統合します。
- SAM3.1は1プロンプトずつ実行し、結果をOR合成します。
- SAM3.1の減算プロンプトは同じ入力素材レシピで検出し、肯定側のプロンプトマスクから差し引いてから合成モードを適用します。
- 小領域除去と上端接続フィルタは空ラベル/空プロンプトだけに適用します。人物、三脚、カスタムプロンプトのマスクは空用後処理では削除されません。
- この機能は第三者モデル重みおよびデータセット由来checkpointを使います。別ライセンス/利用条件については [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) を参照してください。
