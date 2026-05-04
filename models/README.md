# Local Model Files

This directory is reserved for local model weights and downloaded checkpoints.
Model files are not committed to this repository and are not included in release
ZIP archives.

Recommended YOLO/SAM placement:

```text
models/
  ultralytics/
    yolo26m.pt
    yolo26l.pt
    yolo26x.pt
    sam2.1_l.pt
```

Mask2Former sky-mask placement:

```text
models/
  mask2former-swin-large-ade-semantic/
    config.json
    preprocessor_config.json
    model.safetensors
```

Legacy YOLO/SAM `.pt` files in the repository root are still detected for
compatibility, but new local files should be placed under `models/ultralytics/`.

## 日本語

このディレクトリは、ローカルのモデル重みやダウンロード済みチェックポイント用です。
モデルファイルはGitリポジトリにもリリースZIPにも含めません。

YOLO/SAMの新しい標準配置は `models/ultralytics/` です。既存互換のため、
リポジトリ直下の `.pt` も引き続き読み込みますが、新しく配置する場合は
`models/ultralytics/` を使ってください。
