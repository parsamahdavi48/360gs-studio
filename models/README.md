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

YOLO26-sem semantic-mask placement:

```text
models/
  ultralytics/
    yolo26s-sem.pt
```

SAM3.1 prompt-mask placement:

```text
models/
  sam3.1/
    sam3.1_multiplex.pt
    LICENSE
    README.md
```

The GUI can download `sam3.1_multiplex.pt` into this location after the user has
Hugging Face access to `facebook/sam3.1` and accepts the SAM License. Local
manual placement still works.

Legacy YOLO/SAM `.pt` files in the repository root are still detected for
compatibility, but new local files should be placed under `models/ultralytics/`.

## 日本語

このディレクトリは、ローカルのモデル重みやダウンロード済みチェックポイント用です。
モデルファイルはGitリポジトリにもリリースZIPにも含めません。

YOLO/SAMの新しい標準配置は `models/ultralytics/` です。既存互換のため、
リポジトリ直下の `.pt` も引き続き読み込みますが、新しく配置する場合は
`models/ultralytics/` を使ってください。

YOLO26-semセマンティックマスクは `models/ultralytics/yolo26s-sem.pt`、
SAM3.1プロンプトマスクは `models/sam3.1/sam3.1_multiplex.pt` を使います。
SAM3.1は、Hugging Faceで `facebook/sam3.1` の利用申請とSAM Licenseへの同意が
完了していれば、GUIからこの場所へダウンロードできます。手動配置も引き続き使えます。
