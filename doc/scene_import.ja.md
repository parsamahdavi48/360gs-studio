# シーン取り込み

`シーン取り込み` は、既存フォルダ内の `images/`, `masks/`, `output/` をこのアプリのシーンとして再登録します。動画抽出や変換を実行した履歴にはせず、外部アセットの登録として記録します。

## 動作

1. 上部ヘッダーの `シーン取り込み` を押します。
2. 取り込むフォルダを選びます。
3. アプリが現在のフォルダ内容を全量走査し、外部取り込み由来の管理情報を作り直します。

確認ダイアログは出しません。画像、マスク、`output/` 内の実アセットは削除しません。

## 再取り込み

同じフォルダで再度 `シーン取り込み` を実行すると、追加ではなく再登録になります。前回の外部取り込み由来メタデータは現在のフォルダ内容で置き換わります。

置き換える主な情報:

- `_stechdrive/frames/selected_frames.csv`
- `_stechdrive/sources/image_sets.json` 内の外部取り込みレコード
- `_stechdrive/masks/` 内の外部取り込みマスク記録
- `_stechdrive/step4/export_settings.json`
- `_stechdrive/step4/dataset_runs.json` 内の外部取り込みレコード

置き換え前の管理情報は `_stechdrive/imports/backups/` に保存されます。

## 検証

取り込み時には、マスクの対応、画像サイズ、`output/transforms.json` の参照、4x4行列、`pointcloud.ply` の有無などを確認します。警告は処理を止めず、`_stechdrive/imports/scene_imports.json` と画面下部ログに記録されます。
