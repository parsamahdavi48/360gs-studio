# apply_frame_decisions.py - フレーム確認結果の適用

## 概要

`apply_frame_decisions.py` は、`_stechdrive/frames/selected_frames.csv` の `keep` / `drop` 判定を画像フォルダへ反映します。

GUIのStep 2で使う本線は `--finalize-in-place` です。

- `scene_dir/images` を直接更新する
- `drop` の画像を削除する
- `keep` の画像ファイル名は既定では維持する
- 必要な場合は `--renumber-kept-images` で採用画像をCSV順に連番化する
- `_stechdrive/frames/selected_frames.csv` を採用フレームだけに書き換える
- 書き換え前のCSVバックアップを作る

## 使い方

画像フォルダ内で確定する基本形:

```bash
python apply_frame_decisions.py ./scene01 --finalize-in-place
```

適用前に `images/` をバックアップしてから確定:

```bash
python apply_frame_decisions.py ./scene01 --finalize-in-place --backup-dir _stechdrive/frames/backups/images
```

マスク生成やStep 4出力の前に、採用画像を連番化して確定:

```bash
python apply_frame_decisions.py ./scene01 --finalize-in-place --renumber-kept-images
```

別フォルダへ採用フレームだけをコピーするモード:

```bash
python apply_frame_decisions.py ./scene01 --output metashape_images --clean-output
```

## オプション

- `scene_dir`: `_stechdrive/frames/selected_frames.csv` と `images/` を含むシーンフォルダ
- `--csv`: `_stechdrive/frames/` 内のCSVファイル名、または絶対パス。既定値は `selected_frames.csv`
- `--finalize-in-place`: `images/` 内で直接確定し、`_stechdrive/frames/selected_frames.csv` を採用行だけに書き換える
- `--backup-dir`: `--finalize-in-place` と同時に指定すると、変更前の `images/` をこのフォルダへフルコピーする。既存フォルダを置き換えるのは、`backups` や `images_backup` などバックアップ用と分かるパスの場合だけ。空なら画像バックアップなし
- `--renumber-kept-images`: `--finalize-in-place` と同時に指定すると、採用画像をCSV順で `images/frame_000001.ext` からの連番にリネームし、フレーム/ソース台帳も更新する。マスク、`output/`、Step 4メタデータが既にある場合は停止する
- `--output`: コピー先フォルダ名。既定値は `metashape_images`
- `--clean-output`: コピー先にある既存画像を削除してからコピーする。コピー先モード専用

## 出力

`--finalize-in-place` の場合:

- `scene_dir/images/*`: 採用フレームだけが残る。ファイル名は `--renumber-kept-images` を使わない限り維持
- `scene_dir/images/frame_000001.*`, `frame_000002.*`, ...: `--renumber-kept-images` 使用時のみ作る連番ファイル。元の拡張子は維持
- `scene_dir/_stechdrive/frames/backups/images/*`: `--backup-dir` 指定時に作る変更前 `images/` の全コピー
- `scene_dir/_stechdrive/frames/selected_frames.csv`: 採用行だけに書き換えたCSV
- `scene_dir/_stechdrive/frames/backups/selected_frames.before_finalize.csv`: 書き換え前CSVのバックアップ。既に存在する場合は連番つきで作成
- `scene_dir/_stechdrive/frames/selected_frames_keep.csv`: 採用行だけのCSV
- `_stechdrive/frames/extract_sessions.json` と `_stechdrive/sources/image_sets.json`: 採用画像を連番化した場合はパス台帳を更新

コピー先モードの場合:

- `scene_dir/<output>/*`: 採用フレームだけのコピー
- `scene_dir/<output>/selected_frames_keep.csv`

## 注意

- `--finalize-in-place` で画像バックアップを指定しない場合、削除された `drop` 画像は復元できません。
- CSVバックアップは常に作成されますが、画像自体を復元したい場合は `--backup-dir` を指定してください。
- 採用フレームが1枚もない場合や、採用行が参照する画像が欠けている場合はエラーで停止します。
- `--renumber-kept-images` は `--finalize-in-place` 専用です。Step 3やStep 4の成果物が旧ファイル名を参照している可能性があるため、`masks/`、マスク台帳、`output/`、`_stechdrive/step4/` がある状態では実行しません。
