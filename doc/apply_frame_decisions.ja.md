# apply_frame_decisions.py - フレーム確認結果の適用

## 概要

`apply_frame_decisions.py` は、`_stechdrive/frames/selected_frames.csv` の `keep` / `drop` 判定を画像フォルダへ反映します。

GUIのStep 2で使う本線は `--finalize-in-place` です。

- `scene_dir/images` を直接更新する
- `drop` の画像を削除する
- `keep` の画像ファイル名は元フレーム番号を含めて維持する
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

別フォルダへ採用フレームだけをコピーするモード:

```bash
python apply_frame_decisions.py ./scene01 --output metashape_images --clean-output
```

## オプション

- `scene_dir`: `_stechdrive/frames/selected_frames.csv` と `images/` を含むシーンフォルダ
- `--csv`: `_stechdrive/frames/` 内のCSVファイル名、または絶対パス。既定値は `selected_frames.csv`
- `--finalize-in-place`: `images/` 内で直接確定し、`_stechdrive/frames/selected_frames.csv` を採用行だけに書き換える
- `--backup-dir`: `--finalize-in-place` と同時に指定すると、変更前の `images/` をこのフォルダへフルコピーする。既存フォルダを置き換えるのは、`backups` や `images_backup` などバックアップ用と分かるパスの場合だけ。空なら画像バックアップなし
- `--output`: コピー先フォルダ名。既定値は `metashape_images`
- `--clean-output`: コピー先にある既存画像を削除してからコピーする。コピー先モード専用

## 出力

`--finalize-in-place` の場合:

- `scene_dir/images/<prefix>_<source_frame_index>.*`: 採用フレームだけが残る。ファイル名は維持
- `scene_dir/_stechdrive/frames/backups/images/*`: GUIのバックアップON、または上の例の `--backup-dir` 指定時に作る変更前 `images/` の全コピー
- `scene_dir/_stechdrive/frames/selected_frames.csv`: 採用行だけに書き換えたCSV
- `scene_dir/_stechdrive/frames/backups/selected_frames.before_finalize.csv`: 書き換え前CSVのバックアップ。既に存在する場合は連番つきで作成
- `scene_dir/_stechdrive/frames/selected_frames_keep.csv`: 採用行だけのCSV

コピー先モードの場合:

- `scene_dir/<output>/*`: 採用フレームだけのコピー
- `scene_dir/<output>/selected_frames_keep.csv`

## 注意

- `--finalize-in-place` で画像バックアップを指定しない場合、削除された `drop` 画像は復元できません。
- CSVバックアップは常に作成されますが、画像自体を復元したい場合は `--backup-dir` を指定してください。
- 採用フレームが1枚もない場合や、採用行が参照する画像が欠けている場合はエラーで停止します。
