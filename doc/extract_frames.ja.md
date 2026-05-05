# extract_frames.py - ペア解析つきFFmpegフレーム抽出

## 概要

`extract_frames.py` は、エクイレクタングラー360°動画からSfM/3DGS向けの静止画を抽出し、Step 2で確認できる `selected_frames.csv` を作成します。

このツールの目的は、動画の全フレームを機械的に切り出すことではありません。SfMに使いやすいだけの視点変化を残しつつ、似すぎたフレーム、ブレたフレーム、特徴点が弱いフレームを整理し、後工程で確認しやすい画像セットを作ることです。

## 抽出戦略

360°動画は短時間でも大量のフレームを含みます。すべてをSfMに入れると処理が重くなり、似た画像が増えすぎてマッチングや再構成が不安定になることがあります。一方で、単純に大きく間引くと、視差やカバレッジが足りず、SfMに必要な手がかりを失います。

このアプリでは、まず固定間隔を基準にして抽出候補を作ります。固定間隔を基準にすることで、枚数や動画全体のカバレッジを予測しやすくします。そのうえで、直前に採用したフレームと次の候補フレームを比較し、必要に応じて候補を除外したり、中間フレームを追加したりします。

360°のエクイレクタングラー画像では、カメラの向きが変わるだけでも画像上の差分が大きく見えます。しかし、向きの変化だけではSfMに有効な視差とは限りません。そのため、ペア解析では水平回転、つまりyaw方向のズレを推定して補正し、その後に残る差分を見ます。これにより、単なる向きの違いではなく、実際に視点や見え方が変わった区間を拾いやすくしています。

自動判定は最終決定ではありません。ブレ、低テクスチャ、追跡できる特徴点の少なさなど、SfMで問題になりそうなフレームは `selected_frames.csv` に確認フラグとして残します。Step 2ではそれらを画像で確認し、採用/除外を調整できます。

`drop` 判定のフレームも画像として出力します。これは、後から確認して必要なら採用へ戻せるようにするためです。自動処理で不可逆に捨てるのではなく、SfMに使う画像セットを人が確認して仕上げる、という設計です。

## 現在の解析方式

現在の本線は、固定間隔を基準にしたペア解析です。ペア解析では、次に判断する候補フレームと直前の採用フレームを比較します。yaw補正後の残差変化で冗長除外や中間追加を判断し、候補地点だけで疎な特徴点追跡と鮮明度確認を行って、Step 2の確認フラグを作ります。

`--quick-extract` だけが解析を行わない経路です。ペア解析と変化補正をスキップし、指定した固定間隔で直接抽出します。解析処理を飛ばして、指定間隔で素早く動画を切り出したい場合に使います。

## 必要なもの

- FFmpeg (`ffmpeg`) と FFprobe (`ffprobe`) がPATH上にあること
- Pythonモジュール: NumPy、OpenCV

## 使い方

基本のペア解析抽出:

```bash
python extract_frames.py input.mp4 ./scene01 --interval-sec 1.0
```

変化補正つきペア解析:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --interval-sec 1.0 \
  --fixed-smart \
  --min-gap-sec 0.5 \
  --max-gap-sec 2.0
```

指定間隔で素早く切り出すクイック抽出:

```bash
python extract_frames.py input.mp4 ./scene01 \
  --interval-sec 1.0 \
  --quick-extract
```

ファイル名の接頭辞を指定:

```bash
python extract_frames.py input.mp4 ./scene01 --filename-prefix walk01
```

枚数推定だけを実行:

```bash
python extract_frames.py input.mp4 ./scene01 --estimate-only --print-summary-json
```

## 主なオプション

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--interval-sec` | `0.5` | 固定間隔の基準秒数。GUI既定値は `1.0` |
| `--quick-extract` | off | 解析や変化補正を行わず、固定間隔で抽出 |
| `--fixed-smart` | off | ペア解析による変化補正を有効化。冗長な候補の除外、変化が大きい区間への中間追加、最大間隔の安全採用を行う |
| `--min-gap-sec` | `0.25` | ペア解析で中間追加するときの最小間隔 |
| `--max-gap-sec` | `2.0` | 採用間隔が空きすぎないようにする安全上限 |
| `--fixed-smart-max-inserts-per-interval` | `2` | 1つの固定間隔内に追加できる中間候補の最大数 |
| `--pair-motion-profile` | `walk` | 自動しきい値のプロファイル。`walk` は近距離・歩行の1秒基準、`drone` は遠景・空撮向けに低めの残差しきい値を使う |
| `--pair-drop-threshold` | `-1` | この残差未満なら冗長候補として除外。負値は間隔とプロファイルから自動算出 |
| `--pair-add-threshold` | `-1` | この残差以上なら中間候補を追加。負値は間隔とプロファイルから自動算出 |
| `--pair-track-min-count` | `36` | 採用ペアの追跡点数がこれ未満なら `weak_match` としてStep 2確認対象にする |
| `--pair-track-min-confidence` | `0.25` | 追跡点数とカバレッジから見た信頼度しきい値 |
| `--analysis-width` | `1920` | 候補地点の特徴点追跡と鮮明度確認に使う横幅。yaw/残差の監視は内部で最大1280pxに抑える |
| `--image-ext` | `jpg` | 出力画像形式 |
| `--jpg-quality` | `2` | FFmpegのJPEG品質。小さいほど高品質 |
| `--output-mode` | `overwrite` | `selected_frames.csv` と `extract_sessions.json` の扱い。`overwrite`、`append`、`replace-video` |
| `--estimate-only` | off | 画像を書き出さず、選別と枚数表示だけを行う |

## 出力

`output_dir` の下に出力します。

- `images/<prefix>_<source_frame_index>.jpg` または `.png`
- `selected_frames.csv`
- `extract_report.json`
- `extract_sessions.json`

`selected_frames.csv` の主な列:

- `original_index`, `final_index`, `timestamp_sec`
- `status`: `ok`, `novelty_added`, `redundant_drop`, `gap_forced`, `motion_blur`, `low_texture`, `weak_match`
- `decision`: `keep` または `drop`。Step 2で編集する列
- `analysis_pipeline`: `pair` または `quick`
- `selection_reason`: `initial`, `fixed_interval`, `novelty_added`, `redundant_drop`, `gap_forced`, `endpoint`, `quick_extract`
- `residual_score`, `raw_change_score`, `yaw_shift_deg`, `track_count`, `track_coverage`, `match_confidence`
- `blur_score_final`, `sharpness_baseline`, `sharpness_ratio`
- `pair_motion_profile`, `pair_drop_threshold`, `pair_add_threshold`
- `output_file`

`--quick-extract` の場合、ペア解析を行わないため解析スコア列は空になります。

## 補足

- `drop` 行も画像として抽出されます。Step 2で確認してから適用できるようにするためです。
- 既定のファイル名接頭辞は入力動画ファイルのstemです。`--filename-prefix` で変更できます。
- 反復確認中は `--image-ext jpg` が高速です。
- このスクリプトは既存のマスクファイルを変更しません。
