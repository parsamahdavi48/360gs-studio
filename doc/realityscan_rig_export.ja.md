# realityscan_rig_export.py - RealityScan リグパッケージ出力

`realityscan_rig_export.py` は、360度入力から透視投影画像を切り出し、RealityScan のリグ読込向けに対応する XMP サイドカーを出力します。

## 目的

次をまとめて出力したいときに使います。
- 360画像からの視点別透視投影画像
- 各画像と同名対応の XMP サイドカー
- RealityScan 用のリグ情報（`Rig` / `RigInstance` / `RigPoseIndex`）
- 同一フォルダ内のマスクレイヤー（`<image_name>.mask.png`）

## 基本実行

```bash
python realityscan_rig_export.py <scene_dir> <rs_output_root> --views-json <views_config.json>
```

例:

```bash
python realityscan_rig_export.py ./scene01 ./scene01/realityscan_rig --views-json ./scene01/realityscan_rig/views_config.json --mask_dir ./scene01/masks --mask_from_alpha
```

## 入力要件

- `<scene_dir>/<transforms.json>` が必要（`--json` で変更可）
- `camera_model` は `EQUIRECTANGULAR` であること
- `frames[].file_path` が実在画像を指していること
- 各フレームに有効な `4x4` の `transform_matrix` があること

## 出力構成

`<rs_output_root>` 配下に一式を出力します。

- `inputs/`
  - 透視投影画像
  - 対応 XMP (`<image_stem>.xmp`)
  - 任意マスク (`<image_name>.mask.png`)
- `views_config.json`
- `manifest.csv`
- `realityscan_project.json`

RealityScan には `inputs/` フォルダを読み込ませてください。

## 主なオプション

- `--views-json`: 有効視点リスト（`name`, `yaw`, `pitch`）
- `--fov`: 切り出しFOV（既定 `90`）
- `--pose_prior`: XMP姿勢Prior（`initial|exact|locked`, 既定 `exact`）
- `--calibration_prior`: XMP内部パラメータPrior（`initial|fixed|exact|locked`, 既定 `fixed`）
- `--focal35mm`: XMPの焦点距離上書き（未指定時はFOVから自動換算。`90deg -> 18mm`）
- `--mask_dir`: 入力マスクフォルダ
- `--mask_from_alpha`: RGBA入力時にアルファからマスク生成
- `--invert_masks`: 出力マスクの白黒反転
- `--no_transform`: 入力姿勢の軸変換を無効化

## 注意

- アルファ由来マスクと外部マスクが両方ある場合は AND 合成して出力します。
- `Rig` はエクスポート全体で共通、`RigInstance` は入力フレームごとに生成します。
- 同じ視点レイアウトなら、再出力してもリグIDは安定します。
