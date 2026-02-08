# cubemap_tools_gui.py — キューブマップ変換ラッパーGUI

`cubemap_tools_gui.py` は `cubemap_transforms_json.py` をプレビュー付きで実行する PySide6 GUI です。

## 目的

このGUIは次の用途向けです。
- `FOV=90°` 固定で運用したい
- 複数ピッチ行（例: `-30,0,30`）を使いたい
- ピッチごとに6視点スロットのON/OFFを選びたい
- エクイレクタングラー画像上で切り出し範囲を確認したい
- 既存マスクを半透明で重ねて確認したい

## 起動

```bash
python cubemap_tools_gui.py --scene-dir ./scene01
```

Windows推奨:

```bat
start_cubemap_tools_gui.bat
```

## 主な入力項目

- `Scene Directory`:
  - `transforms.json` と `images/` を含む作業フォルダ。
- `Output Directory`:
  - 出力先。既定は `<scene>/cubic`。
- `Transforms JSON`:
  - 入力JSONファイル名。既定は `transforms.json`。
- `Mask Directory`:
  - 変換時のマスク入力先。プレビュー合成にも使用。
- `Yaw Offset (deg)`:
  - スロット角度の基準値。各スロットは `offset + slot*60`。
- `Pitch Rows (deg CSV)`:
  - ピッチ一覧。例: `-30,0,30`。
- `FOV`:
  - このGUIでは `90.0` 固定。
- `Preview Image`:
  - プレビュー対象のエクイレクタングラー画像。
- `Mask Overlay (%)`:
  - マスク重ね表示の不透明度。

## 視点選択

- `Apply Pitch Rows` で各ピッチ行に6スロット（`S0..S5`）が生成されます。
- 各チェックボックスで出力対象をON/OFFできます。
- 典型例:
  - pitch `0`: 6視点すべてON
  - pitch `+/-30`: 必要なスロットだけON

## 実行オプション

- `Extract mask from alpha (--mask_from_alpha)`
- `Transforms only (--no_image)`
- `No axis transform (--no_transform)`
- `Allow duplicate (--duplicate)`

## 実行時の挙動

- 実行時に `<output_dir>/views_config.json` を生成し、
  `cubemap_transforms_json.py --fov 90 --views-json <そのファイル>` を呼び出します。
- OFFのスロットは `enabled=false` として保存され、変換時に無視されます。

## 注意

- 1視点もONになっていない場合は実行できません。
- プレビューは確認用で、最終出力は `views_config.json` の内容に従います。
