# mask_tools_gui.py — マスク生成ラッパーGUI

## 概要

`mask_tools_gui.py` は、以下2つのスクリプトをまとめて実行する PySide6 GUI です。

- `yolo_mask.py`（人物マスク生成）
- `stitch_mask.py`（スティッチ領域マスク付与）

抽出済みフレーム（`images/`）からマスク（`masks/`）を作る用途を想定しています。

## 起動

```bash
python mask_tools_gui.py --scene-dir ./scene01
```

Windows では以下も使えます。

```bat
start_mask_tools_gui.bat
```

## 主な入力項目

- `Scene Directory`:
  - ベースフォルダ。`Apply Scene Paths` で `images` / `masks` を自動入力します。
- `Images Directory`:
  - `yolo_mask.py` の入力画像フォルダ。
- `Masks Directory`:
  - `yolo_mask.py` の出力先。
  - `stitch_mask.py` 実行時は入力/出力の両方として使います。
- `YOLO Level`:
  - `yolo_mask.py --level`（0〜3）
- `YOLO Expand (px)`:
  - `yolo_mask.py --expand`
- `YOLO Add Ext`:
  - `yolo_mask.py --add_ext`
- `Stitch FOV (deg)`:
  - `stitch_mask.py --fov`
- `Stitch Workers`:
  - `stitch_mask.py --workers`

## 各値の意味と調整指針

### `YOLO Level`（デフォルト: `1`）

- 何を制御するか:
  - 人物検出の探索強度です。値が大きいほど追加の局所処理が増え、検出漏れに強くなります。
- 値を上げると:
  - 検出は強くなる傾向
  - 処理時間とVRAM使用量は増加
- 目安:
  - `0`: 最速だが漏れやすい
  - `1`: 速度と精度のバランス
  - `2~3`: 高精度重視（重い）

### `YOLO Expand (px)`（デフォルト: `2`）

- 何を制御するか:
  - 検出された人物領域を何ピクセル膨張/収縮させるか
- 符号の意味:
  - 正の値: 膨張（人物周辺を広めに黒くする）
  - 負の値: 収縮（人物領域を細くする）
  - `0`: 形状変更なし
- 値を上げると:
  - 人物の取り残しは減るが、背景を巻き込みやすくなる

### `YOLO Add Ext`（デフォルト: OFF）

- 何を制御するか:
  - 出力名を `元ファイル名.png` にするか
- OFF:
  - `frame_000001.jpg` -> `frame_000001.png`
- ON:
  - `frame_000001.jpg` -> `frame_000001.jpg.png`
- 通常は OFF 推奨（名前が自然で扱いやすい）

### `Stitch FOV (deg)`（デフォルト: `175.0`）

- 何を制御するか:
  - 魚眼の有効視野角（度）。`stitch_mask.py` では内部的に `fov/2` を境界角として使用します。
- 値を下げると:
  - マスクされるスティッチ境界領域が広がる（保守的）
- 値を上げると:
  - マスク領域が狭まる（攻める）
- 注意:
  - 撮影時に手ブレ補正や水平補正を強くかけた素材では、幾何がずれて想定どおりに効かない場合があります。

### `Stitch Workers`（デフォルト: `CPUコア数`）

- 何を制御するか:
  - `stitch_mask.py` の並列プロセス数
- 値を上げると:
  - 速くなる可能性はあるが、CPU使用率が高くなりPC操作性は下がる
- 実用目安:
  - 作業しながら回すなら `コア数-2` 程度
  - バッチ処理専用なら `コア数` でも可

## デフォルト値は妥当か？

結論として、現状デフォルトは「まず動かす」用途として妥当です。

- `YOLO Level=1`:
  - 速度と精度のバランスがよく、初期値として適切
- `YOLO Expand=2`:
  - 人物境界の取り残しを減らしつつ、過剰マスクになりにくい
- `Stitch FOV=175`:
  - 360カメラの一般的設定に沿った無難な値
- `Workers=CPUコア数`:
  - 最速寄り。重い場合は手動で下げればよい

## 8Kフレーム（7680x3840）での実務プリセット

まずは次の値から開始するのを推奨します。

- `YOLO Level=1`
- `YOLO Expand=2`
- `YOLO Add Ext=OFF`
- `Stitch FOV=175`
- `Stitch Workers=コア数-2`（操作しながら実行する場合）

仕上がりを見て調整:

- 人物マスク漏れが目立つ: `Level 1 -> 2`、`Expand 2 -> 3~4`
- 背景を削りすぎる: `Expand 2 -> 1 or 0`
- スティッチ境界の破綻が残る: `FOV 175 -> 170`
- マスクしすぎる: `FOV 175 -> 178~180`

## 実行ボタン

1. `Run YOLO Mask`:
  - `yolo_mask.py` のみ実行
2. `Run Stitch Mask`:
  - `stitch_mask.py` のみ実行
3. `Run YOLO + Stitch`:
  - YOLO完了後にStitchを連続実行

## 補足

- GUIは内部でCLIスクリプトをそのままサブプロセス実行するため、挙動はCLIと一致します。
- 実行ログは下部ログパネルで確認できます。
