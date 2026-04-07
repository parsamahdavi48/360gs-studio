# cubemap_transforms_json.py : 全球(360°パノラマ)画像用transforms.jsonのキューブマップ変換

このスクリプトは [metashape_360_lfs](https://github.com/tetraface/metashape_360_lfs) が変換した **360度画像用** `transforms.json` ファイルをさらにキューブマップ用に変換します。

つまり、以下の変換が可能です:

Metashape (Standard/Professional) > xml/pointcloud > transforms.json > キューブマップ > 3DGSソフト ([Jawset Postshot](https://www.jawset.com/), [Brush](https://github.com/ArthurBrussee/brush), [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)など)


## ディレクトリ構造

### 入力ディレクトリの例

```
(入力ディレクトリ)/
├─ metashape.xml
├─ metashape.ply
├─ transforms.json
├─ pointcloud.ply (オプション)
├─ images/
│ ├─ image_000.jpg (または .png)
│ └─ image_001.jpg
│ └─ ...
└─ masks/ # (オプション)
  ├─ image_000.png (または .jpg.png, .png.png)
  └─ image_001.png
  └─ ...
```

| ファイル | 説明 |
|------|-------------|
|metashape.ply| Metashape [ファイル > エクスポート > ポイントクラウドをエクスポート] で出力|
|metashape.xml| Metashape [ファイル > エクスポート > カメラをエクスポート] で出力|
|transforms.json| metashape_360_lfsで変換|
|pointcloud.ply| metashape_360_lfsで変換 (オプション)|

### 出力ディレクトリ例

```
(出力ディレクトリ)/
├─ transforms.json
├─ images/
│ ├─ image_000_nx.jpg (または .png)
│ ├─ image_000_ny.jpg
│ ├─ image_000_nz.jpg
│ ├─ image_000_px.jpg
│ ├─ image_000_py.jpg
│ ├─ image_000_pz.jpg
│ ├─ image_001_nx.jpg
│ └─ ...
└─ masks/
  ├─ image_000_nx.png
  └─ ...
```


## 使用例

### 基本的な使用法

カレントディレクトリにある transforms.json とimagesディレクトリ内の画像を変換: (masksディレクトリがあればそれも変換)
```
python metashape_360_lfs.py --images images --xml metashape.xml --output .
python cubemap_transforms_json.py .
```

### 詳細

出力ディレクトリを指定:
```
python cubemap_transforms_json.py . ./cubic
```

各オプション:

```
python cubemap_transforms_json.py . ./cubic \
  --yaw 45 \
  --stitch 2.5 \
  --fov 90
```

カスタム視点リスト（name/yaw/pitch）を使う場合:

```bash
python cubemap_transforms_json.py . ./cubic --views-json views_config.json --fov 90
```

 `--yaw 45 --stitch DEGREE` を指定することで、2つの魚眼画像間の縫い目部分がキューブマップ画像の中心を横切るのを防ぎます。これらのオプションは、カメラの傾きやステッチングなどの**補正なし**で出力されたInsta360やOSMO 360の画像に効果的です。

次の画像はキューブマップの各面と２つの魚眼画像の辺縁部が全球画像のうちどの領域を占めるかを図示しています。

![Example: --yaw 0](../images/yaw0.jpg)<br>
*--yaw 0*

![Example: --yaw 45](../images/yaw45.jpg)<br>
*--yaw 45*

![Example: --yaw 45 --stitch 2.5 --fov 91.5](../images/yaw45_s2_5_f91_5.jpg)<br>
*--yaw 45 --stitch 2.5 --fov 91.5*


### Brush向け

デフォルトでは、Postshot に適した座標軸変換が行われます。Brushの場合、 `--brush` を指定してください。

```
python metashape_360_lfs.py --images images --xml metashape.xml --output .
python cubemap_transforms_json.py . ./cubic --brush
```

### LichtFeld Studio向け

LichtFeld Studioの場合、 `--no_transform` を指定してください。

```
python metashape_360_lfs.py --images images --xml metashape.xml \
  --ply metashape.ply --output .
python cubemap_transforms_json.py . ./cubic --no_tranform
```

### オプション一覧

|オプション|引数|説明|
|------|---|-----------|
|--json|ファイル名|transforms.jsonを別名で扱う場合のファイル名 (default='transforms.json')|
|--mask_dir|ディレクトリ名|入力マスク画像ディレクトリ (default='<input_dir>/masks')|
|--mask_from_alpha|(no)|Extract masks from alpha channel in images|
|--invert_masks|(no)|出力マスクの白黒極性を反転します|
|--yaw|角度°|水平方向の角度をシフトします (default=45.0 degrees)|
|--stitch|角度°|スティッチング領域を除外するための角度 (default=0.0 degrees)|
|--fov|角度°|各キューブマップ面の画像のFOV (default=90.0 degrees)|
|--output_scale|倍率|出力面サイズの入力画像高さに対する倍率 (default=0.5、等倍は `1.0`)|
|--views-json|パス|カスタム視点リストJSON（`[{name,yaw,pitch,enabled}]` または `{\"views\":[...]}`）を使用|
|--no_bottom|(no)|キューブマップの底面を除外して出力|
|--no_top|(no)|キューブマップの上面を除外して出力|
|--no_image|(no)|画像の変換を行わず、transforms.json の変換のみ行います|
|--no_transform|(no)|座標軸変換を行いません|
|--brush|(no)|Brush向けの座標変換を行います|
|--duplicate|(no)|マージされたチャンク間で同名の画像を許可|

## 3DGSソフトウェアへのインポート

各ソフトウェアで以下のファイルをインポートしてください:

### Postshot / Brush

- metashape.ply (`Metashape`で出力)
- transforms.json (出力ディレクトリ内)
- images (出力ディレクトリ内)
- masks (出力ディレクトリ内: オプション)

### LichtFeld Studio

- pointcloud.ply (`metashape_360_lfs`で変換)
- transforms.json (出力ディレクトリ内)
- images (出力ディレクトリ内)
- masks (出力ディレクトリ内: オプション)
