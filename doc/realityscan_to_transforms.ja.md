# realityscan_to_transforms.py : RealityScan CSV を NeRF transforms に変換

[English](realityscan_to_transforms.md)

このスクリプトは、RealityScan から書き出したカメラCSVと、任意の
RealityScan PLYを、NeRF系のデータセットへ変換します。

```text
transforms.json
pointcloud.ply
images/   (参照するだけでコピーしない)
masks/    (対応するマスクがある場合だけ参照)
```

RealityScanでcubemap画像と通常の透視画像を混ぜて再アラインし、その結果を
COLMAP形式へ組み替えずにNeRF/3DGS系ツールへ渡したい場合のための変換です。

## 例

RealityScan出力フォルダの `images/` と `masks/` をコピーせず、同じフォルダに
LichtFeld向きのJSONとPLYを書きます。

```powershell
python -m core.realityscan_to_transforms `
  D:\3DGS\sakume\output\realityscan\rs_sakume.csv `
  D:\3DGS\sakume\output\realityscan `
  --ply D:\3DGS\sakume\output\realityscan\rs_sakume.ply `
  --json-name transforms_lfs.json `
  --pointcloud-name pointcloud_lfs.ply `
  --target-profile lichtfeld
```

変換先フォルダをこの用途専用にする場合は、`--json-name transforms.json` と
`--pointcloud-name pointcloud.ply` を使います。

## 混在カメラ

CSVの全行を変換します。cubemap面だけに絞りません。

- 歪み係数が0のフレームは、フレーム単位の `PINHOLE` カメラとして書きます。
- RealityScanの歪み係数があるフレームは、フレーム単位の `OPENCV` カメラとして
  `k1` から `k4` と `p1` / `p2` を書きます。
- `w`, `h`, `fl_x`, `fl_y`, `cx`, `cy` はフレームごとに書くため、cubemap画像と
  通常画像で解像度や内部パラメータが違っても落としません。

`<CSVのフォルダ>/transforms.json` がある場合は、対応するフレームのメタデータだけ
流用します。カメラ姿勢は常にRealityScan CSVから作ります。

`transforms.json` のトップレベルのカメラ情報は、最も枚数が多いカメラグループから
選びます。フレーム単位の内部パラメータを読まないツールでも、CSVの先頭フレームではなく
主なカメラ設定で読み込まれるようにするためです。

## LichtFeldプロファイル

既定は `--target-profile lichtfeld` です。このツールキットの
Metashape→LichtFeld cubemap出力と同じ最終向きになるようにカメラ姿勢を変換し、
`pointcloud.ply` にはLichtFeldの読み込み処理に合わせたファイル座標変換を適用します。
LichtFeldは `transforms.json` 読み込み時に点群へ独自の軸変換をかけるため、
PLYに使う行列はJSONカメラ姿勢の行列と同一ではありません。

RealityScan CSVの座標系のまま残したい場合だけ `--target-profile realityscan` を使います。

現行のLichtFeldは、`transforms.json` のカメラモデル、画像サイズ、焦点距離、主点、
歪み係数をトップレベルから読みます。フレーム単位のこれらの値は使われません。そのため、
混在内部パラメータのJSONは、対応ツール向けには全行を保持できますが、LichtFeldの
JSON読み込みでは「最多のピンホールカメラ設定」ひとつとして扱われます。cubemap画像と
歪み付き通常画像をLichtFeldで完全に正しく混ぜるには、画像ごと/カメラグループごとの
カメラ情報を持てるCOLMAP形式の出力が必要です。このツールキットでは
`realityscan_to_lfs_colmap.py` を使います。

## 補足

既定では既存画像を参照するだけです。`--image-path-mode` で `images/...`、
出力フォルダからの相対パス、絶対パスを選べます。

`masks/<画像stem>.png` がある場合は、フレームに `mask_path` を書きます。
NeRF系出力では、RealityScan用の `image.jpg.mask.png` レイヤマスクは不要です。
