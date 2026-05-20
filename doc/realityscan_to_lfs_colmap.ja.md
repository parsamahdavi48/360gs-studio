# realityscan_to_lfs_colmap.py : RealityScan CSV/PLY を LichtFeld COLMAP へ変換

[English](realityscan_to_lfs_colmap.md)

このスクリプトは、RealityScanから書き出したカメラCSVとPLYから、LichtFeld Studioで
読めるCOLMAP textデータセットを直接作ります。

RealityScanでcubemap画像と通常画像を混ぜてアラインした場合、LichtFeldの
`transforms.json` ルートではカメラ内部パラメータを正しく混在できません。COLMAP形式なら
複数カメラを `cameras.txt` に持てるため、この用途はこちらを使います。

## 出力構成

既定の出力先は専用Datasetフォルダの `output/realityscan/lfs_colmap/` です。
既存の画像とマスクはコピーせずリンクし、COLMAP sparseファイルだけを書きます。

```text
output/realityscan/
├─ images/              既存の元画像
├─ masks/               既存の元マスク
├─ transforms.json      このルートでは使わない場合がある
└─ lfs_colmap/
   ├─ images/           ../images へのリンク
   ├─ masks/            ../masks へのリンク。masksがある場合のみ
   └─ sparse/
      └─ 0/
         ├─ cameras.txt
         ├─ images.txt
         ├─ points3D.txt
         └─ points3D.ply
```

LichtFeldでは `output/realityscan/lfs_colmap/` をデータセットフォルダとして読み込みます。
`images.txt` は元画像名と元拡張子を保持するため、JPG/PNG混在のまま扱えます。

## 入力

CSVはRealityScanのRegistrationからInternal/External形式で書き出したカメラCSVを使います。
PLYは同じアライン結果・同じ座標状態で書き出した点群PLYを指定してください。
PLYはASCIIとバイナリのどちらでも扱えますが、点群の頂点に `x/y/z` が含まれている必要があります。

## 例

```powershell
python realityscan_to_lfs_colmap.py `
  D:\3DGS\sakume\output\realityscan\rs_sakume.csv `
  --ply D:\3DGS\sakume\output\realityscan\rs_sakume.ply
```

LichtFeldの学習時 `Undistort` を使わずに済ませたい場合は、RealityScan CSV上で
歪み係数を持つ行だけを事前undistortして書き出せます。

```powershell
python realityscan_to_lfs_colmap.py `
  D:\3DGS\sakume\output\realityscan\rs_sakume.csv `
  --ply D:\3DGS\sakume\output\realityscan\rs_sakume.ply `
  --pre-undistort-distorted-images
```

このオプションを使う場合、既定の出力rootは
`output/realityscan/lfs_colmap_undistorted/` になります。歪みあり画像と対応マスクは
そのDataset内へremapし、すでにピンホールであるcubemap画像は可能ならハードリンクで参照します。
対応するCOLMAPカメラは、undistort後の内部パラメータを持つ `PINHOLE` として書きます。

## 座標の扱い

カメラ姿勢はCOLMAP/OpenCVの `images.txt` として書きます。RealityScanのCSVカメラ中心と
RealityScan PLYは同じ生worldにいるため、既定では両方をLichtFeld COLMAP用worldへ
回転します。

点群は `sparse/0/points3D.ply` に書き、既定でX軸+90度を適用します。これは
RealityScanからCOLMAP経由でLichtFeldへ渡すときの実測結果に合わせたものです。
カメラ姿勢も既定で同じX軸+90度を適用し、カメラ中心と `points3D.ply` が同じworldに
残るようにします。すでに目的のCOLMAP worldに変換済みのカメラ姿勢を使う場合だけ
`--camera-rotation-x-deg 0` を指定してください。

`points3D.txt` は空のCOLMAP text点群として作ります。`points3D.ply` がある場合、
LichtFeldはそちらを優先して読みます。

## マスク

LichtFeldはデータセットroot直下の `masks/`, `mask/`, `segmentation/`,
`dynamic_masks/` を探します。専用rootの `lfs_colmap/masks/` は、存在する場合
`output/realityscan/masks/` へのリンクになります。マスク画像のサイズは対応する画像と
一致している必要があります。

`--pre-undistort-distorted-images` を使う場合、歪みあり画像のマスクも最近傍で同じremapを行い、
白=keep、黒=excludeの2値に戻して書きます。すでにピンホールの画像に対応するマスクは
ハードリンクまたはコピーでそのまま使います。

## 注意

ローカルのLichtFeldソースでは、COLMAPローダーとBlender/NeRFローダーの優先度が同点です。
そのため、`sparse/0/` と `transforms.json` が同じDataset rootにある構成は曖昧です。
このスクリプトは既定でその構成を拒否します。通常は専用の `lfs_colmap/` rootを使い、
意図的に混在rootへ書く場合だけ `--allow-mixed-loader-root` を指定してください。
