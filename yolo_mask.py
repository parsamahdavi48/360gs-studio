import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from yolo_mask_utils import EXPAND_DEFAULT, EXPAND_MAX, EXPAND_MIN, clamp_expand_px

CLASS_IDS: list[int] = [0]
LEVEL = 1
PROJECTION = "equirect"
EXPAND = EXPAND_DEFAULT

yolo = None
sam = None
px, py = None, None
ux, uy = None, None
is_bottom = None
proc_count = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make mask images for removing humans in the scene")
    parser.add_argument("images_dir", nargs="?", help="Input directory containing image files (default='./images')")
    parser.add_argument("output_dir", nargs="?", help="Output directory for storing PNG mask images (default='./masks')")
    parser.add_argument(
        "--add-ext",
        "--add_ext",
        action="store_true",
        dest="add_ext",
        help="Add a file extension forcibly (ex: hoge.jpg.png)",
    )
    parser.add_argument("--level", type=int, default=1, help="Detection level [0:3] (default=1)")
    parser.add_argument(
        "--expand",
        type=int,
        default=EXPAND_DEFAULT,
        help=f"Expand detected regions by N pixels (default={EXPAND_DEFAULT}, clamped {EXPAND_MIN}..{EXPAND_MAX})",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0",
        help="YOLO class ids (comma-separated, ex: '0,2,3'; default='0' person)",
    )
    parser.add_argument(
        "--projection",
        choices=("equirect", "normal"),
        default="equirect",
        help="Source image projection. equirect enables 360 panorama-specific bottom re-detection; normal disables it.",
    )
    return parser


def parse_classes(text):
    tokens = [t.strip() for t in str(text).split(",") if t.strip()]
    if len(tokens) == 0:
        raise ValueError("No class ids were provided")
    ids = []
    for tok in tokens:
        cls_id = int(tok)
        if cls_id < 0:
            raise ValueError(f"Class id must be >= 0: {cls_id}")
        ids.append(cls_id)
    return sorted(set(ids))


def should_run_bottom_redetection(level: int, projection: str) -> bool:
    return int(level) >= 1 and projection == "equirect"


def yolo_model_name(level: int) -> str:
    return "yolo26l.pt" if int(level) >= 2 else "yolo26m.pt"


def load_models(level: int) -> None:
    global yolo, sam
    from ultralytics import YOLO, SAM

    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo = YOLO(os.path.join(script_dir, yolo_model_name(level)))
    sam = SAM(os.path.join(script_dir, "sam2.1_l.pt"))


# =========================
# YOLO/SAM2によるマスク抽出
# =========================
def add_yolo_mask(img, mask, has_mask=0):
    global yolo, sam
    if yolo is None or sam is None:
        raise RuntimeError("YOLO/SAM models are not loaded")

    # ---------- YOLO: 指定クラス検出 ----------
    results = yolo(img, conf=0.3, classes=CLASS_IDS, verbose=False)

    bboxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes.xyxy:
            bboxes.append(box.tolist())
    if len(bboxes) == 0:
        return mask, has_mask

    # ---------- SAM2: マスク生成 ----------
    sam_results = sam(
        img,
        bboxes=bboxes,
        verbose=False,
    )

    for m in sam_results[0].masks.data:
        m = m.cpu().numpy().astype(np.uint8) * 255
        mask = np.maximum(mask, m)
    return mask, has_mask + 1


# パノラマから下方向を抽出
def get_bottom_from_pano(pano_img, size=1024):
    global px, py
    h, w = pano_img.shape[:2]

    if px is None or py is None:
        # 下方向の座標系 (u, v) を作成
        u = np.linspace(-1, 1, size)
        v = np.linspace(-1, 1, size)
        U, V = np.meshgrid(u, v)

        # キューブマップの底面から3Dベクトルへの変換
        # 底面なので X=U, Y=V, Z=-1
        X = U
        Y = V
        Z = -np.ones_like(U)

        # 3Dベクトルからパノラマ座標 (経度, 緯度) へ
        lon = np.arctan2(Y, X)
        lat = np.arctan2(Z, np.sqrt(X**2 + Y**2))

        # ピクセル座標へ変換
        px = ((lon + np.pi) / (2 * np.pi) * (w - 1))
        py = ((np.pi/2 - lat) / np.pi * (h - 1))

    # 再サンプリング
    bottom_img = cv2.remap(pano_img, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR)
    return bottom_img


# 下方向画像をパノラマに戻す
def back_to_pano_from_bottom(bottom_img, pano_width, pano_height):
    global ux, uy, is_bottom
    """
    キューブマップの底面画像をパノラマ形状に引き延ばして戻す
    (底面以外の範囲は白 255 で埋める)
    """
    bsize = bottom_img.shape[0]

    if ux is None or uy is None:
        # パノラマの全ピクセルの3Dベクトルを計算
        lon = np.linspace(-np.pi, np.pi, pano_width)
        lat = np.linspace(np.pi / 2, -np.pi / 2, pano_height)
        Lon, Lat = np.meshgrid(lon, lat)

        X = np.cos(Lat) * np.cos(Lon)
        Y = np.cos(Lat) * np.sin(Lon)
        Z = np.sin(Lat)

        # Zが負（下方向）かつ、底面(Z=-1)の面に投影したときに範囲内にあるピクセルを探す
        # 投影点 (u, v) = (X/|Z|, Y/|Z|)  ※Zは常に負
        is_bottom = (Z < 0) & (np.abs(Z) >= np.abs(X)) & (np.abs(Z) >= np.abs(Y))

        # 投影点 (u, v) = (X/|Z|, Y/|Z|) を計算
        # ※ Z=0 での除算を防ぐため、is_bottom の領域のみ計算
        U = np.zeros_like(Z)
        V = np.zeros_like(Z)
        U[is_bottom] = X[is_bottom] / np.abs(Z[is_bottom])
        V[is_bottom] = Y[is_bottom] / np.abs(Z[is_bottom])

        # キューブマップ座標 (-1~1) -> ピクセル座標
        ux = (U + 1) / 2 * (bsize - 1)
        uy = (V + 1) / 2 * (bsize - 1)

    # 背景を0で作成
    res_pano = np.zeros(
        (pano_height, pano_width, 3) if len(bottom_img.shape) == 3 else (pano_height, pano_width),
        dtype=np.uint8,
    )

    # マッピング実行
    mapped = cv2.remap(
        bottom_img,
        ux.astype(np.float32),
        uy.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 底面判定されたピクセルのみ、マッピング結果を上書き
    res_pano[is_bottom] = mapped[is_bottom]

    return res_pano


# =========================
# メイン処理
# =========================
def process_file(input_dir, output_dir, fname, add_ext=True):
    print(f"Processing: {fname}", flush=True)

    # 画像読み込み
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # マスク初期化
    mask = np.zeros((h, w), dtype=np.uint8)

    # 全体で人物検出
    mask, has_mask = add_yolo_mask(img, mask)

    # 水平線付近の高品質抽出
    if LEVEL >= 2:
        # [ni, nj, top_y, bottom_h]
        level_defs = [
            [4, 1, 0.25, 0.75],
            [8, 2, 0.25, 0.75],
            [16, 6, 0.20, 0.80],
        ]
        level_idx = min(LEVEL - 2, len(level_defs) - 1)
        ni, nj = level_defs[level_idx][0], level_defs[level_idx][1]
        top_y = int(h * level_defs[level_idx][2])
        bottom_y = int(h * level_defs[level_idx][3])
        subw = int(w // ni)
        subh = int((bottom_y - top_y) / nj)
        pad = 20  # 重なり部分
        global proc_count
        for i in range(ni):
            x1 = max(0, i * subw - pad)
            x2 = min(w, x1 + subw + pad)
            for j in range(nj):
                y1 = max(0, top_y + j * subh - pad)
                y2 = min(h, y1 + subh + pad)
                # 一部を切り出して検出
                print(f"  Processing region {i * nj + j}/{ni * nj} ...", flush=True)
                if proc_count == 0:
                    print(f"  HQ extraction: region [{y1}:{y2}, {x1}:{x2}]", flush=True)
                subimg = img[y1:y2, x1:x2]
                submask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                submask, has_submask = add_yolo_mask(subimg, submask)

                # 元の画像に反映
                if has_submask > 0:
                    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], submask)
                    has_mask += has_submask
        proc_count += 1

    # 下方向のみ展開画像で再検出（エクイレクタングラー360画像専用）
    if should_run_bottom_redetection(LEVEL, PROJECTION):
        bsize = int(w / 4)
        bottom = get_bottom_from_pano(img, size=bsize)
        bottom_mask = np.zeros((bsize, bsize), dtype=np.uint8)
        bottom_mask, has_bottom = add_yolo_mask(bottom, bottom_mask)
        if has_bottom > 0:
            bottom_mask = back_to_pano_from_bottom(bottom_mask, w, h)
            mask = np.maximum(mask, bottom_mask)
            has_mask += has_bottom

    if has_mask > 0 and EXPAND > 0:
        # 検出領域を指定pxぶん膨張
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=EXPAND)
    elif has_mask > 0 and EXPAND < 0:
        # 負値は検出領域を収縮
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=-EXPAND)

    # ここで反転（背景=白 / 人物=黒）
    mask = 255 - mask

    # ---------- 保存 ----------
    if add_ext:
        outname = fname + ".png"
    else:
        outname = os.path.splitext(fname)[0] + ".png"
    out_path = os.path.join(output_dir, outname)
    cv2.imwrite(out_path, mask)


def main(argv: list[str] | None = None) -> int:
    global CLASS_IDS, LEVEL, PROJECTION, EXPAND, proc_count
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.images_dir if args.images_dir else "images"
    output_dir = args.output_dir if args.output_dir else "masks"
    add_ext = args.add_ext
    LEVEL = args.level
    PROJECTION = args.projection
    EXPAND = clamp_expand_px(args.expand)
    proc_count = 0
    if EXPAND != args.expand:
        print(f"Clamped --expand from {args.expand} to {EXPAND}", flush=True)

    if not os.path.isdir(input_dir) and not os.path.isfile(input_dir):
        print("python yolo_mask.py {images_dir} {masks_dir}", flush=True)
        print(os.getcwd(), flush=True)
        return 1

    try:
        CLASS_IDS = parse_classes(args.classes)
    except Exception as e:
        print(f"Invalid --classes value: {e}", flush=True)
        return 1

    print("YOLO classes:", ",".join(str(x) for x in CLASS_IDS), flush=True)
    print(f"Projection: {PROJECTION}", flush=True)

    load_models(LEVEL)

    # =========================
    # 連番画像を処理
    # =========================
    if os.path.isdir(input_dir):
        # サブディレクトリを含めて処理
        base = Path(input_dir)
        dirs = [p.relative_to(base) for p in [base, *base.rglob("*")] if p.is_dir()]
        tasks = []
        for subdir in dirs:
            dir = input_dir if subdir == "." else os.path.join(input_dir, subdir)
            task_output_dir = output_dir if subdir == "." else os.path.join(output_dir, subdir)
            os.makedirs(task_output_dir, exist_ok=True)

            image_files = sorted([
                f for f in os.listdir(dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
            ])
            tasks.extend((dir, task_output_dir, fname) for fname in image_files)

        total = len(tasks)
        print(f"[progress] 0/{total}", flush=True)
        for done, (dir, task_output_dir, fname) in enumerate(tasks, start=1):
            process_file(dir, task_output_dir, fname, add_ext)
            print(f"Processed: {fname}", flush=True)
            print(f"[progress] {done}/{total}", flush=True)
    else:
        # 単一ファイルの処理
        fname = os.path.basename(input_dir)
        source_dir = os.path.dirname(input_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("[progress] 0/1", flush=True)
        process_file(source_dir, output_dir, fname, add_ext)
        print(f"Processed: {fname}", flush=True)
        print("[progress] 1/1", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
