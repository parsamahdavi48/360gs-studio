import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from image_io import imread_unicode, imwrite_unicode

# 進捗バー表示用 (インストールされていない場合はエラー回避)
try:
    from tqdm import tqdm
except ImportError:
    # tqdmがない場合はダミー関数を使用
    def tqdm(iterable, **kwargs):
        return iterable

# --- 設定 ---
parser = argparse.ArgumentParser(description="Add stitching remover to mask images (Parallelized)")
parser.add_argument("input_dir", nargs="?", help="Input directory of mask images")
parser.add_argument("output_dir", nargs="?", help="Output directory for cubified data (default=[input_dir])")
parser.add_argument("--single", nargs=2, type=int, metavar=("w", "h"), help="Output single mask file with the specified width & height (Ex:--single 7680 3840)")
parser.add_argument("--fov", type=float, default=None, help="Field of View of fisheye lens (legacy, default=175.0 degrees)")
parser.add_argument("--boundary-width", type=float, default=5.0, help="Total stitch boundary mask width in degrees (default=5.0, equivalent to --fov 175)")
parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")

# --- グローバル変数（ワーカープロセス内でのマスク共有用） ---
shared_base_mask = None


def boundary_width_to_fov(boundary_width_deg):
    """Convert total stitch boundary mask width to the legacy fisheye FOV value."""
    return 180.0 - boundary_width_deg


def boundary_width_to_limit_angle(boundary_width_deg):
    """Return the per-lens angular keep limit used by create_angular_stitched_mask()."""
    return boundary_width_to_fov(boundary_width_deg) / 2.0


def fov_to_boundary_width(fov_deg):
    """Convert a legacy fisheye FOV value to total stitch boundary mask width."""
    return max(0.0, 180.0 - fov_deg)


def resolve_limit_angle(fov_deg, boundary_width_deg):
    if fov_deg is not None:
        return fov_deg / 2.0
    return boundary_width_to_limit_angle(boundary_width_deg)

def init_worker(mask):
    """
    各ワーカープロセスの初期化時にベースマスクをグローバル変数にセットする。
    これにより、タスクごとに巨大な配列をPickle転送するコストを防ぐ。
    """
    global shared_base_mask
    shared_base_mask = mask

def create_angular_stitched_mask(width, height, limit_angle_deg):
    """
    パノラマ画像上の各ピクセルについて、前後レンズの中心からの角度を計算し、
    指定した角度(limit_angle_deg)を超える領域をマスクする。
    """
    lon = np.linspace(-np.pi, np.pi, width)
    lat = np.linspace(np.pi / 2, -np.pi / 2, height)
    Lon, Lat = np.meshgrid(lon, lat)

    X = np.cos(Lat) * np.cos(Lon)

    # 前方レンズ中心: (1, 0, 0), 後方レンズ中心: (-1, 0, 0)
    cos_theta_front = X
    cos_theta_back = -X

    angle_front = np.arccos(np.clip(cos_theta_front, -1.0, 1.0))
    angle_back = np.arccos(np.clip(cos_theta_back, -1.0, 1.0))

    limit_rad = np.radians(limit_angle_deg)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[(angle_front <= limit_rad) | (angle_back <= limit_rad)] = 255

    return mask

def process_single_image(file_info):
    """
    1枚の画像を処理する関数（ワーカープロセスで実行）
    """
    input_path, output_path = file_info
    
    # 画像読み込み
    img = imread_unicode(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"Skipped (Read Error): {os.path.basename(input_path)}"
    
    # グローバル変数のマスクを使用
    global shared_base_mask
    if shared_base_mask is None:
        return "Error: Base mask not initialized"

    # サイズ不一致チェック（念のため）
    if img.shape != shared_base_mask.shape:
        # リサイズするかスキップするかですが、ここではAND処理のため安全にスキップまたはリサイズ
        # 今回はベースマスクに合わせてリサイズして処理する例
        img = cv2.resize(
            img,
            (shared_base_mask.shape[1], shared_base_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # 合成
    final_mask = cv2.bitwise_and(img, shared_base_mask)
    
    # 書き出し
    if not imwrite_unicode(output_path, final_mask):
        return f"Skipped (Write Error): {os.path.basename(output_path)}"
    return None # 成功時はNoneを返す

def process_existing_masks_parallel(input_dir, output_dir, limit_angle_deg, max_workers):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    if not files:
        return

    # 最初の1枚からサイズを取得してベースマスクを作成
    first_img_path = os.path.join(input_dir, files[0])
    sample = imread_unicode(first_img_path, cv2.IMREAD_GRAYSCALE)
    if sample is None:
        print(f"Error reading sample image: {first_img_path}")
        return

    h, w = sample.shape
    print(f"Generating base mask ({w}x{h}) for directory: {input_dir} ...")
    base_mask = create_angular_stitched_mask(w, h, limit_angle_deg)

    # 処理タスクのリスト作成
    tasks = []
    for filename in files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        tasks.append((in_path, out_path))

    print(f"Processing {len(tasks)} images with {max_workers} workers...")
    print(f"[progress] 0/{len(tasks)}", flush=True)

    # 並列処理の実行
    # initializerを使うことで、base_maskを各プロセスに一度だけ渡す
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(base_mask,)) as executor:
        # tqdmでプログレスバーを表示
        results = []
        for done, res in enumerate(tqdm(executor.map(process_single_image, tasks), total=len(tasks), unit="img"), start=1):
            results.append(res)
            print(f"[progress] {done}/{len(tasks)}", flush=True)

    # エラーがあった場合のみ表示
    for res in results:
        if res:
            print(res)

def main():
    args = parser.parse_args()

    if args.input_dir:
        INPUT_DIR = args.input_dir
    elif os.path.isdir("masks"):
        INPUT_DIR = "masks"
    else:
        print("Not found 'masks' directory")
        sys.exit()

    SINGLE_SIZE = args.single if args.single else None
    OUTPUT_DIR = args.output_dir if args.output_dir else INPUT_DIR
    if args.fov is None and not (0.0 <= args.boundary_width < 180.0):
        parser.error("--boundary-width must be greater than or equal to 0 and less than 180 degrees")
    if args.fov is not None and args.fov <= 0:
        parser.error("--fov must be greater than 0 degrees")
    limit_angle_deg = resolve_limit_angle(args.fov, args.boundary_width)
    
    # 並列数（CPUコア数または指定値）
    workers = args.workers

    if SINGLE_SIZE:
        if not os.path.exists(INPUT_DIR):
            os.makedirs(INPUT_DIR)
        w, h = SINGLE_SIZE
        print(f"Generating single mask {w}x{h}...")
        mask = create_angular_stitched_mask(w, h, limit_angle_deg)
        if not imwrite_unicode(os.path.join(INPUT_DIR, "single_mask.png"), mask):
            print(f"Error writing single mask: {os.path.join(INPUT_DIR, 'single_mask.png')}")
            sys.exit(1)
        print("Processed: single_mask.png")
    else:
        base = Path(INPUT_DIR)
        # サブディレクトリ構造を維持して探索
        dirs = [p.relative_to(base) for p in [base, *base.rglob("*")] if p.is_dir()]
        
        for subdir in dirs:
            in_dir = INPUT_DIR if subdir == "." else os.path.join(INPUT_DIR, subdir)
            out_dir = OUTPUT_DIR if subdir == "." else os.path.join(OUTPUT_DIR, subdir)
            
            # ディレクトリ内にPNGがあるか簡易チェック
            has_png = any(f.lower().endswith('.png') for f in os.listdir(in_dir))
            if has_png:
                process_existing_masks_parallel(in_dir, out_dir, limit_angle_deg, workers)

if __name__ == "__main__":
    # Windows等のマルチプロセッシング対応のため main() をガード内で呼び出す
    main()
