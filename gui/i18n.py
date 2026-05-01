"""UI文字列 — ロケール自動判定で日本語/英語を切り替え。

モジュールロード時にシステムロケールを判定し、日本語環境なら JA、
それ以外なら EN の文字列をモジュール変数としてエクスポートする。
"""
from __future__ import annotations

import locale
import os

# ---------------------------------------------------------------------------
# 文字列テーブル
# ---------------------------------------------------------------------------

_JA: dict[str, str] = {
    # App
    "APP_TITLE": "3DGS Studio",
    "STEP1_TITLE": "1. フレーム抽出",
    "STEP2_TITLE": "2. フレーム確認",
    "STEP3_TITLE": "3. マスク生成",
    "STEP4_TITLE": "4. キューブマップ変換",

    # Common
    "BROWSE": "参照...",
    "SCENE_DIR": "シーンフォルダ",
    "OUTPUT_DIR": "出力フォルダ",
    "RUN": "実行",
    "CANCEL": "キャンセル",
    "CLOSE": "閉じる",
    "STATUS_IDLE": "待機中",
    "STATUS_RUNNING": "実行中",
    "STATUS_DONE": "完了",
    "STATUS_FAILED": "失敗",
    "STATUS_CANCELED": "キャンセル済み",
    "BUSY_MSG": "別のプロセスが実行中です。",
    "INVALID_INPUT": "入力エラー",

    # Step 1
    "INPUT_VIDEO": "入力動画",
    "EXTRACTION_MODE": "抽出モード",
    "MODE_CHANGE": "変化検出",
    "MODE_FIXED": "固定間隔",
    "CHANGE_THRESHOLD": "変化閾値",
    "MIN_GAP": "最小間隔 (秒)",
    "MAX_GAP": "最大間隔 (秒)",
    "INTERVAL": "間隔 (秒)",
    "ANALYSIS_WIDTH": "解析幅 (px)",
    "BLUR_PERCENTILE": "ブラーパーセンタイル",
    "BLUR_WINDOW": "ブラーウィンドウ",
    "IMAGE_FORMAT": "画像形式",
    "JPEG_QUALITY": "JPEG品質",
    "FFMPEG_PATH": "ffmpeg パス",
    "FFPROBE_PATH": "ffprobe パス",
    "FILENAME_PREFIX": "ファイル名接頭辞",
    "EXTRACT_FRAMES": "フレーム抽出",
    "VIDEO_INFO": "動画情報",
    "FRAME_ESTIMATE": "フレーム数推定",
    "INSTANT_ESTIMATE": "即時推定",
    "SAMPLED_ESTIMATE": "サンプル推定",
    "NO_VIDEO": "動画が選択されていません",

    # Step 2
    "OPEN_REVIEW": "レビューGUIを開く",
    "EXPORT_KEEP": "選択フレームをエクスポート",
    "FINALIZE_INPLACE": "インプレースで確定",
    "STEP2_WORKFLOW": "Step 1 で抽出  ──  レビュー+選別  ──  Step 3 (マスク生成) へ",
    "NEXT_STEP_MASK_NOTICE": "選別が完了したら Step 3 (マスク生成) へ進みます。\n人物・スティッチ・白飛びマスクを Metashape SfM の前に生成することで SfM 精度が向上します。",
    "METASHAPE_NOTICE": "マスク生成完了後、Metashape で SfM を実行してください。\n生成された masks/ フォルダを Metashape の per-image マスクとしてインポートすると、人物・スティッチ・白飛び領域が特徴点マッチングから除外され、SfM 精度が大きく向上します。\n完了後、Step 4 でキューブマップ変換に進みます。",
    "CSV_FILE": "CSVファイル名",
    "EXPORT_DIR": "エクスポート先",

    # Step 3
    "IMAGES_DIR": "画像フォルダ",
    "MASKS_DIR": "マスクフォルダ",
    "YOLO_LEVEL": "YOLOレベル",
    "YOLO_EXPAND": "マスク拡張 (px)",
    "YOLO_ADD_EXT": "拡張子を付加",
    "YOLO_CLASSES": "検出クラス",
    "STITCH_FOV": "スティッチFOV (度)",
    "STITCH_WORKERS": "ワーカー数",
    "RUN_YOLO": "YOLO実行",
    "RUN_STITCH": "スティッチ実行",
    "RUN_YOLO_STITCH": "YOLO + スティッチ実行",
    "CLASS_PRESET_PERSON": "人物のみ",
    "CLASS_PRESET_VEHICLES": "人物+車両",
    "CLASS_PRESET_ALL": "全選択",
    "CLASS_PRESET_CLEAR": "クリア",
    "OVEREXPOSURE": "白飛びマスク",
    "OVEREXPOSURE_THRESHOLD": "白飛び閾値 (RGB)",
    "OVEREXPOSURE_DILATE": "膨張半径 (px)",
    "RUN_OVEREXPOSURE": "白飛びマスク実行",
    "RUN_ALL": "全マスク実行",

    # Step 4
    "JSON_NAME": "transforms.json 名",
    "TARGET_PROFILE": "出力プロファイル",
    "PROFILE_POSTSHOT": "Postshot",
    "PROFILE_BRUSH": "Brush",
    "PROFILE_LICHTFELD": "LichtFeld",
    "PROFILE_CUSTOM": "カスタム",
    "VIEW_MODE": "ビューモード",
    "VIEW_CUSTOM": "カスタムグリッド",
    "VIEW_CUBE6": "Cube6",
    "YAW_OFFSET": "ヨーオフセット (度)",
    "YAW_SLOTS": "ヨースロット数",
    "PITCH_ROWS": "ピッチ行 (CSV)",
    "OUTPUT_SCALE": "出力スケール",
    "NO_IMAGE": "画像なし (JSONのみ)",
    "NO_TRANSFORM": "transforms不要",
    "DUPLICATE": "重複許可",
    "INVERT_MASKS": "マスク反転",
    "MASK_DIR": "マスクフォルダ",
    "MASK_FROM_ALPHA": "アルファからマスク",
    "METASHAPE_PREPROCESS": "Metashape前処理",
    "METASHAPE_XML": "Metashape XML",
    "METASHAPE_PLY": "PLYファイル",
    "SCALE_FACTOR": "スケール係数",
    "USE_PLY": "PLYを使用",
    "NO_FIX_ROTATION": "回転補正なし",
    "RUN_CUBEMAP": "キューブマップ変換",
    "PREVIEW": "プレビュー",

    # ViewConfig / Preview labels
    "VIEW_MODE_LABEL": "ビューモード:",
    "CUSTOM_GRID": "カスタムグリッド",
    "CUBE6_LABEL": "Cube6 (4面+上下)",
    "YAW_OFFSET_LABEL": "ヨーオフセット:",
    "YAW_SLOTS_LABEL": "ヨースロット:",
    "PITCH_ROWS_LABEL": "ピッチ行:",
    "APPLY": "適用",
    "DROP_TOP": "上面(+90)を除外",
    "DROP_BOTTOM": "底面(-90)を除外",
    "SELECT_ALL": "全選択",
    "DESELECT_ALL": "全解除",
    "SELECTED_VIEWS": "選択ビュー",
    "PITCH_SLOT_HEADER": "ピッチ / スロット",
    "PREVIEW_IMAGE_LABEL": "プレビュー画像:",
    "AUTO": "自動",
    "RELOAD": "更新",
    "MASK_OPACITY_LABEL": "マスク透過率:",
    "MASK_IMAGE_LABEL": "マスク画像:",
    "CLEAR": "クリア",
    "PREVIEW_OVERLAY_SECTION": "プレビューオーバーレイ設定",
    "EXCEED": "超過",
    "HIGH": "多い",
    "NO_PREVIEW": "プレビュー画像が選択されていません",
    "NO_PREVIEW_FOUND": "プレビュー画像が見つかりません",
    "PREVIEW_LOAD_FAIL": "画像の読み込みに失敗しました",

    # Step1 extra labels
    "VIDEO_INFO_LOAD": "動画情報読込",
    "SAMPLE_REFRESH": "サンプル推定を更新",
    "VIDEO_LABEL_DEFAULT": "動画: -",
    "ADVANCED_SETTINGS": "詳細設定",
    "AUTO_PREFIX_HINT": "自動 (動画ファイル名)",
    "FRAMES_UNIT": "フレーム",
    "THIN_MOTION_THRESHOLD": "立ち止まり間引き閾値",
    "THIN_MOTION_HINT": "推奨 0.6。0 で無効化。直前 keep フレームからの累積モーションがこれ未満なら drop されます。0.3-1.0 が典型範囲。",
    "NO_CACHE": "解析キャッシュを使わない",
    "NO_CACHE_HINT": "既定（チェックなし）= キャッシュを使う。同じ動画の再実行が高速化されます。チェックすると毎回フル解析（遅い、デバッグ用途）。",

    # Step2 extra labels
    "PREPROCESS_RUN_LABEL": "Metashape前処理を実行",
    "ADD_EXT_LABEL": "マスクに拡張子を付加",

    # Step3 section headers
    "YOLO_SECTION": "YOLO 人物検出",
    "STITCH_OVEREXP_SECTION": "スティッチ / 白飛び設定",

    # Step4 extra
    "OUTPUT_DETAIL": "出力詳細",
    "CONVERSION_OPTIONS": "変換オプション",
    "MS_IMAGES_LABEL": "画像フォルダ",

    # Step4 advanced output (added 2026-05)
    "ADVANCED_OUTPUT_SECTION": "高度な出力設定",
    "YAW_OFFSET_PER_FRAME": "フレーム別ヨー回転 (度)",
    "YAW_OFFSET_PER_FRAME_HINT": "0=無効, 30推奨。フレームごとに cubemap を回転して 3DGS 学習の安定性を向上",
    "OUTPUT_FORMAT": "出力フォーマット",
    "OUTPUT_FORMAT_AUTO": "自動 (入力に合わせる)",
    "JPG_QUALITY": "JPG/WebP 品質 (1-100)",
    "EXPORT_COLMAP": "COLMAP テキスト形式も出力",
    "EXPORT_COLMAP_HINT": "PostShot/Brush/公式 gaussian-splatting 等向け cameras.txt + images.txt + points3D.txt",
}

_EN: dict[str, str] = {
    # App
    "APP_TITLE": "3DGS Studio",
    "STEP1_TITLE": "1. Frame Extraction",
    "STEP2_TITLE": "2. Frame Review",
    "STEP3_TITLE": "3. Mask Generation",
    "STEP4_TITLE": "4. Cubemap Conversion",

    # Common
    "BROWSE": "Browse...",
    "SCENE_DIR": "Scene Folder",
    "OUTPUT_DIR": "Output Folder",
    "RUN": "Run",
    "CANCEL": "Cancel",
    "CLOSE": "Close",
    "STATUS_IDLE": "Idle",
    "STATUS_RUNNING": "Running",
    "STATUS_DONE": "Done",
    "STATUS_FAILED": "Failed",
    "STATUS_CANCELED": "Canceled",
    "BUSY_MSG": "Another process is running.",
    "INVALID_INPUT": "Invalid Input",

    # Step 1
    "INPUT_VIDEO": "Input Video",
    "EXTRACTION_MODE": "Extraction Mode",
    "MODE_CHANGE": "Change-Based",
    "MODE_FIXED": "Fixed Interval",
    "CHANGE_THRESHOLD": "Change Threshold",
    "MIN_GAP": "Min Gap (sec)",
    "MAX_GAP": "Max Gap (sec)",
    "INTERVAL": "Interval (sec)",
    "ANALYSIS_WIDTH": "Analysis Width (px)",
    "BLUR_PERCENTILE": "Blur Percentile",
    "BLUR_WINDOW": "Blur Window",
    "IMAGE_FORMAT": "Image Format",
    "JPEG_QUALITY": "JPEG Quality",
    "FFMPEG_PATH": "ffmpeg Path",
    "FFPROBE_PATH": "ffprobe Path",
    "FILENAME_PREFIX": "Filename Prefix",
    "EXTRACT_FRAMES": "Extract Frames",
    "VIDEO_INFO": "Video Info",
    "FRAME_ESTIMATE": "Frame Estimate",
    "INSTANT_ESTIMATE": "Instant Estimate",
    "SAMPLED_ESTIMATE": "Sampled Estimate",
    "NO_VIDEO": "No video selected",

    # Step 2
    "OPEN_REVIEW": "Open Review GUI",
    "EXPORT_KEEP": "Export Keep Frames",
    "FINALIZE_INPLACE": "Finalize In-Place",
    "STEP2_WORKFLOW": "Step 1 Extract  ──  Review + Select  ──  Proceed to Step 3 (Mask Generation)",
    "NEXT_STEP_MASK_NOTICE": "After selection, proceed to Step 3 (Mask Generation).\nGenerating person / stitch / overexposure masks before Metashape SfM significantly improves SfM accuracy.",
    "METASHAPE_NOTICE": "After mask generation, run Metashape SfM.\nImport the generated masks/ folder as per-image masks in Metashape so that people, stitching seams, and blown-out highlights are excluded from feature matching. This significantly improves SfM accuracy.\nAfter SfM, proceed to Step 4 for cubemap conversion.",
    "CSV_FILE": "CSV Filename",
    "EXPORT_DIR": "Export Folder",

    # Step 3
    "IMAGES_DIR": "Images Folder",
    "MASKS_DIR": "Masks Folder",
    "YOLO_LEVEL": "YOLO Level",
    "YOLO_EXPAND": "Mask Expand (px)",
    "YOLO_ADD_EXT": "Add Extension",
    "YOLO_CLASSES": "Detection Classes",
    "STITCH_FOV": "Stitch FOV (deg)",
    "STITCH_WORKERS": "Workers",
    "RUN_YOLO": "Run YOLO",
    "RUN_STITCH": "Run Stitch",
    "RUN_YOLO_STITCH": "Run YOLO + Stitch",
    "CLASS_PRESET_PERSON": "Person Only",
    "CLASS_PRESET_VEHICLES": "Person+Vehicles",
    "CLASS_PRESET_ALL": "Select All",
    "CLASS_PRESET_CLEAR": "Clear",
    "OVEREXPOSURE": "Overexposure Mask",
    "OVEREXPOSURE_THRESHOLD": "Overexposure Threshold (RGB)",
    "OVEREXPOSURE_DILATE": "Dilate Radius (px)",
    "RUN_OVEREXPOSURE": "Run Overexposure Mask",
    "RUN_ALL": "Run All Masks",

    # Step 4
    "JSON_NAME": "transforms.json Name",
    "TARGET_PROFILE": "Output Profile",
    "PROFILE_POSTSHOT": "Postshot",
    "PROFILE_BRUSH": "Brush",
    "PROFILE_LICHTFELD": "LichtFeld",
    "PROFILE_CUSTOM": "Custom",
    "VIEW_MODE": "View Mode",
    "VIEW_CUSTOM": "Custom Grid",
    "VIEW_CUBE6": "Cube6",
    "YAW_OFFSET": "Yaw Offset (deg)",
    "YAW_SLOTS": "Yaw Slots",
    "PITCH_ROWS": "Pitch Rows (CSV)",
    "OUTPUT_SCALE": "Output Scale",
    "NO_IMAGE": "No Image (JSON only)",
    "NO_TRANSFORM": "No Transform",
    "DUPLICATE": "Allow Duplicate",
    "INVERT_MASKS": "Invert Masks",
    "MASK_DIR": "Mask Folder",
    "MASK_FROM_ALPHA": "Mask from Alpha",
    "METASHAPE_PREPROCESS": "Metashape Preprocess",
    "METASHAPE_XML": "Metashape XML",
    "METASHAPE_PLY": "PLY File",
    "SCALE_FACTOR": "Scale Factor",
    "USE_PLY": "Use PLY",
    "NO_FIX_ROTATION": "No Rotation Fix",
    "RUN_CUBEMAP": "Run Cubemap Convert",
    "PREVIEW": "Preview",

    # ViewConfig / Preview labels
    "VIEW_MODE_LABEL": "View Mode:",
    "CUSTOM_GRID": "Custom Grid",
    "CUBE6_LABEL": "Cube6 (4 sides + top/bottom)",
    "YAW_OFFSET_LABEL": "Yaw Offset:",
    "YAW_SLOTS_LABEL": "Yaw Slots:",
    "PITCH_ROWS_LABEL": "Pitch Rows:",
    "APPLY": "Apply",
    "DROP_TOP": "Drop Top (+90)",
    "DROP_BOTTOM": "Drop Bottom (-90)",
    "SELECT_ALL": "Select All",
    "DESELECT_ALL": "Deselect All",
    "SELECTED_VIEWS": "Selected Views",
    "PITCH_SLOT_HEADER": "Pitch / Slot",
    "PREVIEW_IMAGE_LABEL": "Preview Image:",
    "AUTO": "Auto",
    "RELOAD": "Reload",
    "MASK_OPACITY_LABEL": "Mask Opacity:",
    "MASK_IMAGE_LABEL": "Mask Image:",
    "CLEAR": "Clear",
    "PREVIEW_OVERLAY_SECTION": "Preview Overlay Settings",
    "EXCEED": "exceeded",
    "HIGH": "high",
    "NO_PREVIEW": "No preview image selected",
    "NO_PREVIEW_FOUND": "Preview image not found",
    "PREVIEW_LOAD_FAIL": "Failed to load image",

    # Step1 extra labels
    "VIDEO_INFO_LOAD": "Load Video Info",
    "SAMPLE_REFRESH": "Refresh Sampled Estimate",
    "VIDEO_LABEL_DEFAULT": "Video: -",
    "ADVANCED_SETTINGS": "Advanced Settings",
    "AUTO_PREFIX_HINT": "auto (video filename)",
    "FRAMES_UNIT": "frames",
    "THIN_MOTION_THRESHOLD": "Stationary thinning threshold",
    "THIN_MOTION_HINT": "Recommended 0.6. Set to 0 to disable. Drops frames whose cumulative motion since last kept frame is below this. 0.3-1.0 is a typical range.",
    "NO_CACHE": "Skip analysis cache",
    "NO_CACHE_HINT": "Default (unchecked) = cache is used. Re-runs of the same video are much faster. Check to force full re-analysis every time (slow, mainly for debugging).",

    # Step2 extra labels
    "PREPROCESS_RUN_LABEL": "Run Metashape Preprocess",
    "ADD_EXT_LABEL": "Add extension to mask",

    # Step3 section headers
    "YOLO_SECTION": "YOLO Person Detection",
    "STITCH_OVEREXP_SECTION": "Stitch / Overexposure Settings",

    # Step4 extra
    "OUTPUT_DETAIL": "Output Detail",
    "CONVERSION_OPTIONS": "Conversion Options",
    "MS_IMAGES_LABEL": "Images Folder",

    # Step4 advanced output (added 2026-05)
    "ADVANCED_OUTPUT_SECTION": "Advanced Output Settings",
    "YAW_OFFSET_PER_FRAME": "Per-Frame Yaw Step (deg)",
    "YAW_OFFSET_PER_FRAME_HINT": "0=disabled, 30=recommended. Rotates cubemap per frame to improve 3DGS training stability",
    "OUTPUT_FORMAT": "Output Format",
    "OUTPUT_FORMAT_AUTO": "auto (match input)",
    "JPG_QUALITY": "JPG/WebP Quality (1-100)",
    "EXPORT_COLMAP": "Also export COLMAP text",
    "EXPORT_COLMAP_HINT": "Generate cameras.txt + images.txt + points3D.txt for PostShot/Brush/official gaussian-splatting",
}

# ---------------------------------------------------------------------------
# ツールチップテーブル
# ---------------------------------------------------------------------------

_TIPS_JA: dict[str, str] = {
    "SCENE_DIR": "作業対象のシーンフォルダ。images/, masks/ などのサブフォルダが自動認識されます",
    "RUN": "現在のタブの処理を開始します",
    "CANCEL": "実行中の処理を中断します",
    "INPUT_VIDEO": "エクイレクタングラー形式の360度動画ファイルを選択",
    "MODE_FIXED": "一定の秒数間隔でフレームを抽出。SfMの精度が安定しやすい (推奨: 0.8〜1.0秒)",
    "MODE_CHANGE": "フレーム間の画像変化量に基づいて自動選択。歩行撮影では固定間隔の方が安定",
    "INTERVAL": "フレーム間の秒数。0.8〜1.0秒がSfMに最適。短すぎると枚数が膨大に",
    "CHANGE_THRESHOLD": "変化検出の感度。小さいほど敏感 (多くのフレームを選択)。0.01〜0.12が目安",
    "MIN_GAP": "選択フレーム間の最小間隔 (秒)。連続した似たフレームの選択を防ぐ",
    "MAX_GAP": "選択フレーム間の最大間隔 (秒)。変化が少ない区間でも最低限のフレームを確保",
    "IMAGE_FORMAT": "出力画像の形式。jpgはファイルサイズ小、pngは無劣化",
    "JPEG_QUALITY": "ffmpegの-q:v値。1=最高品質、31=最低品質。2-5推奨",
    "VIDEO_INFO_BTN": "ffprobeで動画の解像度・FPS・長さを取得し、フレーム数を推定",
    "ANALYSIS_WIDTH": "変化検出・ブラー計算に使うデコード幅。大きいほど精度上がるが遅い",
    "BLUR_PERCENTILE": "下位N%のブラースコアを閾値とする。25=下位25%がブレ判定の候補に",
    "BLUR_WINDOW": "ブレ置換の探索範囲 (前後フレーム数)。0=自動 (FPS と最小間隔から計算)。明示指定する場合は 4-12 程度",
    "FFMPEG_PATH": "ffmpegの実行パス。PATHに通っていれば 'ffmpeg' でOK",
    "FFPROBE_PATH": "ffprobeの実行パス。動画情報の取得に使用",
    "FILENAME_PREFIX": "出力ファイル名の接頭辞。空欄なら動画ファイル名を自動使用",
    "SAMPLE_BTN": "動画の一部をサンプリングしてフレーム数を再推定 (変化検出モードのみ)",
    "CSV_FILE": "Step1で生成されたフレーム選択CSVファイル名",
    "EXPORT_DIR": "keepフレームのコピー先フォルダ名。'images'ならインプレース処理",
    "OPEN_REVIEW": "フレーム画像を1枚ずつ確認し、keep/dropを編集するGUIを開く\nB/Shift+Bでブラーワースト順ナビ、閾値一括dropも可能",
    "EXPORT_KEEP": "CSVでkeepとマークされたフレームだけを指定フォルダにコピー",
    "FINALIZE_INPLACE": "images/内のdropフレームを削除し、keepフレームを連番リネーム。元に戻せないので注意",
    "IMAGES_DIR": "エクイレクタングラー画像が入ったフォルダ (通常 images/)",
    "MASKS_DIR": "マスク画像の出力先フォルダ (通常 masks/)。既存マスクがあれば合成",
    "RUN_ALL": "YOLO人物検出 → スティッチマスク → 白飛びマスクの全工程を順番に実行",
    "RUN_YOLO_STITCH": "YOLO人物検出 → スティッチマスクの2工程を実行 (白飛びは含まない)",
    "YOLO_LEVEL": "0: YOLO直接 (高速)\n1: YOLO+SAM2 (標準、推奨)\n2: 水平帯高品質+SAM2\n3: 全方向高品質+SAM2",
    "YOLO_EXPAND": "検出マスクを指定ピクセル分膨張させて、人物の輪郭に余裕を持たせる",
    "YOLO_ADD_EXT": "マスクファイル名を image.jpg.png のように元の拡張子を残す形式にする",
    "STITCH_FOV": "360カメラの魚眼レンズFOV (度)。スティッチ境界の外側をマスク。一般的な360カメラは190前後",
    "STITCH_WORKERS": "並列処理のワーカー数。CPUコア数が目安",
    "OVEREXPOSURE_THRESHOLD": "RGB全チャンネルがこの値を超えるピクセルを白飛びと判定 (200-254)",
    "OVEREXPOSURE_DILATE": "白飛び領域を膨張させるピクセル数。ハロー/フリンジ対策。0で無効",
    "OUTPUT_DIR_CUBEMAP": "キューブマップ変換後の画像とtransforms.jsonの出力先フォルダ",
    "TARGET_PROFILE": "出力先の3DGSソフトウェアに合わせた座標変換とPLY設定のプリセット",
    "OUTPUT_SCALE": "出力キューブマップ面のサイズ。Half=入力高さの半分、Full=等倍",
    "JSON_NAME": "出力するカメラパラメータJSONのファイル名",
    "MASK_DIR_CUBEMAP": "入力マスク画像のフォルダ。キューブマップ変換時にマスクも一緒に変換",
    "MASK_FROM_ALPHA": "RGBA画像のアルファチャンネルからマスクを自動生成",
    "NO_IMAGE": "transforms.jsonのみ生成し、画像変換をスキップ (テスト用)",
    "NO_TRANSFORM": "座標軸変換を無効化。LichtFeld Studioで必要",
    "DUPLICATE": "マージされたチャンク間で同名画像を許可",
    "INVERT_MASKS": "出力マスクの白黒を反転 (ソフトウェアの要求に合わせて)",
    "PREPROCESS_CB": "Metashape XMLからtransforms.jsonを生成してからキューブマップ変換を行う",
    "MS_IMAGES": "MetashapeでSfMに使用した画像のフォルダ",
    "MS_XML": "Metashapeからエクスポートしたカメラパラメータ XML ファイル",
    "MS_PLY": "Metashapeからエクスポートした点群PLYファイル (LichtFeldでは必須)",
    "USE_PLY": "前処理でPLYの座標変換も行う。LichtFeldプロファイルでは自動ON",
    "SCALE_FACTOR": "座標系のスケール係数。通常は1.0のまま",
    "NO_FIX_ROTATION": "前処理の回転補正を無効化。通常はOFFのまま",
    "VIEW_MODE": "カスタムグリッド: ピッチ/ヨーを自由に設定\nCube6: 標準キューブマップ6面 (前後左右+上下)",
    "YAW_OFFSET": "全ビューのヨー角にオフセットを加算 (度)。スティッチ線を避けるために45度推奨",
    "YAW_SLOTS": "水平方向の分割数 (4-8)。360度をN等分した角度でビューを配置",
    "PITCH_ROWS": "垂直方向のピッチ角をカンマ区切りで指定 (度)。-90〜90。例: -30,0,30",
    "APPLY_BTN": "ピッチ行とヨースロットの変更をグリッドに反映",
    "CUBE6_DROP_TOP": "天頂面 (真上) をキューブマップから除外。空しか映らない場合に",
    "CUBE6_DROP_BOTTOM": "底面 (真下) をキューブマップから除外。三脚/撮影者が映る場合に",
    "PREVIEW_SAMPLE": "プレビューに表示するエクイレクタングラー画像のパス",
    "PREVIEW_BROWSE": "プレビュー画像を手動で選択",
    "PREVIEW_AUTO": "シーンフォルダ内の最初の画像を自動選択",
    "PREVIEW_RELOAD": "シーンフォルダの画像リストを再スキャン",
    "PREVIEW_SLIDER": "シーン内の画像を順番にスライドして切り替え",
    "MASK_OPACITY": "プレビュー上のマスク領域の赤オーバーレイ透過率 (0=非表示、100=不透明)",
    "MASK_IMAGE": "特定のマスク画像を手動指定。空欄ならマスクフォルダから自動検索",
    "MASK_IMAGE_BROWSE": "マスク画像ファイルを選択",
    "MASK_IMAGE_CLEAR": "手動指定をクリアして自動検索に戻す",
}

_TIPS_EN: dict[str, str] = {
    "SCENE_DIR": "Working scene folder. Subfolders like images/, masks/ are auto-detected",
    "RUN": "Start processing for the current tab",
    "CANCEL": "Abort the running process",
    "INPUT_VIDEO": "Select an equirectangular 360-degree video file",
    "MODE_FIXED": "Extract frames at a fixed time interval. More stable for SfM (recommended: 0.8-1.0 sec)",
    "MODE_CHANGE": "Auto-select frames based on visual change. Fixed interval is usually more reliable for walking shots",
    "INTERVAL": "Seconds between frames. 0.8-1.0 sec optimal for SfM. Too short = too many frames",
    "CHANGE_THRESHOLD": "Change detection sensitivity. Lower = more frames selected. 0.01-0.12 typical",
    "MIN_GAP": "Minimum seconds between selected frames. Prevents picking redundant similar frames",
    "MAX_GAP": "Maximum seconds between selected frames. Ensures minimum coverage in static areas",
    "IMAGE_FORMAT": "Output format. jpg = smaller files, png = lossless",
    "JPEG_QUALITY": "ffmpeg -q:v value. 1 = best quality, 31 = worst. Recommended: 2-5",
    "VIDEO_INFO_BTN": "Probe video resolution, FPS, and duration with ffprobe",
    "ANALYSIS_WIDTH": "Decode width for change/blur scoring. Higher = more accurate but slower",
    "BLUR_PERCENTILE": "Bottom N% blur scores become the threshold. 25 = bottom 25% are blur candidates",
    "BLUR_WINDOW": "Search range (frames) for blur replacement. 0 = auto (computed from FPS and min-gap). 4-12 is typical when set explicitly",
    "FFMPEG_PATH": "ffmpeg executable path. 'ffmpeg' works if it's on PATH",
    "FFPROBE_PATH": "ffprobe executable path. Used for video metadata probing",
    "FILENAME_PREFIX": "Output filename prefix. Leave empty to use the video filename",
    "SAMPLE_BTN": "Re-estimate frame count by sampling the video (change-based mode only)",
    "CSV_FILE": "Frame selection CSV filename generated by Step 1",
    "EXPORT_DIR": "Destination folder for keep frames. 'images' triggers in-place processing",
    "OPEN_REVIEW": "Open a GUI to review frames one by one and edit keep/drop decisions\nB/Shift+B for blur worst-order nav, threshold bulk drop available",
    "EXPORT_KEEP": "Copy only frames marked as keep in the CSV to the specified folder",
    "FINALIZE_INPLACE": "Delete dropped frames in images/ and renumber kept frames. Cannot be undone",
    "IMAGES_DIR": "Folder containing equirectangular images (typically images/)",
    "MASKS_DIR": "Mask output folder (typically masks/). Existing masks are AND-merged",
    "RUN_ALL": "Run YOLO person detection, stitch mask, and overexposure mask in sequence",
    "RUN_YOLO_STITCH": "Run YOLO person detection then stitch mask (no overexposure)",
    "YOLO_LEVEL": "0: YOLO direct (fast)\n1: YOLO+SAM2 (standard, recommended)\n2: High-quality horizontal band+SAM2\n3: Full high-quality+SAM2",
    "YOLO_EXPAND": "Dilate detection mask by N pixels to cover person edges",
    "YOLO_ADD_EXT": "Name mask files as image.jpg.png (keeping the original extension)",
    "STITCH_FOV": "Fisheye lens FOV in degrees. Masks regions beyond the stitch boundary. Typical 360 cameras: ~190",
    "STITCH_WORKERS": "Number of parallel workers. Use CPU core count as a guide",
    "OVEREXPOSURE_THRESHOLD": "Pixels with all RGB channels above this value are flagged as blown-out (200-254)",
    "OVEREXPOSURE_DILATE": "Dilate blown-out regions by N pixels to cover halos/fringes. 0 = disable",
    "OUTPUT_DIR_CUBEMAP": "Output folder for cubemap images and transforms.json",
    "TARGET_PROFILE": "Coordinate transform and PLY preset for the target 3DGS software",
    "OUTPUT_SCALE": "Cubemap face size. Half = half input height, Full = same as input height",
    "JSON_NAME": "Output camera parameter JSON filename",
    "MASK_DIR_CUBEMAP": "Input mask folder. Masks are converted alongside cubemap images",
    "MASK_FROM_ALPHA": "Auto-generate masks from RGBA image alpha channels",
    "NO_IMAGE": "Generate transforms.json only, skip image conversion (for testing)",
    "NO_TRANSFORM": "Disable axis transform. Required for LichtFeld Studio",
    "DUPLICATE": "Allow duplicate image filenames across merged chunks",
    "INVERT_MASKS": "Invert output mask polarity (match your software's convention)",
    "PREPROCESS_CB": "Generate transforms.json from Metashape XML before cubemap conversion",
    "MS_IMAGES": "Folder of images used for SfM in Metashape",
    "MS_XML": "Metashape-exported camera parameter XML file",
    "MS_PLY": "Metashape-exported point cloud PLY file (required for LichtFeld)",
    "USE_PLY": "Include PLY coordinate transform in preprocess. Auto-ON for LichtFeld profile",
    "SCALE_FACTOR": "Coordinate system scale factor. Usually leave at 1.0",
    "NO_FIX_ROTATION": "Disable rotation correction in preprocess. Usually leave OFF",
    "VIEW_MODE": "Custom Grid: freely set pitch/yaw angles\nCube6: standard 6-face cubemap (front/back/left/right + top/bottom)",
    "YAW_OFFSET": "Add offset to all view yaw angles (degrees). 45 recommended to avoid stitch seams",
    "YAW_SLOTS": "Horizontal divisions (4-8). Places views at 360/N degree intervals",
    "PITCH_ROWS": "Vertical pitch angles, comma-separated (degrees). -90 to 90. Example: -30,0,30",
    "APPLY_BTN": "Apply pitch row and yaw slot changes to the view grid",
    "CUBE6_DROP_TOP": "Exclude the top face (+90) from cubemap. Useful when only sky is visible",
    "CUBE6_DROP_BOTTOM": "Exclude the bottom face (-90) from cubemap. Useful when tripod/photographer is visible",
    "PREVIEW_SAMPLE": "Path of the equirectangular image shown in preview",
    "PREVIEW_BROWSE": "Manually select a preview image",
    "PREVIEW_AUTO": "Auto-select the first image in the scene folder",
    "PREVIEW_RELOAD": "Rescan the scene folder for images",
    "PREVIEW_SLIDER": "Slide through images in the scene sequentially",
    "MASK_OPACITY": "Red overlay opacity for masked regions in preview (0=hidden, 100=opaque)",
    "MASK_IMAGE": "Manually specify a mask image. Leave empty for auto-detection from mask folder",
    "MASK_IMAGE_BROWSE": "Select a mask image file",
    "MASK_IMAGE_CLEAR": "Clear manual selection and revert to auto-detection",
}

# ---------------------------------------------------------------------------
# ロケール判定 & モジュール変数エクスポート
# ---------------------------------------------------------------------------

def _detect_lang() -> str:
    """環境変数 → システムロケールの順で言語を判定。'ja' or 'en'。"""
    override = os.environ.get("STUDIO_LANG", "").strip().lower()
    if override in ("ja", "jp", "ja_jp"):
        return "ja"
    if override in ("en", "en_us", "en_gb"):
        return "en"
    try:
        loc = locale.getdefaultlocale()[0] or ""
    except Exception:
        loc = ""
    return "ja" if loc.lower().startswith("ja") else "en"


LANG = _detect_lang()
_table = _JA if LANG == "ja" else _EN
_tips = _TIPS_JA if LANG == "ja" else _TIPS_EN


def t(key: str) -> str:
    """文字列キーからローカライズ済みテキストを取得。"""
    return _table.get(key, key)


def tip(key: str) -> str:
    """ツールチップキーからローカライズ済みテキストを取得。"""
    return _tips.get(key, "")


# モジュール変数として全キーを公開 (既存コードとの互換性)
def _export_module_vars() -> None:
    g = globals()
    for key, value in _table.items():
        g[key] = value

_export_module_vars()
