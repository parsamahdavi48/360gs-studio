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
    "APP_TITLE": "STechDrive 3DGS Utils",
    "APP_SUBTITLE": "Prepare 360 video frames and masks for Metashape SfM and 3DGS training",
    "WORKFLOW_LABEL": "ワークフロー",
    "STEP1_DESC": "動画からSfM向けのフレームを抽出",
    "STEP2_DESC": "抽出フレームを確認して採用/除外を確定",
    "STEP3_DESC": "人物・スティッチ境界・白飛びをマスク",
    "STEP4_DESC": "Metashape結果をキューブマップ出力へ変換",
    "STEP1_TITLE": "1. フレーム抽出",
    "STEP2_TITLE": "2. フレーム確認",
    "STEP3_TITLE": "3. マスク生成",
    "STEP4_TITLE": "4. キューブマップ変換",

    # Common
    "BROWSE": "参照...",
    "SCENE_DIR": "シーンフォルダ",
    "SCENE_DIR_PLACEHOLDER": "シーンフォルダを選択...",
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
    "INPUT_VIDEO_PLACEHOLDER": "360度動画を選択...",
    "VIDEO_FILE_FILTER": "動画ファイル (*.mp4 *.mov *.mkv *.avi *.m4v);;すべて (*.*)",
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
    "OPEN_REVIEW": "フレーム確認を開く",
    "EXPORT_KEEP": "採用フレームをエクスポート",
    "FINALIZE_INPLACE": "画像フォルダ内で確定",
    "FINALIZE_BUTTON": "選別を確定 (除外分を削除)",
    "FINALIZE_BUTTON_HINT": "除外にしたフレームを images/ から削除し、採用フレームを連番で再採番します。不可逆。バックアップが必要なら左のチェックボックスを ON に。",
    "BACKUP_BEFORE_FINALIZE": "実行前に images_backup/ にバックアップ",
    "BACKUP_BEFORE_FINALIZE_HINT": "ON: 確定前に images/ を images_backup/ にフルコピー（既存バックアップは上書き）。OFF: バックアップなし（容量節約・復元不可）。",
    "STEP2_WORKFLOW": "Step 1 で抽出  ──  確認+選別  ──  Step 3 (マスク生成) へ",

    # --- review_frames.py (Step 2 レビュー GUI) ---
    "REVIEW_TITLE": "フレーム確認",
    "REVIEW_DECISION_PREFIX": "選別: ",
    "REVIEW_DECISION_KEEP": "採用",
    "REVIEW_DECISION_DROP": "除外",
    "REVIEW_INFO_YES": "確認対象",
    "REVIEW_INFO_NO": "通常",
    "REVIEW_PROBLEMS_FORMAT": "確認対象: {n} 件 (置換済み={r}, 置換不可={f}, 自動間引き={t}) | 現在: {cur}",
    "REVIEW_INFO_FORMAT": (
        "時刻: {ts}   |   ブレ: {blur}   |   変化: {change}\n"
        "フレーム抽出時の処理: {process}"
    ),
    "REVIEW_BLUR_VALUE_FORMAT": "{score:.1f} (中央値 {median:.0f} の {pct}%)",
    "REVIEW_BLUR_VALUE_NO_MEDIAN": "{score:.1f}",
    "REVIEW_PROCESS_OK": "通常フレーム (品質基準を満たしています)",
    "REVIEW_PROCESS_REPLACED": "置換済み (元フレーム {orig} から、近くのフレーム {final} に置換)",
    "REVIEW_PROCESS_FALLBACK": "置換不可 (ブレを検出しましたが、代わりのフレームが見つかりませんでした)",
    "REVIEW_PROCESS_THINNED": "自動間引き (動きが少ない区間として除外扱い)",
    "REVIEW_BTN_PREV": "前 (←)",
    "REVIEW_BTN_NEXT": "次 (→)",
    "REVIEW_BTN_PREV_PROBLEM": "前の確認対象 (Shift+F)",
    "REVIEW_BTN_NEXT_PROBLEM": "次の確認対象 (F)",
    "REVIEW_BTN_PROBLEM_TIP": (
        "フレーム抽出時に確認対象として記録されたフレーム\n"
        "（置換済み・置換不可・自動間引き）を撮影順に巡回します。"
    ),
    "REVIEW_BTN_TOGGLE": "採用/除外 切替 (Space)",
    "REVIEW_BTN_JUMP": "番号へ移動",
    "REVIEW_BTN_SAVE": "保存 (S)",
    "REVIEW_JUMP_PLACEHOLDER": "番号",
    "REVIEW_BLUR_THRESHOLD_LABEL": "  ブレ評価値の閾値:",
    "REVIEW_BLUR_DROP_BTN": "閾値以下をまとめて除外",
    "REVIEW_BLUR_THRESHOLD_PLACEHOLDER": "ブレ評価値",
    "REVIEW_NO_PROBLEMS": "確認対象フレームはありません",
    "REVIEW_IMAGE_NOT_FOUND": "画像が見つかりません:\n{path}",
    "REVIEW_IMAGE_LOAD_FAILED": "画像の読み込みに失敗しました:\n{path}",
    "REVIEW_SEQ_INTEGER_ERR": "番号は整数で指定してください。",
    "REVIEW_OUT_OF_RANGE_ERR": "番号は 1 〜 {max} の範囲で指定してください。",
    "REVIEW_INVALID_INPUT": "入力エラー",
    "REVIEW_INFO_HEADER": "情報",
    "REVIEW_THRESHOLD_HEADER": "閾値",
    "REVIEW_THRESHOLD_NEED_INPUT": "閾値を入力してから再度押してください。\n現在のフレームのブレ評価値を入力欄に入れました。",
    "REVIEW_THRESHOLD_NUMERIC_ERR": "数値を入力してください。",
    "REVIEW_BULK_DROP_HEADER": "一括除外",
    "REVIEW_BULK_DROP_RESULT": "ブレ評価値が {thr} 以下のフレーム {n} 件を除外にしました。\n保存 (S) で CSV に反映してください。",
    "REVIEW_SAVED_HEADER": "保存しました",
    "REVIEW_SAVED_BODY": "更新先: {path}\n採用={k}, 除外={d}",
    "REVIEW_HELP_HEADER": "── 使い方 ──",
    "REVIEW_HELP_BODY": (
        "1 枚ずつフレームを確認し、3DGS 学習や Metashape の SfM に使うフレームを選別します。\n"
        "残すフレームは「採用」、使わないフレームは「除外」です。変更は「保存 (S)」を押すまで CSV には書き戻されません。\n"
        "保存後、メイン画面の Step 2 で「選別を確定」を押すと、除外フレームが images/ から削除され、採用フレームだけが連番になります。\n\n"
        "【推奨レビュー手順】\n"
        "  1. F キーで確認対象のフレームを順番に確認します。\n"
        "     確認対象には、置換済み・置換不可・自動間引きのフレームが含まれます。\n"
        "  2. 画像下の色帯で状態を確認します。\n"
        "     オレンジ = ブレを検出したが置換不可 / 青 = 処理内容の確認 / 緑 = 通常フレーム\n"
        "  3. オレンジのフレームは画像を見て、ブレが目立つ場合だけ Space で除外にします。\n"
        "  4. 確認が終わったら S で保存し、メイン画面で「選別を確定」を実行します。\n\n"
        "フレーム抽出時点で、ブレが強いフレームは近くのブレが少ないフレームへ自動置換されています。\n"
        "そのため、実際に除外判断が必要になりやすいのは「置換不可」のフレームです。置換済みや通常フレームは、見た目に問題がなければ採用のままで構いません。\n\n"
        "【除外の目安】\n"
        "  ・人物、車両、揺れる枝など、静止シーンとして扱えない動く被写体\n"
        "  ・置換不可フレームのうち、画像確認でブレが目立つもの\n"
        "  ・レンズフレアや強い太陽光で、大部分の画素が飽和しているもの\n"
        "  ・360度画像のステッチ継ぎ目が大きく破綻しているもの\n"
        "  ・露出過多または露出不足で、真っ白・真っ黒な領域が大半を占めるもの\n"
        "判断に迷ったら採用のままにしてください。後から再レビューできます。\n\n"
        "【画像操作】\n"
        "  マウスホイール    カーソル位置を中心に拡大 / 縮小\n"
        "  ドラッグ          拡大時にパン (画像をつかんで移動)\n"
        "  0 キー            ズームをリセット (フィット表示)\n\n"
        "【キー操作】\n"
        "  ← / →    前後のフレーム\n"
        "  Space    現在のフレームを採用/除外に切替\n"
        "  F / Shift+F  確認対象のフレームを撮影順に巡回\n"
        "  S        CSV へ保存\n"
        "  Q        終了 (未保存の変更は破棄)\n\n"
        "「閾値以下をまとめて除外」は手動の一括処理です。ブレ評価値を見て、\n"
        "明らかに低い値（例: 50 以下）をまとめて除外したい場合だけ使ってください。"
    ),
    "REVIEW_ADVISORY_FALLBACK": "要確認: フレーム抽出時にブレを検出しましたが、置換先が見つかりませんでした。画像を確認し、ブレが目立つ場合は除外してください。",
    "REVIEW_ADVISORY_REPLACED": "情報: フレーム抽出時に、ブレが少ない近くのフレームへ置換済みです。通常は採用のままで問題ありません。",
    "REVIEW_ADVISORY_THINNED": "情報: 動きが少ない区間として自動間引きされ、現在は除外扱いです。必要なら Space で採用に戻せます。",
    "REVIEW_ADVISORY_NORMAL": "通常フレームです。フレーム抽出時の品質基準を満たしています。",
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
    "STITCH_BOUNDARY_WIDTH": "境界マスク幅 (度)",
    "STITCH_WORKERS": "ワーカー数",
    "RUN_MASKS": "マスク作成実行",
    "MASK_TASKS_LABEL": "作成するマスク:",
    "MASK_TASK_YOLO": "YOLO検出",
    "MASK_TASK_STITCH": "スティッチ境界",
    "MASK_TASK_OVEREXPOSURE": "白飛び",
    "MASK_TASK_REQUIRED": "作成するマスクを1つ以上選択してください。",
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
    "STITCH_PREVIEW_SECTION": "スティッチ境界プレビュー",
    "STITCH_PREVIEW_STATUS_FORMAT": "境界 {width:g}° / 内部FOV {fov:g}°",
    "STITCH_PREVIEW_INVALID_WIDTH": "境界マスク幅は 0 以上 180 未満で指定",
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
    "THIN_MOTION_HINT": "推奨 0.6。0 で無効化。直前の採用フレームからの累積モーションがこれ未満なら除外扱いになります。0.3-1.0 が典型範囲。",
    "NO_CACHE": "解析キャッシュを使わない",
    "NO_CACHE_HINT": "既定（チェックなし）= キャッシュを使う。同じ動画の再実行が高速化されます。チェックすると毎回フル解析（遅い、デバッグ用途）。",

    # Step2 extra labels
    "PREPROCESS_RUN_LABEL": "Metashape前処理を実行",
    "ADD_EXT_LABEL": "マスクに拡張子を付加",

    # Step3 section headers
    "YOLO_SECTION": "YOLO 検出設定",
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
    "APP_TITLE": "STechDrive 3DGS Utils",
    "APP_SUBTITLE": "Prepare 360 video frames and masks for Metashape SfM and 3DGS training",
    "WORKFLOW_LABEL": "Workflow",
    "STEP1_DESC": "Extract SfM-ready frames from a 360 video",
    "STEP2_DESC": "Review extracted frames and finalize keep/drop choices",
    "STEP3_DESC": "Mask people, stitch seams, and overexposed regions",
    "STEP4_DESC": "Convert Metashape results into cubemap outputs",
    "STEP1_TITLE": "1. Frame Extraction",
    "STEP2_TITLE": "2. Frame Review",
    "STEP3_TITLE": "3. Mask Generation",
    "STEP4_TITLE": "4. Cubemap Conversion",

    # Common
    "BROWSE": "Browse...",
    "SCENE_DIR": "Scene Folder",
    "SCENE_DIR_PLACEHOLDER": "Select a scene folder...",
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
    "INPUT_VIDEO_PLACEHOLDER": "Select a 360 video...",
    "VIDEO_FILE_FILTER": "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v);;All Files (*.*)",
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
    "FINALIZE_BUTTON": "Finalize (Delete drops + renumber)",
    "FINALIZE_BUTTON_HINT": "Delete dropped frames from images/ and renumber kept frames sequentially. Irreversible. Enable the backup checkbox if you want a safety copy.",
    "BACKUP_BEFORE_FINALIZE": "Back up to images_backup/ before finalize",
    "BACKUP_BEFORE_FINALIZE_HINT": "ON: snapshot images/ to images_backup/ before finalizing (existing backup is replaced). OFF: no backup (saves disk; cannot be undone).",
    "STEP2_WORKFLOW": "Step 1 Extract  ──  Review + Select  ──  Proceed to Step 3 (Mask Generation)",

    # --- review_frames.py (Step 2 review GUI) ---
    "REVIEW_TITLE": "Frame Review",
    "REVIEW_DECISION_PREFIX": "Decision: ",
    "REVIEW_DECISION_KEEP": "Keep",
    "REVIEW_DECISION_DROP": "Drop",
    "REVIEW_INFO_YES": "yes",
    "REVIEW_INFO_NO": "no",
    "REVIEW_PROBLEMS_FORMAT": "Review targets: {n} (replaced={r}, fallback={f}, thinned={t}) | Current: {cur}",
    "REVIEW_INFO_FORMAT": (
        "Time: {ts}   |   Blur: {blur}   |   Change: {change}\n"
        "extract action: {process}"
    ),
    "REVIEW_BLUR_VALUE_FORMAT": "{score:.1f} ({pct}% of median {median:.0f})",
    "REVIEW_BLUR_VALUE_NO_MEDIAN": "{score:.1f}",
    "REVIEW_PROCESS_OK": "kept as-is (passed extract's quality bar)",
    "REVIEW_PROCESS_REPLACED": "auto-swapped (original idx={orig} was blurry; replaced with idx={final})",
    "REVIEW_PROCESS_FALLBACK": "kept blurry (extract found no sharper neighbor)",
    "REVIEW_PROCESS_THINNED": "auto-thinned in stationary cluster",
    "REVIEW_BTN_PREV": "Prev (←)",
    "REVIEW_BTN_NEXT": "Next (→)",
    "REVIEW_BTN_PREV_PROBLEM": "extract problem ← (Shift+F)",
    "REVIEW_BTN_NEXT_PROBLEM": "extract problem → (F)",
    "REVIEW_BTN_PROBLEM_TIP": (
        "Cycle through frames extract_frames flagged automatically\n"
        "(replaced / fallback_keep / thinned), in CSV (capture time) order."
    ),
    "REVIEW_BTN_TOGGLE": "Toggle Keep/Drop (Space)",
    "REVIEW_BTN_JUMP": "Jump to Seq",
    "REVIEW_BTN_SAVE": "Save (S)",
    "REVIEW_JUMP_PLACEHOLDER": "seq",
    "REVIEW_BLUR_THRESHOLD_LABEL": "  Blur threshold:",
    "REVIEW_BLUR_DROP_BTN": "Drop below threshold",
    "REVIEW_BLUR_THRESHOLD_PLACEHOLDER": "blur score",
    "REVIEW_NO_PROBLEMS": "No problem frames found.",
    "REVIEW_IMAGE_NOT_FOUND": "Image not found:\n{path}",
    "REVIEW_IMAGE_LOAD_FAILED": "Failed to load image:\n{path}",
    "REVIEW_SEQ_INTEGER_ERR": "Seq must be an integer.",
    "REVIEW_OUT_OF_RANGE_ERR": "Seq must be between 1 and {max}.",
    "REVIEW_INVALID_INPUT": "Invalid input",
    "REVIEW_INFO_HEADER": "Info",
    "REVIEW_THRESHOLD_HEADER": "Threshold",
    "REVIEW_THRESHOLD_NEED_INPUT": "Enter a threshold and press the button again.\nThe current frame's score has been pre-filled.",
    "REVIEW_THRESHOLD_NUMERIC_ERR": "Enter a numeric value.",
    "REVIEW_BULK_DROP_HEADER": "Bulk Drop",
    "REVIEW_BULK_DROP_RESULT": "Marked {n} frames with blur_score ≦ {thr} as drop.\nPress Save (S) to commit to CSV.",
    "REVIEW_SAVED_HEADER": "Saved",
    "REVIEW_SAVED_BODY": "Updated {path}\nkeep={k}, drop={d}",
    "REVIEW_HELP_HEADER": "── How to use ──",
    "REVIEW_HELP_BODY": (
        "Walk through the frames one by one and mark unwanted ones as drop.\n"
        "Edits stay in memory until you press Save (S), which writes them back to selected_frames.csv.\n"
        "Then return to the main GUI Step 2 and press \"Finalize\" to apply the changes to images/.\n\n"
        "[Recommended review flow] Shortest path to good SfM:\n"
        "  1. Press F to walk through extract-flagged frames\n"
        "     (frames extract marked as 'replaced' / 'fallback_keep' / 'thinned')\n"
        "  2. Read the advisory color band under the image:\n"
        "     orange = fallback_keep (consider drop) / blue = info (usually keep) / green = normal\n"
        "  3. For orange frames, look at the image. If actually blurry, press Space to drop.\n"
        "  4. Press S to save, then go back to the main GUI and click Finalize.\n"
        "  Note: extract has already detected and replaced blurry frames. The truly\n"
        "  problematic ones are 'fallback_keep' (couldn't be replaced). 'ok' / 'replaced'\n"
        "  passed extract's quality bar; usually keep them.\n\n"
        "[When to drop] Frames that hurt SfM (Metashape) or 3DGS training:\n"
        "  - Moving subjects: people, vehicles, swaying branches (3DGS assumes static scenes)\n"
        "  - fallback_keep frames that look visibly blurry\n"
        "  - Lens flare / strong sun: most pixels saturated\n"
        "  - Major stitching seam errors (360 dual-fisheye)\n"
        "  - Over- / under-exposure: most of the image is white or black\n"
        "When in doubt, keep it. You can always re-review later.\n\n"
        "[Image]\n"
        "  Mouse wheel      zoom in / out at the cursor position\n"
        "  Click and drag   pan the image when zoomed in\n"
        "  0 key            reset zoom to fit window\n\n"
        "[Keyboard]\n"
        "  ← / →     previous / next frame\n"
        "  Space     toggle keep/drop on the current frame\n"
        "  F / Shift+F  extract-flagged frames (replaced / fallback_keep / thinned) in capture order\n"
        "  S         save changes to CSV\n"
        "  Q         quit (unsaved changes are discarded)\n\n"
        "The 'Drop below threshold' button is a manual absolute-threshold tool. Use it if you\n"
        "want to bulk-drop frames whose blur score is clearly low (e.g., below 50)."
    ),
    "REVIEW_ADVISORY_FALLBACK": "⚠ extract could not replace this blurry frame — review and consider drop",
    "REVIEW_ADVISORY_REPLACED": "ⓘ Auto-swapped to a sharper neighbor at extract time (usually keep)",
    "REVIEW_ADVISORY_THINNED": "ⓘ Auto-thinned in a stationary cluster (marked drop). Press Space to flip back to keep",
    "REVIEW_ADVISORY_NORMAL": "Normal quality (passed extract's quality bar)",
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
    "STITCH_BOUNDARY_WIDTH": "Boundary Mask Width (deg)",
    "STITCH_WORKERS": "Workers",
    "RUN_MASKS": "Run Mask Creation",
    "MASK_TASKS_LABEL": "Masks to create:",
    "MASK_TASK_YOLO": "YOLO Detection",
    "MASK_TASK_STITCH": "Stitch Seam",
    "MASK_TASK_OVEREXPOSURE": "Overexposure",
    "MASK_TASK_REQUIRED": "Select at least one mask to create.",
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
    "STITCH_PREVIEW_SECTION": "Stitch Seam Preview",
    "STITCH_PREVIEW_STATUS_FORMAT": "Seam {width:g} deg / internal FOV {fov:g} deg",
    "STITCH_PREVIEW_INVALID_WIDTH": "Boundary width must be >= 0 and < 180",
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
    "YOLO_SECTION": "YOLO Detection Settings",
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
    "EXPORT_DIR": "採用フレームのコピー先フォルダ名。'images'ならインプレース処理",
    "OPEN_REVIEW": "フレーム画像を1枚ずつ確認し、採用/除外を編集するGUIを開く\n確認対象の巡回と、ブレ評価値の閾値による一括除外ができます",
    "EXPORT_KEEP": "CSVで採用にしたフレームだけを指定フォルダにコピー",
    "FINALIZE_INPLACE": "images/内の除外フレームを削除し、採用フレームを連番リネーム。元に戻せないので注意",
    "IMAGES_DIR": "エクイレクタングラー画像が入ったフォルダ (通常 images/)",
    "MASKS_DIR": "マスク画像の出力先フォルダ (通常 masks/)。既存マスクがあれば合成",
    "RUN_MASKS": "選択したマスク処理を YOLO検出 → スティッチ境界 → 白飛び の順に実行",
    "MASK_TASK_YOLO": "YOLO/SAMで人物などを検出してマスクに追加。初期状態は人物のみ",
    "MASK_TASK_STITCH": "スティッチ境界をマスクに追加。手ブレ補正、方向ロック、AIスティッチなどで境界位置が動く素材では通常OFF",
    "MASK_TASK_OVEREXPOSURE": "白飛びした画素を検出してマスクに追加",
    "RUN_ALL": "YOLO人物検出 → スティッチマスク → 白飛びマスクの全工程を順番に実行",
    "RUN_YOLO_STITCH": "YOLO人物検出 → スティッチマスクの2工程を実行 (白飛びは含まない)",
    "YOLO_LEVEL": "0: YOLO直接 (高速)\n1: YOLO+SAM2 (標準、推奨)\n2: 水平帯高品質+SAM2\n3: 全方向高品質+SAM2",
    "YOLO_EXPAND": "検出マスクを指定ピクセル分膨張させます。横ドラッグで調整可能。GUI範囲は -64〜256 px",
    "YOLO_ADD_EXT": "マスクファイル名を image.jpg.png のように元の拡張子を残す形式にする",
    "STITCH_BOUNDARY_WIDTH": "除外するスティッチ境界帯の合計幅。横ドラッグで調整できます。GUIでは安全のため0〜30度に制限。5度は従来のFOV 175相当",
    "STITCH_WORKERS": "並列処理のワーカー数。横ドラッグで調整可能。CPUコア数が目安",
    "OVEREXPOSURE_THRESHOLD": "RGB全チャンネルがこの値を超えるピクセルを白飛びと判定。横ドラッグで調整可能。GUI範囲は 0〜255",
    "OVEREXPOSURE_DILATE": "白飛び領域を膨張させるピクセル数。横ドラッグで調整可能。0で無効、GUI範囲は 0〜128",
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
    "RUN_MASKS": "Run the selected mask steps in this order: YOLO detection, stitch seam, overexposure",
    "MASK_TASK_YOLO": "Detect people or selected classes with YOLO/SAM and add them to masks. Default class is person only",
    "MASK_TASK_STITCH": "Add stitch seam masks. Usually keep OFF for stabilized, direction-locked, or AI-stitched footage where seam positions move",
    "MASK_TASK_OVEREXPOSURE": "Detect blown-out pixels and add them to masks",
    "RUN_ALL": "Run YOLO person detection, stitch mask, and overexposure mask in sequence",
    "RUN_YOLO_STITCH": "Run YOLO person detection then stitch mask (no overexposure)",
    "YOLO_LEVEL": "0: YOLO direct (fast)\n1: YOLO+SAM2 (standard, recommended)\n2: High-quality horizontal band+SAM2\n3: Full high-quality+SAM2",
    "YOLO_EXPAND": "Dilate detection masks by N pixels. Drag horizontally to adjust. GUI range: -64 to 256 px",
    "YOLO_ADD_EXT": "Name mask files as image.jpg.png (keeping the original extension)",
    "STITCH_BOUNDARY_WIDTH": "Total stitch seam band to exclude. Drag horizontally to adjust. The GUI clamps this to 0-30 degrees for safety. 5 degrees equals legacy FOV 175",
    "STITCH_WORKERS": "Number of parallel workers. Drag horizontally to adjust. Use CPU core count as a guide",
    "OVEREXPOSURE_THRESHOLD": "Pixels with all RGB channels above this value are flagged as blown-out. Drag horizontally to adjust. GUI range: 0-255",
    "OVEREXPOSURE_DILATE": "Dilate blown-out regions by N pixels. Drag horizontally to adjust. 0 = disabled; GUI range: 0-128",
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
