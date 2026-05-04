"""UI文字列 — ロケール自動判定で日本語/英語を切り替え。

モジュールロード時にシステムロケールを判定し、日本語環境なら JA、
それ以外なら EN の文字列をモジュール変数としてエクスポートする。
"""
from __future__ import annotations

import locale
import os
import textwrap

# ---------------------------------------------------------------------------
# 文字列テーブル
# ---------------------------------------------------------------------------

_JA: dict[str, str] = {
    # App
    "APP_TITLE": "STechDrive 3DGS Utils",
    "APP_SUBTITLE": "Prepare frames, masks, and SfM datasets for 3DGS workflows",
    "WORKFLOW_LABEL": "ワークフロー",
    "STEP1_DESC": "動画からSfM向けのフレームを抽出",
    "STEP2_DESC": "抽出フレームを確認して採用/除外を確定",
    "STEP3_DESC": "人物・スティッチ境界・白飛びをマスク",
    "STEP4_DESC": "SfM結果または抽出画像を視点画像として書き出し",
    "STEP1_TITLE": "1. フレーム抽出",
    "STEP2_TITLE": "2. フレーム確認",
    "STEP3_TITLE": "3. マスク生成",
    "STEP4_TITLE": "4. 書き出し",
    "STEP1_NAV": "1\n抽出",
    "STEP2_NAV": "2\n確認",
    "STEP3_NAV": "3\nマスク",
    "STEP4_NAV": "4\n書き出し",

    # Common
    "BROWSE": "参照...",
    "SCENE_DIR": "シーンフォルダ",
    "SCENE_DIR_PLACEHOLDER": "まず作業シーンフォルダを選択...",
    "SCENE_REQUIRED_ACTION_HINT": "ヘッダーのシーンフォルダを指定してください。",
    "CLEAR_SCENE_DIR": "シーンフォルダをクリア",
    "CLEAR_SCENE_DIR_HINT": "現在のシーンフォルダ指定を解除します。ファイルは削除しません。",
    "OUTPUT_DIR": "出力フォルダ",
    "RUN": "実行",
    "GENERATE": "生成",
    "EXPORT": "書き出し",
    "CANCEL": "キャンセル",
    "CLOSE": "閉じる",
    "STATUS_IDLE": "待機中",
    "STATUS_RUNNING": "実行中",
    "STATUS_DONE": "完了",
    "STATUS_FAILED": "失敗",
    "STATUS_CANCELED": "キャンセル済み",
    "BUSY_MSG": "別のプロセスが実行中です。",
    "INVALID_INPUT": "入力エラー",
    "UNSAFE_SCENE_PATH_TITLE": "シーンフォルダのパスを変更してください",
    "UNSAFE_SCENE_PATH_BODY": (
        "シーンフォルダのパスに、画像処理ライブラリや外部ツールで失敗しやすい要素があります。\n\n"
        "{reasons}\n\n"
        "現在のパス:\n{path}\n\n"
        "英数字だけの短いフォルダ名に変更してください。\n\n"
        "例:\nD:\\work\\scene01\nD:\\projects\\site_001"
    ),
    "UNSAFE_PATH_REASON_NON_ASCII": "日本語などの非 ASCII 文字が含まれています。",
    "UNSAFE_PATH_REASON_TOO_LONG": "パスが長すぎます ({length} 文字、目安 {limit} 文字未満)。",
    "UNSAFE_PATH_REASON_CONTROL_CHARS": "制御文字が含まれています ({value})。",
    "UNSAFE_PATH_REASON_QUOTE": "ダブルクォートが含まれています。",
    "UNSAFE_PATH_REASON_UNKNOWN": "安全性を確認できない文字が含まれています。",

    # Step 1
    "INPUT_VIDEO": "入力動画",
    "INPUT_VIDEO_PLACEHOLDER": "360度動画を選択 (複数可)...",
    "VIDEO_FILE_FILTER": "動画ファイル (*.mp4 *.mov *.mkv *.avi *.m4v);;すべて (*.*)",
    "MODE_CHANGE": "自動間隔",
    "MODE_CHANGE_SHORT": "自動間隔",
    "MODE_FIXED": "固定間隔",
    "MODE_FIXED_SHORT": "固定間隔",
    "FIXED_SMART": "変化補正",
    "FIXED_SMART_ESTIMATE": "解析後に補正",
    "QUICK_EXTRACT": "クイック抽出",
    "QUICK_EXTRACT_ESTIMATE": "クイック抽出",
    "CHANGE_THRESHOLD": "変化しきい値",
    "CHANGE_THRESHOLD_SHORT": "変化",
    "MIN_GAP": "最小間隔 (秒)",
    "MIN_GAP_SHORT": "最小",
    "MAX_GAP": "最大間隔 (秒)",
    "MAX_GAP_SHORT": "最大",
    "INTERVAL": "間隔 (秒)",
    "INTERVAL_SHORT": "間隔",
    "SECONDS_SUFFIX": "秒",
    "ANALYSIS_WIDTH": "解析幅 (px)",
    "QUALITY_MIN_SCORE": "品質確認スコア",
    "QUALITY_MIN_IMPROVEMENT": "代替フレーム選択基準",
    "IMAGE_FORMAT": "画像形式",
    "JPEG_QUALITY": "JPEG品質",
    "FFMPEG_PATH": "ffmpeg パス",
    "FFPROBE_PATH": "ffprobe パス",
    "FILENAME_PREFIX": "ファイル名接頭辞",
    "EXTRACT_FRAMES": "フレーム抽出を実行",
    "VIDEO_INFO": "動画情報",
    "FRAME_ESTIMATE": "フレーム数推定",
    "INSTANT_ESTIMATE": "即時推定",
    "SAMPLED_ESTIMATE": "サンプル推定",
    "NO_VIDEO": "動画が選択されていません",
    "EXTRACT_OUTPUT_MODE": "画像保存方法",
    "EXTRACT_OUTPUT_APPEND": "新規のみ追加",
    "EXTRACT_OUTPUT_REPLACE_VIDEO": "リセットして上書き",
    "CLEAR_INPUT_VIDEO": "入力動画をクリア",
    "CLEAR_INPUT_VIDEO_HINT": "入力動画の選択と動画情報・フレーム数推定をクリアします。ファイルは削除しません。",
    "EXTRACT_READY_SECTION": "実行前チェック",
    "EXTRACT_READY_NO_VIDEO": "入力動画を選択してください。",
    "EXTRACT_READY_VIDEO_NOT_FOUND": "入力動画が見つかりません。",
    "EXTRACT_READY_NO_SCENE": "上部のシーンフォルダを指定してください。",
    "EXTRACT_READY_BAD_ANALYSIS_WIDTH": "解析幅は0以上の整数で入力してください。",
    "EXTRACT_READY_NO_VIDEO_INFO": "動画情報を読み込んでください。",
    "EXTRACT_READY_DUPLICATE_VIDEO": "この動画はすでに抽出済みです ({n} セッション)。既存結果を使う場合は Step 2 へ進むか、再抽出する場合は「リセットして上書き」を選んでください。",
    "EXTRACT_READY_DUPLICATE_REPLACE": "同じ動画の既存セッション {n} 件をリセットし、現在の設定で上書きします。",
    "EXTRACT_READY_QUEUE_OK": "準備完了: {n} 件の動画を順番に抽出します。",
    "EXTRACT_READY_QUEUE_PARTIAL": "準備完了: 未抽出の {n} 件を実行します。抽出済み {skipped} 件はスキップします。",
    "EXTRACT_READY_QUEUE_ALL_DUPLICATE": "選択した {n} 件はすべて抽出済みです。必要なら「リセットして上書き」を選んでください。",
    "EXTRACT_READY_QUEUE_REPLACE": "{n} 件の動画を順番に抽出します。抽出済み {replace} 件は置き換えます。",
    "EXTRACT_READY_OK": "準備完了: フレーム抽出を実行できます。",
    "VIDEO_QUEUE_LABEL_FORMAT": "動画キュー: {total} 件  |  実行 {queued} 件  |  スキップ {skipped} 件",
    "QUEUE_ESTIMATE_FORMAT": "キュー {queued} 件 (抽出済みスキップ {skipped} 件)",
    "VIDEO_INFO_SINGLE_FORMAT": "{width}x{height}  |  {fps:.2f} fps  |  {duration}  |  約{frames}フレーム",
    "VIDEO_INFO_MULTI_HEADER_FORMAT": "動画キュー: {total} 件  |  実行 {queued} 件  |  スキップ {skipped} 件  |  取得 {probed} 件",
    "VIDEO_INFO_MULTI_ITEM_FORMAT": "{name}: {width}x{height}  |  {fps:.2f} fps  |  {duration}  |  約{frames}フレーム",
    "VIDEO_INFO_FAILED_SUFFIX": "  |  取得失敗 {failed} 件",
    "FIXED_INTERVAL_ESTIMATE_FORMAT": "固定間隔 {interval}秒: 約{count}枚",
    "FIXED_INTERVAL_ESTIMATE_MULTI_HEADER_FORMAT": "固定間隔 {interval}秒",
    "FIXED_INTERVAL_ESTIMATE_MULTI_ITEM_FORMAT": "{name}: 約{count}枚",
    "FIXED_INTERVAL_ESTIMATE_MULTI_TOTAL_FORMAT": "合計: 約{count}枚 ({videos}件)",
    "ESTIMATE_MISSING_INFO_SUFFIX": "  |  未取得 {missing} 件",

    # Step 2
    "EXPORT_KEEP": "採用フレームをエクスポート",
    "FINALIZE_INPLACE": "画像フォルダ内で確定",
    "FINALIZE_BUTTON": "選別を確定 (除外分を削除)",
    "FINALIZE_BUTTON_HINT": (
        "除外にしたフレームを images/ から削除します。採用フレームのファイル名は維持します。\n"
        "不可逆。バックアップが必要なら左のチェックボックスを ON に。"
    ),
    "BACKUP_BEFORE_FINALIZE": "適用前に images_backup/ にバックアップ",
    "BACKUP_BEFORE_FINALIZE_HINT": (
        "ON: 適用前に images/ を images_backup/ にフルコピー（既存バックアップは上書き）。\n"
        "OFF: バックアップなし（容量節約・復元不可）。"
    ),
    "ACTION_FINALIZE_REVIEW": "適用",
    "REVIEW_EMBED_NO_SCENE": "ヘッダーのシーンフォルダに抽出済みフォルダを指定すると、selected_frames.csv を自動で読み込みます。",
    "REVIEW_EMBED_EMPTY": "Step 1 の抽出完了後、または抽出済みのシーンフォルダを指定すると、ここにフレーム確認ビューが自動表示されます。",
    "REVIEW_EMBED_MISSING": "CSVが見つかりません:\n{path}",

    # --- review_frames.py (Step 2 レビュー GUI) ---
    "REVIEW_TITLE": "フレーム確認",
    "REVIEW_PREVIEW_MODE_SINGLE": "1枚プレビュー",
    "REVIEW_PREVIEW_MODE_THUMBNAILS": "サムネイル一覧",
    "REVIEW_DECISION_PREFIX": "選別: ",
    "REVIEW_DECISION_KEEP": "採用",
    "REVIEW_DECISION_DROP": "除外",
    "REVIEW_INFO_YES": "対象",
    "REVIEW_INFO_NO": "通常",
    "REVIEW_PROBLEMS_FORMAT": "確認対象 {n}件 | 追加 {a} / 差し替え {r} / 低品質 {f} / 間引き {t} | 表示中 {cur}",
    "REVIEW_FRAME_SLIDER_TIP": "CSV内のフレームを順番に切り替えます。",
    "REVIEW_FRAME_POSITION_FORMAT": "{seq} / {total} : {name}",
    "REVIEW_INFO_FORMAT": "動画位置: {ts}  |  品質スコア: {quality}",
    "REVIEW_QUALITY_ORIGINAL_FORMAT": "元:{score}",
    "REVIEW_QUALITY_THRESHOLD_FORMAT": "基準:{score}",
    "REVIEW_FLAG_TIP": "採用フラグを切り替えます。変更はすぐCSVに反映されます。",
    "REVIEW_RESET_DECISION_TIP": "このフレームの採用フラグを読み込み時の状態へ戻します。",
    "REVIEW_BTN_PREV": "前 (←)",
    "REVIEW_BTN_NEXT": "次 (→)",
    "REVIEW_BTN_PREV_PROBLEM": "前の確認対象 (Shift+F)",
    "REVIEW_BTN_NEXT_PROBLEM": "次の確認対象 (F)",
    "REVIEW_BTN_PROBLEM_TIP": (
        "フレーム抽出時に確認対象として記録されたフレーム\n"
        "（差し替え・低品質・間引き）を撮影順に巡回します。"
    ),
    "REVIEW_NO_PROBLEMS": "確認対象フレームはありません",
    "REVIEW_IMAGE_NOT_FOUND": "画像が見つかりません:\n{path}",
    "REVIEW_IMAGE_LOAD_FAILED": "画像の読み込みに失敗しました:\n{path}",
    "REVIEW_INVALID_INPUT": "入力エラー",
    "REVIEW_INFO_HEADER": "情報",
    "REVIEW_SAVE_FAILED_HEADER": "保存に失敗しました",
    "REVIEW_SAVE_FAILED_BODY": "採用フラグをCSVに反映できませんでした:\n{error}",
    "REVIEW_ADVISORY_FALLBACK": "要確認: 探索範囲内の代表候補が低品質",
    "REVIEW_ADVISORY_REPLACED": "差し替え済み: 近傍のSfM向けフレームを使用",
    "REVIEW_ADVISORY_SMART_ADDED": "変化補正: 動きが大きいため追加",
    "REVIEW_ADVISORY_THINNED": "自動間引き: 動きが少ないため除外中",
    "REVIEW_ADVISORY_NORMAL": "通常: 品質基準OK",
    "NEXT_STEP_MASK_NOTICE": "除外予定の画像が残っている場合、または採用/除外を変更した場合は、\n下部の「適用」で画像フォルダへ反映します。\n「適用」が無効なら、そのまま Step 3 (マスク生成) へ進めます。",
    "METASHAPE_NOTICE": "マスク生成後は、用途に合わせて次へ進みます。\nMetashapeルートでは masks/ をマスクとして読み込み、SfM後にStep 4で3DGS向けに書き出します。\nCOLMAPルートではStep 4で視点画像を書き出し、必要に応じてCOLMAP/GLOMAPを実行します。",
    "EXPORT_DIR": "エクスポート先",

    # Step 3
    "IMAGES_DIR": "画像フォルダ",
    "MASKS_DIR": "マスクフォルダ",
    "YOLO_LEVEL": "YOLOレベル",
    "YOLO_LEVEL_COMPACT": "レベル",
    "YOLO_LEVEL_FAST": "0 高速",
    "YOLO_LEVEL_STANDARD": "1 標準",
    "YOLO_LEVEL_QUALITY": "2 高品質",
    "YOLO_LEVEL_BEST": "3 最高",
    "MASK_QUALITY": "品質",
    "MASK_QUALITY_STANDARD": "標準",
    "MASK_QUALITY_HIGH": "高品質",
    "MASK_QUALITY_BEST": "最高",
    "YOLO_EXPAND": "マスク拡張 (px)",
    "YOLO_EXPAND_COMPACT": "拡張",
    "PERSON_MODEL": "人物モデル",
    "PERSON_MODEL_YOLO_SAM": "YOLO/SAM2.1",
    "PERSON_MODEL_SAM31": "SAM3.1",
    "YOLO_BOTTOM_ENHANCE": "下部検出強化",
    "YOLO_BOTTOM_STANDARD": "標準",
    "YOLO_BOTTOM_STRONG": "高",
    "YOLO_BOTTOM_MAX": "最高",
    "YOLO_ADD_EXT": "拡張子を付加",
    "YOLO_CLASSES": "検出クラス",
    "STITCH_BOUNDARY_WIDTH": "境界マスク幅 (度)",
    "STITCH_BOUNDARY_WIDTH_COMPACT": "境界幅",
    "STITCH_WORKERS": "ワーカー数",
    "STITCH_WORKERS_COMPACT": "ワーカー",
    "RUN_MASKS": "選択したマスクを再生成",
    "MASK_TASKS_LABEL": "マスク:",
    "ADDITIONAL_MASKS_LABEL": "オプション:",
    "MASK_IMAGE_TYPE": "画像タイプ:",
    "MASK_IMAGE_TYPE_EQUIRECT": "360°",
    "MASK_IMAGE_TYPE_NORMAL": "通常",
    "MASK_TASK_YOLO": "人物",
    "MASK_TASK_STITCH": "スティッチ",
    "MASK_TASK_OVEREXPOSURE": "白飛び",
    "MASK_TASK_SKY": "空",
    "MASK_TASK_CUSTOM": "カスタム",
    "MASK_TASK_REQUIRED": "作成するマスクを1つ以上選択してください。",
    "MASK_PHASE_PRIMARY": "主マスク",
    "MASK_PHASE_STITCH": "スティッチ",
    "MASK_PHASE_OVEREXPOSURE": "白飛び",
    "MASK_PHASE_CUSTOM": "カスタム",
    "MASK_PHASE_INIT": "初期マスク",
    "MASK_MODEL": "モデル",
    "CUSTOM_MASK_SECTION": "カスタムマスク",
    "CUSTOM_MASK_FILE": "マスク画像",
    "CUSTOM_MASK_NOT_SELECTED": "未選択",
    "CUSTOM_MASK_BROWSE": "読み込み",
    "CUSTOM_MASK_CLEAR": "解除",
    "CUSTOM_MASK_SELECT_FILE": "カスタムマスク画像を選択",
    "CUSTOM_MASK_FILE_FILTER": "PNG画像 (*.png)",
    "CUSTOM_MASK_REQUIRED": "カスタムマスク画像を選択してください。",
    "CUSTOM_MASK_NOT_FOUND": "カスタムマスク画像が見つかりません: {path}",
    "MASK_READY_OK": "準備完了: マスクを生成できます。",
    "MASK_READY_EXTERNAL_IMAGES": "準備完了: selected_frames.csv なしの外部画像としてマスクを生成します。",
    "MASK_READY_SCENE_NOT_FOUND": "シーンフォルダが見つかりません。",
    "MASK_READY_NO_IMAGES_DIR": "シーンフォルダ内に images/ がありません。Step 1 で抽出するか、外部画像を images/ に配置してください。",
    "MASK_READY_NO_IMAGES": "images/ に対象画像がありません。Step 1 で抽出するか、外部画像を images/ に配置してください。",
    "MASK_READY_NO_CSV": "selected_frames.csv が見つかりません。Step 1 でフレーム抽出を実行してください。",
    "EXTERNAL_IMAGES_SECTION": "外部画像",
    "EXTERNAL_IMAGES_HINT": "通常動画や一眼の連番画像を使う場合は、画像フォルダから追加します。追加先は現在のシーンフォルダの images/ です。",
    "EXTERNAL_IMAGES_ADD": "画像フォルダから追加",
    "EXTERNAL_IMAGES_OPEN": "images/を開く",
    "EXTERNAL_IMAGES_SELECT_FOLDER": "追加する画像フォルダを選択",
    "EXTERNAL_IMAGES_SELECT_SCENE": "シーンフォルダを選択",
    "EXTERNAL_IMAGES_SCENE_REQUIRED_TITLE": "シーンフォルダが必要です",
    "EXTERNAL_IMAGES_SCENE_REQUIRED_MESSAGE": "外部画像を追加するには、まずシーンフォルダが必要です。新しいシーンフォルダを選択しますか？",
    "EXTERNAL_IMAGES_SOURCE_IS_TARGET": "選択したフォルダは現在の images/ です。コピーは不要です。",
    "EXTERNAL_IMAGES_SOURCE_NOT_FOUND": "画像フォルダが見つかりません: {path}",
    "EXTERNAL_IMAGES_RESULT_TITLE": "外部画像を追加",
    "EXTERNAL_IMAGES_RESULT": "追加 {added} 件 / スキップ {skipped} 件",
    "MASK_PENDING_DROPS_ERROR": "除外予定の画像が画像フォルダに {n} 件残っています。Step 2 で「適用」してからマスクを生成してください。\n{files}",
    "MASK_UNTRACKED_IMAGES_ERROR": "selected_frames.csv に載っていない画像が画像フォルダに {n} 件あります。古い抽出結果が混在している可能性があります。\n{files}",
    "RUN_YOLO": "YOLO実行",
    "RUN_STITCH": "スティッチ実行",
    "RUN_YOLO_STITCH": "YOLO + スティッチ実行",
    "CLASS_PRESET_PERSON": "人物のみ",
    "CLASS_PRESET_VEHICLES": "人物+車両",
    "CLASS_PRESET_ALL": "全選択",
    "CLASS_PRESET_CLEAR": "クリア",
    "DETECTION_TARGET_SECTION": "検出対象",
    "YOLO_CLASS_LIST_SECTION": "検出対象",
    "ADE20K_CLASS_LIST_SECTION": "検出対象",
    "SAM31_PROMPT_SECTION": "検出対象",
    "SAM31_APPLY_MODE": "適用",
    "SAM31_APPLY_REPLACE": "再生成",
    "SAM31_APPLY_ADD": "加算",
    "SAM31_APPLY_SUBTRACT": "減算",
    "SAM31_CUSTOM_PROMPT_PLACEHOLDER": "追加プロンプト: tripod, hand; selfie stick",
    "SAM31_SUBTRACT_PROMPT_PLACEHOLDER": "減算プロンプト: pictogram, logo",
    "OVEREXPOSURE": "白飛びマスク",
    "OVEREXPOSURE_THRESHOLD": "白飛び閾値 (RGB)",
    "OVEREXPOSURE_THRESHOLD_COMPACT": "閾値",
    "OVEREXPOSURE_DILATE": "膨張半径 (px)",
    "OVEREXPOSURE_DILATE_COMPACT": "膨張",
    "RUN_OVEREXPOSURE": "白飛びマスク実行",
    "SKY_MODEL": "モデル",
    "SKY_MODEL_MASK2FORMER": "Mask2Former",
    "SKY_MODEL_SAM31": "SAM3.1",
    "SKY_MODEL_SAM31_MISSING": "SAM3.1 checkpointが見つかりません: models/sam3.1/sam3.1_multiplex.pt",
    "SKY_MODE": "方式",
    "SKY_MODE_FULL": "高品質",
    "SKY_MODE_HYBRID": "直+上部",
    "SKY_MODE_DIRECT": "直処理",
    "SKY_MODE_TOP": "上部投影",
    "SKY_MODE_BOTTOM": "下部投影",
    "SKY_INFERENCE_SIZE": "推論サイズ",
    "SKY_EXPAND": "空拡張 (px)",
    "SKY_MIN_SCORE": "最小スコア",
    "SKY_MIN_AREA": "小領域除去",
    "SKY_MODEL_DETAILS_SECTION": "モデル詳細",
    "SKY_POSTPROCESS_SECTION": "空マスク設定",
    "SKY_TOP_CONNECTED": "上端接続のみ",
    "RUN_ALL": "全マスク実行",

    # Step 4
    "JSON_NAME": "transforms.json 名",
    "EXPORT_METHOD": "書き出し方式",
    "EXPORT_METHOD_COMPACT": "選択:",
    "METHOD_METASHAPE_IMPORT": "Metashape",
    "METHOD_COLMAP_EXPORT": "COLMAP",
    "COLMAP_PIPELINE_SECTION": "COLMAP実行設定",
    "RUN_COLMAP_SFM": "書き出し後にCOLMAPを実行",
    "COLMAP_EXECUTABLE": "COLMAP実行ファイル",
    "GLOMAP_EXECUTABLE": "GLOMAP実行ファイル",
    "COLMAP_MATCHER_COMPACT": "Matcher:",
    "COLMAP_MAPPER_COMPACT": "Mapper:",
    "COLMAP_MATCHER_SEQUENTIAL": "Sequential",
    "COLMAP_MATCHER_EXHAUSTIVE": "Exhaustive",
    "COLMAP_MAPPER_INCREMENTAL": "Incremental",
    "COLMAP_MAPPER_GLOBAL": "Global",
    "COLMAP_MAPPER_GLOMAP": "GLOMAP",
    "PHASE_COLMAP_RIG_EXPORT": "COLMAP用視点画像を書き出し",
    "PHASE_COLMAP_FEATURE": "COLMAP Feature",
    "PHASE_COLMAP_RIG_CONFIG": "COLMAP Rig設定",
    "PHASE_COLMAP_MATCH": "COLMAP Matcher",
    "PHASE_COLMAP_MAPPER": "COLMAP Mapper",
    "COLMAP_EXEC_NOT_FOUND": "COLMAP実行ファイルが見つかりません。インストール先の colmap.exe を指定してください: {path}",
    "GLOMAP_EXEC_NOT_FOUND": "GLOMAP実行ファイルが見つかりません。GLOMAPを使う場合は glomap.exe を指定してください: {path}",
    "TARGET_PROFILE": "出力プリセット",
    "PROFILE_POSTSHOT": "Postshot",
    "PROFILE_BRUSH": "Brush",
    "PROFILE_LICHTFELD": "LichtFeld",
    "PROFILE_CUSTOM": "カスタム",
    "PROFILE_CUSTOM_HINT": "カスタム: 手動設定",
    "AXIS_TRANSFORM": "座標変換",
    "AXIS_TRANSFORM_POSTSHOT": "Postshot",
    "AXIS_TRANSFORM_BRUSH": "Brush",
    "AXIS_TRANSFORM_NONE": "変換なし",
    "VIEW_MODE": "ビューモード",
    "VIEW_CUSTOM": "カスタムグリッド",
    "VIEW_CUBE6": "Cube6",
    "YAW_OFFSET": "Yawオフセット (度)",
    "YAW_SLOTS": "Yawスロット数",
    "PITCH_ROWS": "Pitch行数",
    "OUTPUT_SCALE": "画像サイズ",
    "INVERT_MASKS": "マスク反転",
    "NO_IMAGE": "画像とマスク変換なし",
    "EXPORT_TARGETS": "出力",
    "EXPORT_IMAGES": "画像",
    "EXPORT_MASKS": "マスク",
    "MASK_DIR": "マスクフォルダ",
    "METASHAPE_PREPROCESS": "Metashapeインポート設定",
    "METASHAPE_XML": "カメラXML",
    "METASHAPE_PLY": "点群PLY",
    "SCALE_FACTOR": "スケール係数",
    "SCALE_FACTOR_COMPACT": "スケール",
    "MS_USE_PLY": "PLY使用",
    "NO_FIX_ROTATION": "回転補正なし",
    "RUN_CUBEMAP": "視点画像を書き出し",
    "PREVIEW": "プレビュー",

    # ViewConfig / Preview labels
    "VIEW_MODE_LABEL": "プリセット:",
    "CUSTOM_GRID": "カスタムグリッド",
    "CUBE6_LABEL": "Cube6",
    "YAW_OFFSET_LABEL": "Yawオフセット:",
    "YAW_SLOTS_LABEL": "Yaw:",
    "PITCH_ROWS_LABEL": "Pitch行:",
    "YAW_SLOT_ADD": "Yaw列を追加",
    "YAW_SLOT_REMOVE": "Yaw列を削除",
    "YAW_SLOT_COUNT_FORMAT": "Yaw {count}",
    "PITCH_ROW_ADD": "Pitch行を追加",
    "PITCH_ROW_REMOVE": "Pitch行を削除",
    "PITCH_ROW_COUNT_FORMAT": "Pitch {count}",
    "APPLY": "適用",
    "SELECT_ALL": "全選択",
    "DESELECT_ALL": "全解除",
    "SELECTED_VIEWS": "選択ビュー",
    "VIEW_SELECTION_SECTION": "ビュー選択グリッド",
    "VIEW_SELECTION_COMPACT_SECTION": "出力視点",
    "OUTPUT_IMAGE_COUNT_LABEL": "出力画像",
    "OUTPUT_IMAGE_COUNT_FORMAT": "{count}枚",
    "OUTPUT_RESET_TITLE": "出力フォルダをリセット",
    "OUTPUT_RESET_MESSAGE": "出力フォルダに既存ファイルがあります。\n\n{path}\n\n古い画像や transforms.json が混在しないよう、中身を削除してから書き出します。続行しますか？",
    "OUTPUT_PARTIAL_RESET_TITLE": "書き出し対象をリセット",
    "OUTPUT_PARTIAL_RESET_MESSAGE": "書き出し対象のフォルダに既存ファイルがあります。\n\n{paths}\n\n対象フォルダだけを削除してから書き出します。続行しますか？",
    "PITCH_SLOT_HEADER": "Pitch / Slot",
    "PREVIEW_IMAGE_LABEL": "プレビュー画像:",
    "PREVIEW_IMAGE_POSITION_FORMAT": "{seq} / {total} : {name}",
    "AUTO": "自動",
    "RELOAD": "更新",
    "MASK_PREVIEW_BUTTON": "マスクプレビュー",
    "MASK_PREVIEW_CLEAR_BUTTON": "プレビュー解除",
    "MASK_PREVIEW_VISIBILITY_BUTTON": "プレビュー表示",
    "YOLO_PREVIEW_BUTTON": "YOLOプレビュー",
    "MASK_REPROCESS_CURRENT_BUTTON": "マスク再生成",
    "MASK_REPROCESS_SELECTED_BUTTON": "マスク{count}枚再生成",
    "MASK_REPROCESS_SELECTED_FALLBACK_BUTTON": "マスク再生成",
    "MASK_REPROCESS_CURRENT_RUNNING": "再生成中...",
    "MASK_REPROCESS_CURRENT_DONE": "再生成しました: {name}",
    "MASK_REPROCESS_SELECTED_PROGRESS": "再生成中: {done}/{total} {name}",
    "MASK_REPROCESS_SELECTED_DONE": "再生成しました: {done}/{total}枚",
    "MASK_REPROCESS_SELECTED_FAILED": "再生成に失敗しました: {failed}/{total}枚",
    "MASK_REPROCESS_CURRENT_FAILED": "表示中の画像の再生成に失敗しました",
    "MASK_REPROCESS_NO_BASE_MASK": "マスクのベースを作成できませんでした。現在の設定を確認してください。",
    "MASK_OVERLAY_TOGGLE": "マスク表示",
    "MASK_OPACITY_LABEL": "マスク透過率:",
    "MASK_IMAGE_LABEL": "マスク画像:",
    "CLEAR": "クリア",
    "PREVIEW_OVERLAY_SECTION": "プレビューオーバーレイ設定",
    "MASK_PREVIEW_SECTION": "マスクプレビュー",
    "MASK_PREVIEW_MODE_SINGLE": "1枚プレビュー",
    "MASK_PREVIEW_MODE_THUMBNAILS": "サムネイル一覧",
    "MASK_PREVIEW_THUMBNAIL_STATUS": "サムネイル一覧: {count}枚",
    "MASK_PREVIEW_TEMP": "マスク: プレビュー結果",
    "MASK_PREVIEW_RUNNING": "マスクプレビュー生成中...",
    "MASK_PREVIEW_FAILED": "マスクプレビューに失敗しました",
    "MASK_PREVIEW_CLEARED": "プレビューを解除しました",
    "MASK_PREVIEW_NO_IMAGE": "プレビュー画像を選択してください",
    "MASK_PREVIEW_NO_SCENE_HELP": "シーンフォルダを選択すると、images/ の画像をここに表示します。",
    "MASK_PREVIEW_EMPTY_HELP": "Step 1でフレームを抽出するか、images/ に画像を配置すると、ここにマスクプレビューを表示します。",
    "MASK_PREVIEW_YOLO_EXISTING": "主マスク: 既存",
    "MASK_PREVIEW_YOLO_TEMP": "主マスク: プレビュー",
    "MASK_PREVIEW_YOLO_PENDING": "主マスク: 実行後に反映",
    "MASK_PREVIEW_YOLO_RUNNING": "主マスク実行中...",
    "MASK_PREVIEW_YOLO_FAILED": "主マスクプレビューに失敗しました",
    "MASK_PREVIEW_YOLO_NO_IMAGE": "プレビュー画像を選択してください",
    "MASK_PREVIEW_SKY_EXISTING": "空: 既存マスク",
    "MASK_PREVIEW_SKY_PENDING": "空: 実行後に反映",
    "YOLO_SAM_LICENSE_NOTICE_TITLE": "YOLO/SAMモデルの利用条件",
    "YOLO_SAM_LICENSE_NOTICE_BODY": (
        "YOLO/SAMマスク機能では、第三者が提供するモデルファイルおよびライブラリを使用します。\n\n"
        "このアプリ本体のソースコードはMIT Licenseですが、YOLO/SAM機能で使用されるモデルおよび関連ライブラリには別のライセンス条件が適用されます。\n\n"
        "- Ultralytics YOLO / ultralytics: AGPL-3.0 または Ultralytics Enterprise License\n"
        "- Meta SAM2/SAM2.1: Apache License 2.0\n"
        "- Meta SAM3.1: SAM License（SAM3.1人物モデルを選ぶ場合）\n\n"
        "モデル重みはこのアプリには同梱されていません。初回使用時にユーザー環境へダウンロードされる場合があります。\n\n"
        "商用利用、再配布、社内展開、製品組み込み等における各ライセンス条件への適合は、利用者の責任で確認してください。"
    ),
    "YOLO_SAM_LICENSE_NOTICE_DONT_SHOW_AGAIN": "次回からこの確認を表示しない",
    "YOLO_SAM_LICENSE_NOTICE_CONTINUE": "確認して続行",
    "YOLO_SAM_LICENSE_NOTICE_CANCELED": "YOLO/SAMの実行をキャンセルしました",
    "SKY_LICENSE_NOTICE_TITLE": "セマンティックマスクモデルの利用条件",
    "SKY_LICENSE_NOTICE_BODY": (
        "セマンティックマスク機能では、第三者が提供するMask2Former ADE20Kモデルファイル、"
        "Transformers関連ライブラリ、またはユーザーが配置したMeta SAM3.1チェックポイントを使用します。\n\n"
        "このアプリ本体のソースコードはMIT Licenseですが、マスク生成で使用されるモデル、"
        "SAM Materials、関連ライブラリ、学習元データセットには別のライセンス条件や利用条件が適用されます。\n\n"
        "- Mask2Former: MIT License\n"
        "- Transformers / safetensors: Apache License 2.0\n"
        "- Meta SAM3.1: SAM License\n"
        "- ADE20K dataset: データセット側の利用条件が適用されます\n\n"
        "モデル重みはこのアプリには同梱されていません。初回使用時にユーザー環境へダウンロードされる場合があります。\n\n"
        "商用利用、再配布、社内展開、製品組み込み等における各ライセンス条件への適合は、利用者の責任で確認してください。"
    ),
    "SKY_LICENSE_NOTICE_CANCELED": "セマンティックマスクの実行をキャンセルしました",
    "MASK_PREVIEW_STITCH_STATUS": "スティッチ {width:g}°",
    "MASK_PREVIEW_OVEREXP_STATUS": "白飛び RGB>{threshold} +{dilate}px",
    "MASK_PREVIEW_CUSTOM_STATUS": "カスタム",
    "MASK_PREVIEW_CUSTOM_INVALID": "カスタム: 読み込み不可/サイズ不一致",
    "MASK_PREVIEW_INVALID_STITCH_WIDTH": "境界マスク幅は 0 以上 180 未満で指定",
    "MASK_PREVIEW_NO_ACTIVE_MASK": "プレビュー対象のマスクなし",
    "CUBEMAP_PREVIEW_SECTION": "視点プレビュー",
    "EXCEED": "超過",
    "HIGH": "多い",
    "NO_PREVIEW": "プレビュー画像が選択されていません",
    "NO_PREVIEW_FOUND": "プレビュー画像が見つかりません",
    "PREVIEW_LOAD_FAIL": "画像の読み込みに失敗しました",

    # Step1 extra labels
    "SAMPLE_REFRESH": "サンプル推定を更新",
    "VIDEO_LABEL_DEFAULT": "動画: -",
    "ADVANCED_SETTINGS": "詳細設定",
    "AUTO_SELECTION_SECTION": "SfM品質確認",
    "AUTO_SELECTION_HINT": "抽出候補をSfM向けにスコアリングし、必要に応じて代替フレーム選択と品質確認を行います。",
    "AUTO_PREFIX_HINT": "自動 (動画ファイル名)",
    "FRAMES_UNIT": "フレーム",

    # Step2 extra labels
    "ADD_EXT_LABEL": "マスクに拡張子を付加",

    # Step3 section headers
    "YOLO_SECTION": "マスク設定",
    "STITCH_OVEREXP_SECTION": "スティッチ / 白飛び設定",
    "MASK_TAB_YOLO": "マスク設定",
    "MASK_TAB_OPTIONS": "オプション",
    "MASK_TAB_STITCH_OVEREXP": "スティッチ/白飛び",
    "MASK_TAB_SKY": "空",
    "MASK_TAB_CUSTOM": "カスタムマスク",

    # Step4 extra
    "OUTPUT_DETAIL": "出力詳細",
    "MS_IMAGES_LABEL": "画像フォルダ",
    "STEP4_TAB_METASHAPE": "変換設定",
    "STEP4_TAB_VIEW_EXPORT": "投影視点",
    "STEP4_TAB_COLMAP": "変換設定",

    # Step4 advanced output (added 2026-05)
    "ADVANCED_OUTPUT_SECTION": "視点書き出し設定",
    "YAW_OFFSET_PER_FRAME": "Yaw回転",
    "YAW_OFFSET_PER_FRAME_HINT": (
        "0=無効, 30推奨。-180〜180度でクランプ。ドラッグで調整できます。\n"
        "フレームごとに cubemap のYawを回転して 3DGS 学習の安定性を向上"
    ),
    "OUTPUT_FORMAT": "出力フォーマット",
    "OUTPUT_FORMAT_COMPACT": "フォーマット:",
    "OUTPUT_FORMAT_AUTO": "自動",
    "OUTPUT_BIT_DEPTH": "出力ビット深度",
    "OUTPUT_BIT_DEPTH_COMPACT": "ビット深度:",
    "OUTPUT_BIT_DEPTH_8": "8bit",
    "OUTPUT_BIT_DEPTH_SOURCE": "元画像",
    "JPG_QUALITY": "JPG/WebP 品質 (1-100)",
    "JPG_QUALITY_COMPACT": "JPG/WebP:",
    "EXPORT_COLMAP": "COLMAP形式モデルを追加出力",
    "EXPORT_COLMAP_HINT": (
        "output/transforms.json とPLYから\n"
        "cameras.txt / images.txt / points3D.txt を output/colmap/ に作成します。\n"
        "COLMAPで再SfMするための画像書き出しではありません。"
    ),
}

_EN: dict[str, str] = {
    # App
    "APP_TITLE": "STechDrive 3DGS Utils",
    "APP_SUBTITLE": "Prepare frames, masks, and SfM datasets for 3DGS workflows",
    "WORKFLOW_LABEL": "Workflow",
    "STEP1_DESC": "Extract SfM-ready frames from a 360 video",
    "STEP2_DESC": "Review extracted frames and finalize keep/drop choices",
    "STEP3_DESC": "Mask people, sky, stitch seams, and overexposed regions",
    "STEP4_DESC": "Export viewpoint images from SfM results or extracted frames",
    "STEP1_TITLE": "1. Frame Extraction",
    "STEP2_TITLE": "2. Frame Review",
    "STEP3_TITLE": "3. Mask Generation",
    "STEP4_TITLE": "4. Export",
    "STEP1_NAV": "1\nExtract",
    "STEP2_NAV": "2\nReview",
    "STEP3_NAV": "3\nMask",
    "STEP4_NAV": "4\nExport",

    # Common
    "BROWSE": "Browse...",
    "SCENE_DIR": "Scene Folder",
    "SCENE_DIR_PLACEHOLDER": "Select the working scene folder first...",
    "SCENE_REQUIRED_ACTION_HINT": "Set the scene folder in the header first.",
    "CLEAR_SCENE_DIR": "Clear Scene Folder",
    "CLEAR_SCENE_DIR_HINT": "Clear the current scene folder selection. No files are deleted.",
    "OUTPUT_DIR": "Output Folder",
    "RUN": "Run",
    "GENERATE": "Generate",
    "EXPORT": "Export",
    "CANCEL": "Cancel",
    "CLOSE": "Close",
    "STATUS_IDLE": "Idle",
    "STATUS_RUNNING": "Running",
    "STATUS_DONE": "Done",
    "STATUS_FAILED": "Failed",
    "STATUS_CANCELED": "Canceled",
    "BUSY_MSG": "Another process is running.",
    "INVALID_INPUT": "Invalid Input",
    "UNSAFE_SCENE_PATH_TITLE": "Change the scene folder path",
    "UNSAFE_SCENE_PATH_BODY": (
        "The scene folder path contains characters or length that can fail in image libraries or external tools.\n\n"
        "{reasons}\n\n"
        "Current path:\n{path}\n\n"
        "Use a short folder path with ASCII letters and numbers only.\n\n"
        "Examples:\nD:\\work\\scene01\nD:\\projects\\site_001"
    ),
    "UNSAFE_PATH_REASON_NON_ASCII": "The path contains non-ASCII characters.",
    "UNSAFE_PATH_REASON_TOO_LONG": "The path is too long ({length} characters; target is under {limit}).",
    "UNSAFE_PATH_REASON_CONTROL_CHARS": "The path contains control characters ({value}).",
    "UNSAFE_PATH_REASON_QUOTE": "The path contains a double quote.",
    "UNSAFE_PATH_REASON_UNKNOWN": "The path contains characters that cannot be verified as safe.",

    # Step 1
    "INPUT_VIDEO": "Input Video",
    "INPUT_VIDEO_PLACEHOLDER": "Select 360 video(s)...",
    "VIDEO_FILE_FILTER": "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v);;All Files (*.*)",
    "MODE_CHANGE": "Auto Interval",
    "MODE_CHANGE_SHORT": "Auto",
    "MODE_FIXED": "Fixed Interval",
    "MODE_FIXED_SHORT": "Fixed",
    "FIXED_SMART": "Motion",
    "FIXED_SMART_ESTIMATE": "adjusted after analysis",
    "QUICK_EXTRACT": "Quick extract",
    "QUICK_EXTRACT_ESTIMATE": "quick extract",
    "CHANGE_THRESHOLD": "Change Threshold",
    "CHANGE_THRESHOLD_SHORT": "Diff",
    "MIN_GAP": "Min Gap (sec)",
    "MIN_GAP_SHORT": "Min",
    "MAX_GAP": "Max Gap (sec)",
    "MAX_GAP_SHORT": "Max",
    "INTERVAL": "Interval (sec)",
    "INTERVAL_SHORT": "Interval",
    "SECONDS_SUFFIX": "s",
    "ANALYSIS_WIDTH": "Analysis Width (px)",
    "QUALITY_MIN_SCORE": "Quality Review Score",
    "QUALITY_MIN_IMPROVEMENT": "Alternate Frame Criterion",
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
    "EXTRACT_OUTPUT_MODE": "Image Save Mode",
    "EXTRACT_OUTPUT_APPEND": "Add New Only",
    "EXTRACT_OUTPUT_REPLACE_VIDEO": "Reset and Overwrite",
    "CLEAR_INPUT_VIDEO": "Clear Input Videos",
    "CLEAR_INPUT_VIDEO_HINT": "Clear selected input videos, video info, and frame estimates. No files are deleted.",
    "EXTRACT_READY_SECTION": "Pre-run Check",
    "EXTRACT_READY_NO_VIDEO": "Select an input video.",
    "EXTRACT_READY_VIDEO_NOT_FOUND": "Input video was not found.",
    "EXTRACT_READY_NO_SCENE": "Set the scene folder at the top.",
    "EXTRACT_READY_BAD_ANALYSIS_WIDTH": "Analysis width must be an integer greater than or equal to 0.",
    "EXTRACT_READY_NO_VIDEO_INFO": "Load video information.",
    "EXTRACT_READY_DUPLICATE_VIDEO": "This video has already been extracted ({n} session(s)). Go to Step 2 to use the existing result, or choose Reset and Overwrite to re-extract it.",
    "EXTRACT_READY_DUPLICATE_REPLACE": "Ready to reset {n} existing session(s) for this video and overwrite them with the current settings.",
    "EXTRACT_READY_QUEUE_OK": "Ready: {n} videos will be extracted sequentially.",
    "EXTRACT_READY_QUEUE_PARTIAL": "Ready: {n} new videos will run. {skipped} already-extracted videos will be skipped.",
    "EXTRACT_READY_QUEUE_ALL_DUPLICATE": "All {n} selected videos have already been extracted. Choose Reset and Overwrite if needed.",
    "EXTRACT_READY_QUEUE_REPLACE": "{n} videos will be extracted sequentially. {replace} already-extracted videos will be replaced.",
    "EXTRACT_READY_OK": "Ready: frame extraction can run.",
    "VIDEO_QUEUE_LABEL_FORMAT": "Video queue: {total}  |  run {queued}  |  skip {skipped}",
    "QUEUE_ESTIMATE_FORMAT": "queue {queued} videos (skip {skipped} extracted)",
    "VIDEO_INFO_SINGLE_FORMAT": "{width}x{height}  |  {fps:.2f} fps  |  {duration}  |  approx. {frames} frames",
    "VIDEO_INFO_MULTI_HEADER_FORMAT": "Video queue: {total}  |  run {queued}  |  skip {skipped}  |  probed {probed}",
    "VIDEO_INFO_MULTI_ITEM_FORMAT": "{name}: {width}x{height}  |  {fps:.2f} fps  |  {duration}  |  approx. {frames} frames",
    "VIDEO_INFO_FAILED_SUFFIX": "  |  {failed} probe failed",
    "FIXED_INTERVAL_ESTIMATE_FORMAT": "Fixed {interval}s: approx. {count} images",
    "FIXED_INTERVAL_ESTIMATE_MULTI_HEADER_FORMAT": "Fixed {interval}s",
    "FIXED_INTERVAL_ESTIMATE_MULTI_ITEM_FORMAT": "{name}: approx. {count} images",
    "FIXED_INTERVAL_ESTIMATE_MULTI_TOTAL_FORMAT": "Total: approx. {count} images ({videos} videos)",
    "ESTIMATE_MISSING_INFO_SUFFIX": "  |  {missing} not probed",

    # Step 2
    "EXPORT_KEEP": "Export Keep Frames",
    "FINALIZE_INPLACE": "Finalize In-Place",
    "FINALIZE_BUTTON": "Finalize (Delete Drops)",
    "FINALIZE_BUTTON_HINT": (
        "Delete dropped frames from images/ and preserve kept filenames.\n"
        "Irreversible. Enable the backup checkbox if you want a safety copy."
    ),
    "BACKUP_BEFORE_FINALIZE": "Back up to images_backup/ before Apply",
    "BACKUP_BEFORE_FINALIZE_HINT": (
        "ON: snapshot images/ to images_backup/ before Apply (existing backup is replaced).\n"
        "OFF: no backup (saves disk; cannot be undone)."
    ),
    "ACTION_FINALIZE_REVIEW": "Apply",
    "REVIEW_EMBED_NO_SCENE": "Set an extracted scene folder in the header to automatically load selected_frames.csv.",
    "REVIEW_EMBED_EMPTY": "After Step 1 extraction completes, or after you select an extracted scene folder, the frame review view appears here automatically.",
    "REVIEW_EMBED_MISSING": "CSV not found:\n{path}",

    # --- review_frames.py (Step 2 review GUI) ---
    "REVIEW_TITLE": "Frame Review",
    "REVIEW_PREVIEW_MODE_SINGLE": "Single Preview",
    "REVIEW_PREVIEW_MODE_THUMBNAILS": "Thumbnails",
    "REVIEW_DECISION_PREFIX": "Decision: ",
    "REVIEW_DECISION_KEEP": "Keep",
    "REVIEW_DECISION_DROP": "Drop",
    "REVIEW_INFO_YES": "target",
    "REVIEW_INFO_NO": "normal",
    "REVIEW_PROBLEMS_FORMAT": "Review {n} | add {a} / replace {r} / low {f} / skip {t} | now {cur}",
    "REVIEW_FRAME_SLIDER_TIP": "Slide through frames in the CSV.",
    "REVIEW_FRAME_POSITION_FORMAT": "{seq} / {total} : {name}",
    "REVIEW_INFO_FORMAT": "Video position: {ts}  |  Quality score: {quality}",
    "REVIEW_QUALITY_ORIGINAL_FORMAT": "orig:{score}",
    "REVIEW_QUALITY_THRESHOLD_FORMAT": "threshold:{score}",
    "REVIEW_FLAG_TIP": "Toggle the keep flag. Changes are written to the CSV immediately.",
    "REVIEW_RESET_DECISION_TIP": "Reset this frame's keep flag to the state loaded from the CSV.",
    "REVIEW_BTN_PREV": "Prev (←)",
    "REVIEW_BTN_NEXT": "Next (→)",
    "REVIEW_BTN_PREV_PROBLEM": "extract problem ← (Shift+F)",
    "REVIEW_BTN_NEXT_PROBLEM": "extract problem → (F)",
    "REVIEW_BTN_PROBLEM_TIP": (
        "Cycle through frames extract_frames flagged automatically\n"
        "(representative replacement / low quality / thinned), in CSV order."
    ),
    "REVIEW_NO_PROBLEMS": "No problem frames found.",
    "REVIEW_IMAGE_NOT_FOUND": "Image not found:\n{path}",
    "REVIEW_IMAGE_LOAD_FAILED": "Failed to load image:\n{path}",
    "REVIEW_INVALID_INPUT": "Invalid input",
    "REVIEW_INFO_HEADER": "Info",
    "REVIEW_SAVE_FAILED_HEADER": "Save failed",
    "REVIEW_SAVE_FAILED_BODY": "Could not write the keep flag to the CSV:\n{error}",
    "REVIEW_ADVISORY_FALLBACK": "Review: no high-quality representative in search window",
    "REVIEW_ADVISORY_REPLACED": "Replaced: using a more SfM-ready nearby frame",
    "REVIEW_ADVISORY_SMART_ADDED": "Motion-adjusted: added for high motion",
    "REVIEW_ADVISORY_THINNED": "Auto-thinned: low motion, currently dropped",
    "REVIEW_ADVISORY_NORMAL": "Normal: quality OK",
    "NEXT_STEP_MASK_NOTICE": "If drop-marked images remain, or if you changed keep/drop choices, press Apply at the bottom to write them into the image folder.\nIf Apply is disabled, proceed directly to Step 3 (Mask Generation).",
    "METASHAPE_NOTICE": "After mask generation, continue with the route that matches your dataset.\nFor the Metashape route, import masks/ as masks, run SfM, then use Step 4 for 3DGS export.\nFor the COLMAP route, use Step 4 to export viewpoint images and optionally run COLMAP/GLOMAP.",
    "EXPORT_DIR": "Export Folder",

    # Step 3
    "IMAGES_DIR": "Images Folder",
    "MASKS_DIR": "Masks Folder",
    "YOLO_LEVEL": "YOLO Level",
    "YOLO_LEVEL_COMPACT": "Level",
    "YOLO_LEVEL_FAST": "0 Fast",
    "YOLO_LEVEL_STANDARD": "1 Standard",
    "YOLO_LEVEL_QUALITY": "2 Quality",
    "YOLO_LEVEL_BEST": "3 Best",
    "MASK_QUALITY": "Qual.",
    "MASK_QUALITY_STANDARD": "Standard",
    "MASK_QUALITY_HIGH": "High",
    "MASK_QUALITY_BEST": "Best",
    "YOLO_EXPAND": "Mask Expand (px)",
    "YOLO_EXPAND_COMPACT": "Exp.",
    "PERSON_MODEL": "Person Model",
    "PERSON_MODEL_YOLO_SAM": "YOLO/SAM2.1",
    "PERSON_MODEL_SAM31": "SAM3.1",
    "YOLO_BOTTOM_ENHANCE": "Bottom Enhance",
    "YOLO_BOTTOM_STANDARD": "Standard",
    "YOLO_BOTTOM_STRONG": "High",
    "YOLO_BOTTOM_MAX": "Max",
    "YOLO_ADD_EXT": "Add Extension",
    "YOLO_CLASSES": "Detection Classes",
    "STITCH_BOUNDARY_WIDTH": "Boundary Mask Width (deg)",
    "STITCH_BOUNDARY_WIDTH_COMPACT": "Seam",
    "STITCH_WORKERS": "Workers",
    "STITCH_WORKERS_COMPACT": "Workers",
    "RUN_MASKS": "Regenerate Selected Masks",
    "MASK_TASKS_LABEL": "Masks:",
    "ADDITIONAL_MASKS_LABEL": "Options:",
    "MASK_IMAGE_TYPE": "Image Type:",
    "MASK_IMAGE_TYPE_EQUIRECT": "360°",
    "MASK_IMAGE_TYPE_NORMAL": "Normal",
    "MASK_TASK_YOLO": "Person",
    "MASK_TASK_STITCH": "Stitch",
    "MASK_TASK_OVEREXPOSURE": "Overexp",
    "MASK_TASK_SKY": "Sky",
    "MASK_TASK_CUSTOM": "Custom",
    "MASK_TASK_REQUIRED": "Select at least one mask to create.",
    "MASK_PHASE_PRIMARY": "Primary Mask",
    "MASK_PHASE_STITCH": "Stitch",
    "MASK_PHASE_OVEREXPOSURE": "Overexposure",
    "MASK_PHASE_CUSTOM": "Custom",
    "MASK_PHASE_INIT": "Init Masks",
    "MASK_MODEL": "Model",
    "CUSTOM_MASK_SECTION": "Custom Mask",
    "CUSTOM_MASK_FILE": "Mask Image",
    "CUSTOM_MASK_NOT_SELECTED": "Not selected",
    "CUSTOM_MASK_BROWSE": "Load",
    "CUSTOM_MASK_CLEAR": "Remove",
    "CUSTOM_MASK_SELECT_FILE": "Select custom mask image",
    "CUSTOM_MASK_FILE_FILTER": "PNG images (*.png)",
    "CUSTOM_MASK_REQUIRED": "Select a custom mask image.",
    "CUSTOM_MASK_NOT_FOUND": "Custom mask image was not found: {path}",
    "MASK_READY_OK": "Ready: masks can be generated.",
    "MASK_READY_EXTERNAL_IMAGES": "Ready: masks will be generated for external images without selected_frames.csv.",
    "MASK_READY_SCENE_NOT_FOUND": "Scene folder was not found.",
    "MASK_READY_NO_IMAGES_DIR": "images/ was not found in the scene folder. Run Step 1 extraction or place external images in images/.",
    "MASK_READY_NO_IMAGES": "No supported images were found in images/. Run Step 1 extraction or place external images in images/.",
    "MASK_READY_NO_CSV": "selected_frames.csv was not found. Run Step 1 frame extraction first.",
    "EXTERNAL_IMAGES_SECTION": "External Images",
    "EXTERNAL_IMAGES_HINT": "For normal video frames or still-camera image sequences, add an image folder here. Images are copied to images/ in the current scene folder.",
    "EXTERNAL_IMAGES_ADD": "Add Image Folder",
    "EXTERNAL_IMAGES_OPEN": "Open images/",
    "EXTERNAL_IMAGES_SELECT_FOLDER": "Select image folder to add",
    "EXTERNAL_IMAGES_SELECT_SCENE": "Select scene folder",
    "EXTERNAL_IMAGES_SCENE_REQUIRED_TITLE": "Scene folder required",
    "EXTERNAL_IMAGES_SCENE_REQUIRED_MESSAGE": "External images need a scene folder first. Select a new scene folder now?",
    "EXTERNAL_IMAGES_SOURCE_IS_TARGET": "The selected folder is the current images/ folder. No copy is needed.",
    "EXTERNAL_IMAGES_SOURCE_NOT_FOUND": "Image folder was not found: {path}",
    "EXTERNAL_IMAGES_RESULT_TITLE": "Add External Images",
    "EXTERNAL_IMAGES_RESULT": "Added {added} / skipped {skipped}",
    "MASK_PENDING_DROPS_ERROR": "{n} drop-marked images still exist in the image folder. Apply Step 2 before generating masks.\n{files}",
    "MASK_UNTRACKED_IMAGES_ERROR": "{n} images in the image folder are not listed in selected_frames.csv. Old extraction results may be mixed in.\n{files}",
    "RUN_YOLO": "Run YOLO",
    "RUN_STITCH": "Run Stitch",
    "RUN_YOLO_STITCH": "Run YOLO + Stitch",
    "CLASS_PRESET_PERSON": "Person Only",
    "CLASS_PRESET_VEHICLES": "Person+Vehicles",
    "CLASS_PRESET_ALL": "Select All",
    "CLASS_PRESET_CLEAR": "Clear",
    "DETECTION_TARGET_SECTION": "Detection Targets",
    "YOLO_CLASS_LIST_SECTION": "Detection Targets",
    "ADE20K_CLASS_LIST_SECTION": "Detection Targets",
    "SAM31_PROMPT_SECTION": "Detection Targets",
    "SAM31_APPLY_MODE": "Op.",
    "SAM31_APPLY_REPLACE": "Replace",
    "SAM31_APPLY_ADD": "Add",
    "SAM31_APPLY_SUBTRACT": "Subtract",
    "SAM31_CUSTOM_PROMPT_PLACEHOLDER": "Add prompts: tripod, hand; selfie stick",
    "SAM31_SUBTRACT_PROMPT_PLACEHOLDER": "Subtract prompts: pictogram, logo",
    "OVEREXPOSURE": "Overexposure Mask",
    "OVEREXPOSURE_THRESHOLD": "Overexposure Threshold (RGB)",
    "OVEREXPOSURE_THRESHOLD_COMPACT": "Threshold",
    "OVEREXPOSURE_DILATE": "Dilate Radius (px)",
    "OVEREXPOSURE_DILATE_COMPACT": "Dilate",
    "RUN_OVEREXPOSURE": "Run Overexposure Mask",
    "SKY_MODEL": "Model",
    "SKY_MODEL_MASK2FORMER": "Mask2Former",
    "SKY_MODEL_SAM31": "SAM3.1",
    "SKY_MODEL_SAM31_MISSING": "SAM3.1 checkpoint not found: models/sam3.1/sam3.1_multiplex.pt",
    "SKY_MODE": "Mode",
    "SKY_MODE_FULL": "High Quality",
    "SKY_MODE_HYBRID": "Direct+Top",
    "SKY_MODE_DIRECT": "Direct",
    "SKY_MODE_TOP": "Top View",
    "SKY_MODE_BOTTOM": "Bottom View",
    "SKY_INFERENCE_SIZE": "Size",
    "SKY_EXPAND": "Sky Expand (px)",
    "SKY_MIN_SCORE": "Min Score",
    "SKY_MIN_AREA": "Min Area",
    "SKY_MODEL_DETAILS_SECTION": "Model Details",
    "SKY_POSTPROCESS_SECTION": "Sky Mask",
    "SKY_TOP_CONNECTED": "Top edge only",
    "RUN_ALL": "Run All Masks",

    # Step 4
    "JSON_NAME": "transforms.json Name",
    "EXPORT_METHOD": "Export Method",
    "EXPORT_METHOD_COMPACT": "Select:",
    "METHOD_METASHAPE_IMPORT": "Metashape",
    "METHOD_COLMAP_EXPORT": "COLMAP",
    "COLMAP_PIPELINE_SECTION": "COLMAP Run Settings",
    "RUN_COLMAP_SFM": "Run COLMAP after export",
    "COLMAP_EXECUTABLE": "COLMAP Executable",
    "GLOMAP_EXECUTABLE": "GLOMAP Executable",
    "COLMAP_MATCHER_COMPACT": "Matcher:",
    "COLMAP_MAPPER_COMPACT": "Mapper:",
    "COLMAP_MATCHER_SEQUENTIAL": "Sequential",
    "COLMAP_MATCHER_EXHAUSTIVE": "Exhaustive",
    "COLMAP_MAPPER_INCREMENTAL": "Incremental",
    "COLMAP_MAPPER_GLOBAL": "Global",
    "COLMAP_MAPPER_GLOMAP": "GLOMAP",
    "PHASE_COLMAP_RIG_EXPORT": "Export COLMAP View Images",
    "PHASE_COLMAP_FEATURE": "COLMAP Feature",
    "PHASE_COLMAP_RIG_CONFIG": "COLMAP Rig Setup",
    "PHASE_COLMAP_MATCH": "COLMAP Matcher",
    "PHASE_COLMAP_MAPPER": "COLMAP Mapper",
    "COLMAP_EXEC_NOT_FOUND": "COLMAP executable was not found. Select the installed colmap executable: {path}",
    "GLOMAP_EXEC_NOT_FOUND": "GLOMAP executable was not found. Select glomap executable when using GLOMAP: {path}",
    "TARGET_PROFILE": "Output Preset",
    "PROFILE_POSTSHOT": "Postshot",
    "PROFILE_BRUSH": "Brush",
    "PROFILE_LICHTFELD": "LichtFeld",
    "PROFILE_CUSTOM": "Custom",
    "PROFILE_CUSTOM_HINT": "Custom: manual settings",
    "AXIS_TRANSFORM": "Axis Transform",
    "AXIS_TRANSFORM_POSTSHOT": "Postshot",
    "AXIS_TRANSFORM_BRUSH": "Brush",
    "AXIS_TRANSFORM_NONE": "None",
    "VIEW_MODE": "View Mode",
    "VIEW_CUSTOM": "Custom Grid",
    "VIEW_CUBE6": "Cube6",
    "YAW_OFFSET": "Yaw Offset (deg)",
    "YAW_SLOTS": "Yaw Slots",
    "PITCH_ROWS": "Pitch Rows",
    "OUTPUT_SCALE": "Image Size",
    "INVERT_MASKS": "Invert Masks",
    "NO_IMAGE": "Skip Image/Mask Conversion",
    "EXPORT_TARGETS": "Output",
    "EXPORT_IMAGES": "Images",
    "EXPORT_MASKS": "Masks",
    "MASK_DIR": "Mask Folder",
    "METASHAPE_PREPROCESS": "Metashape Import Settings",
    "METASHAPE_XML": "Camera XML",
    "METASHAPE_PLY": "Point Cloud PLY",
    "SCALE_FACTOR": "Scale Factor",
    "SCALE_FACTOR_COMPACT": "Scale",
    "MS_USE_PLY": "PLY",
    "NO_FIX_ROTATION": "No rot. fix",
    "RUN_CUBEMAP": "Export View Images",
    "PREVIEW": "Preview",

    # ViewConfig / Preview labels
    "VIEW_MODE_LABEL": "Preset:",
    "CUSTOM_GRID": "Custom Grid",
    "CUBE6_LABEL": "Cube6",
    "YAW_OFFSET_LABEL": "Yaw Offset:",
    "YAW_SLOTS_LABEL": "Yaw:",
    "PITCH_ROWS_LABEL": "Pitch Rows:",
    "YAW_SLOT_ADD": "Add Yaw Column",
    "YAW_SLOT_REMOVE": "Remove Yaw Column",
    "YAW_SLOT_COUNT_FORMAT": "Yaw {count}",
    "PITCH_ROW_ADD": "Add Pitch Row",
    "PITCH_ROW_REMOVE": "Remove Pitch Row",
    "PITCH_ROW_COUNT_FORMAT": "Pitch {count}",
    "APPLY": "Apply",
    "SELECT_ALL": "Select All",
    "DESELECT_ALL": "Deselect All",
    "SELECTED_VIEWS": "Selected Views",
    "VIEW_SELECTION_SECTION": "View Selection Grid",
    "VIEW_SELECTION_COMPACT_SECTION": "Output Views",
    "OUTPUT_IMAGE_COUNT_LABEL": "Output Images",
    "OUTPUT_IMAGE_COUNT_FORMAT": "{count}",
    "OUTPUT_RESET_TITLE": "Reset Output Folder",
    "OUTPUT_RESET_MESSAGE": "The output folder already contains files.\n\n{path}\n\nTo avoid mixing old images or transforms.json, its contents will be deleted before export. Continue?",
    "OUTPUT_PARTIAL_RESET_TITLE": "Reset Export Targets",
    "OUTPUT_PARTIAL_RESET_MESSAGE": "The selected export target folders already contain files.\n\n{paths}\n\nOnly those target folders will be deleted before export. Continue?",
    "PITCH_SLOT_HEADER": "Pitch / Slot",
    "PREVIEW_IMAGE_LABEL": "Preview Image:",
    "PREVIEW_IMAGE_POSITION_FORMAT": "{seq} / {total} : {name}",
    "AUTO": "Auto",
    "RELOAD": "Reload",
    "MASK_PREVIEW_BUTTON": "Mask Preview",
    "MASK_PREVIEW_CLEAR_BUTTON": "Clear Preview",
    "MASK_PREVIEW_VISIBILITY_BUTTON": "Show Preview",
    "YOLO_PREVIEW_BUTTON": "YOLO Preview",
    "MASK_REPROCESS_CURRENT_BUTTON": "Regenerate Mask",
    "MASK_REPROCESS_SELECTED_BUTTON": "Regenerate {count} Masks",
    "MASK_REPROCESS_SELECTED_FALLBACK_BUTTON": "Regenerate Masks",
    "MASK_REPROCESS_CURRENT_RUNNING": "Regenerating...",
    "MASK_REPROCESS_CURRENT_DONE": "Regenerated: {name}",
    "MASK_REPROCESS_SELECTED_PROGRESS": "Regenerating: {done}/{total} {name}",
    "MASK_REPROCESS_SELECTED_DONE": "Regenerated: {done}/{total} images",
    "MASK_REPROCESS_SELECTED_FAILED": "Failed to regenerate: {failed}/{total} images",
    "MASK_REPROCESS_CURRENT_FAILED": "Failed to regenerate the current image",
    "MASK_REPROCESS_NO_BASE_MASK": "Could not create the mask base. Check the current settings.",
    "MASK_OVERLAY_TOGGLE": "Mask Overlay",
    "MASK_OPACITY_LABEL": "Mask Opacity:",
    "MASK_IMAGE_LABEL": "Mask Image:",
    "CLEAR": "Clear",
    "PREVIEW_OVERLAY_SECTION": "Preview Overlay Settings",
    "MASK_PREVIEW_SECTION": "Mask Preview",
    "MASK_PREVIEW_MODE_SINGLE": "Single Preview",
    "MASK_PREVIEW_MODE_THUMBNAILS": "Thumbnails",
    "MASK_PREVIEW_THUMBNAIL_STATUS": "Thumbnail list: {count} images",
    "MASK_PREVIEW_TEMP": "Mask: preview result",
    "MASK_PREVIEW_RUNNING": "Generating mask preview...",
    "MASK_PREVIEW_FAILED": "Mask preview failed",
    "MASK_PREVIEW_CLEARED": "Preview cleared",
    "MASK_PREVIEW_NO_IMAGE": "Select a preview image",
    "MASK_PREVIEW_NO_SCENE_HELP": "Select a scene folder to show images from images/ here.",
    "MASK_PREVIEW_EMPTY_HELP": "Run Step 1 frame extraction or place images in images/ to show mask previews here.",
    "MASK_PREVIEW_YOLO_EXISTING": "Primary: existing mask",
    "MASK_PREVIEW_YOLO_TEMP": "Primary: preview result",
    "MASK_PREVIEW_YOLO_PENDING": "Primary: after generation",
    "MASK_PREVIEW_YOLO_RUNNING": "Running primary mask...",
    "MASK_PREVIEW_YOLO_FAILED": "Primary mask preview failed",
    "MASK_PREVIEW_YOLO_NO_IMAGE": "Select a preview image",
    "MASK_PREVIEW_SKY_EXISTING": "Sky: existing mask",
    "MASK_PREVIEW_SKY_PENDING": "Sky: after generation",
    "YOLO_SAM_LICENSE_NOTICE_TITLE": "YOLO/SAM Model Terms",
    "YOLO_SAM_LICENSE_NOTICE_BODY": (
        "The YOLO/SAM mask feature uses third-party model files and libraries.\n\n"
        "This application's own source code is licensed under the MIT License, but the models and related libraries used by the YOLO/SAM feature are governed by separate license terms.\n\n"
        "- Ultralytics YOLO / ultralytics: AGPL-3.0 or Ultralytics Enterprise License\n"
        "- Meta SAM2/SAM2.1: Apache License 2.0\n"
        "- Meta SAM3.1: SAM License (when the SAM3.1 person backend is selected)\n\n"
        "Model weights are not included with this application. They may be downloaded to the user's environment on first use.\n\n"
        "Users are responsible for ensuring that commercial use, redistribution, internal deployment, or product integration complies with the applicable license terms."
    ),
    "YOLO_SAM_LICENSE_NOTICE_DONT_SHOW_AGAIN": "Do not show this notice again",
    "YOLO_SAM_LICENSE_NOTICE_CONTINUE": "Continue",
    "YOLO_SAM_LICENSE_NOTICE_CANCELED": "YOLO/SAM run canceled",
    "SKY_LICENSE_NOTICE_TITLE": "Semantic Mask Model Terms",
    "SKY_LICENSE_NOTICE_BODY": (
        "The semantic mask feature uses third-party Mask2Former ADE20K model files, "
        "Transformers-related libraries, or a user-provided Meta SAM3.1 checkpoint.\n\n"
        "This application's own source code is licensed under the MIT License, but the models, SAM Materials, "
        "related libraries, and training dataset used by mask generation are governed by separate license terms.\n\n"
        "- Mask2Former: MIT License\n"
        "- Transformers / safetensors: Apache License 2.0\n"
        "- Meta SAM3.1: SAM License\n"
        "- ADE20K dataset: governed by the dataset provider's terms\n\n"
        "Model weights are not included with this application. They may be downloaded to the user's environment on first use.\n\n"
        "Users are responsible for confirming compliance for commercial use, redistribution, internal deployment, or product integration."
    ),
    "SKY_LICENSE_NOTICE_CANCELED": "Semantic mask run canceled",
    "MASK_PREVIEW_STITCH_STATUS": "Stitch {width:g} deg",
    "MASK_PREVIEW_OVEREXP_STATUS": "Overexposure RGB>{threshold} +{dilate}px",
    "MASK_PREVIEW_CUSTOM_STATUS": "Custom",
    "MASK_PREVIEW_CUSTOM_INVALID": "Custom: unreadable/size mismatch",
    "MASK_PREVIEW_INVALID_STITCH_WIDTH": "Boundary width must be >= 0 and < 180",
    "MASK_PREVIEW_NO_ACTIVE_MASK": "No active mask preview",
    "CUBEMAP_PREVIEW_SECTION": "View Preview",
    "EXCEED": "exceeded",
    "HIGH": "high",
    "NO_PREVIEW": "No preview image selected",
    "NO_PREVIEW_FOUND": "Preview image not found",
    "PREVIEW_LOAD_FAIL": "Failed to load image",

    # Step1 extra labels
    "SAMPLE_REFRESH": "Refresh Sampled Estimate",
    "VIDEO_LABEL_DEFAULT": "Video: -",
    "ADVANCED_SETTINGS": "Advanced",
    "AUTO_SELECTION_SECTION": "SfM Quality Check",
    "AUTO_SELECTION_HINT": "Scores extracted candidates for SfM and, when needed, selects alternate frames and flags low-quality frames for review.",
    "AUTO_PREFIX_HINT": "auto (video filename)",
    "FRAMES_UNIT": "frames",

    # Step2 extra labels
    "ADD_EXT_LABEL": "Add extension to mask",

    # Step3 section headers
    "YOLO_SECTION": "Mask Settings",
    "STITCH_OVEREXP_SECTION": "Stitch / Overexposure Settings",
    "MASK_TAB_YOLO": "Mask Settings",
    "MASK_TAB_OPTIONS": "Options",
    "MASK_TAB_STITCH_OVEREXP": "Stitch/Overexp.",
    "MASK_TAB_SKY": "Sky",
    "MASK_TAB_CUSTOM": "Custom Mask",

    # Step4 extra
    "OUTPUT_DETAIL": "Output Detail",
    "MS_IMAGES_LABEL": "Images Folder",
    "STEP4_TAB_METASHAPE": "Conversion",
    "STEP4_TAB_VIEW_EXPORT": "Projection Views",
    "STEP4_TAB_COLMAP": "Conversion",

    # Step4 advanced output (added 2026-05)
    "ADVANCED_OUTPUT_SECTION": "View Export Settings",
    "YAW_OFFSET_PER_FRAME": "Yaw Step",
    "YAW_OFFSET_PER_FRAME_HINT": (
        "0=disabled, 30=recommended. Clamped to -180 to 180 degrees. Drag horizontally to adjust.\n"
        "Rotates cubemap per frame to improve 3DGS training stability"
    ),
    "OUTPUT_FORMAT": "Output Format",
    "OUTPUT_FORMAT_COMPACT": "Format:",
    "OUTPUT_FORMAT_AUTO": "Auto",
    "OUTPUT_BIT_DEPTH": "Output Bit Depth",
    "OUTPUT_BIT_DEPTH_COMPACT": "Bit Depth:",
    "OUTPUT_BIT_DEPTH_8": "8-bit",
    "OUTPUT_BIT_DEPTH_SOURCE": "Source",
    "JPG_QUALITY": "JPG/WebP Quality (1-100)",
    "JPG_QUALITY_COMPACT": "JPG/WebP:",
    "EXPORT_COLMAP": "Add COLMAP Text Model",
    "EXPORT_COLMAP_HINT": (
        "Create cameras.txt / images.txt / points3D.txt in output/colmap/\n"
        "from output/transforms.json and PLY.\n"
        "This is not a COLMAP SfM image export."
    ),
}

# ---------------------------------------------------------------------------
# ツールチップテーブル
# ---------------------------------------------------------------------------

_TIPS_JA: dict[str, str] = {
    "SCENE_DIR": "作業対象のシーンフォルダ。再開時は selected_frames.csv と images/ がある抽出済みフォルダを指定します。images/, masks/ などのサブフォルダが自動認識されます",
    "RUN": "現在のタブの処理を開始します",
    "CANCEL": "実行中の処理を中断します",
    "INPUT_VIDEO": "エクイレクタングラー形式の360度動画ファイルを選択。参照ダイアログでは複数ファイルを選択できます",
    "EXTRACT_OUTPUT_MODE": "新規のみ追加: 未抽出の動画だけを画像フォルダへ追加します。リセットして上書き: 同じ動画の前回抽出結果を削除し、現在の設定で作り直します。他の動画は残します",
    "MODE_FIXED": "指定した秒数ごとにフレームを抽出します。横ドラッグで調整可能。推奨は0.8〜1.0秒、UI範囲は0.05〜60秒",
    "FIXED_SMART": (
        "固定間隔を基準に、変化が少ない候補をスキップし、変化が大きい区間には追加候補を入れます。\n"
        "輝度差だけでなく、疎な特徴点追跡によるモーションも使うため、SfMで意味のある視差を拾いやすくなります"
    ),
    "QUICK_EXTRACT": (
        "短いテストSfMをすぐ試したいときに使います。\n"
        "細かな自動選別より待ち時間を短くし、指定した間隔の結果を素早く確認できます"
    ),
    "MODE_CHANGE": "画像の変化量に応じて抽出間隔を自動調整します。最小/最大間隔で極端な枚数増減を防ぎます",
    "INTERVAL": "固定間隔で使うフレーム間隔。単位は秒。横ドラッグで調整可能。推奨は0.8〜1.0秒、UI範囲は0.05〜60秒",
    "CHANGE_THRESHOLD": (
        "自動間隔で使う変化しきい値。単位は正規化スコアで、隣接解析フレームの平均輝度差 / 255 です。\n"
        "範囲は0.000〜1.000。小さいほど敏感に反応して抽出枚数が増え、大きいほど大きな変化だけを採用します。\n"
        "目安は0.010〜0.120、既定値は0.040です。横ドラッグで調整可能"
    ),
    "MIN_GAP": "変化量で補正するときの最小間隔。単位は秒。追加候補はこの秒数より近くには入りません。UI範囲は0.05〜60秒",
    "MAX_GAP": "変化量で補正するときの安全間隔。単位は秒。低変化スキップで採用フレームが空きすぎるのを防ぎます。UI範囲は0.05〜60秒",
    "IMAGE_FORMAT": "出力画像の形式。jpgはファイルサイズ小、pngは無劣化",
    "JPEG_QUALITY": "ffmpegの-q:v値。1=最高品質、31=最低品質。2-5推奨。横ドラッグで調整可能",
    "ANALYSIS_WIDTH": "変化検出・品質評価に使うデコード幅。大きいほど精度は上がるが遅くなります",
    "QUALITY_MIN_SCORE": (
        "Step 2で品質確認として表示する基準スコア。範囲は0.00〜1.00、単位は正規化スコアです。\n"
        "品質スコア = 特徴点数、特徴点の画面内分布、シャープネス、コントラスト、白飛び/黒つぶれペナルティの合成値。\n"
        "代替フレーム選択後の品質スコアがこの値未満なら、Step 2で確認対象になります。既定値: 0.35"
    ),
    "QUALITY_MIN_IMPROVEMENT": (
        "元フレームではなく近傍の代替フレームを選ぶために必要な品質スコア差。範囲は0.00〜1.00です。\n"
        "候補品質スコア - 元品質スコア がこの値以上のときだけ、代替フレームを採用します。\n"
        "大きいほど代替選択は控えめ、小さいほど積極的になります。既定値: 0.08"
    ),
    "FFMPEG_PATH": "ffmpegの実行パス。PATHに通っていれば 'ffmpeg' でOK",
    "FFPROBE_PATH": "ffprobeの実行パス。動画情報の取得に使用",
    "FILENAME_PREFIX": "出力ファイル名の接頭辞。空欄なら動画ファイル名を自動使用",
    "SAMPLE_BTN": "動画の一部をサンプリングしてフレーム数を再推定 (自動間隔モードのみ)",
    "EXPORT_DIR": "採用フレームのコピー先フォルダ名。'images'ならインプレース処理",
    "EXPORT_KEEP": "CSVで採用にしたフレームだけを指定フォルダにコピー",
    "FINALIZE_INPLACE": "images/内の除外フレームを削除し、採用フレームのファイル名は維持。元に戻せないので注意",
    "IMAGES_DIR": "シーンフォルダ内の images/ を自動使用します。Step 1 のフレーム抽出結果が入る標準フォルダです",
    "MASKS_DIR": "シーンフォルダ内の masks/ を自動使用します。存在しない場合は生成時に作成されます",
    "RUN_MASKS": "主マスク生成と、現在ONの追加マスク処理で masks/ を再生成します。前回OFFだった追加処理結果は残りません",
    "MASK_TASK_YOLO": "主マスク生成は常に実行されます。モデルと対象はマスク設定タブで選びます",
    "MASK_TASK_STITCH": "スティッチ境界をマスクに追加。手ブレ補正、方向ロック、AIスティッチなどで境界位置が動く素材では通常OFF",
    "MASK_TASK_STITCH_DISABLED_NORMAL": "スティッチ境界は360°エクイレクタングラー画像専用です。通常画像では使いません",
    "MASK_TASK_OVEREXPOSURE": "白飛びした画素を検出してマスクに追加。室内照明では消しすぎる場合があるため必要な時だけON",
    "MASK_TASK_SKY": "選択した空検出モデルで空領域を検出してマスクに追加。SfMで空の特徴点を避けたい場合に使います",
    "MASK_TASK_CUSTOM": "ユーザーが用意したPNG静的マスクを最後にAND合成します。8bit/16bitのグレー/RGB/RGBAを0/255に二値化し、白=採用、黒=除外として扱います。サイズが一致する画像だけに適用し、不一致はスキップします",
    "CUSTOM_MASK_FILE": "全フレームに適用するPNGカスタムマスク。8bitは128以上、16bitは32768以上を白として二値化します。RGB/RGBAはグレースケール化し、アルファは無視します。サイズ不一致は自動リサイズせずスキップします",
    "CUSTOM_MASK_BROWSE": "カスタムマスク画像を選択します。選択するとカスタムマスク処理もONになります",
    "CUSTOM_MASK_CLEAR": "選択中のカスタムマスクを解除し、カスタムマスク処理をOFFにします",
    "MASK_IMAGE_TYPE": "360°画像か通常画像かを選びます。通常画像ではスティッチ境界と360°底面再検出を使いません",
    "MASK_IMAGE_TYPE_EQUIRECT": "エクイレクタングラー360°画像。スティッチ境界マスクと下方向の再検出を使えます",
    "MASK_IMAGE_TYPE_NORMAL": "通常の動画フレームやデジタル一眼の連番画像。スティッチ境界と360°底面再検出を無効化します",
    "EXTERNAL_IMAGES_SECTION": "通常画像をこのシーンの images/ に取り込むための補助操作です。画像タイプが通常のときだけ表示します",
    "EXTERNAL_IMAGES_ADD": "選択したフォルダ直下の対応画像を現在のシーンフォルダの images/ にコピーします。同名ファイルはスキップします",
    "EXTERNAL_IMAGES_OPEN": "現在のシーンフォルダの images/ を開きます。なければ作成します",
    "RUN_ALL": "YOLO人物検出 → スティッチマスク → 白飛びマスクの全工程を順番に実行",
    "RUN_YOLO_STITCH": "YOLO人物検出 → スティッチマスクの2工程を実行 (白飛びは含まない)",
    "PERSON_MODEL": "主マスク生成に使うモデル。YOLO/SAM2.1はCOCOクラス、Mask2FormerはADE20Kクラス、SAM3.1は英語プロンプトを使います",
    "PERSON_MODEL_SAM31": "models/sam3.1/sam3.1_multiplex.pt がある場合に使えるテキストプロンプト型のマスクbackendです",
    "SAM31_APPLY_MODE": "SAM3.1専用。今回のプロンプト検出結果をどう書き込むかを選びます。再生成は今回結果で書き直し、加算は既存マスクがあればそこへ検出領域を黒で追加、減算は既存マスク上の検出領域を白に戻します。漏れの追加や誤検出の取り消しに使います",
    "SAM31_CUSTOM_PROMPT": "SAM3.1へ追加で投げる英語プロンプト。カンマ、セミコロンで区切れます。区切り前後の半角スペースは無視し、selfie stickのような語内スペースは保持します",
    "SAM31_SUBTRACT_PROMPT": "SAM3.1の検出結果から差し引く英語プロンプト。personがピクトグラムやロゴを拾う場合などに使います。区切り文字とスペースの扱いは追加プロンプトと同じです",
    "YOLO_LEVEL": "YOLO/SAM2.1人物検出の探索強度です。360°画像では「2 高品質」を推奨します。処理時間を優先する確認用は「1 標準」、それでも人物が漏れる場合は「3 最高」を使います。通常画像では「1 標準」から始めるのが目安です。Mask2Former/SAM3.1では使いません",
    "MASK_QUALITY": "主マスク生成に渡す入力素材の密度です。標準は全体直処理中心、高品質は360°の上部/下部投影と人物向けタイルを追加、最高はより細かいタイルと強い下部補助を使います。通常画像では360°専用の投影補助は使いません",
    "YOLO_EXPAND": "主マスク境界を固定ピクセルで補正します。既定は0px。横ドラッグで調整可能。安全範囲は -16〜32px",
    "YOLO_BOTTOM_ENHANCE": "YOLO/SAM2.1で360°画像の真下付近に写る撮影者・三脚・手元の検出漏れを減らします。下部が十分にマスクされているなら「標準」。真上から見た撮影者が漏れる場合は「高」。それでも漏れる場合だけ「最高」を使います。Mask2Former/SAM3.1では投影補助を使います",
    "YOLO_ADD_EXT": "マスクファイル名を image.jpg.png のように元の拡張子を残す形式にする",
    "STITCH_BOUNDARY_WIDTH": "除外するスティッチ境界帯の合計幅。横ドラッグで調整できます。GUIでは安全のため0〜30度に制限。5度は従来のFOV 175相当",
    "STITCH_WORKERS": "並列処理のワーカー数。横ドラッグで調整可能。CPUコア数が目安",
    "OVEREXPOSURE_THRESHOLD": "RGB全チャンネルがこの8bit相当値を超えるピクセルを白飛びと判定。16bit画像では同じ比率に換算。GUI範囲は 1〜254",
    "OVEREXPOSURE_DILATE": "白飛び領域を膨張させるピクセル数。既定は1px。0で無効、GUI範囲は 0〜128",
    "SKY_MODEL": "主マスク生成に使うモデルは上のモデル欄で選びます",
    "SKY_MODEL_SAM31": "models/sam3.1/sam3.1_multiplex.pt がある場合に使える実験バックエンドです",
    "SKY_MODE": "投影補助の方式。高品質はエクイレクタングラー直処理、上部投影、下部投影を合成します",
    "SKY_INFERENCE_SIZE": "Mask2Formerの入力サイズ。大きいほど境界が安定しやすくなりますが、GPUメモリと処理時間が増えます。SAM3.1では現在1008固定です",
    "SKY_EXPAND": "検出した空マスクをピクセル単位で拡張/収縮します。正の値で空の除外範囲を広げ、負の値で狭めます",
    "SKY_MIN_SCORE": "Mask2Former専用。0.00〜1.00のクラス信頼度しきい値です。0で無効。上げるほど曖昧な画素を捨て、誤検出は減りますが漏れも増えます",
    "SKY_MIN_AREA": "空マスクだけに適用します。画像面積に対してこの割合未満の小さな空候補を除去します。0%で無効。木の隙間など細い空を残したい場合は0%推奨",
    "SKY_TOP_CONNECTED": "空マスクだけに適用します。画像上端に接している空の連結成分だけ残す強いフィルタです。人物や他の対象には適用しません",
    "OUTPUT_DIR_CUBEMAP": "シーンフォルダ内の output/ を自動使用します。出力 transforms.json もこのフォルダに固定されます",
    "OUTPUT_DIR_COLMAP_PROJECT": "シーンフォルダ内の output/colmap_rig/ が完成COLMAPプロジェクトです。COLMAP実行後はこのフォルダを3DGSアプリへ渡します",
    "RUN_CUBEMAP": "現在の書き出し方式と視点設定で画像、マスク、必要なメタデータを書き出します",
    "EXPORT_METHOD": "MetashapeのSfM結果を3DGS向けに書き出すか、抽出済み画像からCOLMAP向け視点画像を書き出すかを選びます",
    "METHOD_METASHAPE_IMPORT": "MetashapeでSfM済みのカメラXMLと点群PLYを読み込み、3DGSアプリ向けの視点画像・マスク・transforms.jsonを書き出します",
    "METHOD_COLMAP_EXPORT": "抽出済みのエクイレクタングラー画像とマスクから、COLMAP Rig形式の視点画像とrig_config.jsonを書き出します",
    "RUN_COLMAP_SFM": "ONにすると、視点画像を書き出した後に feature_extractor → rig_configurator → matcher → mapper を連続実行します。重い処理なので必要な時だけONにします",
    "COLMAP_EXECUTABLE": "この環境で使う colmap.exe のパス。空欄ならPATH上の colmap.exe を探します",
    "GLOMAP_EXECUTABLE": "GLOMAP mapperを選ぶ場合に使う glomap.exe のパス。COLMAPのGlobal Mapperを使う場合は不要です",
    "COLMAP_MATCHER": "Matcher。Sequentialは高速で動画の連番フレーム向け。Exhaustiveは全ペアを照合するため精度が出る場合がありますが、枚数が増えると極端に遅く、数十時間規模になることがあります",
    "COLMAP_MAPPER": "Mapper。GlobalはCOLMAP 4.0以降に統合されたGLOMAP系のグローバルSfMで、高速なため既定推奨。Incrementalは従来のCOLMAP mapperで堅実ですが遅め。GLOMAPは外部glomap.exe用の互換選択肢です",
    "TARGET_PROFILE": "出力先の3DGSソフトウェアに合わせた座標変換とPLY設定のプリセット",
    "AXIS_TRANSFORM": "出力先に合わせてカメラ座標軸を変換します。プリセット値から変更すると出力プリセットはカスタムになります",
    "OUTPUT_SCALE": "キューブマップ1面の画像サイズ。Fullは入力画像の高さ、Normalは90度画像中央部の角度解像度を元画像に近づける自動サイズ、Halfは軽量出力です。最終品質はFull推奨",
    "EXPORT_TARGETS": "視点画像とマスクのどちらを書き出すかを選びます。マスクだけ作り直した場合は画像をOFF、マスクをONにします。両方OFFではカメラ情報だけ更新します",
    "EXPORT_IMAGES": "視点画像を output/images/ に書き出します。OFFにすると既存の画像を残したまま、マスクやカメラ情報だけ更新できます",
    "EXPORT_MASKS": "マスクを output/masks/ に書き出します。OFFにすると既存のマスクを残したまま、画像やカメラ情報だけ更新できます",
    "JSON_NAME": "出力するカメラパラメータJSONのファイル名",
    "MASK_DIR_CUBEMAP": "入力マスク画像のフォルダ。キューブマップ変換時にマスクも一緒に変換",
    "INVERT_MASKS": "出力マスクの白黒を反転。通常はOFF。出力先で逆極性が必要な場合のみON",
    "OUTPUT_FORMAT": "書き出す視点画像の形式。自動は元画像の形式を維持します。JPG/WebPでは品質設定が使われます",
    "OUTPUT_BIT_DEPTH": "出力画像のビット深度。8bitは対応アプリが多く安定、元画像はPNG/TIFFのビット深度を維持します",
    "JPG_QUALITY": "JPG/WebPの圧縮品質。1が低品質、100が高品質です。PNG/TIFFには影響しません",
    "NO_IMAGE": "画像とマスクを再変換せず、transforms.jsonだけを更新します。既存の output/ 内の画像と masks/ は削除しません",
    "REVIEW_PREVIEW_MODE_SINGLE": "選択中の1枚を大きく表示し、採用/除外フラグを細部確認しながら切り替えます",
    "REVIEW_PREVIEW_MODE_THUMBNAILS": "抽出済みフレームを一覧で表示します。クリックまたは矢印キーで表示中の画像を選択できます",
    "MS_IMAGES": "カメラXML内の画像名に対応する画像フォルダ。シーンフォルダ内の images/ を自動使用します",
    "MS_XML": "MetashapeからエクスポートしたカメラポーズXML。シーンフォルダ内の metashape.xml / cameras.xml / 最初のXMLを自動候補にします",
    "MS_PLY": "Metashapeからエクスポートした点群PLY。LichtFeldではカメラと同じ座標系に変換した pointcloud.ply を作るために使用します",
    "MS_USE_PLY": "Metashapeインポート時に --ply を渡します。LichtFeldではON、Postshot/Brushでは通常OFFです。変更するとカスタムになります",
    "SCALE_FACTOR": "カメラ位置と点群座標に掛けるスケール係数。通常は1.0のまま",
    "NO_FIX_ROTATION": "Metashapeデータ読み込み時の向き補正を無効化。通常はOFFのまま",
    "VIEW_MODE": "Cube6: 4列 x 3行グリッドの標準6面プリセット。上下は既定でS3を使います\nカスタムグリッド: Pitch/Yawを自由に設定",
    "VIEW_SELECTION_SECTION": "書き出す視点をオン/オフできます。項目にマウスを重ねると、プレビュー上で対応する視点をハイライトします",
    "YAW_OFFSET": "全ビューのYaw角にオフセットを加算 (度)。-180〜180度でクランプ。ドラッグで調整できます。スティッチ線を避けるために45度推奨",
    "YAW_SLOTS": "水平方向の分割数 (4-8)。ビュー選択グリッド上の +/- で列数を増減します",
    "PITCH_ROWS": "垂直方向のPitch行数 (1-5)。グリッド上の追加/削除ボタンで行を増減し、左端のPitch角はドラッグで調整できます",
    "APPLY_BTN": "Pitch行とYawスロットの変更をグリッドに反映",
    "PREVIEW_SAMPLE": "プレビューに表示するエクイレクタングラー画像のパス",
    "PREVIEW_BROWSE": "プレビュー画像を手動で選択",
    "PREVIEW_AUTO": "シーンフォルダ内の最初の画像を自動選択",
    "PREVIEW_RELOAD": "シーンフォルダの画像リストを再スキャン",
    "PREVIEW_SLIDER": "シーン内の画像を順番にスライドして切り替え",
    "MASK_PREVIEW_MODE_SINGLE": "選択中の1枚を大きく表示します。細部確認や表示中の再生成に使います",
    "MASK_PREVIEW_MODE_THUMBNAILS": "画像と既存マスクを一覧で表示します。クリックまたは矢印キーで表示中の画像を選択できます",
    "MASK_PREVIEW_BUTTON": "現在表示中の1枚だけ、現在ONのマスク処理で一時プレビューを作成します。masks/ には保存しません",
    "MASK_PREVIEW_CLEAR_BUTTON": "一時プレビューを破棄し、保存済みマスクの表示へ戻します",
    "MASK_PREVIEW_VISIBILITY_BUTTON": "生成済みの一時プレビューと、masks/ の保存済み表示を切り替えます。ONはプレビュー結果、OFFは保存済みマスクです。保存マスクがない場合は画像のみ表示します。プレビュー自体は削除しません",
    "YOLO_PREVIEW_BUTTON": "現在表示中の1枚だけYOLO/SAMを実行し、結果をプレビューに重ねます。マスクフォルダには保存しません",
    "MASK_REPROCESS_CURRENT_BUTTON": "1枚プレビューでは現在表示中の1枚、サムネイル一覧では選択中の画像を現在の設定で masks/ に再生成します。SAM3.1の加算/減算モードでは既存マスクに対して現在の検出結果を適用します",
    "MASK_OVERLAY_TOGGLE": "赤いマスクオーバーレイの表示/非表示を切り替えます。表示時の透過率は45%固定です",
    "MASK_OPACITY": "プレビュー上のマスク領域の赤オーバーレイ透過率 (0=非表示、100=不透明)",
    "MASK_IMAGE": "特定のマスク画像を手動指定。空欄ならマスクフォルダから自動検索",
    "MASK_IMAGE_BROWSE": "マスク画像ファイルを選択",
    "MASK_IMAGE_CLEAR": "手動指定をクリアして自動検索に戻す",
}

_TIPS_EN: dict[str, str] = {
    "SCENE_DIR": "Working scene folder. When resuming, select the extracted folder that contains selected_frames.csv and images/. Subfolders like images/, masks/ are auto-detected",
    "RUN": "Start processing for the current tab",
    "CANCEL": "Abort the running process",
    "INPUT_VIDEO": "Select equirectangular 360-degree video files. The browse dialog supports multiple selection",
    "EXTRACT_OUTPUT_MODE": "Add New Only: add only videos that have not been extracted yet. Reset and Overwrite: remove prior results for the same video and rebuild them with the current settings. Other videos are kept.",
    "MODE_FIXED": "Extract frames every N seconds. Drag horizontally to adjust. Recommended: 0.8-1.0 sec; UI range: 0.05-60 sec",
    "FIXED_SMART": (
        "Keeps the fixed interval baseline, skips low-change candidates, and inserts extra candidates in high-motion ranges.\n"
        "Uses sparse feature tracking as well as luma difference, so motion that matters to SfM is easier to catch"
    ),
    "QUICK_EXTRACT": (
        "Use this when you want a short test SfM run quickly.\n"
        "It favors shorter wait time over fine automatic picking, so you can check the fixed-interval result sooner"
    ),
    "MODE_CHANGE": "Automatically adjusts extraction interval from image change, with min/max gaps as safety limits",
    "INTERVAL": "Fixed extraction interval in seconds. Drag horizontally to adjust. Recommended: 0.8-1.0 sec; UI range: 0.05-60 sec",
    "CHANGE_THRESHOLD": (
        "Change threshold used by auto interval mode. Unit: normalized score = mean absolute luma difference between adjacent analysis frames / 255.\n"
        "Range: 0.000-1.000. Lower values are more sensitive and produce more frames; higher values require larger changes.\n"
        "Typical: 0.010-0.120; default: 0.040. Drag horizontally to adjust"
    ),
    "MIN_GAP": "Minimum spacing for motion adjustment in seconds. Extra candidates are not inserted closer than this. UI range: 0.05-60 sec",
    "MAX_GAP": "Safety spacing for motion adjustment in seconds. Low-change skipping will not leave kept frames farther apart than this. UI range: 0.05-60 sec",
    "IMAGE_FORMAT": "Output format. jpg = smaller files, png = lossless",
    "JPEG_QUALITY": "ffmpeg -q:v value. 1 = best quality, 31 = worst. Recommended: 2-5. Drag horizontally to adjust",
    "ANALYSIS_WIDTH": "Decode width for change and quality scoring. Higher = more accurate but slower",
    "QUALITY_MIN_SCORE": (
        "Quality-score threshold used to flag frames for review in Step 2. Range: 0.00-1.00; unit: normalized score.\n"
        "Quality combines feature count, feature spread, sharpness, contrast, and blown/black exposure penalty.\n"
        "If the final representative is below this value after alternate-frame selection, Step 2 marks it for review. Default: 0.35"
    ),
    "QUALITY_MIN_IMPROVEMENT": (
        "Quality-score gain required before choosing a nearby alternate frame. Range: 0.00-1.00.\n"
        "An alternate frame is used only when candidate score - original score is at least this value.\n"
        "Higher is more conservative; lower is more aggressive. Default: 0.08"
    ),
    "FFMPEG_PATH": "ffmpeg executable path. 'ffmpeg' works if it's on PATH",
    "FFPROBE_PATH": "ffprobe executable path. Used for video metadata probing",
    "FILENAME_PREFIX": "Output filename prefix. Leave empty to use the video filename",
    "SAMPLE_BTN": "Re-estimate frame count by sampling the video (Auto Interval mode only)",
    "EXPORT_DIR": "Destination folder for keep frames. 'images' triggers in-place processing",
    "EXPORT_KEEP": "Copy only frames marked as keep in the CSV to the specified folder",
    "FINALIZE_INPLACE": "Delete dropped frames in images/ and preserve kept filenames. Cannot be undone",
    "IMAGES_DIR": "Automatically uses images/ inside the scene folder. This is the standard output folder from Step 1.",
    "MASKS_DIR": "Automatically uses masks/ inside the scene folder. It is created during generation if missing.",
    "RUN_MASKS": "Regenerate masks/ with the primary mask generator and the extra mask steps currently enabled. Results from extra steps that are now off are not kept.",
    "MASK_TASK_YOLO": "The primary mask generator always runs. Choose its model and targets in the Mask Settings tab",
    "MASK_TASK_STITCH": "Add stitch seam masks. Usually keep OFF for stabilized, direction-locked, or AI-stitched footage where seam positions move",
    "MASK_TASK_STITCH_DISABLED_NORMAL": "Stitch seam masks are only for equirectangular 360° images and are not used for normal images",
    "MASK_TASK_OVEREXPOSURE": "Detect blown-out pixels and add them to masks",
    "MASK_TASK_SKY": "Detect sky regions with the selected sky model and add them to masks. Use this to avoid sky features before SfM",
    "MASK_TASK_CUSTOM": "AND-merge a user-provided static PNG mask as the final step. 8-bit/16-bit grayscale, RGB, or RGBA inputs are binarized to 0/255. White means keep and black means exclude. It applies only to images with matching dimensions; mismatches are skipped",
    "CUSTOM_MASK_FILE": "PNG custom mask applied to every frame with matching dimensions. 8-bit values >=128 and 16-bit values >=32768 become white. RGB/RGBA inputs are converted to grayscale and alpha is ignored. Mismatches are skipped without auto-resizing",
    "CUSTOM_MASK_BROWSE": "Select a custom mask image. Selecting a file also enables custom mask processing",
    "CUSTOM_MASK_CLEAR": "Clear the selected custom mask and disable custom mask processing",
    "MASK_IMAGE_TYPE": "Choose whether the source images are equirectangular 360° images or normal images. Normal mode disables stitch seams and 360° bottom re-detection",
    "MASK_IMAGE_TYPE_EQUIRECT": "Equirectangular 360° images. Enables stitch seam masks and bottom-view re-detection",
    "MASK_IMAGE_TYPE_NORMAL": "Normal video frames or still-camera image sequences. Disables stitch seam masks and 360° bottom re-detection",
    "EXTERNAL_IMAGES_SECTION": "Helper controls for importing normal images into this scene's images/ folder. Shown only in Normal image mode",
    "EXTERNAL_IMAGES_ADD": "Copy supported images directly inside the selected folder to the current scene's images/. Existing filenames are skipped",
    "EXTERNAL_IMAGES_OPEN": "Open the current scene's images/ folder. It is created if missing",
    "RUN_ALL": "Run YOLO person detection, stitch mask, and overexposure mask in sequence",
    "RUN_YOLO_STITCH": "Run YOLO person detection then stitch mask (no overexposure)",
    "PERSON_MODEL": "Primary mask model. YOLO/SAM2.1 uses COCO classes, Mask2Former uses ADE20K classes, and SAM3.1 uses English text prompts",
    "PERSON_MODEL_SAM31": "Text-prompt mask backend available when models/sam3.1/sam3.1_multiplex.pt exists",
    "SAM31_APPLY_MODE": "SAM3.1 only. Choose how to write the current prompt detections. Replace writes the current result, Add blackens detected regions on top of an existing mask when available, and Subtract turns detected regions white in an existing mask. Use this to add misses or undo false detections",
    "SAM31_CUSTOM_PROMPT": "Additional English prompts for SAM3.1. Separate prompts with commas or semicolons. Spaces around separators are ignored; spaces inside prompts such as selfie stick are kept",
    "SAM31_SUBTRACT_PROMPT": "English prompts to subtract from SAM3.1 detections. Use this when person also catches pictograms or logos. Separators and spaces are handled like the add prompt field",
    "YOLO_LEVEL": "Detection strength for the YOLO/SAM2.1 person backend. For 360° images, start with 2 Quality. Use 1 Standard for faster checks, and 3 Best only if people still leak through. For normal images, start with 1 Standard. Not used by Mask2Former/SAM3.1",
    "MASK_QUALITY": "Controls the input view recipe for primary masks. Standard mostly uses direct full-image inference, High adds 360° top/bottom projection and person tiles, and Best uses denser tiles with stronger bottom assist. Normal images skip 360°-specific projection assist",
    "YOLO_EXPAND": "Adjust the primary mask boundary by fixed pixels. Default is 0px. Drag horizontally to adjust. Safe range: -16 to 32px",
    "YOLO_BOTTOM_ENHANCE": "With YOLO/SAM2.1, reduces missed masks near the bottom of 360° images, such as top-down photographers, tripods, and hands. Use Standard when the bottom is already masked well. Use High when top-down photographers remain. Use Max only if they still remain. Mask2Former/SAM3.1 use projection assist instead",
    "YOLO_ADD_EXT": "Name mask files as image.jpg.png (keeping the original extension)",
    "STITCH_BOUNDARY_WIDTH": "Total stitch seam band to exclude. Drag horizontally to adjust. The GUI clamps this to 0-30 degrees for safety. 5 degrees equals legacy FOV 175",
    "STITCH_WORKERS": "Number of parallel workers. Drag horizontally to adjust. Use CPU core count as a guide",
    "OVEREXPOSURE_THRESHOLD": "Pixels with all RGB channels above this 8-bit-equivalent value are flagged as blown-out. 16-bit images are scaled to the same ratio. GUI range: 1-254",
    "OVEREXPOSURE_DILATE": "Dilate blown-out regions by N pixels. Default is 1px. 0 = disabled; GUI range: 0-128",
    "SKY_MODEL": "Choose the primary mask model in the model field above",
    "SKY_MODEL_SAM31": "Experimental backend available when models/sam3.1/sam3.1_multiplex.pt exists",
    "SKY_MODE": "Projection-assist mode. High Quality combines direct equirectangular inference with top and bottom projection views",
    "SKY_INFERENCE_SIZE": "Mask2Former input size. Larger values can improve boundaries but use more GPU memory and time. SAM3.1 currently uses fixed 1008",
    "SKY_EXPAND": "Expand or shrink the detected sky mask in pixels. Positive values exclude more sky; negative values keep a tighter boundary",
    "SKY_MIN_SCORE": "Mask2Former only. Class-confidence threshold from 0.00 to 1.00. 0 disables it. Higher values reject ambiguous pixels but can increase misses",
    "SKY_MIN_AREA": "Applies only to the sky mask. Removes sky candidates smaller than this image-area ratio. 0% disables it. Use 0% when you want to keep thin sky gaps",
    "SKY_TOP_CONNECTED": "Applies only to the sky mask. Keeps only sky components touching the top image edge. This is a strong filter and is never applied to people or other targets",
    "OUTPUT_DIR_CUBEMAP": "Automatically uses output/ inside the scene folder. The output transforms.json is fixed to this folder.",
    "OUTPUT_DIR_COLMAP_PROJECT": "output/colmap_rig/ inside the scene folder is the finished COLMAP project. After COLMAP finishes, pass this folder to 3DGS apps.",
    "RUN_CUBEMAP": "Export images, masks, and required metadata with the current export method and view settings",
    "EXPORT_METHOD": "Choose whether to export a 3DGS dataset from Metashape SfM results or viewpoint images for COLMAP from extracted frames",
    "METHOD_METASHAPE_IMPORT": "Read Metashape SfM camera XML and point cloud PLY, then export viewpoint images, masks, and transforms.json for 3DGS apps.",
    "METHOD_COLMAP_EXPORT": "Export COLMAP Rig viewpoint images, masks, and rig_config.json from extracted equirectangular images.",
    "RUN_COLMAP_SFM": "When enabled, runs feature_extractor, rig_configurator, matcher, and mapper after exporting the views. Keep it off unless you want to start the heavy SfM step.",
    "COLMAP_EXECUTABLE": "Path to the colmap executable for this machine. Leave empty to resolve colmap from PATH.",
    "GLOMAP_EXECUTABLE": "Path to glomap executable when using the legacy GLOMAP mapper. Not needed for COLMAP Global Mapper.",
    "COLMAP_MATCHER": "Matcher. Sequential is fast and suited to ordered video frames. Exhaustive can improve coverage but compares all pairs and can become extremely slow, even tens of hours on large sets.",
    "COLMAP_MAPPER": "Mapper. Global is the COLMAP 4.0+ integrated GLOMAP-style global SfM path and is the recommended default for speed. Incremental is the classic COLMAP mapper and is more conservative but slower. GLOMAP uses an external legacy glomap executable.",
    "TARGET_PROFILE": "Coordinate transform and PLY preset for the target 3DGS software",
    "AXIS_TRANSFORM": "Transform camera axes for the target app. Changing the preset value switches the output preset to Custom",
    "OUTPUT_SCALE": "Cubemap face size. Full uses the input image height, Normal matches the center angular resolution of a 90-degree view to the source image, and Half is a lightweight output. Full is recommended for final quality",
    "EXPORT_TARGETS": "Choose whether to write view images, masks, or only camera metadata. Turn Images off and Masks on when you only rebuilt masks in Step 3.",
    "EXPORT_IMAGES": "Write viewpoint images to output/images/. Turn off to keep existing images and update masks or camera metadata only.",
    "EXPORT_MASKS": "Write masks to output/masks/. Turn off to keep existing masks and update images or camera metadata only.",
    "JSON_NAME": "Output camera parameter JSON filename",
    "MASK_DIR_CUBEMAP": "Input mask folder. Masks are converted alongside cubemap images",
    "INVERT_MASKS": "Invert output mask polarity. Usually keep OFF; enable only when the target app expects the opposite polarity",
    "OUTPUT_FORMAT": "Viewpoint image format. Auto preserves the source image format. JPG/WebP use the quality value.",
    "OUTPUT_BIT_DEPTH": "Output image bit depth. 8-bit is broadly compatible; Source preserves PNG/TIFF source bit depth.",
    "JPG_QUALITY": "JPG/WebP compression quality. 1 is lowest, 100 is highest. PNG/TIFF are not affected.",
    "NO_IMAGE": "Update transforms.json without reconverting images or masks. Existing images and masks inside output/ are preserved",
    "REVIEW_PREVIEW_MODE_SINGLE": "Show the selected frame large while checking or toggling its keep/drop flag",
    "REVIEW_PREVIEW_MODE_THUMBNAILS": "Show extracted frames as a thumbnail list. Click or use arrow keys to choose the current image",
    "MS_IMAGES": "Image folder matching the filenames in the camera XML. The GUI automatically uses images/ inside the scene folder",
    "MS_XML": "Metashape-exported camera pose XML. The GUI auto-suggests metashape.xml, cameras.xml, or the first XML in the scene folder",
    "MS_PLY": "Metashape-exported point cloud PLY. LichtFeld uses this to create pointcloud.ply in the same coordinate system as the cameras",
    "MS_USE_PLY": "Pass --ply during Metashape import. ON for LichtFeld and usually OFF for Postshot/Brush. Changing this switches the output preset to Custom",
    "SCALE_FACTOR": "Scale factor applied to camera positions and point coordinates. Usually leave at 1.0",
    "NO_FIX_ROTATION": "Disable orientation correction when importing Metashape data. Usually leave OFF",
    "VIEW_MODE": "Cube6: standard six-face preset on a 4 x 3 grid. Top/bottom use S3 by default\nCustom Grid: freely set pitch/yaw angles",
    "VIEW_SELECTION_SECTION": "Turn export viewpoints on or off. Hover a viewpoint to highlight the matching area in the preview.",
    "YAW_OFFSET": "Add offset to all view yaw angles (degrees). Clamped to -180 to 180. Drag horizontally to adjust. 45 is recommended to avoid stitch seams",
    "YAW_SLOTS": "Horizontal divisions (4-8). Use the +/- controls on the view grid to add or remove columns",
    "PITCH_ROWS": "Number of vertical pitch rows (1-5). Use the grid add/remove buttons to change rows, and drag the left pitch angle fields to adjust values",
    "APPLY_BTN": "Apply pitch row and yaw slot changes to the view grid",
    "PREVIEW_SAMPLE": "Path of the equirectangular image shown in preview",
    "PREVIEW_BROWSE": "Manually select a preview image",
    "PREVIEW_AUTO": "Auto-select the first image in the scene folder",
    "PREVIEW_RELOAD": "Rescan the scene folder for images",
    "PREVIEW_SLIDER": "Slide through images in the scene sequentially",
    "MASK_PREVIEW_MODE_SINGLE": "Show the selected image large for detail checks and current-image regeneration",
    "MASK_PREVIEW_MODE_THUMBNAILS": "Show images with existing masks as a thumbnail list. Click or use arrow keys to choose the current image",
    "MASK_PREVIEW_BUTTON": "Build a temporary preview for the displayed image using the currently enabled mask steps. It is not saved to masks/",
    "MASK_PREVIEW_CLEAR_BUTTON": "Discard the temporary preview and return to the saved mask display",
    "MASK_PREVIEW_VISIBILITY_BUTTON": "Switch between the generated temporary preview and the saved display from masks/. ON shows the preview result; OFF shows the saved mask, or the image alone when no saved mask exists. The preview is kept",
    "YOLO_PREVIEW_BUTTON": "Run YOLO/SAM for the currently displayed image only and overlay the result. It is not saved to the mask folder",
    "MASK_REPROCESS_CURRENT_BUTTON": "In single preview, regenerate the current image. In thumbnails, regenerate selected images with the current settings and save to masks/. With SAM3.1 Add/Subtract mode, the current detections are applied to the existing mask.",
    "MASK_OVERLAY_TOGGLE": "Toggle the red mask overlay. When visible, opacity is fixed at 45%.",
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
    for env_name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        env_locale = os.environ.get(env_name, "").strip().lower()
        if env_locale.startswith(("ja", "japanese")):
            return "ja"
        if env_locale.startswith(("en", "english")):
            return "en"
    try:
        loc = locale.getlocale()[0] or ""
    except Exception:
        loc = ""
    loc = loc.lower()
    return "ja" if loc.startswith(("ja", "japanese")) else "en"


LANG = _detect_lang()
_table = _JA if LANG == "ja" else _EN
_tips = _TIPS_JA if LANG == "ja" else _TIPS_EN
_TOOLTIP_WRAP_WIDTH = 46 if LANG == "ja" else 78
_PATH_SEPARATORS = set("/\\")
_JA_FORBIDDEN_LINE_START = set("。、，．,.!?！？:：;；)]）】〕〉》」』”’/\\")


def _extend_path_separator_break(line: str, start: int, end: int) -> int:
    while start < end < len(line):
        if line[end] in _JA_FORBIDDEN_LINE_START:
            end += 1
            continue
        if line[end - 1] in _PATH_SEPARATORS:
            end += 1
            continue
        break
    return end


def _wrap_ja_tooltip_line(line: str) -> list[str]:
    if len(line) <= _TOOLTIP_WRAP_WIDTH:
        return [line]
    wrapped: list[str] = []
    start = 0
    while start < len(line):
        end = min(len(line), start + _TOOLTIP_WRAP_WIDTH)
        end = _extend_path_separator_break(line, start, end)
        wrapped.append(line[start:end])
        start = end
    return wrapped


def _wrap_tooltip(text: str) -> str:
    lines: list[str] = []
    for line in text.split("\n"):
        if not line or len(line) <= _TOOLTIP_WRAP_WIDTH:
            lines.append(line)
            continue
        if LANG == "ja":
            lines.extend(_wrap_ja_tooltip_line(line))
            continue
        lines.extend(
            textwrap.wrap(
                line,
                width=_TOOLTIP_WRAP_WIDTH,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


def t(key: str) -> str:
    """文字列キーからローカライズ済みテキストを取得。"""
    return _table.get(key, key)


def tip(key: str) -> str:
    """ツールチップキーからローカライズ済みテキストを取得。"""
    return _wrap_tooltip(_tips.get(key, ""))


# モジュール変数として全キーを公開 (既存コードとの互換性)
def _export_module_vars() -> None:
    g = globals()
    for key, value in _table.items():
        g[key] = value

_export_module_vars()
