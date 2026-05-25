# 保守向けアーキテクチャ

このリポジトリは、Windows向け統合GUIアプリとして保守します。再利用する実装は責務別の `core/` モジュールへ置き、アプリ内部の処理はGUIから型付きcoreジョブとして直接呼び出します。

## エントリポイント

root直下の互換 `*.py` ラッパーは、アプリ構造に含めません。新しいGUI処理をリポジトリ内Pythonスクリプトのコマンド文字列として追加しないでください。まず型付きpayloadとcore runnerを作ります。堅牢性のためにプロセス境界が必要な場合も、rootラッパーではなく `core` モジュールのエントリポイントを使います。

共通実装は `core/` に置きます。テストでは実装契約とGUIが作るジョブpayloadを確認します。公開CLI互換は、リリースノートで明示したラッパーを除き、GUIワークフローより優先しません。

## リリースに含める範囲

リリースZIPには、GUIランタイム、`core/`、`gui/`、ドキュメント、requirements、modelsメタデータ、セットアップ/検証用ファイルを含めます。`scripts/` 配下でエンドユーザー向けリリースに含めるのは、`setup_windows.bat` と `update_venv.bat` が使う `scripts/update_venv.py` と `scripts/check_venv.py` だけです。

その他の `scripts/*.py` は、開発者向けの診断ツールまたは薄いCLIアダプタとしてリポジトリには残してよいものとします。ただし、リリースZIPからは除外します。GUIランタイムは引き続き、`scripts/` のファイルを起動せず、型付きcore jobまたは `python -m core.<module>` のworker境界を使います。

## Core の契約

- `core/video_info.py` は、フレーム抽出と後続処理で共有する動画メタデータ dataclass を担当します。
- `core/frame_pair_analysis.py` は、ペアフレームのメトリクス、ブレ/追跡リスク閾値、依存チェック、選択解析を担当します。コマンドヘルプや無関係なテストで任意依存が必須にならないよう、OpenCV/numpy import はガードしてください。
- `core/extract_frames.py` は、FFmpeg/FFprobe による抽出、キャッシュ、静止/類似フレーム間引き、判定 CSV を担当します。Step 1 からは frame job runner 経由で typed `ExtractFramesOptions` を渡し、`run_extract_frames()` を呼び出します。`main()` は CLI 用アダプタに限定します。
- 静止画連番フォルダの取り込みも同じ frame job 契約を使います。`core/image_sequence_import_cli.py` は取り込み payload を作り、`core/frame_job_runner.py` に実行を委譲します。対応する `scripts/` 入口は開発者向けの薄いラッパーだけにし、リリースZIPからは除外します。
- `core/frame_renumbering.py` は、Step 2 の採用画像連番化契約を担当します。下流成果物によるブロック判定、衝突を避けるリネーム計画/適用、パス変更時のフレーム/ソース画像台帳更新をここに集約します。
- `core/apply_frame_decisions.py` は Step 2 の採用/除外適用実装を担当します。`core/apply_frame_decisions_cli.py` はCLI用アダプタだけにし、GUI/frame job からは `core/frame_job_runner.py` 経由で `apply_decisions()` を呼びます。
- `core/scene_inventory.py` は、シーン内画像、マスク、投影タイプ、ソースID、通常カメラ既定値を読み取る共有契約です。ソース単位で処理対象を限定する機能は、ファイル名推測ではなく `SceneInventory.source_groups()` を使って判断します。
- `core/scene_import*.py` は、外部シーンの再登録を担当します。取り込みはアプリ定義のシーンフォルダだけを走査し、外部取り込み由来メタデータを全量再登録として置き換え、実際の画像/マスク/出力アセットは削除しません。
- `core/app_job.py`、`core/workflow_job_runner.py`、`core/sfm_job_runner.py`、`core/frame_job_runner.py`、`core/dataset_job_runner.py` はGUI内部ジョブ実行を担当します。GUIステップは、アプリ内部のPython処理には `AppJob` を返し、raw command list は COLMAP、FFmpeg、SphereSfMバイナリ、学習アプリCLIなど外部実行ファイルに限定します。GPU負荷の大きいマスク生成は、モデルメモリやクラッシュをGUIプロセスから隔離するため `python -m core.<module>` で実行してよいものとします。
- `core/*_job_spec.py` は、バージョン付きjob payloadの生成と検証を担当します。必須フィールド、値の範囲、視点セットの構造は、jobファイルを書き出す前または実行前に検証します。これはフレーム抽出とレビュー確定にも、workflow/SfM/dataset jobsにも適用します。共通の検証ヘルパーは `core/job_payload_validation.py` に置きます。
- `core/workflow_job_cli.py` は、バージョン付き workflow job の開発/CLI用アダプタです。`core/workflow_job_runner.py` の薄いパーサ層として保ちます。
- `core/mask_job_spec.py` は Step 3 のマスクコマンドpayloadを担当します。GPU負荷の大きいマスク処理は引き続き `python -m core.<module>` の別プロセスで実行しますが、GUIのコマンド構築はまず検証済みmask payloadを作り、それをコマンドへ変換します。
- `core/yolo_mask.py` のYOLO/SAM実行設定は `YoloMaskRuntimeSettings` で正規化し、グローバル状態の更新は `apply_runtime_settings()` に集約します。既存の処理関数が参照する互換グローバルは残しますが、新しい設定追加はこの入口を通します。
- AprilTagスケール推定は `core/apriltag_scale_estimate.py` が実行実装、`core/apriltag_scale_job_spec.py` がGUIからのpayload検証とコマンド生成を担当します。推定はキャンセル可能な長時間処理なので別プロセスで実行してよいものとしますが、GUIから `scripts/` 配下を直接起動しません。
- SphereSfMのプロジェクト準備、GPU preflight、sparse model変換は `core/spheresfm_project.py`、`core/spheresfm_gpu_preflight.py`、`core/spheresfm_to_transforms.py` が担当します。コマンドライン用アダプタは対応する `core/*_cli.py` に置きます。対応する `scripts/` 配下のファイルは開発/CLI用の薄い入口であり、リリースZIPからは除外し、runtime実装を持たせません。
- COLMAP mixed project の作成は `core/colmap_mixed_project.py` が担当し、開発/CLI入口は `core/colmap_mixed_project_cli.py` に置きます。対応する `scripts/` 配下のファイルは開発者向けの薄いラッパーだけにし、GUIルートではバージョン付きSfM job payloadと `core/sfm_job_runner.py` を使います。
- マスク系モジュールでは、リポジトリ全体のマスク極性を守ります。白は使用可能ピクセル、黒は除外ピクセルです。マスク合成は、明示的に別操作として文書化しない限り AND 型を維持します。
- Metashape の座標変換は `core/metashape_coordinates.py` に集約します。Metashape XML のカメラ姿勢や PLY 点群を変換するルートでは、軸変換行列を個別実装せずこのモジュールを使います。投影/PINHOLEのMetashape NeRF/COLMAPデータセット作成は dataset job で直接実行します。Step 4 の Metashape 前処理ジョブは、RealityScan再アライン用出力とERP 360°直接出力のように中間のエクイレクタングラー `transforms.json` が必要なルートに限定します。GUI ルートは旧 upstream の Metashape converter に依存しません。
- Metashape 由来の NeRF/COLMAP データセット出力は、バージョン付き dataset job payload を使います。`core/metashape_dataset_cli.py` は開発/CLI用アダプタであり、直接CLI実行とGUIジョブ実行が同じ契約になるよう `core/dataset_job_runner.py` に実行を委譲します。
- RealityScan から LichtFeld 用 COLMAP への変換実装は `core/realityscan_to_lfs_colmap.py` に置きます。コマンドライン解析は `core/realityscan_to_lfs_colmap_cli.py` が担当し、GUI実行ではバージョン付き dataset job payload と `core/dataset_job_runner.py` を使います。
- RealityScan CSV/PLY から NeRF 系 `transforms.json` への変換実装は `core/realityscan_to_transforms.py` に置き、コマンドライン解析は `core/realityscan_to_transforms_cli.py` が担当します。
- Cubemapの視点セットとRemap仕様は `core/cubemap_view_spec.py` に集約します。デフォルトCube6、カスタム視点JSON、入力サイズ/FOV/出力サイズの検証はここを通し、画像変換実装側に個別の視点パースを増やしません。
- Cubemap のコマンドライン解析は `core/cubemap_transforms_json_cli.py` に置きます。`core/cubemap_transforms_json.py` はCLI入口と旧import向けの薄い互換facadeです。新しい orchestration では CLI 引数処理を複製せず、分割済み実装モジュールを直接 import するか workflow/dataset job payload を使います。
- COLMAP text 変換のコマンドライン解析は `core/transforms_to_colmap_cli.py` に置き、`core/transforms_to_colmap.py` は変換実装を担当します。
- キューブマップと COLMAP 出力では、座標プロファイルの意味を維持します。Postshot は標準キューブマップ変換、Brush は Brush 変換、LichtFeld のキューブマップ出力は Cubemap CLI で最終向き補正済みの `transforms.json` と `pointcloud.ply` を作ります。LichtFeld GUT向けERP 360°出力はキューブマップ化せずエクイレクタングラー入力を使い、直接データセット作成時に同じ最終向き補正を適用します。RealityScan再アライン用出力はStep 4のMetashapeルートで扱い、Metashapeインポート時の座標変換を相殺し、MetashapeのY-up姿勢をRealityScanのZ-up local Euclidean軸へ写してから `output/realityscan/` に cubemap 画像とXMPサイドカーを書き出します。RealityScan 側でアライン後に点群を再生成する前提なので、Metashape PLY は必須にせず渡しません。

## GUI の契約

GUI ステップは薄い orchestration 層にします。UI ラベル、ヒント、警告、ツールチップは `gui/i18n_ja.py` と `gui/i18n_en.py` に置き、日本語と英語の意味を揃えてください。

Step 3 のマスク生成は責務別に分割します。

- `gui/steps/step3_mask.py` はページレイアウト、コントロール、状態接続を担当します。
- `gui/steps/step3_mask_actions.py` はプレビュー、再生成、選択画像処理を担当します。
- `gui/steps/mask_commands.py` はコマンド構築を担当します。
- `gui/steps/mask_postprocess.py` は保存済みマスクの後処理を担当します。

SfMルート選択、データセット変換、学習起動は責務別に分割します。

- `gui/steps/sfm_step.py` は Step 4 のルートカードを担当します。
- `gui/steps/dataset_step.py` は Step 5 のデータセットツール一覧と、選択中ツールへの実行契約の委譲を担当します。
- `gui/steps/step4_contracts.py` は安定したルート/プロファイル/出力定数と診断判定を担当します。
- `gui/steps/step4_command_plan.py` は実行ファイル解決とコマンド計画を担当します。
- `gui/steps/step4_pipeline.py` はルート準備状態とサブ工程遷移を担当します。
- `gui/steps/step4_paths.py` は出力パス契約、cleanup/reset、成果物検証、sparse model 探索を担当します。
- `gui/steps/step4_manifest.py` は export settings と実行 manifest の保存を担当します。
- `gui/steps/step4_runtime.py` は実行完了処理、進捗解析、Metashape 入力検出、後続アクションを担当します。
- `gui/steps/step4_training.py` は学習バックエンド UI、設定、起動コマンドを担当します。
- `gui/steps/step4_widgets.py` は Step 4 で使う小さな再利用ウィジェットを担当します。
- `gui/steps/step4_cubemap.py` は Step 5 のデータセット作成ツールで使う mixin を合成する orchestration クラスです。新しいワークフロー実装は、明確に該当する責務モジュールがある場合はそこへ追加してください。

## 検証ルール

Python コードを変更したら Ruff を使います。

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

フォーマットは、リポジトリに既存の未整形差分が残っている場合、今回変更したファイルだけに絞ってください。

広いリファクタのコミット前にはフルテストを実行します。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

リリース公開前には、次のリリースゲートをすべて通します。

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\create_release_zip.py
```

リリースZIP作成コマンドに含まれるsetup preflight検証は、オーナーから明示的に指示されない限り無効化しません。

Codex や CI 風の offscreen 環境で PySide GUI を確認する場合は、`QT_QPA_PLATFORM=offscreen` を設定し、`MainWindow` を作る前に `apply_theme(app)` を呼びます。テーマ適用で Meiryo UI など日本語表示可能な Windows フォントが設定され、日本語ラベルが四角表示になる問題を避けられます。
