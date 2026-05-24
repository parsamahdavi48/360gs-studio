# 保守向けアーキテクチャ

このリポジトリは、Windows向け統合GUIアプリとして保守します。再利用する実装は責務別の `core/` モジュールへ置き、アプリ内部の処理はGUIから型付きcoreジョブとして直接呼び出します。

## エントリポイント

root直下の互換 `*.py` ラッパーは、アプリ構造に含めません。新しいGUI処理をリポジトリ内Pythonスクリプトのコマンド文字列として追加しないでください。まず型付きpayloadとcore runnerを作ります。堅牢性のためにプロセス境界が必要な場合も、rootラッパーではなく `core` モジュールのエントリポイントを使います。

共通実装は `core/` に置きます。テストでは実装契約とGUIが作るジョブpayloadを確認します。公開CLI互換は、リリースノートで明示したラッパーを除き、GUIワークフローより優先しません。

## Core の契約

- `core/video_info.py` は、フレーム抽出と後続処理で共有する動画メタデータ dataclass を担当します。
- `core/frame_pair_analysis.py` は、ペアフレームのメトリクス、ブレ/追跡リスク閾値、依存チェック、選択解析を担当します。コマンドヘルプや無関係なテストで任意依存が必須にならないよう、OpenCV/numpy import はガードしてください。
- `core/extract_frames.py` は、FFmpeg/FFprobe による抽出、キャッシュ、静止/類似フレーム間引き、判定 CSV を担当します。Step 1 からは frame job runner 経由で呼び出します。
- `core/frame_renumbering.py` は、Step 2 の採用画像連番化契約を担当します。下流成果物によるブロック判定、衝突を避けるリネーム計画/適用、パス変更時のフレーム/ソース画像台帳更新をここに集約します。
- `core/scene_import*.py` は、外部シーンの再登録を担当します。取り込みはアプリ定義のシーンフォルダだけを走査し、外部取り込み由来メタデータを全量再登録として置き換え、実際の画像/マスク/出力アセットは削除しません。
- `core/app_job.py`、`core/workflow_job_runner.py`、`core/sfm_job_runner.py`、`core/frame_job_runner.py`、`core/dataset_job_runner.py` はGUI内部ジョブ実行を担当します。GUIステップは、アプリ内部のPython処理には `AppJob` を返し、raw command list は COLMAP、FFmpeg、SphereSfMバイナリ、学習アプリCLIなど外部実行ファイルに限定します。GPU負荷の大きいマスク生成は、モデルメモリやクラッシュをGUIプロセスから隔離するため `python -m core.<module>` で実行してよいものとします。
- `core/*_job_spec.py` は、バージョン付きjob payloadの生成と検証を担当します。必須フィールド、値の範囲、視点セットの構造は、jobファイルを書き出す前または実行前に検証します。共通の検証ヘルパーは `core/job_payload_validation.py` に置きます。
- SphereSfMのプロジェクト準備、GPU preflight、sparse model変換は `core/spheresfm_project.py`、`core/spheresfm_gpu_preflight.py`、`core/spheresfm_to_transforms.py` が担当します。対応する `scripts/` 配下のファイルは開発/CLI用の薄い入口であり、runtime実装を持たせません。
- マスク系モジュールでは、リポジトリ全体のマスク極性を守ります。白は使用可能ピクセル、黒は除外ピクセルです。マスク合成は、明示的に別操作として文書化しない限り AND 型を維持します。
- Metashape の座標変換は `core/metashape_coordinates.py` に集約します。Metashape XML のカメラ姿勢や PLY 点群を変換するルートでは、軸変換行列を個別実装せずこのモジュールを使います。Step 4 の Metashape 前処理ジョブは `core/metashape_preprocess.py` で中間のエクイレクタングラー `transforms.json` を作成します。GUI ルートは旧 upstream の Metashape converter に依存しません。
- キューブマップと COLMAP 出力では、座標プロファイルの意味を維持します。Postshot は標準キューブマップ変換、Brush は Brush 変換、LichtFeld のキューブマップ出力は Cubemap CLI で最終向き補正済みの `transforms.json` と `pointcloud.ply` を作ります。LichtFeld GUT向けERP 360°出力はキューブマップ化せずエクイレクタングラー入力を使い、直接データセット作成時に同じ最終向き補正を適用します。RealityScan再アライン用出力はStep 4のMetashapeルートで扱い、Metashapeインポート時の座標変換を相殺し、MetashapeのY-up姿勢をRealityScanのZ-up local Euclidean軸へ写してから `output/realityscan/` に cubemap 画像とXMPサイドカーを書き出します。RealityScan 側でアライン後に点群を再生成する前提なので、Metashape PLY は必須にせず渡しません。

## GUI の契約

GUI ステップは薄い orchestration 層にします。UI ラベル、ヒント、警告、ツールチップは `gui/i18n_ja.py` と `gui/i18n_en.py` に置き、日本語と英語の意味を揃えてください。

Step 3 のマスク生成は責務別に分割します。

- `gui/steps/step3_mask.py` はページレイアウト、コントロール、状態接続を担当します。
- `gui/steps/step3_mask_actions.py` はプレビュー、再生成、選択画像処理を担当します。
- `gui/mask/mask_commands.py` はコマンド構築を担当します。
- `gui/mask/mask_postprocess.py` は保存済みマスクの後処理を担当します。

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
