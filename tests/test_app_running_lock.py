import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.scene_import import SceneImportResult
from gui import i18n
from gui.app import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def _process_events_until(app: QApplication, predicate, timeout_sec: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def test_main_window_locks_workflow_controls_while_process_runs(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(str(tmp_path))

    try:
        assert window.scene_browse.isEnabled()
        assert window.clear_scene_btn.isEnabled()
        assert window.stack.isEnabled()
        assert all(btn.isEnabled() for btn in window.step_buttons)

        window.runner.start_queue([("sleep", [sys.executable, "-c", "import time; time.sleep(0.2)"])])
        app.processEvents()

        assert window.runner.is_running()
        assert window.cancel_btn.isEnabled()
        assert not window.scene_browse.isEnabled()
        assert not window.clear_scene_btn.isEnabled()
        assert not window.scene_menu_btn.isEnabled()
        assert not window.open_scene_action.isEnabled()
        assert not window.clear_scene_action.isEnabled()
        assert not window.import_scene_action.isEnabled()
        assert not window.stack.isEnabled()
        assert all(not btn.isEnabled() for btn in window.step_buttons)

        assert _process_events_until(app, lambda: not window.runner.is_running())

        assert window.scene_browse.isEnabled()
        assert window.clear_scene_btn.isEnabled()
        assert window.scene_menu_btn.isEnabled()
        assert window.open_scene_action.isEnabled()
        assert window.clear_scene_action.isEnabled()
        assert window.import_scene_action.isEnabled()
        assert window.stack.isEnabled()
        assert all(btn.isEnabled() for btn in window.step_buttons)
    finally:
        window.shutdown()
        if window.runner.is_running():
            window.runner.cancel()
            _process_events_until(app, lambda: not window.runner.is_running())


def test_scene_import_runs_off_gui_thread_and_reports_start(tmp_path: Path, monkeypatch) -> None:
    app = _app()
    window = MainWindow(str(tmp_path))

    def fake_import(scene: Path, **_kwargs) -> SceneImportResult:
        time.sleep(0.2)
        return SceneImportResult(
            scene_dir=scene,
            import_id="import_test",
            status="ok",
            image_count=1,
            mask_count=0,
            output_image_count=0,
            output_mask_count=0,
            output_shape="",
            dataset_kind="",
            warnings=(),
            errors=(),
            backup_dir=None,
            report_path=scene / "_stechdrive" / "imports" / "scene_imports.json",
            selected_frames_csv=None,
            export_settings_json=scene / "_stechdrive" / "step4" / "export_settings.json",
        )

    monkeypatch.setattr("gui.app.import_scene", fake_import)

    try:
        started_at = time.monotonic()
        window._start_scene_import(str(tmp_path))
        assert time.monotonic() - started_at < 0.1
        app.processEvents()

        assert window._scene_import_running
        assert not window.scene_browse.isEnabled()
        assert not window.scene_menu_btn.isEnabled()
        assert not window.open_scene_action.isEnabled()
        assert not window.clear_scene_action.isEnabled()
        assert not window.import_scene_action.isEnabled()
        assert not window.stack.isEnabled()
        assert window.cancel_btn.isEnabled()
        assert i18n.t("IMPORT_SCENE_STARTED").format(scene=str(tmp_path)) in window.log_panel.toPlainText()
        assert window.progress.status_label.text() == i18n.t("IMPORT_SCENE_RUNNING")

        assert _process_events_until(app, lambda: not window._scene_import_running)

        assert window.scene_browse.isEnabled()
        assert window.scene_menu_btn.isEnabled()
        assert window.open_scene_action.isEnabled()
        assert window.clear_scene_action.isEnabled()
        assert window.import_scene_action.isEnabled()
        assert window.stack.isEnabled()
        assert "Scene import:" in window.log_panel.toPlainText()
        assert window.progress.status_label.text() == i18n.t("IMPORT_SCENE_DONE").format(
            images=1,
            masks=0,
            output_images=0,
        )
    finally:
        window.shutdown()


def test_scene_import_finish_defers_current_step_refresh(tmp_path: Path, monkeypatch) -> None:
    app = _app()
    scene = tmp_path / "scene"
    scene.mkdir()
    window = MainWindow("")

    result = SceneImportResult(
        scene_dir=scene,
        import_id="import_test",
        status="ok",
        image_count=1,
        mask_count=0,
        output_image_count=0,
        output_mask_count=0,
        output_shape="",
        dataset_kind="",
        warnings=(),
        errors=(),
        backup_dir=None,
        report_path=scene / "_stechdrive" / "imports" / "scene_imports.json",
        selected_frames_csv=None,
        export_settings_json=scene / "_stechdrive" / "step4" / "export_settings.json",
    )

    calls = {"set_scene_dir": 0, "activated": 0}

    def count_set_scene_dir(_path: str) -> None:
        calls["set_scene_dir"] += 1

    def count_activated() -> None:
        calls["activated"] += 1

    try:
        window._set_current_step(2)
        monkeypatch.setattr(window.step3, "set_scene_dir", count_set_scene_dir)
        monkeypatch.setattr(window.step3, "on_activated", count_activated)

        window._on_scene_import_finished(result, "", False)
        app.processEvents()

        assert window.scene_browse.text() == str(scene)
        assert calls == {"set_scene_dir": 0, "activated": 0}
        assert window._step_scene_sync_deferred(window.step3)
        assert not window.run_btn.isEnabled()
        assert i18n.t("IMPORT_SCENE_DEFERRED_REFRESH") in window.log_panel.toPlainText()

        window._set_current_step(2)

        assert calls == {"set_scene_dir": 1, "activated": 1}
        assert not window._step_scene_sync_deferred(window.step3)
    finally:
        window.shutdown()


def test_scene_import_cancel_requests_worker_and_keeps_metadata_unchanged(tmp_path: Path, monkeypatch) -> None:
    app = _app()
    window = MainWindow(str(tmp_path))

    def fake_import(scene: Path, **kwargs) -> SceneImportResult:
        progress_callback = kwargs["progress_callback"]
        cancel_token = kwargs["cancel_token"]
        progress_callback("fake scan started")
        while not cancel_token.is_cancelled():
            time.sleep(0.01)
        cancel_token.check_cancelled()
        raise AssertionError("unreachable")

    monkeypatch.setattr("gui.app.import_scene", fake_import)

    try:
        window._start_scene_import(str(tmp_path))
        app.processEvents()
        assert _process_events_until(app, lambda: "fake scan started" in window.log_panel.toPlainText())
        assert window._scene_import_running
        assert window.cancel_btn.isEnabled()

        window._on_cancel()

        assert i18n.t("IMPORT_SCENE_CANCELING") in window.log_panel.toPlainText()
        assert _process_events_until(app, lambda: not window._scene_import_running)
        assert window.progress.status_label.text() == i18n.STATUS_CANCELED
        assert i18n.t("IMPORT_SCENE_CANCELED") in window.log_panel.toPlainText()
        assert not (tmp_path / "_stechdrive" / "imports" / "scene_imports.json").exists()
    finally:
        window.shutdown()
