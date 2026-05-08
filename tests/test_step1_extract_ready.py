import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QToolButton

from extract_sessions import build_session_record, load_manifest, save_manifest
from gui import i18n
from gui.app import MainWindow
from gui.steps.step1_extract import ExtractStep
from scene_project import source_video_record, upsert_source_videos


def _app():
    return QApplication.instance() or QApplication([])


def _video_info() -> dict:
    return {
        "width": 7680,
        "height": 3840,
        "fps": 29.97,
        "duration_sec": 10.0,
        "total_frames": 300,
    }


def _make_ready(step: ExtractStep, video: Path, scene: Path) -> None:
    step.video_browse.line_edit.blockSignals(True)
    step.video_browse.set_text(str(video))
    step.video_browse.line_edit.blockSignals(False)
    step.set_scene_dir(str(scene))
    step.video_info = _video_info()
    step._update_ready_status()


def _write_session(scene: Path, video: Path, prefix: str = "input") -> None:
    manifest = load_manifest(scene)
    sessions = list(manifest.get("sessions", []))
    sessions.append(
        build_session_record(
            session_id=f"existing-session-{prefix}",
            input_video=video,
            video_info=_video_info(),
            mode="fixed",
            filename_prefix=prefix,
            image_ext="jpg",
            output_files=[f"images/{prefix}_0001.jpg"],
            selected_count=1,
            dropped_count=0,
        )
    )
    save_manifest(
        scene,
        {
            "version": 1,
            "sessions": sessions,
        },
    )


def _select_videos(step: ExtractStep, videos: list[Path], scene: Path) -> None:
    step.set_scene_dir(str(scene))
    step.video_browse.set_text("; ".join(str(video) for video in videos))
    step._update_ready_status()


def test_extract_run_disabled_until_video_is_selected() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert not step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_NO_VIDEO")
    assert step.primary_action_tooltip() == i18n.t("EXTRACT_READY_NO_VIDEO")


def test_extract_run_enabled_when_required_inputs_are_ready(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())

    _make_ready(step, video, tmp_path)

    assert step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_OK")
    assert step.primary_action_tooltip() == i18n.tip("RUN")


def test_extract_step_shows_standard_images_folder_when_scene_is_set(tmp_path: Path) -> None:
    _app()
    step = ExtractStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))

    labels = [label.text() for label in step.findChildren(QLabel)]
    assert str(tmp_path / "images") in labels

    step.set_scene_dir("")

    assert step.images_path_label.text() == "-"


def test_extract_step_autoloads_registered_source_videos_when_scene_is_set(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path / "scene"
    source = tmp_path / "source"
    scene.mkdir()
    source.mkdir()
    video = source / "registered.mp4"
    video.write_bytes(b"dummy")
    upsert_source_videos(scene, [source_video_record(video, _video_info())])
    step = ExtractStep(Path.cwd())
    monkeypatch.setattr(step, "_probe_video_info_for_path", lambda _path: _video_info())

    step.set_scene_dir(str(scene))

    assert step.video_browse.text() == str(video)
    assert step.video_info == _video_info()


def test_extract_step_autoloads_scene_videos_when_no_registry_exists(tmp_path: Path, monkeypatch) -> None:
    _app()
    video = tmp_path / "camera.MP4"
    ignored = tmp_path / "images" / "render.mov"
    video.write_bytes(b"dummy")
    ignored.parent.mkdir()
    ignored.write_bytes(b"generated")
    step = ExtractStep(Path.cwd())
    monkeypatch.setattr(step, "_probe_video_info_for_path", lambda _path: _video_info())

    step.set_scene_dir(str(tmp_path))

    assert step.video_browse.text() == str(video)
    assert step.video_info == _video_info()


def test_extract_step_does_not_autoload_over_existing_video_selection(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path / "scene"
    source = tmp_path / "source"
    scene.mkdir()
    source.mkdir()
    selected = source / "selected.mp4"
    candidate = scene / "candidate.mp4"
    selected.write_bytes(b"selected")
    candidate.write_bytes(b"candidate")
    step = ExtractStep(Path.cwd())
    monkeypatch.setattr(step, "_probe_video_info_for_path", lambda _path: _video_info())

    step.video_browse.set_text(str(selected))
    step.set_scene_dir(str(scene))

    assert step.video_browse.text() == str(selected)


def test_extract_run_disabled_for_invalid_analysis_width(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)

    step.analysis_width_edit.setText("wide")

    assert not step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_BAD_ANALYSIS_WIDTH")


def test_extract_quick_mode_ignores_invalid_analysis_width(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)

    step.analysis_width_edit.setText("wide")
    step.quick_extract_cb.setChecked(True)

    assert step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_OK")


def test_main_window_run_button_follows_extract_readiness(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    window = MainWindow(str(tmp_path))

    assert window.stack.currentIndex() == 0
    assert not window.run_btn.isEnabled()

    _make_ready(window.step1, video, tmp_path)

    assert window.run_btn.isEnabled()
    window.close()


def test_main_window_auto_sets_scene_from_single_video_when_scene_is_empty(tmp_path: Path, monkeypatch) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    window = MainWindow()
    monkeypatch.setattr(window.step1, "_probe_video_info_for_path", lambda _path: _video_info())

    window.step1.video_browse.set_text(str(video))

    assert window.scene_browse.text() == str(tmp_path)
    assert window.step1.scene_dir == str(tmp_path)
    window.close()


def test_main_window_auto_sets_scene_from_multiple_videos_in_same_folder(tmp_path: Path, monkeypatch) -> None:
    _app()
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mov"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    window = MainWindow()
    monkeypatch.setattr(window.step1, "_probe_video_info_for_path", lambda _path: _video_info())

    window.step1.video_browse.set_text(f"{video_a}; {video_b}")

    assert window.scene_browse.text() == str(tmp_path)
    assert window.step1.scene_dir == str(tmp_path)
    window.close()


def test_main_window_does_not_auto_set_scene_for_videos_from_different_folders(tmp_path: Path, monkeypatch) -> None:
    _app()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    video_a = first / "a.mp4"
    video_b = second / "b.mov"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    window = MainWindow()
    monkeypatch.setattr(window.step1, "_probe_video_info_for_path", lambda _path: _video_info())

    window.step1.video_browse.set_text(f"{video_a}; {video_b}")

    assert window.scene_browse.text() == ""
    assert window.step1.scene_dir == ""
    window.close()


def test_main_window_does_not_overwrite_existing_scene_from_video_selection(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path / "scene"
    source = tmp_path / "source"
    scene.mkdir()
    source.mkdir()
    video = source / "input.mp4"
    video.write_bytes(b"dummy")
    window = MainWindow(str(scene))
    monkeypatch.setattr(window.step1, "_probe_video_info_for_path", lambda _path: _video_info())

    window.step1.video_browse.set_text(str(video))

    assert window.scene_browse.text() == str(scene)
    assert window.step1.scene_dir == str(scene)
    window.close()


def test_main_window_clear_scene_button_clears_header_scene(tmp_path: Path) -> None:
    _app()
    window = MainWindow(str(tmp_path))

    assert window.scene_browse.text() == str(tmp_path)
    assert isinstance(window.scene_browse.browse_button, QToolButton)
    assert window.scene_browse.browse_button.objectName() == "iconToolButton"
    assert window.scene_browse.browse_button.text() == ""
    assert not window.scene_browse.browse_button.icon().isNull()
    assert window.scene_browse.browse_button.accessibleName() == i18n.BROWSE
    assert window.clear_scene_btn.isEnabled()
    assert isinstance(window.clear_scene_btn, QToolButton)
    assert window.clear_scene_btn.text() == ""
    assert window.clear_scene_btn.accessibleName() == i18n.t("CLEAR_SCENE_DIR")
    assert window.clear_scene_btn.toolTip() == i18n.t("CLEAR_SCENE_DIR_HINT")

    window.clear_scene_btn.click()

    assert window.scene_browse.text() == ""
    assert window.step1.scene_dir == ""
    assert not window.clear_scene_btn.isEnabled()
    window.close()


def test_extract_clear_input_videos_clears_auto_scene(tmp_path: Path, monkeypatch) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    window = MainWindow()
    monkeypatch.setattr(window.step1, "_probe_video_info_for_path", lambda _path: _video_info())

    window.step1.video_browse.set_text(str(video))
    assert window.scene_browse.text() == str(tmp_path)
    assert window.step1.video_info is not None

    window.step1.clear_video_btn.click()

    assert window.step1.video_browse.text() == ""
    assert window.step1.video_info is None
    assert window.scene_browse.text() == ""
    assert window.step1.scene_dir == ""
    assert isinstance(window.step1.clear_video_btn, QToolButton)
    assert window.step1.clear_video_btn.text() == ""
    assert window.step1.clear_video_btn.accessibleName() == i18n.t("CLEAR_INPUT_VIDEO")
    assert window.step1.clear_video_btn.toolTip() == i18n.t("CLEAR_INPUT_VIDEO_HINT")
    window.close()


def test_extract_clear_input_videos_keeps_manual_scene(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene = tmp_path / "scene"
    source = tmp_path / "source"
    scene.mkdir()
    source.mkdir()
    video = source / "input.mp4"
    video.write_bytes(b"dummy")
    window = MainWindow(str(scene))
    monkeypatch.setattr(window.step1, "_probe_video_info_for_path", lambda _path: _video_info())

    window.step1.video_browse.set_text(str(video))
    window.step1.clear_video_btn.click()

    assert window.step1.video_browse.text() == ""
    assert window.scene_browse.text() == str(scene)
    assert window.step1.scene_dir == str(scene)
    window.close()


def test_extract_run_disabled_when_same_video_would_be_appended(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    _write_session(tmp_path, video)
    step = ExtractStep(Path.cwd())

    _make_ready(step, video, tmp_path)

    assert step._extract_output_mode() == "append"
    assert not step.primary_action_enabled()
    assert i18n.t("EXTRACT_READY_DUPLICATE_VIDEO").split("{n}")[0] in step.ready_status_label.text()


def test_extract_replace_same_video_enables_run_and_sets_cli_mode(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    _write_session(tmp_path, video)
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)

    step.output_mode_combo.setCurrentIndex(1)
    cmd = step._build_extract_cmd()

    assert step.primary_action_enabled()
    assert cmd[cmd.index("--output-mode") + 1] == "replace-video"
    assert "--allow-duplicate-video" not in cmd


def test_extract_multi_select_queues_only_unextracted_videos(tmp_path: Path) -> None:
    _app()
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.MOV"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    _write_session(tmp_path, video_a, prefix="a")
    step = ExtractStep(Path.cwd())

    _select_videos(step, [video_a, video_b], tmp_path)
    commands = step.build_commands()

    assert step.primary_action_enabled()
    assert len(commands) == 1
    assert commands[0][0] == "extract: b.MOV"
    cmd = commands[0][1]
    assert cmd[3] == str(video_b)
    assert cmd[cmd.index("--output-mode") + 1] == "append"
    assert cmd[cmd.index("--filename-prefix") + 1] == "b"


def test_extract_multi_select_build_commands_rechecks_missing_videos(tmp_path: Path) -> None:
    _app()
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mov"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    step = ExtractStep(Path.cwd())

    _select_videos(step, [video_a, video_b], tmp_path)
    video_b.unlink()

    try:
        step.build_commands()
    except ValueError as exc:
        assert i18n.t("EXTRACT_READY_VIDEO_NOT_FOUND") in str(exc)
        assert str(video_b) in str(exc)
    else:
        raise AssertionError("missing video should stop command construction")


def test_extract_multi_select_disables_when_all_videos_are_already_extracted(tmp_path: Path) -> None:
    _app()
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mov"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    _write_session(tmp_path, video_a, prefix="a")
    _write_session(tmp_path, video_b, prefix="b")
    step = ExtractStep(Path.cwd())

    _select_videos(step, [video_a, video_b], tmp_path)

    assert not step.primary_action_enabled()
    assert i18n.t("EXTRACT_READY_QUEUE_ALL_DUPLICATE").split("{n}")[0] in step.ready_status_label.text()


def test_extract_multi_select_replace_mode_queues_all_videos(tmp_path: Path) -> None:
    _app()
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mov"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    _write_session(tmp_path, video_a, prefix="a")
    step = ExtractStep(Path.cwd())

    _select_videos(step, [video_a, video_b], tmp_path)
    step.output_mode_combo.setCurrentIndex(1)
    commands = step.build_commands()

    assert step.primary_action_enabled()
    assert [phase for phase, _cmd in commands] == ["extract: a.mp4", "extract: b.mov"]
    assert [cmd[cmd.index("--output-mode") + 1] for _phase, cmd in commands] == ["replace-video", "replace-video"]


def test_extract_output_mode_has_only_add_and_reextract() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert step.output_mode_combo.count() == 2
    assert step.output_mode_combo.itemText(0) == i18n.t("EXTRACT_OUTPUT_APPEND")
    assert step.output_mode_combo.itemText(1) == i18n.t("EXTRACT_OUTPUT_REPLACE_VIDEO")
    assert step.output_mode_combo.itemData(0) == "append"
    assert step.output_mode_combo.itemData(1) == "replace-video"
    assert step.output_mode_combo.maximumWidth() == 180 or step.output_mode_combo.width() <= 180


def test_extract_video_info_status_follows_output_mode(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    _write_session(tmp_path, video)
    step = ExtractStep(Path.cwd())

    _make_ready(step, video, tmp_path)
    step._update_video_info_label()

    assert step.output_mode_combo.currentData() == "append"
    assert step.video_info_label.text().startswith(i18n.t("VIDEO_QUEUE_STATUS_SKIP"))

    step.output_mode_combo.setCurrentIndex(1)

    assert step.video_info_label.text().startswith(i18n.t("VIDEO_QUEUE_STATUS_REEXTRACT"))


def test_extract_queue_finish_refreshes_append_state_after_success(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)

    assert step.primary_action_enabled()

    _write_session(tmp_path, video)
    step.on_queue_finished(True)

    assert not step.primary_action_enabled()
    assert i18n.t("EXTRACT_READY_DUPLICATE_VIDEO").split("{n}")[0] in step.ready_status_label.text()
    assert step.video_info_label.text().startswith(i18n.t("VIDEO_QUEUE_STATUS_SKIP"))


def test_extract_queue_finish_revalidates_missing_video_after_failure(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)

    video.unlink()
    step.on_queue_finished(False)

    assert step.video_info is None
    assert not step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_VIDEO_NOT_FOUND")


def test_extract_queue_finish_revalidates_unreadable_video_after_failure(tmp_path: Path, monkeypatch) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)
    monkeypatch.setattr(step, "_probe_video_info_for_path", lambda _path: (_ for _ in ()).throw(RuntimeError("bad video")))

    step.on_queue_finished(False)

    assert step.video_info is None
    assert not step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_NO_VIDEO_INFO")


def test_extract_video_info_button_is_removed() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    buttons = [button.text().strip() for button in step.findChildren(QPushButton)]

    assert "動画情報" not in buttons
    assert "Video Info" not in buttons


def test_extract_single_video_shows_fast_fixed_interval_estimate() -> None:
    _app()
    step = ExtractStep(Path.cwd())
    step.video_info = _video_info()

    step._update_video_info_label()
    step._update_instant_estimate()

    assert step.video_info_label.text() == i18n.t("VIDEO_INFO_SINGLE_FORMAT").format(
        status=i18n.t("VIDEO_QUEUE_STATUS_NEW"),
        projection=i18n.t("VIDEO_PROJECTION_EQUIRECT"),
        width=7680,
        height=3840,
        fps=29.97,
        duration="00:00:10",
        frames="300",
    )
    assert step.instant_estimate_text == (
        i18n.t("FIXED_INTERVAL_ESTIMATE_FORMAT").format(interval="1", count="11")
        + f" ({i18n.t('FIXED_SMART_ESTIMATE')})"
    )


def test_extract_quick_mode_estimate_uses_quick_suffix() -> None:
    _app()
    step = ExtractStep(Path.cwd())
    step.video_info = _video_info()

    step.quick_extract_cb.setChecked(True)
    step._update_instant_estimate()

    assert step.instant_estimate_text == (
        i18n.t("FIXED_INTERVAL_ESTIMATE_FORMAT").format(interval="1", count="11")
        + f" ({i18n.t('QUICK_EXTRACT_ESTIMATE')})"
    )


def test_extract_multi_video_auto_probes_and_shows_total_estimate(tmp_path: Path, monkeypatch) -> None:
    _app()
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mov"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    step = ExtractStep(Path.cwd())

    def fake_probe(video: Path) -> dict:
        if video == video_a:
            return _video_info()
        return {
            "width": 3840,
            "height": 2160,
            "fps": 30.0,
            "duration_sec": 20.0,
            "total_frames": 600,
        }

    monkeypatch.setattr(step, "_probe_video_info_for_path", fake_probe)

    step.set_scene_dir(str(tmp_path))
    step.video_browse.set_text(f"{video_a}; {video_b}")

    assert step.video_info_label.text() == "\n".join(
        [
            i18n.t("VIDEO_INFO_MULTI_HEADER_FORMAT").format(total=2, queued=2, skipped=0, probed=2),
            i18n.t("VIDEO_INFO_MULTI_ITEM_FORMAT").format(
                name="a.mp4",
                status=i18n.t("VIDEO_QUEUE_STATUS_NEW"),
                projection=i18n.t("VIDEO_PROJECTION_EQUIRECT"),
                width=7680,
                height=3840,
                fps=29.97,
                duration="00:00:10",
                frames="300",
            ),
            i18n.t("VIDEO_INFO_MULTI_ITEM_FORMAT").format(
                name="b.mov",
                status=i18n.t("VIDEO_QUEUE_STATUS_NEW"),
                projection=i18n.t("VIDEO_PROJECTION_NORMAL"),
                width=3840,
                height=2160,
                fps=30.0,
                duration="00:00:20",
                frames="600",
            ),
        ]
    )
    assert step.instant_estimate_text == "\n".join(
        [
            i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_HEADER_FORMAT").format(interval="1"),
            i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_ITEM_FORMAT").format(name="a.mp4", count="11"),
            i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_ITEM_FORMAT").format(name="b.mov", count="21"),
            i18n.t("FIXED_INTERVAL_ESTIMATE_MULTI_TOTAL_FORMAT").format(count="32", videos=2)
            + f" ({i18n.t('FIXED_SMART_ESTIMATE')})",
        ]
    )


def test_extract_ffprobe_path_change_reloads_video_info(tmp_path: Path, monkeypatch) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    calls: list[Path] = []

    def fake_probe(path: Path) -> dict:
        calls.append(path)
        return _video_info()

    monkeypatch.setattr(step, "_probe_video_info_for_path", fake_probe)

    step.set_scene_dir(str(tmp_path))
    step.video_browse.set_text(str(video))
    step.ffprobe_browse.set_text("custom-ffprobe")

    assert calls == [video, video]
