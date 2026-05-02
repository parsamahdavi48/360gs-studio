import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from extract_sessions import build_session_record, load_manifest, save_manifest
from gui import i18n
from gui.app import MainWindow
from gui.steps.step1_extract import ExtractStep


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


def test_extract_run_disabled_for_invalid_analysis_width(tmp_path: Path) -> None:
    _app()
    video = tmp_path / "input.mp4"
    video.write_bytes(b"dummy")
    step = ExtractStep(Path.cwd())
    _make_ready(step, video, tmp_path)

    step.analysis_width_edit.setText("wide")

    assert not step.primary_action_enabled()
    assert step.ready_status_label.text() == i18n.t("EXTRACT_READY_BAD_ANALYSIS_WIDTH")


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


def test_extract_output_mode_has_only_add_and_overwrite() -> None:
    _app()
    step = ExtractStep(Path.cwd())

    assert step.output_mode_combo.count() == 2
    assert step.output_mode_combo.itemData(0) == "append"
    assert step.output_mode_combo.itemData(1) == "replace-video"
    assert step.output_mode_combo.maximumWidth() == 180 or step.output_mode_combo.width() <= 180
