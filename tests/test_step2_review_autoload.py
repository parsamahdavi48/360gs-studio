import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit, QPushButton

from gui import i18n
from gui.app import MainWindow
from gui.common.preview_mode_toolbar import PREVIEW_MODE_SINGLE, PREVIEW_MODE_THUMBNAILS
from gui.steps.step2_review import ReviewStep


def _app():
    return QApplication.instance() or QApplication([])


def _write_scene(scene: Path, count: int = 2, drop_indices: set[int] | None = None) -> Path:
    drop_indices = drop_indices or set()
    images = scene / "images"
    images.mkdir(exist_ok=True)
    rows = []
    for seq in range(1, count + 1):
        pixmap = QPixmap(64, 32)
        pixmap.fill(Qt.red if seq % 2 else Qt.blue)
        image_rel = f"images/frame_{seq:06d}.png"
        assert pixmap.save(str(scene / image_rel))
        rows.append(
            {
                "seq": str(seq),
                "output_file": image_rel,
                "decision": "drop" if seq in drop_indices else "keep",
                "status": "ok",
                "timestamp_sec": str(seq),
                "blur_score_final": "100",
                "change_score_final": "0.1",
            }
        )

    csv_path = scene / "selected_frames.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_review_step_waits_for_csv_until_activated(tmp_path: Path) -> None:
    _app()
    step = ReviewStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))
    step.on_activated()

    assert step._review_widget is None
    assert step._loaded_csv_signature is None


def test_review_step_guides_scene_folder_when_unset() -> None:
    _app()
    step = ReviewStep(Path.cwd())

    labels = [label.text() for label in step.findChildren(QLabel)]

    assert i18n.t("REVIEW_EMBED_NO_SCENE") in labels
    assert not step.primary_action_enabled()


def test_review_step_left_pane_guides_apply_before_mask_step() -> None:
    _app()
    step = ReviewStep(Path.cwd())

    labels = [label.text() for label in step.findChildren(QLabel)]
    buttons = [button.text() for button in step.findChildren(QPushButton)]
    checkboxes = [checkbox.text() for checkbox in step.findChildren(QCheckBox)]

    assert all("確認+選別" not in text for text in labels)
    assert all("Review + Select" not in text for text in labels)
    assert all("CSVファイル名" not in text for text in labels)
    assert all("CSV Filename" not in text for text in labels)
    assert "再読み込み" not in buttons
    assert "Reload" not in buttons
    assert "別ウィンドウで開く" not in buttons
    assert "Open Separate Window" not in buttons
    assert not step.findChildren(QLineEdit)
    assert i18n.t("BACKUP_BEFORE_FINALIZE") in checkboxes
    assert i18n.NEXT_STEP_MASK_NOTICE in labels
    notice = next(label for label in step.findChildren(QLabel) if label.text() == i18n.NEXT_STEP_MASK_NOTICE)
    assert notice.alignment() & Qt.AlignLeft
    assert "変更した場合は、\n下部の" in i18n.NEXT_STEP_MASK_NOTICE
    assert i18n.t("ACTION_FINALIZE_REVIEW") in i18n.NEXT_STEP_MASK_NOTICE
    assert "Step 3" in i18n.NEXT_STEP_MASK_NOTICE


def test_review_step_uses_fixed_selected_frames_csv(tmp_path: Path) -> None:
    _app()
    step = ReviewStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))

    assert step._csv_path() == tmp_path / "selected_frames.csv"


def test_review_step_autoloads_csv_when_activated(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path)
    step = ReviewStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))
    assert step._review_widget is None

    step.on_activated()

    assert step._review_widget is not None
    assert step._loaded_csv_signature is not None
    assert not step.primary_action_enabled()


def test_review_step_apply_enabled_only_after_decision_change(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path)
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()

    assert step._review_widget is not None
    assert not step.primary_action_enabled()

    step._review_widget.toggle_decision()

    assert step.primary_action_enabled()

    step._review_widget.reset_decision()

    assert not step.primary_action_enabled()


@pytest.mark.parametrize("flag_journey", ["thumbnail_to_single", "single_to_thumbnail"])
def test_review_step_apply_uses_flag_changed_across_preview_modes(
    tmp_path: Path,
    monkeypatch,
    flag_journey: str,
) -> None:
    _app()
    csv_path = _write_scene(tmp_path, count=3)
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()
    widget = step._review_widget
    assert widget is not None

    if flag_journey == "thumbnail_to_single":
        widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)
        widget.thumbnail_view.setCurrentIndex(widget.thumbnail_model.index(1, 0))
        widget.toggle_decision()
        item = widget.thumbnail_model.item_at(1)
        assert item is not None
        assert item.cache_key[0] == "drop"
        widget.set_preview_mode(PREVIEW_MODE_SINGLE)
        assert widget.preview_mode() == PREVIEW_MODE_SINGLE
        assert not widget.flag_button.isChecked()
    else:
        widget.frame_slider.setValue(1)
        widget.toggle_decision()
        assert not widget.flag_button.isChecked()
        widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)
        item = widget.thumbnail_model.item_at(1)
        assert item is not None
        assert item.cache_key[0] == "drop"
        assert widget.preview_mode() == PREVIEW_MODE_THUMBNAILS

    assert widget.rows[1]["decision"] == "drop"
    assert _read_rows(csv_path)[1]["decision"] == "drop"
    assert step.primary_action_enabled()

    monkeypatch.setattr(step, "_confirm_finalize", lambda: True)
    commands = step.build_commands()
    assert len(commands) == 1
    label, cmd = commands[0]
    assert label == "finalize"
    assert cmd[:3] == [sys.executable, "-u", str(Path.cwd() / "apply_frame_decisions.py")]
    assert cmd[-2:] == [str(tmp_path), "--finalize-in-place"]

    result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in (tmp_path / "images").glob("*.png")) == [
        "frame_000001.png",
        "frame_000003.png",
    ]
    rows = _read_rows(csv_path)
    assert [row["output_file"] for row in rows] == [
        "images/frame_000001.png",
        "images/frame_000003.png",
    ]
    assert [row["decision"] for row in rows] == ["keep", "keep"]


def test_review_step_apply_enabled_when_initial_drop_image_exists(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path, drop_indices={1})
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()

    assert step._review_widget is not None
    assert step.primary_action_enabled()


def test_main_window_disables_apply_until_review_changes(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path)
    window = MainWindow(str(tmp_path))
    window._set_current_step(1)

    assert window.step2._review_widget is not None
    assert not window.run_btn.isEnabled()

    window.step2._review_widget.toggle_decision()

    assert window.run_btn.isEnabled()
    window.close()


def test_main_window_starts_without_scene_dir_and_requires_header_scene() -> None:
    _app()
    window = MainWindow()

    assert window.scene_browse.text() == ""
    assert window.scene_browse.line_edit.placeholderText() == i18n.t("SCENE_DIR_PLACEHOLDER")
    assert not window.run_btn.isEnabled()
    assert window.run_btn.toolTip() == i18n.t("SCENE_REQUIRED_ACTION_HINT")
    window.close()


def test_main_window_enables_apply_when_initial_drop_image_exists(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path, drop_indices={1})
    window = MainWindow(str(tmp_path))
    window._set_current_step(1)

    assert window.step2._review_widget is not None
    assert window.run_btn.isEnabled()
    window.close()


def test_review_step_skips_reload_when_csv_unchanged(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path)
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()
    first_widget = step._review_widget

    step.on_activated()

    assert step._review_widget is first_widget


def test_review_step_clear_pane_shuts_down_embedded_review() -> None:
    _app()
    step = ReviewStep(Path.cwd())
    label = QLabel("review")
    calls: list[bool] = []
    label.shutdown = lambda: calls.append(True)  # type: ignore[attr-defined]
    step.review_layout.addWidget(label)
    step._review_widget = label

    step._clear_review_pane()

    assert calls == [True]
    assert step._review_widget is None


def test_review_step_reloads_when_csv_changes(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path, count=2)
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()
    first_widget = step._review_widget

    _write_scene(tmp_path, count=3)
    step.on_activated()

    assert step._review_widget is not None
    assert step._review_widget is not first_widget
    assert len(step._review_widget.rows) == 3
