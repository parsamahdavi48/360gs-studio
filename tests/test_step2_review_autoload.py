import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QPushButton, QSizePolicy

from core.app_job import AppJob, run_app_job
from core.review_blur_sensitivity import (
    BLUR_REVIEW_MODE_FIELD,
    BLUR_REVIEW_MODE_LOW,
    BLUR_REVIEW_MODE_STANDARD,
    REVIEW_DECISION_OVERRIDE_FIELD,
)
from core.scene_layout import selected_frames_path
from gui import i18n
from gui.app import MainWindow
from gui.common.preview_mode_toolbar import PREVIEW_MODE_PERSPECTIVE, PREVIEW_MODE_SINGLE, PREVIEW_MODE_THUMBNAILS
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

    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _write_blur_scene(scene: Path) -> Path:
    images = scene / "images"
    images.mkdir(exist_ok=True)
    rows = []
    specs = [
        ("0.55", "motion_blur", "drop", "motion_blur"),
        ("0.75", "borderline_blur", "keep", "fixed_interval"),
        ("0.90", "ok", "keep", "fixed_interval"),
    ]
    for seq, (ratio, status, decision, reason) in enumerate(specs, start=1):
        pixmap = QPixmap(64, 32)
        pixmap.fill(Qt.red if seq % 2 else Qt.blue)
        image_rel = f"images/frame_{seq:06d}.png"
        assert pixmap.save(str(scene / image_rel))
        rows.append(
            {
                "seq": str(seq),
                "output_file": image_rel,
                "decision": decision,
                "status": status,
                "analysis_pipeline": "pair",
                "selection_reason": reason,
                "review_required": "1" if status != "ok" else "0",
                "risk_flags": status if "blur" in status else "",
                "timestamp_sec": str(seq),
                "blur_score_final": "55",
                "sharpness_baseline": "100",
                "sharpness_ratio": ratio,
                "change_score_final": "0.1",
            }
        )

    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _write_mixed_projection_scene(scene: Path) -> Path:
    images = scene / "images"
    images.mkdir(exist_ok=True)
    specs = [("equirect.png", 128, 64), ("normal.png", 64, 48)]
    rows = []
    for seq, (name, width, height) in enumerate(specs, start=1):
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.red if seq == 1 else Qt.blue)
        image_rel = f"images/{name}"
        assert pixmap.save(str(scene / image_rel))
        rows.append(
            {
                "seq": str(seq),
                "output_file": image_rel,
                "decision": "keep",
                "status": "ok",
                "timestamp_sec": str(seq),
                "blur_score_final": "100",
                "change_score_final": "0.1",
            }
        )

    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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

    assert all("確認+選別" not in text for text in labels)
    assert all("Review + Select" not in text for text in labels)
    assert all("CSVファイル名" not in text for text in labels)
    assert all("CSV Filename" not in text for text in labels)
    assert "再読み込み" not in buttons
    assert "Reload" not in buttons
    assert "別ウィンドウで開く" not in buttons
    assert "Open Separate Window" not in buttons
    assert not step.findChildren(QLineEdit)
    assert not step.findChildren(QComboBox)[0].isEnabled()
    assert i18n.NEXT_STEP_MASK_NOTICE in labels
    notice = next(label for label in step.findChildren(QLabel) if label.text() == i18n.NEXT_STEP_MASK_NOTICE)
    assert notice.alignment() & Qt.AlignLeft
    assert "\n" in i18n.NEXT_STEP_MASK_NOTICE
    assert i18n.t("ACTION_FINALIZE_REVIEW") in i18n.NEXT_STEP_MASK_NOTICE
    assert "Step 3" in i18n.NEXT_STEP_MASK_NOTICE
    assert i18n.t("REVIEW_BLUR_DETECTION") in labels
    assert i18n.tip("REVIEW_BLUR_DETECTION").splitlines()[0] not in labels
    assert i18n.t("REVIEW_BLUR_DETECTION_STANDARD") in buttons
    assert i18n.t("REVIEW_BLUR_DETECTION_LOW") in buttons
    assert step.blur_mode_label.toolTip() == i18n.tip("REVIEW_BLUR_DETECTION")
    assert step.blur_mode_control.sizePolicy().horizontalPolicy() == QSizePolicy.Fixed


def test_review_step_uses_fixed_selected_frames_csv(tmp_path: Path) -> None:
    _app()
    step = ReviewStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))

    assert step._csv_path() == selected_frames_path(tmp_path)


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


def test_review_step_blur_detection_mode_can_switch_back_and_forth(tmp_path: Path) -> None:
    _app()
    csv_path = _write_blur_scene(tmp_path)
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()

    assert step.blur_mode_buttons[BLUR_REVIEW_MODE_STANDARD].isChecked()
    assert step.blur_mode_buttons[BLUR_REVIEW_MODE_LOW].isEnabled()

    step._on_blur_review_mode_clicked(BLUR_REVIEW_MODE_LOW)
    rows = _read_rows(csv_path)

    assert step.blur_mode_buttons[BLUR_REVIEW_MODE_LOW].isChecked()
    assert [row["status"] for row in rows] == ["borderline_blur", "ok", "ok"]
    assert [row["decision"] for row in rows] == ["keep", "keep", "keep"]
    assert {row[BLUR_REVIEW_MODE_FIELD] for row in rows} == {BLUR_REVIEW_MODE_LOW}

    step._on_blur_review_mode_clicked(BLUR_REVIEW_MODE_STANDARD)
    rows = _read_rows(csv_path)

    assert step.blur_mode_buttons[BLUR_REVIEW_MODE_STANDARD].isChecked()
    assert [row["status"] for row in rows] == ["motion_blur", "ok", "ok"]
    assert [row["decision"] for row in rows] == ["drop", "keep", "keep"]
    assert {row[BLUR_REVIEW_MODE_FIELD] for row in rows} == {BLUR_REVIEW_MODE_STANDARD}


def test_review_step_blur_detection_does_not_overwrite_manual_decisions(tmp_path: Path) -> None:
    _app()
    csv_path = _write_blur_scene(tmp_path)
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()
    widget = step._review_widget
    assert widget is not None

    widget.toggle_decision()
    rows = _read_rows(csv_path)
    assert rows[0]["decision"] == "keep"
    assert rows[0][REVIEW_DECISION_OVERRIDE_FIELD] == "1"

    step._on_blur_review_mode_clicked(BLUR_REVIEW_MODE_LOW)
    step._on_blur_review_mode_clicked(BLUR_REVIEW_MODE_STANDARD)
    rows = _read_rows(csv_path)

    assert rows[0]["status"] == "motion_blur"
    assert rows[0]["decision"] == "keep"
    assert rows[0][REVIEW_DECISION_OVERRIDE_FIELD] == "1"

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
    assert isinstance(cmd, AppJob)
    assert cmd.payload["scene_dir"] == str(tmp_path)
    assert cmd.payload["finalize_in_place"] is True

    run_app_job(cmd)
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


def test_review_perspective_mode_is_disabled_for_normal_images(tmp_path: Path) -> None:
    _app()
    _write_mixed_projection_scene(tmp_path)
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()
    widget = step._review_widget
    assert widget is not None

    assert widget.projection_toggle_btn.isEnabled()
    widget.set_preview_mode(PREVIEW_MODE_PERSPECTIVE)
    assert widget.preview_mode() == PREVIEW_MODE_PERSPECTIVE

    widget._set_index(1)

    assert widget.preview_mode() == PREVIEW_MODE_SINGLE
    assert not widget.projection_toggle_btn.isEnabled()
    assert widget.projection_toggle_btn.toolTip() == i18n.tip("PREVIEW_PROJECTION_TOGGLE_DISABLED")

    widget.set_preview_mode(PREVIEW_MODE_PERSPECTIVE)

    assert widget.preview_mode() == PREVIEW_MODE_SINGLE

    widget._set_index(0)

    assert widget.projection_toggle_btn.isEnabled()
    widget.set_preview_mode(PREVIEW_MODE_PERSPECTIVE)
    assert widget.preview_mode() == PREVIEW_MODE_PERSPECTIVE


def test_review_step_can_renumber_kept_images_on_finalize(tmp_path: Path, monkeypatch) -> None:
    _app()
    csv_path = _write_scene(tmp_path, count=3, drop_indices={2})
    step = ReviewStep(Path.cwd())
    step.set_scene_dir(str(tmp_path))
    step.on_activated()
    step.renumber_kept_images_cb.setChecked(True)
    monkeypatch.setattr(step, "_confirm_finalize", lambda: True)

    commands = step.build_commands()

    assert len(commands) == 1
    _label, cmd = commands[0]
    assert isinstance(cmd, AppJob)
    assert cmd.payload["scene_dir"] == str(tmp_path)
    assert cmd.payload["finalize_in_place"] is True
    assert cmd.payload["renumber_kept_images"] is True
    assert step._pending_review_run is not None
    assert step._pending_review_run["renumber_kept_images"] is True
    assert step._pending_review_run["renamed_images"] == [
        {"from": "images/frame_000003.png", "to": "images/frame_000002.png"}
    ]

    run_app_job(cmd)
    assert sorted(path.name for path in (tmp_path / "images").glob("*.png")) == [
        "frame_000001.png",
        "frame_000002.png",
    ]
    rows = _read_rows(csv_path)
    assert [row["output_file"] for row in rows] == [
        "images/frame_000001.png",
        "images/frame_000002.png",
    ]


def test_review_step_disables_renumber_after_masks_exist(tmp_path: Path) -> None:
    _app()
    _write_scene(tmp_path, count=2)
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "frame_000001.png").write_bytes(b"mask")
    step = ReviewStep(Path.cwd())

    step.set_scene_dir(str(tmp_path))
    step.on_activated()

    assert not step.renumber_kept_images_cb.isEnabled()
    assert i18n.t("REVIEW_RENUMBER_BLOCKED") in step.renumber_kept_images_cb.toolTip()


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
