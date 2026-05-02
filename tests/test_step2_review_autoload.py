import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from gui import i18n
from gui.app import MainWindow
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
    assert "再読み込み" not in buttons
    assert "Reload" not in buttons
    assert i18n.NEXT_STEP_MASK_NOTICE in labels
    assert i18n.t("ACTION_FINALIZE_REVIEW") in i18n.NEXT_STEP_MASK_NOTICE
    assert "Step 3" in i18n.NEXT_STEP_MASK_NOTICE


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
