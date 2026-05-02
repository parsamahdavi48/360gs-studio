import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from gui import i18n
from review_frames import ReviewWidget


def _app():
    return QApplication.instance() or QApplication([])


def _write_scene(tmp_path: Path) -> tuple[Path, Path]:
    scene = tmp_path
    images = scene / "images"
    images.mkdir()
    rows = []
    for seq in range(1, 3):
        pixmap = QPixmap(64, 32)
        pixmap.fill(Qt.red if seq == 1 else Qt.blue)
        image_rel = f"images/frame_{seq:06d}.png"
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
                "quality_min_score": "0.35",
                "quality_score_original": "0.20" if seq == 1 else "0.50",
                "quality_score_final": "0.62" if seq == 1 else "0.50",
            }
        )

    csv_path = scene / "selected_frames.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return scene, csv_path


def _write_scene_with_drop(tmp_path: Path, drop_file_exists: bool = True) -> tuple[Path, Path]:
    scene, csv_path = _write_scene(tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["decision"] = "drop"
    if not drop_file_exists:
        (scene / rows[0]["output_file"]).unlink()
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return scene, csv_path


def _read_decisions(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [row["decision"] for row in csv.DictReader(f)]


def _set_summary_statuses(widget: ReviewWidget) -> None:
    widget.rows[0]["status"] = "smart_added+replaced"
    widget.rows[1]["status"] = "fallback_keep+thinned"
    widget.problem_indices = widget._collect_problem_indices()
    widget._render_current()


def test_review_widget_slider_changes_current_frame(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)

    widget.frame_slider.setValue(1)

    assert widget.index == 1
    assert widget.frame_slider.value() == 1
    assert "2 / 2" in widget.frame_position_label.text()
    assert i18n.t("REVIEW_INFO_FORMAT").format(
        ts="2.00s",
        quality=f"0.50 ({i18n.t('REVIEW_QUALITY_THRESHOLD_FORMAT').format(score='0.35')})",
    ) == widget.info_label.text()
    assert widget.image_view._source_pixmap is not None


def test_review_widget_shows_quality_score_and_original_when_replaced(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)

    assert widget.info_label.text() == i18n.t("REVIEW_INFO_FORMAT").format(
        ts="1.00s",
        quality=(
            "0.62 ("
            f"{i18n.t('REVIEW_QUALITY_ORIGINAL_FORMAT').format(score='0.20')} / "
            f"{i18n.t('REVIEW_QUALITY_THRESHOLD_FORMAT').format(score='0.35')}"
            ")"
        ),
    )


def test_review_summary_label_is_readable_single_line(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    _set_summary_statuses(widget)

    text = widget.problem_summary_label.text()
    assert "\n" not in text
    assert not widget.problem_summary_label.wordWrap()
    assert "代表置換" not in text
    assert "representative" not in text.lower()
    assert "#666" not in widget.problem_summary_label.styleSheet()
    assert "palette(text)" in widget.problem_summary_label.styleSheet()
    assert widget.problem_summary_label.sizeHint().width() <= 760


def test_review_summary_label_is_single_line_in_english(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import os
        import tempfile
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ["STUDIO_LANG"] = "en"
        from pathlib import Path
        from PySide6.QtWidgets import QApplication
        from review_frames import ReviewWidget
        from tests.test_review_widget import _write_scene, _set_summary_statuses

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as d:
            scene, csv_path = _write_scene(Path(d))
            widget = ReviewWidget(scene, csv_path)
            _set_summary_statuses(widget)
            text = widget.problem_summary_label.text()
            assert "\\n" not in text
            assert not widget.problem_summary_label.wordWrap()
            assert "representative" not in text.lower()
            assert "replace" in text
            assert widget.problem_summary_label.sizeHint().width() <= 760
        """
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["STUDIO_LANG"] = "en"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_review_widget_flag_toggle_saves_immediately_and_resets(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)

    widget.toggle_decision()

    assert widget.rows[0]["decision"] == "drop"
    assert _read_decisions(csv_path)[0] == "drop"
    assert not widget.flag_button.isChecked()
    assert not widget.flag_button.icon().isNull()
    assert not widget.reset_decision_button.icon().isNull()
    assert widget.reset_decision_button.isEnabled()

    widget.reset_decision()

    assert widget.rows[0]["decision"] == "keep"
    assert _read_decisions(csv_path)[0] == "keep"
    assert widget.flag_button.isChecked()
    assert not widget.reset_decision_button.isEnabled()


def test_review_widget_tracks_decision_changes(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)

    assert not widget.has_decision_changes()

    widget.toggle_decision()

    assert widget.has_decision_changes()

    widget.reset_decision()

    assert not widget.has_decision_changes()


def test_review_widget_pending_finalize_includes_existing_drop_images(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene_with_drop(tmp_path, drop_file_exists=True)
    widget = ReviewWidget(scene, csv_path)

    assert not widget.has_decision_changes()
    assert widget.has_pending_finalize()
    assert [p.name for p in widget.pending_drop_image_paths()] == ["frame_000001.png"]


def test_review_widget_pending_finalize_ignores_missing_drop_images(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene_with_drop(tmp_path, drop_file_exists=False)
    widget = ReviewWidget(scene, csv_path)

    assert not widget.has_decision_changes()
    assert not widget.has_pending_finalize()
