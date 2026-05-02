import csv
import os
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


def _read_decisions(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [row["decision"] for row in csv.DictReader(f)]


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
