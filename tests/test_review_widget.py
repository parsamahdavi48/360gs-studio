import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication, QToolButton

from core.scene_layout import selected_frames_path
from gui import i18n
from gui.common.perspective_preview import PREVIEW_PROJECTION_EQUIRECT, PREVIEW_PROJECTION_PERSPECTIVE
from gui.common.preview_mode_toolbar import PREVIEW_MODE_PERSPECTIVE, PREVIEW_MODE_SINGLE, PREVIEW_MODE_THUMBNAILS
from review_frames import ReviewWidget, _review_thumbnail_image


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
            }
        )

    csv_path = selected_frames_path(scene)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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
    widget.rows[0]["analysis_pipeline"] = "pair"
    widget.rows[1]["analysis_pipeline"] = "pair"
    widget.rows[0]["status"] = "novelty_added"
    widget.rows[1]["status"] = "motion_blur+redundant_drop"
    widget.rows[1]["decision"] = "drop"
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
    assert i18n.t("REVIEW_INFO_FORMAT").format(ts="2.00s") == widget.info_label.text()
    assert widget.image_view._source_pixmap is not None


def test_review_widget_projection_toggle_renders_square_perspective(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)

    assert widget.preview_projection() == PREVIEW_PROJECTION_EQUIRECT

    widget.projection_toggle_btn.click()
    before = widget._perspective_params
    widget._on_perspective_dragged(10.0, 5.0)

    assert widget.preview_mode() == PREVIEW_MODE_PERSPECTIVE
    assert widget.preview_projection() == PREVIEW_PROJECTION_PERSPECTIVE
    assert widget.projection_toggle_btn.isChecked()
    assert not widget.mode_toolbar.single_preview_btn.isChecked()
    assert widget.image_view._drag_mode == "look"
    assert widget._perspective_params != before
    assert widget.current_pixmap is not None
    assert widget.current_pixmap.width() == widget.current_pixmap.height()


def test_review_widget_perspective_button_leaves_thumbnail_mode(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)

    widget.projection_toggle_btn.click()

    assert widget.preview_mode() == PREVIEW_MODE_PERSPECTIVE
    assert widget.preview_projection() == PREVIEW_PROJECTION_PERSPECTIVE
    assert widget.preview_stack.currentWidget() == widget.image_view
    assert widget.projection_toggle_btn.isChecked()
    assert not widget.mode_toolbar.thumbnail_preview_btn.isChecked()


def test_review_widget_labels_quick_extract_rows_separately(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["analysis_pipeline"] = "quick"
    rows[1]["analysis_pipeline"] = "quick"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    widget = ReviewWidget(scene, csv_path)

    assert widget.advisory_label.text() == i18n.t("REVIEW_ADVISORY_QUICK")
    assert "#ede9fe" in widget.advisory_label.styleSheet()
    assert "#5b21b6" in widget.advisory_label.styleSheet()


def test_review_widget_labels_external_import_rows_separately(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["analysis_pipeline"] = "external_import"
    rows[1]["analysis_pipeline"] = "external_import"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    widget = ReviewWidget(scene, csv_path)

    assert widget.advisory_label.text() == i18n.t("REVIEW_ADVISORY_EXTERNAL_IMPORT")
    assert "#ccfbf1" in widget.advisory_label.styleSheet()
    assert "#0f766e" in widget.advisory_label.styleSheet()


def test_review_widget_advisory_labels_do_not_show_internal_pair_analysis(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["analysis_pipeline"] = "pair"
    rows[0]["status"] = "redundant_drop"
    rows[0]["decision"] = "drop"
    rows[1]["analysis_pipeline"] = "pair"
    rows[1]["status"] = "gap_forced"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    widget = ReviewWidget(scene, csv_path)

    assert "ペア解析" not in widget.advisory_label.text()
    assert "Pair analysis" not in widget.advisory_label.text()
    assert widget.advisory_label.text() == i18n.t("REVIEW_ADVISORY_DROP_REDUNDANT")
    widget._set_index(1)
    assert "ペア解析" not in widget.advisory_label.text()
    assert "Pair analysis" not in widget.advisory_label.text()
    assert widget.advisory_label.text() == i18n.t("REVIEW_ADVISORY_GAP_FORCED")


def test_review_widget_labels_blur_replacements(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["analysis_pipeline"] = "pair"
    rows[0]["status"] = "blur_replacement"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    widget = ReviewWidget(scene, csv_path)

    assert widget.advisory_label.text() == i18n.t("REVIEW_ADVISORY_BLUR_REPLACEMENT")


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
            assert "replace" not in text.lower()
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


def test_review_widget_thumbnail_mode_shows_advisory_ribbons_without_decision_border(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene_with_drop(tmp_path)
    widget = ReviewWidget(scene, csv_path)

    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)

    buttons = widget.mode_toolbar.findChildren(QToolButton)
    assert len(buttons) == 3
    assert widget.preview_mode() == PREVIEW_MODE_THUMBNAILS
    assert widget.preview_stack.currentWidget() == widget.thumbnail_view
    assert widget.mode_toolbar.thumbnail_preview_btn.isChecked()
    assert widget.mode_toolbar.single_preview_btn.accessibleName() == i18n.t("REVIEW_PREVIEW_MODE_SINGLE")
    assert widget.thumbnail_view.selectionMode() == QAbstractItemView.ExtendedSelection
    assert widget.thumbnail_model.rowCount() == 2

    drop_item = widget.thumbnail_model.item_at(0)
    keep_item = widget.thumbnail_model.item_at(1)
    assert drop_item is not None
    assert keep_item is not None
    assert drop_item.cache_key[0] == "drop"
    assert keep_item.cache_key[0] == "keep"
    assert drop_item.cache_key[5] == i18n.t("REVIEW_ADVISORY_SHORT_DROP_MANUAL")
    assert keep_item.cache_key[5] == i18n.t("REVIEW_ADVISORY_SHORT_NORMAL")

    drop_thumb = _review_thumbnail_image(drop_item, widget.thumbnail_model.icon_size())
    keep_thumb = _review_thumbnail_image(keep_item, widget.thumbnail_model.icon_size())
    sample_x = widget.thumbnail_model.icon_size().width() - 8
    sample_y = widget.thumbnail_model.icon_size().height() - 9
    assert drop_thumb.pixelColor(sample_x, sample_y).name().lower() == "#fee2e2"
    assert keep_thumb.pixelColor(sample_x, sample_y).name().lower() == "#dcfce7"
    assert drop_thumb.pixelColor(1, 1).name().lower() not in {"#991b1b", "#22c55e"}
    assert keep_thumb.pixelColor(1, 1).name().lower() not in {"#991b1b", "#22c55e"}


def test_review_thumbnail_uses_single_preview_advisory_colors(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.rows[0]["analysis_pipeline"] = "pair"
    widget.rows[0]["status"] = "novelty_added"
    widget.rows[1]["analysis_pipeline"] = "pair"
    widget.rows[1]["status"] = "motion_blur"
    widget.rows[1]["decision"] = "drop"
    widget._refresh_thumbnail_row(0)
    widget._refresh_thumbnail_row(1)

    size = widget.thumbnail_model.icon_size()
    added_item = widget.thumbnail_model.item_at(0)
    blur_item = widget.thumbnail_model.item_at(1)
    assert added_item is not None
    assert blur_item is not None
    assert added_item.cache_key[5] == i18n.t("REVIEW_ADVISORY_SHORT_NOVELTY_ADDED")
    assert blur_item.cache_key[5] == i18n.t("REVIEW_ADVISORY_SHORT_DROP_BLUR")

    added_thumb = _review_thumbnail_image(added_item, size)
    blur_thumb = _review_thumbnail_image(blur_item, size)
    sample_x = size.width() - 8
    sample_y = size.height() - 9

    assert added_thumb.pixelColor(sample_x, sample_y).name().lower() == "#dbeafe"
    assert blur_thumb.pixelColor(sample_x, sample_y).name().lower() == "#fee2e2"


def test_review_widget_review_controls_are_left_of_mode_toolbar(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    top_row = widget.layout().itemAt(0).layout()

    assert top_row.itemAt(top_row.count() - 1).widget() == widget.mode_toolbar
    assert widget.mode_toolbar.isAncestorOf(widget.projection_toggle_btn)
    assert top_row.itemAt(top_row.count() - 2).widget() == widget.reset_decision_button
    assert top_row.itemAt(top_row.count() - 3).widget() == widget.flag_button


def test_review_widget_thumbnail_selection_changes_current_frame(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)

    widget.thumbnail_view.setCurrentIndex(widget.thumbnail_model.index(1, 0))

    assert widget.index == 1
    assert widget.frame_slider.value() == 1
    assert "2 / 2" in widget.frame_position_label.text()


def test_review_widget_thumbnail_arrow_key_moves_thumbnail_selection(tmp_path: Path) -> None:
    app = _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.show()

    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)
    app.processEvents()
    QTest.keyClick(widget.thumbnail_view, Qt.Key_Right)
    app.processEvents()

    assert not widget.prev_row_shortcut.isEnabled()
    assert not widget.next_row_shortcut.isEnabled()
    assert widget.thumbnail_view.hasFocus()
    assert widget.index == 1
    assert widget.frame_slider.value() == 1
    assert widget.thumbnail_view.selectionModel().currentIndex().row() == 1
    assert sorted(index.row() for index in widget.thumbnail_view.selectionModel().selectedIndexes()) == [1]


def test_review_widget_thumbnail_mode_uses_extended_selection(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)
    selection = widget.thumbnail_view.selectionModel()

    selection.select(widget.thumbnail_model.index(0, 0), QItemSelectionModel.ClearAndSelect)
    selection.select(widget.thumbnail_model.index(1, 0), QItemSelectionModel.Select)

    assert widget.thumbnail_view.selectionMode() == QAbstractItemView.ExtendedSelection
    assert sorted(index.row() for index in selection.selectedIndexes()) == [0, 1]


def test_review_widget_thumbnail_multi_selection_toggles_and_resets_flags(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)
    selection = widget.thumbnail_view.selectionModel()
    selection.select(widget.thumbnail_model.index(0, 0), QItemSelectionModel.ClearAndSelect)
    selection.select(widget.thumbnail_model.index(1, 0), QItemSelectionModel.Select)
    changed_rows: list[int] = []

    def fail_full_reset(*_args, **_kwargs) -> None:
        raise AssertionError("batch flag changes must not rebuild the full thumbnail model")

    def remember_changed(top_left, _bottom_right, _roles) -> None:
        changed_rows.append(top_left.row())

    monkeypatch.setattr(widget.thumbnail_model, "set_items", fail_full_reset)
    widget.thumbnail_model.dataChanged.connect(remember_changed)

    widget.toggle_decision()

    assert [row["decision"] for row in widget.rows] == ["drop", "drop"]
    assert _read_decisions(csv_path) == ["drop", "drop"]
    assert [widget.thumbnail_model.item_at(i).cache_key[0] for i in range(2)] == ["drop", "drop"]
    assert sorted(index.row() for index in selection.selectedIndexes()) == [0, 1]
    assert changed_rows == [0, 1]

    widget.reset_decision()

    assert [row["decision"] for row in widget.rows] == ["keep", "keep"]
    assert _read_decisions(csv_path) == ["keep", "keep"]
    assert [widget.thumbnail_model.item_at(i).cache_key[0] for i in range(2)] == ["keep", "keep"]
    assert sorted(index.row() for index in selection.selectedIndexes()) == [0, 1]
    assert changed_rows == [0, 1, 0, 1]


def test_review_widget_thumbnail_decision_update_refreshes_only_flag_item(tmp_path: Path, monkeypatch) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)
    widget.thumbnail_view.setCurrentIndex(widget.thumbnail_model.index(1, 0))
    changed_rows: list[int] = []

    def fail_full_reset(*_args, **_kwargs) -> None:
        raise AssertionError("flag changes must not rebuild the full thumbnail model")

    def remember_changed(top_left, _bottom_right, _roles) -> None:
        changed_rows.append(top_left.row())

    monkeypatch.setattr(widget.thumbnail_model, "set_items", fail_full_reset)
    widget.thumbnail_model.dataChanged.connect(remember_changed)

    widget.toggle_decision()

    item = widget.thumbnail_model.item_at(1)
    assert item is not None
    assert widget.index == 1
    assert widget.rows[1]["decision"] == "drop"
    assert _read_decisions(csv_path)[1] == "drop"
    assert item.cache_key[0] == "drop"
    assert widget.thumbnail_view.selectionModel().currentIndex().row() == 1
    assert changed_rows == [1]


def test_review_widget_thumbnail_double_click_opens_single_preview(tmp_path: Path) -> None:
    _app()
    scene, csv_path = _write_scene(tmp_path)
    widget = ReviewWidget(scene, csv_path)
    widget.set_preview_mode(PREVIEW_MODE_THUMBNAILS)

    widget.thumbnail_view.doubleClicked.emit(widget.thumbnail_model.index(1, 0))

    assert widget.preview_mode() == PREVIEW_MODE_SINGLE
    assert widget.preview_stack.currentWidget() == widget.image_view
    assert widget.mode_toolbar.single_preview_btn.isChecked()
    assert widget.index == 1
    assert widget.image_view._source_pixmap is not None
