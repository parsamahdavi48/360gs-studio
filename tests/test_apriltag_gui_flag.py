from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_apriltag_scale_tool_is_visible_without_startup_flag() -> None:
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ.pop("STECHDRIVE_ENABLE_APRILTAG", None)

        from PySide6.QtWidgets import QApplication

        from gui import i18n
        from gui.theme import apply_theme
        from gui.steps.apriltag_scale_tool import AprilTagScaleTool

        app = QApplication([])
        apply_theme(app)
        step = AprilTagScaleTool(Path.cwd())
        assert step.primary_action_enabled() is False
        assert step.primary_action_tooltip() == i18n.tip("APRILTAG_TAB_PRIMARY_ACTION")
        assert not hasattr(step, "apriltag_enable_cb")
        assert step.apriltag_id_edit.text() == "7"
        assert step.apriltag_conversion_preset_combo.currentData() == "auto"
        assert step.apriltag_result_label.text() == i18n.t("APRILTAG_RESULT_EMPTY")
        """
    )

    env = os.environ.copy()
    env.pop("STECHDRIVE_ENABLE_APRILTAG", None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_scale_command_targets_existing_cubemap_output(tmp_path: Path) -> None:
    transforms = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 10,
        "h": 10,
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": 4.5,
        "cy": 4.5,
        "frames": [
            {
                "file_path": "images/a.png",
                "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ],
    }
    script = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        scene = Path({str(tmp_path)!r})
        output = scene / "output" / "metashape_cubemap"
        (output / "images").mkdir(parents=True)
        (output / "images" / "a.png").write_bytes(b"image")
        (output / "transforms.json").write_text(json.dumps({transforms!r}), encoding="utf-8")

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.set_scene_dir(str(scene))
        step.apriltag_tag_size_edit.setValue(0.2)
        step.apriltag_id_edit.setText("7, 8")

        cmd = step._build_apriltag_scale_cmd(scene / "_stechdrive" / "step4" / "apriltag_scale_report.json")
        assert cmd[0].endswith("python.exe") or cmd[0].endswith("python")
        assert "scripts\\\\estimate_apriltag_scale.py" in cmd[2] or "scripts/estimate_apriltag_scale.py" in cmd[2]
        assert str(output / "transforms.json") in cmd
        assert cmd[cmd.index("--tag-size-m") + 1] == "0.2"
        assert "--equirect-temp-dir" not in cmd
        assert "--cubemap-pose-preset" not in cmd
        assert cmd.count("--tag-id") == 2
        assert cmd[cmd.index("--tag-id") + 1] == "7"
        assert cmd[cmd.index("--tag-id", cmd.index("--tag-id") + 1) + 1] == "8"
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_scale_command_can_set_conversion_preset(tmp_path: Path) -> None:
    transforms = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 10,
        "h": 10,
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": 4.5,
        "cy": 4.5,
        "frames": [
            {
                "file_path": "images/a.png",
                "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ],
    }
    script = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        scene = Path({str(tmp_path)!r})
        output = scene / "output" / "metashape_cubemap"
        (output / "images").mkdir(parents=True)
        (output / "images" / "a.png").write_bytes(b"image")
        (output / "transforms.json").write_text(json.dumps({transforms!r}), encoding="utf-8")

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.set_scene_dir(str(scene))
        preset_index = step.apriltag_conversion_preset_combo.findData("lichtfeld")
        assert preset_index >= 0
        step.apriltag_conversion_preset_combo.setCurrentIndex(preset_index)

        cmd = step._build_apriltag_scale_cmd(scene / "_stechdrive" / "step4" / "apriltag_scale_report.json")
        assert cmd[cmd.index("--cubemap-pose-preset") + 1] == "lichtfeld"
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_scale_command_rejects_equirectangular_output(tmp_path: Path) -> None:
    transforms = {
        "camera_model": "EQUIRECTANGULAR",
        "frames": [
            {
                "file_path": "images/a.png",
                "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ],
    }
    script = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        scene = Path({str(tmp_path)!r})
        output = scene / "output" / "metashape_cubemap"
        (output / "images").mkdir(parents=True)
        (output / "images" / "a.png").write_bytes(b"image")
        (output / "transforms.json").write_text(json.dumps({transforms!r}), encoding="utf-8")

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.set_scene_dir(str(scene))

        try:
            step._build_apriltag_scale_cmd(scene / "_stechdrive" / "step4" / "apriltag_scale_report.json")
        except ValueError as exc:
            assert "projected Cubemap output" in str(exc)
        else:
            raise AssertionError("EQUIRECTANGULAR output should be rejected")
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_apply_scale_confirms_target_files(tmp_path: Path) -> None:
    transforms = {
        "camera_model": "SIMPLE_PINHOLE",
        "w": 10,
        "h": 10,
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": 4.5,
        "cy": 4.5,
        "ply_file_path": "custom_points.ply",
        "frames": [
            {
                "file_path": "images/a.png",
                "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ],
    }
    script = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication, QMessageBox

        import gui.steps.step4_apriltag as apriltag_mod
        from gui import i18n
        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        scene = Path({str(tmp_path)!r})
        output = scene / "output" / "metashape_cubemap"
        (output / "images").mkdir(parents=True)
        (output / "images" / "a.png").write_bytes(b"image")
        (output / "custom_points.ply").write_text("ply\\n", encoding="ascii")
        (output / "transforms.json").write_text(json.dumps({transforms!r}), encoding="utf-8")

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.set_scene_dir(str(scene))
        step._apriltag_last_scale = 1.25

        captured = {{}}
        def fake_question(parent, title, text, buttons, default):
            captured["title"] = title
            captured["text"] = text
            return QMessageBox.No

        def fail_apply(*_args, **_kwargs):
            raise AssertionError("apply should not run when confirmation is canceled")

        QMessageBox.question = fake_question
        apriltag_mod.apply_scene_output_scale = fail_apply

        step._apply_apriltag_scale()

        assert captured["title"] == i18n.t("APRILTAG_APPLY_CONFIRM_TITLE")
        assert "scale=1.25" in captured["text"]
        assert "transforms.json" in captured["text"]
        assert "custom_points.ply" in captured["text"]
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_background_progress_reaches_main_window() -> None:
    script = textwrap.dedent(
        """
        import os

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui.app import MainWindow
        from gui.theme import apply_theme

        app = QApplication([])
        apply_theme(app)
        window = MainWindow("")
        window.dataset_step.show_tool("scale")
        window.dataset_step.scale_tool.background_task_started.emit("Running scale")
        window.dataset_step.scale_tool.background_line_received.emit("[apriltag] detection start")
        window.dataset_step.scale_tool.background_progress_changed.emit(3, 10)
        assert window.progress.status_label.text() == "Running scale"
        assert window.progress.bar.value() == 3
        assert window.progress.bar.maximum() == 10
        assert "detection start" in window.log_panel.toPlainText()
        window.dataset_step.scale_tool.background_task_finished.emit(True, False)
        window.shutdown()
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_scale_tool_uses_internal_actions() -> None:
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui import i18n
        from gui.theme import apply_theme
        from gui.steps.apriltag_scale_tool import AprilTagScaleTool

        app = QApplication([])
        apply_theme(app)
        step = AprilTagScaleTool(Path.cwd())
        assert step.primary_action_enabled() is False
        assert step.primary_action_tooltip() == i18n.tip("APRILTAG_TAB_PRIMARY_ACTION")
        assert step.apriltag_apply_btn.isEnabled() is False
        assert step.apriltag_scale_row.isHidden()
        assert step.apriltag_copy_scale_btn.isHidden()
        assert step.apriltag_print_page_combo.findData("A4") >= 0
        assert step.apriltag_print_page_combo.findData("A3") >= 0
        assert step.apriltag_print_page_combo.findData("Letter") >= 0
        assert step.apriltag_conversion_preset_combo.toolTip() == i18n.tip("APRILTAG_CONVERSION_PRESET")
        assert step.apriltag_conversion_preset_combo.findData("lichtfeld") >= 0
        assert step.apriltag_conversion_preset_combo.findData("postshot") >= 0
        assert step.apriltag_conversion_preset_combo.findData("brush") >= 0
        assert step.apriltag_conversion_preset_combo.findData("standard") >= 0
        result_tip = step.apriltag_result_label.toolTip()
        assert "観測" in result_tip or "observations" in result_tip
        assert "ペア" in result_tip or "pairs" in result_tip
        assert "inlier" in result_tip
        assert "RMS" in result_tip
        assert "Scaleを反映" in result_tip or "applied as-is" in result_tip
        tooltip = step.apriltag_id_edit.toolTip()
        assert 'src="data:image/png;base64,' in tooltip
        assert 'width="72" height="72"' in tooltip
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_result_wraps_and_copy_button_copies_scale(tmp_path: Path) -> None:
    long_scene = tmp_path / (
        "scene_with_a_long_name_for_scale_backup_wrapping_"
        "abcdefghijklmnopqrstuvwxyz"
    )
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from types import SimpleNamespace

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui import i18n
        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step._show_apriltag_estimate_result(
            1.23456789,
            {{"observation_count": 12, "pair_count": 8, "inlier_count": 7, "rms_residual_m": 0.0123}},
        )
        step._sync_apriltag_controls()
        assert step.apriltag_scale_value_label.text() == "scale=1.23456789"
        assert step.apriltag_scale_row.layout().itemAt(0).widget() is step.apriltag_scale_value_label
        assert step.apriltag_scale_row.layout().itemAt(1).widget() is step.apriltag_copy_scale_btn
        assert not step.apriltag_copy_scale_btn.isHidden()
        assert step.apriltag_copy_scale_btn.isEnabled()
        assert step.apriltag_result_label.wordWrap()
        assert not step.apriltag_result_label.text().startswith("scale=")
        assert step.apriltag_result_label.text().count("\\n") >= 2
        step._copy_apriltag_scale()
        assert QApplication.clipboard().text() == "1.23456789"
        assert step.apriltag_copy_scale_btn.toolTip() == i18n.t("APRILTAG_SCALE_COPIED")

        backup = Path({str(long_scene)!r}) / "output" / "apriltag_scale_backup_20260511_120000" / "transforms.json"
        result = SimpleNamespace(
            scale=1.23456789,
            frames_scaled=123,
            points_scaled=456,
            transforms_json=backup.parent.parent / "transforms.json",
            pointcloud_ply=backup.parent.parent / "custom_points.ply",
            transforms_backup=backup,
        )
        step._show_apriltag_applied_result(result)
        assert step.apriltag_scale_value_label.text() == "scale=1.23456789"
        assert "\\n" in step.apriltag_result_label.text()
        assert not step.apriltag_result_label.text().startswith("scale=")
        assert "バックアップ:" in step.apriltag_result_label.text() or "Backup:" in step.apriltag_result_label.text()
        assert str(backup.parent) in step.apriltag_result_label.toolTip()
        assert step.apriltag_copy_scale_btn.toolTip() == i18n.t("APRILTAG_COPY_SCALE")
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_pdf_exports_to_scene_output_and_wraps_status_path(tmp_path: Path) -> None:
    long_scene = tmp_path / (
        "scene_with_a_long_name_for_apriltag_pdf_status_wrapping_"
        "abcdefghijklmnopqrstuvwxyz"
    )
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        scene = Path({str(long_scene)!r})
        scene.mkdir(parents=True)

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.set_scene_dir(str(scene))
        step._export_apriltag_pdf()

        output = scene / "output"
        pdfs = sorted(output.glob("apriltag_*.pdf"))
        assert len(pdfs) == 1
        assert pdfs[0].parent == output
        assert not (scene / "_stechdrive" / "step4" / "apriltag_targets").exists()
        assert step.apriltag_print_status_label.toolTip() == str(pdfs[0])
        assert "\\n" in step.apriltag_print_status_label.text()
        assert "output" in step.apriltag_print_status_label.text()
        """
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
