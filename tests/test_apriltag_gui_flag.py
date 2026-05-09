from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_apriltag_tab_is_hidden_without_experimental_flag() -> None:
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ.pop("STECHDRIVE_ENABLE_APRILTAG", None)

        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        assert step.settings_tabs.count() == 3
        assert not hasattr(step, "apriltag_enable_cb")
        """
    )

    env = os.environ.copy()
    env.pop("STECHDRIVE_ENABLE_APRILTAG", None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_tab_is_visible_with_experimental_flag() -> None:
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["STECHDRIVE_ENABLE_APRILTAG"] = "1"

        from PySide6.QtWidgets import QApplication

        from gui import i18n
        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        labels = [step.settings_tabs.tabText(i) for i in range(step.settings_tabs.count())]
        assert i18n.t("STEP4_TAB_APRILTAG_SCALE") in labels
        assert step.apriltag_enable_cb.text() == i18n.t("APRILTAG_SCALE_ENABLE")
        assert step.apriltag_result_label.text() == i18n.t("APRILTAG_RESULT_EMPTY")
        """
    )

    env = os.environ.copy()
    env["STECHDRIVE_ENABLE_APRILTAG"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_apriltag_scale_command_targets_existing_cubemap_output(tmp_path: Path) -> None:
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["STECHDRIVE_ENABLE_APRILTAG"] = "1"

        from PySide6.QtWidgets import QApplication

        from gui.theme import apply_theme
        from gui.steps.step4_cubemap import CubemapStep

        scene = Path({str(tmp_path)!r})
        output = scene / "output"
        output.mkdir(parents=True)
        (output / "transforms.json").write_text("{{}}", encoding="utf-8")

        app = QApplication([])
        apply_theme(app)
        step = CubemapStep(Path.cwd())
        step.set_scene_dir(str(scene))
        step.apriltag_enable_cb.setChecked(True)
        step.apriltag_tag_size_edit.setText("0.2")
        step.apriltag_id_edit.setText("7")

        cmd = step._build_apriltag_scale_cmd(scene / "_stechdrive" / "step4" / "apriltag_scale_report.json")
        assert cmd[0].endswith("python.exe") or cmd[0].endswith("python")
        assert "scripts\\\\estimate_apriltag_scale.py" in cmd[2] or "scripts/estimate_apriltag_scale.py" in cmd[2]
        assert str(output / "transforms.json") in cmd
        assert cmd[cmd.index("--tag-size-m") + 1] == "0.2"
        assert cmd[cmd.index("--tag-id") + 1] == "7"
        """
    )

    env = os.environ.copy()
    env["STECHDRIVE_ENABLE_APRILTAG"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
