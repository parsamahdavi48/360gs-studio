"""ステップ基底クラス"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QScrollArea, QWidget

from core.app_job import AppJob
from gui import i18n

SETTINGS_PANE_WIDTH = 480
SETTINGS_PANE_MARGINS = (0, 0, 14, 4)


def configure_settings_scroll(scroll: QScrollArea, width: int = SETTINGS_PANE_WIDTH) -> None:
    """ステップ左側の設定ペインを固定幅・縦スクロール専用に揃える。"""
    scroll.setObjectName("settingsScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setFixedWidth(width)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)


class BaseStepWidget(QWidget):
    """各ワークフローステップの基底クラス。

    サブクラスは以下を実装する:
        build_ui()  -- ステップ固有のUI構築
        set_scene_dir(path)  -- シーンディレクトリ変更への対応
        build_commands()  -- [(phase, [cmd...] | AppJob)] を返す。失敗時は ValueError
        on_line(line)  -- 出力行のパース（プログレス更新など）
    """

    primary_action_state_changed = Signal()
    background_task_started = Signal(str)
    background_line_received = Signal(str)
    background_progress_changed = Signal(int, int)
    background_status_changed = Signal(str)
    background_task_finished = Signal(bool, bool)

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self.scene_dir = ""

    def set_scene_dir(self, path: str) -> None:
        self.scene_dir = path

    def build_commands(self) -> list[tuple[str, list[str] | AppJob]]:
        raise NotImplementedError

    def confirm_commands(self, commands: list[tuple[str, list[str] | AppJob]]) -> bool:
        return True

    def process_log_dir(self) -> Path | None:
        """Return a directory for per-phase process logs, or None to disable file logs."""
        return None

    def primary_action_text(self) -> str:
        return i18n.RUN

    def primary_action_tooltip(self) -> str:
        return i18n.tip("RUN")

    def primary_action_enabled(self) -> bool:
        return True

    def run_primary_action(self) -> bool:
        """Handle the shared primary action button without queueing commands.

        Return True when the action was handled internally. Queue-based steps
        leave this as False so MainWindow can build and queue external commands
        or typed internal AppJob phases.
        """
        return False

    def header_title(self) -> str:
        """Optional title override for the shared central pane header."""
        return ""

    def header_back_enabled(self) -> bool:
        """Return whether the shared header should show an in-step back button."""
        return False

    def header_back_tooltip(self) -> str:
        return ""

    def header_back(self) -> None:
        """Handle the shared header back button for multi-page steps."""
        pass

    def phase_display_name(self, phase: str) -> str:
        return phase

    def phase_status_text(self, phase: str, queue_index: int, queue_total: int) -> str:
        return f"{i18n.STATUS_RUNNING}: {self.phase_display_name(phase)}"

    def on_line(self, line: str) -> tuple[int, int] | None:
        """出力行を解析し、(done, total) を返す。該当なしなら None。"""
        return None

    def on_phase_started(self, phase: str) -> tuple[int, int] | None:
        """フェーズ開始時のプログレス初期化フック（オプション）"""
        return None

    def on_phase_log_started(self, phase: str, path: str) -> None:
        """フェーズログファイル作成時のフック（オプション）"""
        pass

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        """フェーズ完了時のフック（オプション）"""
        pass

    def on_queue_finished(self, success: bool) -> None:
        """全フェーズ完了時のフック（オプション）"""
        pass

    def on_activated(self) -> None:
        """ステップが表示された時のフック（オプション）"""
        pass

    def has_background_task(self) -> bool:
        """Return whether this step owns a currently running internal task."""
        return False

    def cancel_background_task(self) -> None:
        """Cancel a running internal task if the step has one."""
        pass

    def shutdown(self) -> None:
        """Application shutdown hook for child widgets with background work."""
        pass
