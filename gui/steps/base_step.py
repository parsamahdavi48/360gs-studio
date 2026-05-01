"""ステップ基底クラス"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QWidget

from gui import i18n


class BaseStepWidget(QWidget):
    """各ワークフローステップの基底クラス。

    サブクラスは以下を実装する:
        build_ui()  -- ステップ固有のUI構築
        set_scene_dir(path)  -- シーンディレクトリ変更への対応
        build_commands()  -- [(phase, [cmd...])] を返す。失敗時は ValueError
        on_line(line)  -- 出力行のパース（プログレス更新など）
    """

    def __init__(self, base_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self.scene_dir = ""

    def set_scene_dir(self, path: str) -> None:
        self.scene_dir = path

    def build_commands(self) -> list[tuple[str, list[str]]]:
        raise NotImplementedError

    def primary_action_text(self) -> str:
        return i18n.RUN

    def primary_action_tooltip(self) -> str:
        return i18n.tip("RUN")

    def on_line(self, line: str) -> tuple[int, int] | None:
        """出力行を解析し、(done, total) を返す。該当なしなら None。"""
        return None

    def on_phase_finished(self, phase: str, exit_code: int, canceled: bool) -> None:
        """フェーズ完了時のフック（オプション）"""
        pass

    def on_queue_finished(self, success: bool) -> None:
        """全フェーズ完了時のフック（オプション）"""
        pass
