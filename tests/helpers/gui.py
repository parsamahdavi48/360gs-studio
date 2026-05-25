from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from gui.theme import apply_theme


def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def themed_qt_app() -> QApplication:
    app = qt_app()
    apply_theme(app)
    return app


def patch_messagebox_question(monkeypatch, answer=QMessageBox.Yes) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: answer)


def patch_messagebox_warning_collector(monkeypatch, warnings: list[tuple[str, str]]) -> None:
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, text: warnings.append((title, text)))
