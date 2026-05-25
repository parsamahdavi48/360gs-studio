"""Small wrapper around Qt file dialogs.

Tests patch these functions instead of reaching into individual widget modules.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QWidget


def get_existing_directory(parent: QWidget | None, title: str, start_dir: str = "") -> str:
    return QFileDialog.getExistingDirectory(parent, title, start_dir)


def get_open_file_name(
    parent: QWidget | None,
    title: str,
    start_dir: str = "",
    filter_str: str = "",
) -> tuple[str, str]:
    return QFileDialog.getOpenFileName(parent, title, start_dir, filter_str)


def get_open_file_names(
    parent: QWidget | None,
    title: str,
    start_dir: str = "",
    filter_str: str = "",
) -> tuple[list[str], str]:
    return QFileDialog.getOpenFileNames(parent, title, start_dir, filter_str)
