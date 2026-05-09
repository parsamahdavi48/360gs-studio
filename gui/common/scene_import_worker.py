"""Scene import worker that keeps file scanning off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.scene_import import SceneImportResult, import_scene

SceneImporter = Callable[[Path], SceneImportResult]


class SceneImportWorker(QObject):
    finished = Signal(object, str)

    def __init__(self, scene_dir: Path, importer: SceneImporter = import_scene) -> None:
        super().__init__()
        self._scene_dir = scene_dir
        self._importer = importer

    @Slot()
    def run(self) -> None:
        try:
            result = self._importer(self._scene_dir)
        except Exception as exc:  # noqa: BLE001 - surfaced in the GUI log.
            self.finished.emit(None, str(exc))
            return
        self.finished.emit(result, "")
