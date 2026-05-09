"""Scene import worker that keeps file scanning off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.scene_import import SceneImportProgressCallback, SceneImportResult, import_scene
from core.scene_import_contracts import SceneImportCancelled, SceneImportCancelToken

SceneImporter = Callable[..., SceneImportResult]


class SceneImportWorker(QObject):
    progress = Signal(str)
    finished = Signal(object, str, bool)

    def __init__(self, scene_dir: Path, importer: SceneImporter = import_scene) -> None:
        super().__init__()
        self._scene_dir = scene_dir
        self._importer = importer
        self._cancel_token = SceneImportCancelToken()

    def cancel(self) -> None:
        self._cancel_token.request_cancel()

    @Slot()
    def run(self) -> None:
        try:
            progress: SceneImportProgressCallback = self.progress.emit
            result = self._importer(
                self._scene_dir,
                cancel_token=self._cancel_token,
                progress_callback=progress,
            )
        except SceneImportCancelled:
            self.finished.emit(None, "", True)
            return
        except Exception as exc:  # noqa: BLE001 - surfaced in the GUI log.
            self.finished.emit(None, str(exc), False)
            return
        self.finished.emit(result, "", False)
