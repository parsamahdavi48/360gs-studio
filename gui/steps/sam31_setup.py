"""SAM3.1 checkpoint setup UI helpers."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEventLoop, QObject, Qt, QThread, Signal
from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox, QProgressDialog, QWidget

from gui import i18n
from sam31_download import download_sam31_checkpoint


class Sam31DownloadWorker(QObject):
    finished = Signal(str, str)

    def __init__(self, token: str, target_dir: Path) -> None:
        super().__init__()
        self._token = token
        self._target_dir = target_dir

    def run(self) -> None:
        try:
            checkpoint = download_sam31_checkpoint(self._token, self._target_dir)
        except Exception as exc:  # noqa: BLE001 - surfaced in the GUI.
            self.finished.emit("", str(exc))
            return
        finally:
            self._token = ""
        self.finished.emit(str(checkpoint), "")


def ensure_sam31_checkpoint_available(
    parent: QWidget,
    checkpoint_path: Path,
    *,
    on_available: Callable[[], None] | None = None,
) -> bool:
    if checkpoint_path.is_file():
        return True

    result = QMessageBox.question(
        parent,
        i18n.t("SAM31_DOWNLOAD_TITLE"),
        i18n.t("SAM31_DOWNLOAD_BODY").format(path=str(checkpoint_path)),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    if result != QMessageBox.Yes:
        return False

    token, accepted = QInputDialog.getText(
        parent,
        i18n.t("SAM31_TOKEN_TITLE"),
        i18n.t("SAM31_TOKEN_BODY"),
        QLineEdit.Password,
    )
    if not accepted:
        return False
    token = token.strip()
    if not token:
        QMessageBox.warning(parent, i18n.t("SAM31_DOWNLOAD_TITLE"), i18n.t("SAM31_TOKEN_EMPTY"))
        return False

    progress = QProgressDialog(i18n.t("SAM31_DOWNLOAD_PROGRESS"), "", 0, 0, parent)
    progress.setWindowTitle(i18n.t("SAM31_DOWNLOAD_TITLE"))
    progress.setCancelButton(None)
    progress.setWindowModality(Qt.ApplicationModal)
    progress.show()

    result_path = ""
    error_text = ""
    loop = QEventLoop(parent)
    thread = QThread(parent)
    worker = Sam31DownloadWorker(token, checkpoint_path.parent)
    token = ""
    worker.moveToThread(thread)

    def on_finished(path: str, error: str) -> None:
        nonlocal result_path, error_text
        result_path = path
        error_text = error
        thread.quit()
        loop.quit()

    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    loop.exec()
    if thread.isRunning():
        thread.quit()
        thread.wait(3000)

    progress.close()
    if error_text:
        QMessageBox.critical(
            parent,
            i18n.t("SAM31_DOWNLOAD_TITLE"),
            i18n.t("SAM31_DOWNLOAD_FAILED").format(error=error_text),
        )
        return False

    checkpoint = Path(result_path)
    if on_available is not None:
        on_available()
    QMessageBox.information(
        parent,
        i18n.t("SAM31_DOWNLOAD_TITLE"),
        i18n.t("SAM31_DOWNLOAD_COMPLETE").format(path=str(checkpoint)),
    )
    return True
