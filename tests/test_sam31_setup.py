import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from gui.steps import sam31_setup


def _app():
    return QApplication.instance() or QApplication([])


class _FakeProgressDialog:
    closed = False
    shown = False

    def __init__(self, *_args, **_kwargs) -> None:
        type(self).closed = False
        type(self).shown = False

    def setWindowTitle(self, _title: str) -> None:
        pass

    def setCancelButton(self, _button: object) -> None:
        pass

    def setWindowModality(self, _modality: object) -> None:
        pass

    def show(self) -> None:
        type(self).shown = True

    def close(self) -> None:
        type(self).closed = True


def test_ensure_sam31_checkpoint_download_finishes_before_return(tmp_path: Path, monkeypatch) -> None:
    _app()
    parent = QWidget()
    checkpoint_path = tmp_path / "models" / "sam3.1" / "sam3.1_multiplex.pt"
    info_messages: list[str] = []

    def fake_download(token: str, target_dir: Path) -> Path:
        assert token == "secret-token"
        target = target_dir / checkpoint_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"checkpoint")
        return target

    monkeypatch.setattr(
        sam31_setup.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        sam31_setup.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("secret-token", True),
    )
    monkeypatch.setattr(
        sam31_setup.QMessageBox,
        "information",
        lambda _parent, _title, body: info_messages.append(body),
    )
    monkeypatch.setattr(sam31_setup, "QProgressDialog", _FakeProgressDialog)
    monkeypatch.setattr(sam31_setup, "download_sam31_checkpoint", fake_download)

    try:
        assert sam31_setup.ensure_sam31_checkpoint_available(parent, checkpoint_path)
    finally:
        parent.deleteLater()

    assert checkpoint_path.is_file()
    assert _FakeProgressDialog.shown
    assert _FakeProgressDialog.closed
    assert info_messages and str(checkpoint_path) in info_messages[0]
