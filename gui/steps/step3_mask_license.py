"""Step 3 model license and checkpoint checks."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QMessageBox

from gui import i18n
from gui.steps.sam31_setup import ensure_sam31_checkpoint_available
from gui.user_settings import load_user_settings_section, update_user_settings_section

_LICENSE_NOTICE_SECTION = "license_notices"
_YOLO_SAM_NOTICE_VERSION = 3
_YOLO_SAM_NOTICE_KEY = "yolo_sam_models_ack_version"
_SKY_NOTICE_VERSION = 2
_SKY_NOTICE_KEY = "sky_models_ack_version"


class Step3MaskLicenseMixin:
    def confirm_commands(self, commands: list[tuple[str, list[str]]]) -> bool:
        if any(phase.startswith("yolo") for phase, _cmd in commands):
            if self._person_backend_arg() == "yolo_sam":
                if not self._confirm_yolo_sam_license_notice():
                    return False
            else:
                if not self._confirm_sky_license_notice():
                    return False
                if self._uses_sam31_for_primary_mask() and not self._ensure_sam31_checkpoint_available():
                    return False
        return True

    def _uses_sam31_for_primary_mask(self) -> bool:
        return self._person_uses_sam31() or self._sky_backend_arg() == "sam31"

    def _ensure_sam31_checkpoint_available(self) -> bool:
        return ensure_sam31_checkpoint_available(
            self,
            self._sam31_checkpoint_path(),
            on_available=self._refresh_sam31_backend_availability,
        )

    def _refresh_sam31_backend_availability(self) -> None:
        self._update_person_backend_availability()
        self._update_sky_backend_availability()

    def _confirm_yolo_sam_license_notice(self) -> bool:
        if self._yolo_sam_notice_acknowledged():
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(i18n.t("YOLO_SAM_LICENSE_NOTICE_TITLE"))
        box.setText(i18n.t("YOLO_SAM_LICENSE_NOTICE_BODY"))
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)

        remember_cb = QCheckBox(i18n.t("YOLO_SAM_LICENSE_NOTICE_DONT_SHOW_AGAIN"))
        remember_cb.setChecked(True)
        box.setCheckBox(remember_cb)

        continue_btn = box.addButton(
            i18n.t("YOLO_SAM_LICENSE_NOTICE_CONTINUE"),
            QMessageBox.AcceptRole,
        )
        box.addButton(i18n.CANCEL, QMessageBox.RejectRole)
        box.setDefaultButton(continue_btn)

        box.exec()
        if box.clickedButton() != continue_btn:
            return False
        if remember_cb.isChecked():
            self._set_yolo_sam_notice_acknowledged()
        return True

    def _yolo_sam_notice_acknowledged(self) -> bool:
        settings = load_user_settings_section(_LICENSE_NOTICE_SECTION)
        try:
            version = int(settings.get(_YOLO_SAM_NOTICE_KEY, 0))
        except (TypeError, ValueError):
            version = 0
        return version >= _YOLO_SAM_NOTICE_VERSION

    @staticmethod
    def _set_yolo_sam_notice_acknowledged() -> None:
        update_user_settings_section(
            _LICENSE_NOTICE_SECTION,
            {_YOLO_SAM_NOTICE_KEY: _YOLO_SAM_NOTICE_VERSION},
        )

    def _confirm_sky_license_notice(self) -> bool:
        if self._sky_notice_acknowledged():
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(i18n.t("SKY_LICENSE_NOTICE_TITLE"))
        box.setText(i18n.t("SKY_LICENSE_NOTICE_BODY"))
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)

        remember_cb = QCheckBox(i18n.t("YOLO_SAM_LICENSE_NOTICE_DONT_SHOW_AGAIN"))
        remember_cb.setChecked(True)
        box.setCheckBox(remember_cb)

        continue_btn = box.addButton(
            i18n.t("YOLO_SAM_LICENSE_NOTICE_CONTINUE"),
            QMessageBox.AcceptRole,
        )
        box.addButton(i18n.CANCEL, QMessageBox.RejectRole)
        box.setDefaultButton(continue_btn)

        box.exec()
        if box.clickedButton() != continue_btn:
            return False
        if remember_cb.isChecked():
            self._set_sky_notice_acknowledged()
        return True

    def _sky_notice_acknowledged(self) -> bool:
        settings = load_user_settings_section(_LICENSE_NOTICE_SECTION)
        try:
            version = int(settings.get(_SKY_NOTICE_KEY, 0))
        except (TypeError, ValueError):
            version = 0
        return version >= _SKY_NOTICE_VERSION

    @staticmethod
    def _set_sky_notice_acknowledged() -> None:
        update_user_settings_section(
            _LICENSE_NOTICE_SECTION,
            {_SKY_NOTICE_KEY: _SKY_NOTICE_VERSION},
        )
