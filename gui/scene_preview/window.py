"""Standalone read-only scene preview window."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.image_io import imread_unicode
from core.scene_preview import (
    ScenePreviewCamera,
    ScenePreviewDataset,
    load_colmap_preview_dataset,
    load_metashape_preview_dataset,
    load_ply_preview_pointcloud,
    load_realityscan_preview_dataset,
    load_transforms_preview_dataset,
    transform_preview_dataset,
)
from core.scene_preview_cubemap import (
    cubemap_frame_groups_from_preview_cameras,
    face_view_params,
    render_cubemap_equirect,
    virtual_camera_direction,
)
from core.scene_preview_sources import ScenePreviewCandidate, discover_scene_preview_candidates
from gui import i18n
from gui.common.perspective_preview import PerspectiveParams
from gui.scene_preview.camera_image_view import SceneCameraImageView
from gui.scene_preview.pointcloud_view import ScenePointCloudView


class ScenePreviewWidget(QWidget):
    """Embeddable read-only preview for SfM/Step 4 datasets."""

    def __init__(
        self,
        scene_dir: Path | None = None,
        parent: QWidget | None = None,
        *,
        show_scene_controls: bool = True,
    ) -> None:
        super().__init__(parent)
        self._scene_dir: Path | None = None
        self._show_scene_controls = bool(show_scene_controls)
        self._candidates: tuple[ScenePreviewCandidate, ...] = ()
        self._current_candidate: ScenePreviewCandidate | None = None
        self._dataset: ScenePreviewDataset | None = None
        self._selected_camera_id = ""
        self._cubemap_faces_by_camera_key: dict[str, tuple[Any, str]] = {}
        self._cubemap_face_camera_keys: set[str] = set()
        self._cubemap_front_camera_id_by_camera_id: dict[str, str] = {}
        self._cubemap_equirect_cache: dict[str, Any] = {}
        self._cubemap_mask_cache: dict[str, Any] = {}
        self._cubemap_image_cache: dict[Path, Any] = {}
        self._mask_image_cache: dict[Path, Any] = {}
        self._build_ui()
        self.set_scene_dir(scene_dir)

    def set_scene_dir(self, scene_dir: Path | None, *, refresh: bool = True) -> None:
        self._scene_dir = Path(scene_dir) if scene_dir else None
        if refresh:
            self._refresh_candidates()
        else:
            self._clear_dataset()
            self.candidate_list.clear()
            if self._scene_dir is None:
                self.scene_label.setText(i18n.t("SCENE_PREVIEW_NO_SCENE"))
                self.scene_label.setToolTip("")
                self.summary_text.setPlainText(i18n.t("SCENE_PREVIEW_NO_SCENE"))
            else:
                self.scene_label.setText(str(self._scene_dir))
                self.scene_label.setToolTip(str(self._scene_dir))
                self.summary_text.setPlainText(i18n.t("SCENE_PREVIEW_NO_CANDIDATES"))
            self._candidates = ()

    def refresh(self) -> None:
        self._refresh_candidates()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.scene_label = QLabel()
        self.scene_label.setObjectName("pathSummaryValue")
        self.scene_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.open_scene_btn = QPushButton(i18n.t("SCENE_PREVIEW_SELECT_SCENE"))
        self.open_scene_btn.setToolTip(i18n.tip("SCENE_PREVIEW_SELECT_SCENE"))
        self.open_scene_btn.clicked.connect(self._select_scene_dir)
        self.refresh_btn = QPushButton(i18n.t("SCENE_PREVIEW_REFRESH"))
        self.refresh_btn.setToolTip(i18n.tip("SCENE_PREVIEW_REFRESH"))
        self.refresh_btn.clicked.connect(self._refresh_candidates)
        if self._show_scene_controls:
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(8)
            header.addWidget(self.scene_label, stretch=1)
            header.addWidget(self.refresh_btn)
            header.addWidget(self.open_scene_btn)
            root.addLayout(header)
        else:
            self.scene_label.hide()
            self.open_scene_btn.hide()
            self.refresh_btn.hide()

        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.setSpacing(8)
        camera_row.addWidget(QLabel(i18n.t("SCENE_PREVIEW_SELECTED_CAMERA")))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_combo_changed)
        camera_row.addWidget(self.camera_combo, stretch=1)
        root.addLayout(camera_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        candidate_title = QLabel(i18n.t("SCENE_PREVIEW_CANDIDATES"))
        candidate_title.setObjectName("sectionTitle")
        self.candidate_list = QListWidget()
        self.candidate_list.currentRowChanged.connect(self._load_selected_candidate)
        left_layout.addWidget(candidate_title)
        left_layout.addWidget(self.candidate_list, stretch=1)

        right = QSplitter(Qt.Vertical)
        right.setChildrenCollapsible(False)
        view_splitter = QSplitter(Qt.Horizontal)
        view_splitter.setChildrenCollapsible(False)
        self.pointcloud_view = ScenePointCloudView()
        self.pointcloud_view.camera_selected.connect(self._select_camera_from_pointcloud)
        self.camera_image_view = SceneCameraImageView()
        self.camera_image_view.perspective_params_changed.connect(self._on_camera_image_view_params_changed)
        view_splitter.addWidget(self.pointcloud_view)
        view_splitter.addWidget(self.camera_image_view)
        view_splitter.setStretchFactor(0, 1)
        view_splitter.setStretchFactor(1, 1)
        view_splitter.setSizes([560, 560])
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setLineWrapMode(QTextEdit.NoWrap)
        right.addWidget(view_splitter)
        right.addWidget(self.summary_text)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        right.setSizes([430, 160])

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 700])
        root.addWidget(splitter, stretch=1)

    def _select_scene_dir(self) -> None:
        start = str(self._scene_dir or Path.cwd())
        selected = QFileDialog.getExistingDirectory(self, i18n.t("SCENE_PREVIEW_SELECT_SCENE"), start)
        if selected:
            self.set_scene_dir(Path(selected))

    def _refresh_candidates(self) -> None:
        self.candidate_list.clear()
        if self._scene_dir is None:
            self.scene_label.setText(i18n.t("SCENE_PREVIEW_NO_SCENE"))
            self._clear_dataset()
            self.summary_text.setPlainText(i18n.t("SCENE_PREVIEW_NO_SCENE"))
            self._candidates = ()
            return

        self.scene_label.setText(str(self._scene_dir))
        self.scene_label.setToolTip(str(self._scene_dir))
        self._candidates = discover_scene_preview_candidates(self._scene_dir)
        for candidate in self._candidates:
            item = QListWidgetItem(candidate.label)
            item.setToolTip(str(candidate.path))
            self.candidate_list.addItem(item)

        if not self._candidates:
            self._clear_dataset()
            self.summary_text.setPlainText(i18n.t("SCENE_PREVIEW_NO_CANDIDATES"))
            return
        self.candidate_list.setCurrentRow(0)

    def _load_selected_candidate(self, row: int) -> None:
        if row < 0 or row >= len(self._candidates):
            return
        candidate = self._candidates[row]
        try:
            dataset = _load_candidate(candidate)
        except Exception as exc:
            self._current_candidate = None
            self._clear_dataset()
            self.summary_text.setPlainText(
                "\n".join(
                    [
                        i18n.t("SCENE_PREVIEW_LOAD_FAILED"),
                        "",
                        f"{i18n.t('SCENE_PREVIEW_SOURCE')}: {candidate.label}",
                        f"{i18n.t('SCENE_PREVIEW_PATH')}: {candidate.path}",
                        f"{i18n.t('SCENE_PREVIEW_ERROR')}: {exc}",
                    ]
                )
            )
            return
        self._current_candidate = candidate
        self._set_dataset(dataset)

    def _clear_dataset(self) -> None:
        self._current_candidate = None
        self._dataset = None
        self._selected_camera_id = ""
        self._cubemap_faces_by_camera_key = {}
        self._cubemap_face_camera_keys = set()
        self._cubemap_front_camera_id_by_camera_id = {}
        self._cubemap_equirect_cache = {}
        self._cubemap_mask_cache = {}
        self._cubemap_image_cache = {}
        self._mask_image_cache = {}
        self.pointcloud_view.set_dataset(None)
        self.pointcloud_view.set_selected_camera_id("")
        self.pointcloud_view.set_selected_view_ray(None)
        self.camera_image_view.set_camera(None)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.blockSignals(False)

    def _set_dataset(self, dataset: ScenePreviewDataset) -> None:
        self._dataset = dataset
        self._build_cubemap_lookup(dataset)
        self.pointcloud_view.set_dataset(dataset)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for camera in dataset.cameras:
            self.camera_combo.addItem(camera.label, camera.camera_id)
        self.camera_combo.blockSignals(False)
        first_id = self._initial_camera_id(dataset)
        self._select_camera_id(first_id)

    def _on_camera_combo_changed(self, _index: int) -> None:
        self._select_camera_id(str(self.camera_combo.currentData() or ""))

    def _select_camera_id(self, camera_id: str) -> None:
        camera_id = str(camera_id or "")
        self._selected_camera_id = camera_id
        self.pointcloud_view.set_selected_camera_id(camera_id)
        camera = self._camera_by_id(camera_id)
        reconstructed = self._reconstructed_cubemap_preview(camera)
        if reconstructed is None:
            if self._is_cubemap_face_camera(camera):
                self.camera_image_view.set_message(i18n.t("SCENE_PREVIEW_CUBEMAP_REBUILD_FAILED"))
            else:
                self.camera_image_view.set_camera(camera, mask=self._mask_for_camera(camera))
        else:
            image, params, mask = reconstructed
            self.camera_image_view.set_camera(camera, reconstructed_bgr=image, perspective_params=params, mask=mask)
        self._sync_selected_view_ray(camera)
        combo_index = self.camera_combo.findData(camera_id)
        if combo_index >= 0 and self.camera_combo.currentIndex() != combo_index:
            self.camera_combo.blockSignals(True)
            self.camera_combo.setCurrentIndex(combo_index)
            self.camera_combo.blockSignals(False)
        self._update_summary_text()

    def _select_camera_from_pointcloud(self, camera_id: str) -> None:
        camera_id = str(camera_id or "")
        self._select_camera_id(camera_id)

    def _initial_camera_id(self, dataset: ScenePreviewDataset) -> str:
        if not dataset.cameras:
            return ""
        first_id = dataset.cameras[0].camera_id
        return self._cubemap_front_camera_id_by_camera_id.get(first_id, first_id)

    def _on_camera_image_view_params_changed(self, _params: object) -> None:
        self._sync_selected_view_ray(self._camera_by_id(self._selected_camera_id))
        self._update_summary_text()

    def _sync_selected_view_ray(self, camera: ScenePreviewCamera | None) -> None:
        if camera is None:
            self.pointcloud_view.set_selected_view_ray(None)
            return
        self.pointcloud_view.set_selected_view_ray(self._selected_view_direction_for_camera(camera))

    def _view_ray_basis_camera(self, camera: ScenePreviewCamera) -> ScenePreviewCamera:
        front_id = self._cubemap_front_camera_id_by_camera_id.get(camera.camera_id, "")
        return self._camera_by_id(front_id) or camera

    def _camera_by_id(self, camera_id: str) -> ScenePreviewCamera | None:
        if self._dataset is None or not camera_id:
            return None
        return next((camera for camera in self._dataset.cameras if camera.camera_id == camera_id), None)

    def _update_summary_text(self) -> None:
        if self._dataset is None:
            return
        camera = self._camera_by_id(self._selected_camera_id)
        view_direction = self._selected_view_direction_for_camera(camera)
        self.summary_text.setPlainText(
            _format_dataset_summary(
                self._current_candidate,
                self._dataset,
                selected_camera=camera,
                selected_camera_index=self._camera_index(camera),
                selected_view_direction=view_direction,
                cubemap_face=self._cubemap_face_for_camera(camera),
                mask_available=self._mask_available_for_camera(camera),
            )
        )

    def _selected_view_direction_for_camera(self, camera: ScenePreviewCamera | None) -> np.ndarray | None:
        if camera is None:
            return None
        params = self.camera_image_view.perspective_params()
        match = self._cubemap_match_for_camera(camera)
        if match is not None and params is not None:
            group, _face = match
            return virtual_camera_direction(
                group,
                yaw_deg=params.yaw_deg,
                pitch_deg=params.pitch_deg,
                roll_deg=params.roll_deg,
            )
        return _camera_view_direction(camera, params, basis_camera=self._view_ray_basis_camera(camera))

    def _camera_index(self, camera: ScenePreviewCamera | None) -> int:
        if self._dataset is None or camera is None:
            return -1
        return next((index for index, item in enumerate(self._dataset.cameras) if item.camera_id == camera.camera_id), -1)

    def _cubemap_face_for_camera(self, camera: ScenePreviewCamera | None) -> str | None:
        match = self._cubemap_match_for_camera(camera)
        if match is not None:
            return str(match[1])
        if camera is None:
            return None
        return _cubemap_face_for_camera(camera)

    def _cubemap_match_for_camera(self, camera: ScenePreviewCamera | None) -> tuple[Any, str] | None:
        if camera is None:
            return None
        return next(
            (
                self._cubemap_faces_by_camera_key[key]
                for key in _camera_lookup_keys(camera)
                if key in self._cubemap_faces_by_camera_key
            ),
            None,
        )

    def _mask_available_for_camera(self, camera: ScenePreviewCamera | None) -> bool | None:
        if self._dataset is None or camera is None or camera.image_path is None:
            return None
        if self._dataset.mask_root is None:
            return None
        return _resolve_preview_mask_path(camera.image_path, self._dataset.mask_root, self._dataset.image_root) is not None

    def _build_cubemap_lookup(self, dataset: ScenePreviewDataset) -> None:
        self._cubemap_faces_by_camera_key = {}
        self._cubemap_face_camera_keys = set()
        self._cubemap_front_camera_id_by_camera_id = {}
        self._cubemap_equirect_cache = {}
        self._cubemap_mask_cache = {}
        self._cubemap_image_cache = {}
        self._mask_image_cache = {}
        groups = cubemap_frame_groups_from_preview_cameras(dataset.cameras)
        frame_lookup: dict[str, tuple[Any, str]] = {}
        for group in groups:
            for face, frame in group.frames_by_face.items():
                for key in _frame_lookup_keys(frame):
                    frame_lookup[key] = (group, face)
        for camera in dataset.cameras:
            if _cubemap_face_for_camera(camera) is not None:
                self._cubemap_face_camera_keys.update(_camera_lookup_keys(camera))
            match = next((frame_lookup[key] for key in _camera_lookup_keys(camera) if key in frame_lookup), None)
            if match is None:
                continue
            for key in _camera_lookup_keys(camera):
                self._cubemap_faces_by_camera_key[key] = match
        cameras_by_group: dict[str, dict[str, ScenePreviewCamera]] = {}
        for camera in dataset.cameras:
            match = next((self._cubemap_faces_by_camera_key[key] for key in _camera_lookup_keys(camera) if key in self._cubemap_faces_by_camera_key), None)
            if match is None:
                continue
            group, face = match
            cameras_by_group.setdefault(_cubemap_group_key(group), {})[face] = camera
        for cameras_by_face in cameras_by_group.values():
            front = _front_camera_for_cubemap_group(cameras_by_face)
            if front is None:
                continue
            for camera in cameras_by_face.values():
                self._cubemap_front_camera_id_by_camera_id[camera.camera_id] = front.camera_id

    def _is_cubemap_face_camera(self, camera: ScenePreviewCamera | None) -> bool:
        if camera is None:
            return False
        return any(key in self._cubemap_face_camera_keys for key in _camera_lookup_keys(camera))

    def _reconstructed_cubemap_preview(
        self,
        camera: ScenePreviewCamera | None,
    ) -> tuple[Any, PerspectiveParams, np.ndarray | None] | None:
        if camera is None:
            return None
        match = self._cubemap_match_for_camera(camera)
        if match is None:
            return None
        group, face = match
        try:
            cache_key = str(getattr(group, "name", "")) or camera.camera_id
            image = self._cubemap_equirect_cache.get(cache_key)
            if image is None:
                image = render_cubemap_equirect(group, image_cache=self._cubemap_image_cache)
                self._cubemap_equirect_cache[cache_key] = image
            mask = self._reconstructed_cubemap_mask(group, cache_key)
            params = face_view_params(group, face, fov_deg=90.0)
        except Exception:
            return None
        if params is None:
            return None
        yaw, pitch, fov = params
        return image, PerspectiveParams(yaw_deg=float(yaw), pitch_deg=float(pitch), fov_deg=float(fov)), mask

    def _mask_for_camera(self, camera: ScenePreviewCamera | None) -> np.ndarray | None:
        if self._dataset is None or camera is None or camera.image_path is None:
            return None
        return _load_preview_mask(
            camera.image_path,
            self._dataset.mask_root,
            self._dataset.image_root,
            self._mask_image_cache,
            target_size=(camera.width, camera.height),
        )

    def _reconstructed_cubemap_mask(self, group: Any, cache_key: str) -> np.ndarray | None:
        if self._dataset is None or self._dataset.mask_root is None:
            return None
        cached = self._cubemap_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        face_cache: dict[Path, np.ndarray] = {}
        found_mask = False
        for frame in getattr(group, "frames", ()):
            raw_image_path = getattr(frame, "image_path", None)
            if raw_image_path is None:
                continue
            image_path = Path(raw_image_path)
            width = int(getattr(frame, "width", 0) or 0)
            height = int(getattr(frame, "height", 0) or 0)
            mask = _load_preview_mask(
                image_path,
                self._dataset.mask_root,
                self._dataset.image_root,
                self._mask_image_cache,
                target_size=(width, height),
            )
            if mask is None:
                if width <= 0 or height <= 0:
                    continue
                mask = np.full((height, width), 255, dtype=np.uint8)
            else:
                found_mask = True
            face_cache[image_path] = mask
        if not found_mask:
            return None
        rendered = render_cubemap_equirect(group, image_cache=face_cache)
        if rendered.ndim == 3:
            rendered = cv2.cvtColor(rendered[:, :, :3], cv2.COLOR_BGR2GRAY)
        rendered = np.ascontiguousarray(rendered.astype(np.uint8, copy=False))
        self._cubemap_mask_cache[cache_key] = rendered
        return rendered


def _load_candidate(candidate: ScenePreviewCandidate) -> ScenePreviewDataset:
    pointcloud = (
        load_ply_preview_pointcloud(candidate.pointcloud_path)
        if candidate.pointcloud_path is not None
        else None
    )
    if candidate.kind == "output":
        dataset = load_transforms_preview_dataset(
            candidate.path,
            image_root=candidate.image_root,
            mask_root=candidate.mask_root,
            pointcloud=pointcloud,
        )
    elif candidate.kind == "metashape":
        dataset = load_metashape_preview_dataset(
            candidate.path,
            images_dir=candidate.image_root,
            masks_dir=candidate.mask_root,
            pointcloud=pointcloud,
        )
    elif candidate.kind == "realityscan":
        dataset = load_realityscan_preview_dataset(
            candidate.path,
            images_dir=candidate.image_root,
            masks_dir=candidate.mask_root,
            pointcloud=pointcloud,
        )
    else:
        dataset = load_colmap_preview_dataset(
            candidate.path,
            images_dir=candidate.image_root,
            masks_dir=candidate.mask_root,
            pointcloud=pointcloud,
            opengl_camera=candidate.colmap_opengl_camera,
        )
    if candidate.display_transform is None:
        return dataset
    return transform_preview_dataset(
        dataset,
        camera_matrix=candidate.display_transform.camera_matrix,
        pointcloud_matrix=candidate.display_transform.pointcloud_matrix,
        coordinate_note=candidate.display_transform.note,
    )


def _load_preview_mask(
    image_path: Path,
    mask_root: Path | None,
    image_root: Path | None,
    cache: dict[Path, np.ndarray],
    *,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray | None:
    mask_path = _resolve_preview_mask_path(image_path, mask_root, image_root)
    if mask_path is None:
        return None
    cached = cache.get(mask_path)
    if cached is None:
        cached = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if cached is None:
            return None
        cache[mask_path] = cached
    mask = cached
    if target_size is not None:
        width, height = int(target_size[0]), int(target_size[1])
        if width > 0 and height > 0 and mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(mask.astype(np.uint8, copy=True))


def _resolve_preview_mask_path(
    image_path: Path,
    mask_root: Path | None,
    image_root: Path | None,
) -> Path | None:
    if mask_root is None or not mask_root.is_dir():
        return None
    for candidate in _preview_mask_candidates(Path(image_path), Path(mask_root), image_root):
        if candidate.is_file():
            return candidate
    return None


def _preview_mask_candidates(image_path: Path, mask_root: Path, image_root: Path | None) -> tuple[Path, ...]:
    variants: list[Path] = []
    rel = _relative_image_path(image_path, image_root)
    if rel is not None:
        variants.append(rel)
        if rel.parts and rel.parts[0].lower() == "images" and len(rel.parts) > 1:
            variants.append(Path(*rel.parts[1:]))
    if image_path.name:
        variants.append(Path(image_path.name))

    candidates: list[Path] = []
    for variant in variants:
        if variant.is_absolute():
            continue
        candidates.extend(
            [
                mask_root / variant,
                (mask_root / variant).with_suffix(".png"),
                mask_root / variant.parent / f"{variant.name}.png",
                mask_root / variant.parent / f"{variant.stem}.png",
            ]
        )

    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return tuple(result)


def _relative_image_path(image_path: Path, image_root: Path | None) -> Path | None:
    if image_root is not None:
        try:
            return image_path.resolve().relative_to(Path(image_root).resolve())
        except Exception:
            pass
    if not image_path.is_absolute() and image_path.parts:
        return image_path
    return None


def _format_dataset_summary(
    candidate: ScenePreviewCandidate | None,
    dataset: ScenePreviewDataset,
    *,
    selected_camera: ScenePreviewCamera | None = None,
    selected_camera_index: int = -1,
    selected_view_direction: np.ndarray | None = None,
    cubemap_face: str | None = None,
    mask_available: bool | None = None,
) -> str:
    source_label = candidate.label if candidate is not None else dataset.source_kind
    source_path = candidate.path if candidate is not None else dataset.source_path
    lines = [
        i18n.t("SCENE_PREVIEW_DATASET_SUMMARY"),
        f"{i18n.t('SCENE_PREVIEW_SOURCE')}: {source_label}",
        f"{i18n.t('SCENE_PREVIEW_PATH')}: {source_path}",
        f"{i18n.t('SCENE_PREVIEW_IMAGE_ROOT')}: {dataset.image_root or '-'}",
        f"{i18n.t('SCENE_PREVIEW_CAMERAS')}: {len(dataset.cameras)}",
        f"{i18n.t('SCENE_PREVIEW_POINTS')}: {_point_count(dataset)}",
        f"{i18n.t('SCENE_PREVIEW_COORDINATE')}: {dataset.coordinate_note}",
    ]
    lines.extend(["", i18n.t("SCENE_PREVIEW_SELECTED_CAMERA_SUMMARY")])
    if selected_camera is None:
        lines.append(i18n.t("SCENE_PREVIEW_NO_CAMERA"))
        return "\n".join(lines)

    camera_index_text = "-"
    if selected_camera_index >= 0:
        camera_index_text = f"{selected_camera_index + 1} / {len(dataset.cameras)}"
    lines.extend(
        [
            f"{i18n.t('SCENE_PREVIEW_CAMERA_INDEX')}: {camera_index_text}",
            f"{i18n.t('SCENE_PREVIEW_LABEL')}: {selected_camera.label}",
            f"{i18n.t('SCENE_PREVIEW_PROJECTION')}: {selected_camera.projection}",
            f"{i18n.t('SCENE_PREVIEW_IMAGE_SIZE')}: {_format_image_size(selected_camera)}",
            f"{i18n.t('SCENE_PREVIEW_INTRINSICS')}: {_format_intrinsics(selected_camera)}",
            f"{i18n.t('SCENE_PREVIEW_IMAGE')}: {selected_camera.image_path or '-'}",
            f"{i18n.t('SCENE_PREVIEW_MASK')}: {_format_presence(mask_available)}",
            f"{i18n.t('SCENE_PREVIEW_CUBEMAP_FACE')}: {cubemap_face or '-'}",
            f"{i18n.t('SCENE_PREVIEW_POSITION')}: {_format_vector(selected_camera.position)}",
            f"{i18n.t('SCENE_PREVIEW_FORWARD')}: {_format_vector(selected_camera.forward)}",
            f"{i18n.t('SCENE_PREVIEW_VIEW_DIRECTION')}: {_format_optional_vector(selected_view_direction)}",
            f"{i18n.t('SCENE_PREVIEW_UP')}: {_format_vector(selected_camera.up)}",
            f"{i18n.t('SCENE_PREVIEW_WORLD_UP_ROLL')}: {_format_roll(selected_camera.world_up_roll_radians())}",
        ]
    )
    return "\n".join(lines)


def _point_count(dataset: ScenePreviewDataset) -> str:
    if dataset.pointcloud is None:
        return "-"
    return str(int(dataset.pointcloud.points.shape[0]))


def _format_vector(vector: object) -> str:
    values = [float(value) for value in vector]  # type: ignore[operator]
    return f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f})"


def _format_optional_vector(vector: object) -> str:
    if vector is None:
        return "-"
    return _format_vector(vector)


def _format_image_size(camera: ScenePreviewCamera) -> str:
    if camera.width <= 0 or camera.height <= 0:
        return "-"
    return f"{camera.width} x {camera.height}"


def _format_intrinsics(camera: ScenePreviewCamera) -> str:
    if camera.fl_x is None or camera.fl_y is None or camera.cx is None or camera.cy is None:
        return "-"
    return f"fx={camera.fl_x:.3g}, fy={camera.fl_y:.3g}, cx={camera.cx:.3g}, cy={camera.cy:.3g}"


def _format_presence(value: bool | None) -> str:
    if value is None:
        return "-"
    return i18n.t("SCENE_PREVIEW_PRESENT") if value else i18n.t("SCENE_PREVIEW_MISSING")


def _format_roll(roll_radians: float | None) -> str:
    if roll_radians is None:
        return "-"
    return f"{math.degrees(roll_radians):.2f} deg"


def _camera_view_direction(
    camera: ScenePreviewCamera,
    params: PerspectiveParams | None,
    *,
    basis_camera: ScenePreviewCamera | None = None,
) -> np.ndarray:
    basis = basis_camera or camera
    if params is None:
        return _normalized(camera.forward)
    yaw = math.radians(float(params.yaw_deg))
    pitch = math.radians(float(params.pitch_deg))
    local = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ],
        dtype=np.float64,
    )
    direction = basis.right * local[0] + basis.up * local[1] + basis.forward * local[2]
    return _normalized(direction)


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return value / norm


def _camera_lookup_keys(camera: ScenePreviewCamera) -> tuple[str, ...]:
    keys = {
        _lookup_key(camera.camera_id),
        _lookup_key(camera.label),
    }
    if camera.image_path is not None:
        keys.add(_lookup_key(str(camera.image_path)))
        keys.add(_lookup_key(camera.image_path.name))
    return tuple(key for key in keys if key)


def _cubemap_face_for_camera(camera: ScenePreviewCamera) -> str | None:
    candidates = [camera.label, camera.camera_id]
    if camera.image_path is not None:
        candidates.extend([camera.image_path.name, str(camera.image_path)])
    for value in candidates:
        stem = Path(str(value).replace("\\", "/")).stem.lower()
        for face in ("px", "nx", "pz", "nz", "top", "bottom", "py", "ny"):
            if stem.endswith(f"_{face}"):
                return face
    return None


def _frame_lookup_keys(frame: Any) -> tuple[str, ...]:
    image_path = getattr(frame, "image_path", None)
    file_path = str(getattr(frame, "file_path", "") or "")
    frame_id = str(getattr(frame, "frame_id", "") or "")
    keys = {
        _lookup_key(frame_id),
        _lookup_key(file_path),
    }
    if image_path is not None:
        path = Path(image_path)
        keys.add(_lookup_key(str(path)))
        keys.add(_lookup_key(path.name))
    return tuple(key for key in keys if key)


def _cubemap_group_key(group: Any) -> str:
    name = str(getattr(group, "name", "") or "")
    index = str(getattr(group, "group_index", "") or "")
    return f"{index}:{name}"


def _front_camera_for_cubemap_group(cameras_by_face: dict[str, ScenePreviewCamera]) -> ScenePreviewCamera | None:
    for face in ("pz", "front"):
        camera = cameras_by_face.get(face)
        if camera is not None:
            return camera
    for face in ("px", "nx", "nz", "py", "top", "ny", "bottom"):
        camera = cameras_by_face.get(face)
        if camera is not None:
            return camera
    return next(iter(cameras_by_face.values()), None)


def _lookup_key(value: object) -> str:
    text = str(value or "").replace("\\", "/").strip().lower()
    return text


class ScenePreviewWindow(ScenePreviewWidget):
    """Standalone shell for loading SfM/Step 4 preview datasets."""

    def __init__(self, scene_dir: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(scene_dir=scene_dir, parent=parent, show_scene_controls=True)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(i18n.t("SCENE_PREVIEW_WINDOW_TITLE"))
        self.resize(1280, 720)
