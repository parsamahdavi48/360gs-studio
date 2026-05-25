"""Data quality diagnostics for read-only scene preview inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.scene_import_contracts import IMAGE_EXTS
from core.scene_preview import ScenePreviewCamera, ScenePreviewDataset
from core.scene_preview_cubemap import split_cubemap_face

_SIDE_FACES = ("px", "nx", "pz", "nz")
_PY_NY_FACES = ("py", "ny")
_TOP_BOTTOM_FACES = ("top", "bottom")
_FACE_ORDER = ("px", "nx", "pz", "nz", "py", "ny", "top", "bottom")


@dataclass(frozen=True)
class CubemapGroupDiagnostic:
    name: str
    present_faces: tuple[str, ...]
    expected_faces: tuple[str, ...]

    @property
    def missing_faces(self) -> tuple[str, ...]:
        present = set(self.present_faces)
        return tuple(face for face in self.expected_faces if face not in present)

    @property
    def is_complete(self) -> bool:
        return not self.missing_faces


@dataclass(frozen=True)
class ScenePreviewDiagnostics:
    image_count: int = 0
    camera_image_count: int = 0
    images_without_camera: tuple[Path, ...] = ()
    camera_images_missing_on_disk: tuple[str, ...] = ()
    cubemap_groups: tuple[CubemapGroupDiagnostic, ...] = ()

    @property
    def incomplete_cubemap_groups(self) -> tuple[CubemapGroupDiagnostic, ...]:
        return tuple(group for group in self.cubemap_groups if not group.is_complete)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.images_without_camera
            or self.camera_images_missing_on_disk
            or self.incomplete_cubemap_groups
        )

    def cubemap_group_for_camera(self, camera: ScenePreviewCamera | None) -> CubemapGroupDiagnostic | None:
        if camera is None:
            return None
        names = [camera.label, camera.camera_id]
        if camera.image_path is not None:
            names.extend([camera.image_path.name, str(camera.image_path)])
        for value in names:
            parsed = split_cubemap_face(str(value or ""))
            if parsed is None:
                continue
            prefix, _face = parsed
            return next((group for group in self.cubemap_groups if group.name == prefix), None)
        return None


def analyze_scene_preview_dataset(dataset: ScenePreviewDataset) -> ScenePreviewDiagnostics:
    camera_names: list[str] = []
    missing_on_disk: list[str] = []
    for camera in dataset.cameras:
        if camera.image_path is not None:
            camera_names.append(str(camera.image_path))
            if not camera.image_path.is_file():
                missing_on_disk.append(str(camera.image_path))
        elif camera.label:
            camera_names.append(camera.label)

    additional_roots = _additional_image_roots_for_dataset(dataset)
    return analyze_named_camera_images(
        camera_names,
        dataset.image_root,
        additional_image_roots=additional_roots,
        camera_images_missing_on_disk=tuple(missing_on_disk),
        camera_image_count=len({name.casefold() for name in camera_names if name}),
    )


def analyze_named_camera_images(
    camera_names: Iterable[str],
    image_root: Path | None,
    *,
    additional_image_roots: Iterable[Path] = (),
    camera_images_missing_on_disk: tuple[str, ...] | None = None,
    camera_image_count: int | None = None,
) -> ScenePreviewDiagnostics:
    names = tuple(str(name or "") for name in camera_names if str(name or "").strip())
    image_roots = _existing_roots(image_root, additional_image_roots)
    camera_keys: set[str] = set()
    for name in names:
        camera_keys.update(_image_keys_for_roots(Path(name), image_roots))

    image_files = tuple(
        sorted(
            (path for root in image_roots for path in _image_files(root)),
            key=lambda item: str(item).casefold(),
        )
    )
    image_keys: set[str] = set()
    for path in image_files:
        image_keys.update(_image_keys_for_roots(path, image_roots))
    if image_roots and camera_images_missing_on_disk is None:
        camera_images_missing_on_disk = tuple(
            name for name in names if not image_keys.intersection(_image_keys_for_roots(Path(name), image_roots))
        )
    images_without_camera = tuple(
        path for path in image_files if not camera_keys.intersection(_image_keys_for_roots(path, image_roots))
    )
    return ScenePreviewDiagnostics(
        image_count=len(image_files),
        camera_image_count=len({name.casefold() for name in names}) if camera_image_count is None else int(camera_image_count),
        images_without_camera=images_without_camera,
        camera_images_missing_on_disk=tuple(camera_images_missing_on_disk or ()),
        cubemap_groups=_cubemap_groups_from_names(names),
    )


def _additional_image_roots_for_dataset(dataset: ScenePreviewDataset) -> tuple[Path, ...]:
    if dataset.source_kind != "realityscan" or dataset.image_root is None:
        return ()
    from core.realityscan_to_transforms import REALITYSCAN_IMAGE_DIR_NAMES, related_realityscan_asset_roots

    primary = Path(dataset.image_root).resolve(strict=False).as_posix().casefold()
    return tuple(
        root
        for root in related_realityscan_asset_roots(Path(dataset.image_root), REALITYSCAN_IMAGE_DIR_NAMES)
        if root.resolve(strict=False).as_posix().casefold() != primary
    )


def _existing_roots(image_root: Path | None, additional_image_roots: Iterable[Path]) -> tuple[Path, ...]:
    roots: list[Path] = []
    if image_root is not None:
        roots.append(Path(image_root))
    roots.extend(Path(root) for root in additional_image_roots)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        key = root.resolve(strict=False).as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return tuple(deduped)


def _image_files(root: Path | None) -> tuple[Path, ...]:
    if root is None or not Path(root).is_dir():
        return ()
    files: list[Path] = []
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if _is_mask_layer_name(path.name):
            continue
        files.append(path)
    return tuple(sorted(files, key=lambda item: str(item).casefold()))


def _is_mask_layer_name(name: str) -> bool:
    lower = str(name or "").casefold()
    return lower.endswith(".mask.png")


def _image_keys(path: Path, image_root: Path | None) -> set[str]:
    keys: set[str] = set()
    raw = str(path)
    if raw:
        keys.add(_normalize_key(raw))
    if path.name:
        keys.add(_normalize_key(path.name))
    if image_root is not None:
        try:
            keys.add(_normalize_key(str(path.resolve().relative_to(Path(image_root).resolve()))))
        except Exception:
            pass
    return {key for key in keys if key}


def _image_keys_for_roots(path: Path, image_roots: tuple[Path, ...]) -> set[str]:
    keys: set[str] = set()
    if not image_roots:
        keys.update(_image_keys(path, None))
    for root in image_roots:
        keys.update(_image_keys(path, root))
    return keys


def _normalize_key(value: str) -> str:
    return str(value or "").replace("\\", "/").casefold()


def _cubemap_groups_from_names(names: Iterable[str]) -> tuple[CubemapGroupDiagnostic, ...]:
    grouped: dict[str, set[str]] = {}
    for name in names:
        parsed = split_cubemap_face(str(name or ""))
        if parsed is None:
            continue
        prefix, face = parsed
        grouped.setdefault(prefix, set()).add(face)
    diagnostics = []
    for name, faces in sorted(grouped.items()):
        expected = _expected_faces(faces)
        diagnostics.append(
            CubemapGroupDiagnostic(
                name=name,
                present_faces=tuple(face for face in _FACE_ORDER if face in faces),
                expected_faces=expected,
            )
        )
    return tuple(diagnostics)


def _expected_faces(faces: set[str]) -> tuple[str, ...]:
    vertical = _TOP_BOTTOM_FACES if any(face in faces for face in _TOP_BOTTOM_FACES) else _PY_NY_FACES
    return (*_SIDE_FACES, *vertical)
