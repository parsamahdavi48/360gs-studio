from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.sfm_preflight import preflight_spheresfm


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def test_spheresfm_preflight_accepts_same_resolution_erp(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg", (64, 32))
    _write_image(tmp_path / "images" / "b.jpg", (64, 32))

    result = preflight_spheresfm(tmp_path)

    assert result.ok
    assert result.issues == ()


def test_spheresfm_preflight_rejects_mixed_erp_resolutions(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg", (64, 32))
    _write_image(tmp_path / "images" / "b.jpg", (128, 64))

    result = preflight_spheresfm(tmp_path)

    assert not result.ok
    assert [issue.code for issue in result.issues] == ["requires_single_resolution"]


def test_spheresfm_preflight_rejects_normal_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg", (64, 32))
    _write_image(tmp_path / "images" / "normal.jpg", (40, 30))

    result = preflight_spheresfm(tmp_path)

    assert not result.ok
    assert [issue.code for issue in result.issues] == ["requires_equirectangular_only", "requires_single_resolution"]
