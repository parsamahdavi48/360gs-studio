from __future__ import annotations

from pathlib import Path

from core.artifact_registry import load_artifacts
from core.workflow_artifacts import (
    DATASET_KIND_COLMAP_DATASET,
    DATASET_KIND_LICHTFELD_COLMAP,
    DATASET_KIND_NERF_JSON_PLY,
    SFM_KIND_METASHAPE_XML_PLY,
    detect_dataset_kind,
    latest_dataset_root,
    register_dataset_artifact,
    register_sfm_artifact,
)


def _write_colmap_sparse(root: Path) -> None:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse / name).write_text("", encoding="utf-8")


def test_register_dataset_artifact_detects_nerf_and_files(tmp_path: Path) -> None:
    scene = tmp_path
    root = scene / "output" / "metashape_cubemap"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "transforms.json").write_text("{}", encoding="utf-8")
    (root / "pointcloud.ply").write_text("ply\n", encoding="ascii")

    record = register_dataset_artifact(scene, artifact_id="dataset_a", root=root)

    assert record is not None
    assert record.kind == DATASET_KIND_NERF_JSON_PLY
    assert record.files["transforms_json"] == "output/metashape_cubemap/transforms.json"
    assert record.files["images_dir"] == "output/metashape_cubemap/images"
    assert load_artifacts(scene, "dataset")[0].id == "dataset_a"


def test_register_dataset_artifact_detects_colmap_and_lfs_colmap(tmp_path: Path) -> None:
    colmap_root = tmp_path / "output" / "metashape_colmap"
    lfs_root = tmp_path / "output" / "realityscan" / "lfs_colmap"
    _write_colmap_sparse(colmap_root)
    _write_colmap_sparse(lfs_root)

    assert detect_dataset_kind(colmap_root) == DATASET_KIND_COLMAP_DATASET
    assert detect_dataset_kind(lfs_root) == DATASET_KIND_LICHTFELD_COLMAP


def test_register_sfm_artifact_keeps_existing_files_only(tmp_path: Path) -> None:
    scene = tmp_path
    xml = scene / "cameras.xml"
    xml.write_text("<document/>", encoding="utf-8")

    record = register_sfm_artifact(
        scene,
        artifact_id="sfm_a",
        kind=SFM_KIND_METASHAPE_XML_PLY,
        root=scene,
        files={"xml": xml, "ply": scene / "missing.ply"},
    )

    assert record.files == {"xml": "cameras.xml"}


def test_latest_dataset_root_uses_registered_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "output" / "first"
    second = tmp_path / "output" / "second"
    (first / "transforms.json").parent.mkdir(parents=True, exist_ok=True)
    (first / "transforms.json").write_text("{}", encoding="utf-8")
    (second / "transforms.json").parent.mkdir(parents=True, exist_ok=True)
    (second / "transforms.json").write_text("{}", encoding="utf-8")
    register_dataset_artifact(tmp_path, artifact_id="a", root=first)
    register_dataset_artifact(tmp_path, artifact_id="b", root=second)

    assert latest_dataset_root(tmp_path) == second
