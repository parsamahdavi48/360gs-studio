from __future__ import annotations

from pathlib import Path

from core.artifact_registry import load_artifacts, make_artifact_record, upsert_artifact
from core.job_spec import new_job_spec, read_job_spec, write_job_spec
from core.scene_layout import dataset_artifacts_path, jobs_dir, sfm_artifacts_path
from core.scene_project import load_json


def test_artifact_registry_writes_scene_relative_paths(tmp_path: Path) -> None:
    scene = tmp_path
    root = scene / "output" / "metashape_cubemap"
    record = make_artifact_record(
        scene,
        artifact_id="sfm_a",
        kind="metashape_xml_ply",
        root=root,
        files={"xml": scene / "cameras.xml", "ply": scene / "pointcloud.ply"},
        settings={"profile": "test"},
        warnings=["sample warning"],
    )

    upsert_artifact(scene, "sfm", record)

    assert sfm_artifacts_path(scene).is_file()
    loaded = load_artifacts(scene, "sfm")
    assert loaded[0].root == "output/metashape_cubemap"
    assert loaded[0].files == {"xml": "cameras.xml", "ply": "pointcloud.ply"}
    assert loaded[0].settings == {"profile": "test"}
    assert loaded[0].warnings == ("sample warning",)


def test_artifact_registry_upserts_by_id(tmp_path: Path) -> None:
    first = make_artifact_record(tmp_path, artifact_id="dataset_a", kind="nerf_json_ply", root="output/a")
    second = make_artifact_record(tmp_path, artifact_id="dataset_a", kind="colmap_dataset", root="output/b")

    upsert_artifact(tmp_path, "dataset", first)
    upsert_artifact(tmp_path, "dataset", second)

    loaded = load_artifacts(tmp_path, "dataset")
    assert len(loaded) == 1
    assert loaded[0].kind == "colmap_dataset"
    assert load_json(dataset_artifacts_path(tmp_path))["schema_version"] == 1


def test_job_spec_round_trips_under_scene_jobs_dir(tmp_path: Path) -> None:
    spec = new_job_spec("dataset_export", {"source": "sfm_a"})

    path = write_job_spec(tmp_path, spec)

    assert path.parent == jobs_dir(tmp_path)
    loaded = read_job_spec(path)
    assert loaded.id == spec.id
    assert loaded.kind == "dataset_export"
    assert loaded.params == {"source": "sfm_a"}
