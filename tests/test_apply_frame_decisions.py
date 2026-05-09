"""apply_frame_decisions.py の finalize_in_place + バックアップ機能テスト。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from apply_frame_decisions import (
    backup_images_dir,
    finalize_in_place,
    pending_drop_image_paths,
    untracked_image_paths,
)
from core.scene_layout import (
    extract_sessions_path,
    frame_backups_dir,
    selected_frames_keep_path,
    selected_frames_path,
    source_image_sets_path,
)
from core.scene_project import file_identity, write_json


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _make_scene(tmp_path: Path, num_frames: int = 4, drop_indices: list[int] = None) -> Path:
    """テスト用シーンを生成。images/ に N 個のダミー画像、selected_frames.csv に
    drop_indices で指定したインデックスを drop マークする。"""
    drop_indices = drop_indices or []
    scene = tmp_path / "scene"
    scene.mkdir()
    images = scene / "images"
    images.mkdir()

    rows = []
    for i in range(1, num_frames + 1):
        img = images / f"frame_{i:06d}.jpg"
        img.write_bytes(b"fake image " + str(i).encode())
        rows.append(
            {
                "seq": str(i),
                "original_index": str(i * 10),
                "final_index": str(i * 10),
                "status": "ok",
                "decision": "drop" if i in drop_indices else "keep",
                "output_file": f"images/frame_{i:06d}.jpg",
            }
        )

    _write_csv(selected_frames_path(scene), rows)
    return scene


# =============================================================================
# backup_images_dir
# =============================================================================


def test_backup_images_dir_copies_all_files(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"A")
    (images / "b.jpg").write_bytes(b"B")
    (images / "c.png").write_bytes(b"C")

    backup = tmp_path / "images_backup"
    n = backup_images_dir(images, backup)

    assert n == 3
    assert (backup / "a.jpg").read_bytes() == b"A"
    assert (backup / "b.jpg").read_bytes() == b"B"
    assert (backup / "c.png").read_bytes() == b"C"


def test_backup_images_dir_replaces_existing_backup(tmp_path: Path):
    """既存 backup ディレクトリは丸ごと置き換わる。"""
    images = tmp_path / "images"
    images.mkdir()
    (images / "new.jpg").write_bytes(b"new")

    backup = tmp_path / "images_backup"
    backup.mkdir()
    (backup / "old.jpg").write_bytes(b"old")  # 古い残骸

    backup_images_dir(images, backup)

    # 古いファイルは消えて新しい内容のみ
    assert not (backup / "old.jpg").exists()
    assert (backup / "new.jpg").read_bytes() == b"new"


def test_backup_images_dir_refuses_non_backup_existing_target(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "new.jpg").write_bytes(b"new")

    target = tmp_path / "important"
    target.mkdir()
    (target / "old.jpg").write_bytes(b"old")

    with pytest.raises(RuntimeError, match="does not look like a backup"):
        backup_images_dir(images, target)

    assert (target / "old.jpg").read_bytes() == b"old"


def test_backup_images_dir_refuses_target_inside_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "new.jpg").write_bytes(b"new")

    target = images / "backup"

    with pytest.raises(RuntimeError, match="inside images"):
        backup_images_dir(images, target)


def test_backup_images_dir_handles_missing_source(tmp_path: Path):
    """source が存在しない場合は 0 を返してエラーにならない。"""
    backup = tmp_path / "images_backup"
    n = backup_images_dir(tmp_path / "nonexistent", backup)
    assert n == 0
    assert not backup.exists()


# =============================================================================
# finalize_in_place
# =============================================================================


def test_finalize_in_place_drops_and_preserves_keep_filenames(tmp_path: Path):
    """drop 指定された画像が削除され、keep のファイル名は維持される。"""
    scene = _make_scene(tmp_path, num_frames=4, drop_indices=[2, 4])

    finalize_in_place(scene, "selected_frames.csv")

    images = scene / "images"
    files = sorted(p.name for p in images.glob("*.jpg"))
    assert len(files) == 2
    assert "frame_000001.jpg" in files
    assert "frame_000003.jpg" in files

    with selected_frames_path(scene).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["seq"] for row in rows] == ["1", "2"]
    assert [row["output_file"] for row in rows] == [
        "images/frame_000001.jpg",
        "images/frame_000003.jpg",
    ]


def test_finalize_in_place_with_backup(tmp_path: Path):
    """backup_dir 指定時、events 前に images/ がフルコピーされる。"""
    scene = _make_scene(tmp_path, num_frames=4, drop_indices=[2, 4])
    backup_dir = scene / "images_backup"

    finalize_in_place(scene, "selected_frames.csv", backup_dir=backup_dir)

    # backup には削除前の 4 個全部が保持されている
    backup_files = sorted(p.name for p in backup_dir.glob("*.jpg"))
    assert len(backup_files) == 4
    assert "frame_000002.jpg" in backup_files  # drop 対象もバックアップに含まれる
    assert "frame_000004.jpg" in backup_files

    # 一方 images/ 自体は drop 後の 2 個に減り、keep のファイル名は維持される
    images = scene / "images"
    final_files = sorted(p.name for p in images.glob("*.jpg"))
    assert final_files == ["frame_000001.jpg", "frame_000003.jpg"]


def test_finalize_in_place_without_backup_no_backup_dir(tmp_path: Path):
    """backup_dir=None なら images_backup/ は作られない。"""
    scene = _make_scene(tmp_path, num_frames=3, drop_indices=[2])

    finalize_in_place(scene, "selected_frames.csv", backup_dir=None)

    assert not (scene / "images_backup").exists()
    images = scene / "images"
    files = sorted(p.name for p in images.glob("*.jpg"))
    assert files == ["frame_000001.jpg", "frame_000003.jpg"]


def test_finalize_in_place_backup_idempotent(tmp_path: Path):
    """同じ backup_dir に 2 回実行しても問題なく動く（既存 backup を上書き）。"""
    scene = _make_scene(tmp_path, num_frames=3, drop_indices=[3])
    backup_dir = scene / "images_backup"

    finalize_in_place(scene, "selected_frames.csv", backup_dir=backup_dir)
    files_after_first = set(p.name for p in backup_dir.glob("*.jpg"))

    # 1 回目の結果状態から、再度 finalize できるよう CSV を再構築
    images = scene / "images"
    rows = []
    for i, img in enumerate(sorted(images.glob("*.jpg")), start=1):
        rows.append(
            {
                "seq": str(i),
                "original_index": str(i),
                "final_index": str(i),
                "status": "ok",
                "decision": "keep",
                "output_file": f"images/{img.name}",
            }
        )
    _write_csv(selected_frames_path(scene), rows)

    finalize_in_place(scene, "selected_frames.csv", backup_dir=backup_dir)

    # 2 回目の backup は 1 回目の状態（drop 後の 2 個）のフルコピーになる
    files_after_second = set(p.name for p in backup_dir.glob("*.jpg"))
    assert len(files_after_second) == 2  # 1 回目の結果が反映
    # 古い backup（3 個）は消えている
    assert files_after_second != files_after_first


def test_finalize_in_place_creates_csv_backup(tmp_path: Path):
    """CSV のバックアップ (.before_finalize.csv) が作成される。"""
    scene = _make_scene(tmp_path, num_frames=3, drop_indices=[2])

    finalize_in_place(scene, "selected_frames.csv")

    # selected_frames.before_finalize.csv が作られている
    csv_backup_files = list(frame_backups_dir(scene).glob("selected_frames.before_finalize*.csv"))
    assert len(csv_backup_files) >= 1


def test_finalize_in_place_can_renumber_kept_images_and_update_frame_metadata(tmp_path: Path):
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    (images / "clip_0010.jpg").write_bytes(b"keep-a")
    (images / "clip_0020.jpg").write_bytes(b"drop-b")
    (images / "clip_0030.jpg").write_bytes(b"keep-c")
    rows = [
        {
            "seq": "1",
            "original_index": "10",
            "final_index": "10",
            "status": "ok",
            "decision": "keep",
            "output_file": "images/clip_0010.jpg",
        },
        {
            "seq": "2",
            "original_index": "20",
            "final_index": "20",
            "status": "ok",
            "decision": "drop",
            "output_file": "images/clip_0020.jpg",
        },
        {
            "seq": "3",
            "original_index": "30",
            "final_index": "30",
            "status": "ok",
            "decision": "keep",
            "output_file": "images/clip_0030.jpg",
        },
    ]
    _write_csv(selected_frames_path(scene), rows)
    write_json(
        extract_sessions_path(scene),
        {
            "version": 1,
            "sessions": [
                {
                    "id": "session_1",
                    "output_files": [
                        "images/clip_0010.jpg",
                        "images/clip_0020.jpg",
                        "images/clip_0030.jpg",
                    ],
                }
            ],
        },
    )
    write_json(
        source_image_sets_path(scene),
        {
            "version": 1,
            "image_sets": [
                {
                    "id": "imageset_1",
                    "updated_at": "old",
                    "files": [
                        {
                            "source_path": str(images / "clip_0010.jpg"),
                            "scene_path": "images/clip_0010.jpg",
                            "file": file_identity(images / "clip_0010.jpg"),
                            "source_file": file_identity(images / "clip_0010.jpg"),
                        },
                        {
                            "source_path": str(images / "clip_0020.jpg"),
                            "scene_path": "images/clip_0020.jpg",
                            "file": file_identity(images / "clip_0020.jpg"),
                            "source_file": file_identity(images / "clip_0020.jpg"),
                        },
                        {
                            "source_path": str(images / "clip_0030.jpg"),
                            "scene_path": "images/clip_0030.jpg",
                            "file": file_identity(images / "clip_0030.jpg"),
                            "source_file": file_identity(images / "clip_0030.jpg"),
                        },
                    ],
                }
            ],
        },
    )

    finalize_in_place(scene, "selected_frames.csv", renumber_kept_images=True)

    assert sorted(path.name for path in images.glob("*.jpg")) == ["frame_000001.jpg", "frame_000002.jpg"]
    assert (images / "frame_000001.jpg").read_bytes() == b"keep-a"
    assert (images / "frame_000002.jpg").read_bytes() == b"keep-c"
    assert not (images / "clip_0020.jpg").exists()

    selected_rows = _read_csv(selected_frames_path(scene))
    assert [row["output_file"] for row in selected_rows] == [
        "images/frame_000001.jpg",
        "images/frame_000002.jpg",
    ]
    keep_rows = _read_csv(selected_frames_keep_path(scene))
    assert [row["output_file"] for row in keep_rows] == [
        "images/frame_000001.jpg",
        "images/frame_000002.jpg",
    ]

    sessions = json.loads(extract_sessions_path(scene).read_text(encoding="utf-8"))["sessions"]
    assert sessions[0]["output_files"] == [
        "images/frame_000001.jpg",
        "images/clip_0020.jpg",
        "images/frame_000002.jpg",
    ]
    image_set = json.loads(source_image_sets_path(scene).read_text(encoding="utf-8"))["image_sets"][0]
    assert [item["scene_path"] for item in image_set["files"]] == [
        "images/frame_000001.jpg",
        "images/clip_0020.jpg",
        "images/frame_000002.jpg",
    ]
    assert image_set["files"][0]["file"]["path"].endswith("frame_000001.jpg")
    assert image_set["files"][2]["file"]["path"].endswith("frame_000002.jpg")
    assert image_set["files"][0]["source_path"].endswith("frame_000001.jpg")
    assert image_set["files"][2]["source_path"].endswith("frame_000002.jpg")
    assert image_set["files"][0]["source_file"]["path"].endswith("frame_000001.jpg")
    assert image_set["files"][2]["source_file"]["path"].endswith("frame_000002.jpg")
    assert image_set["updated_at"] != "old"


def test_finalize_in_place_renumber_refuses_existing_untracked_target(tmp_path: Path):
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    (images / "clip_a.jpg").write_bytes(b"A")
    (images / "clip_b.jpg").write_bytes(b"B")
    (images / "frame_000001.jpg").write_bytes(b"stale")
    _write_csv(
        selected_frames_path(scene),
        [
            {
                "seq": "1",
                "original_index": "1",
                "final_index": "1",
                "status": "ok",
                "decision": "keep",
                "output_file": "images/clip_a.jpg",
            },
            {
                "seq": "2",
                "original_index": "2",
                "final_index": "2",
                "status": "ok",
                "decision": "keep",
                "output_file": "images/clip_b.jpg",
            },
        ],
    )

    with pytest.raises(RuntimeError, match="Renumber target already exists"):
        finalize_in_place(scene, "selected_frames.csv", renumber_kept_images=True)

    assert (images / "clip_a.jpg").read_bytes() == b"A"
    assert (images / "clip_b.jpg").read_bytes() == b"B"
    assert (images / "frame_000001.jpg").read_bytes() == b"stale"


def test_finalize_in_place_renumber_refuses_downstream_outputs(tmp_path: Path):
    scene = _make_scene(tmp_path, num_frames=3, drop_indices=[2])
    masks = scene / "masks"
    masks.mkdir()
    (masks / "frame_000001.png").write_bytes(b"mask")

    with pytest.raises(RuntimeError, match="downstream outputs"):
        finalize_in_place(scene, "selected_frames.csv", renumber_kept_images=True)

    assert (scene / "images" / "frame_000002.jpg").is_file()


def test_finalize_in_place_raises_on_no_keep(tmp_path: Path):
    """全 drop なら例外。"""
    scene = _make_scene(tmp_path, num_frames=3, drop_indices=[1, 2, 3])

    with pytest.raises(RuntimeError, match="No keep frames"):
        finalize_in_place(scene, "selected_frames.csv")


def test_pending_drop_image_paths_reports_existing_drop_files(tmp_path: Path):
    scene = _make_scene(tmp_path, num_frames=4, drop_indices=[2, 4])

    pending = pending_drop_image_paths(scene)

    assert [p.name for p in pending] == ["frame_000002.jpg", "frame_000004.jpg"]


def test_pending_drop_image_paths_ignores_missing_drop_files(tmp_path: Path):
    scene = _make_scene(tmp_path, num_frames=3, drop_indices=[2])
    (scene / "images" / "frame_000002.jpg").unlink()

    assert pending_drop_image_paths(scene) == []


def test_untracked_image_paths_reports_images_not_in_selected_csv(tmp_path: Path):
    scene = _make_scene(tmp_path, num_frames=2)
    stale = scene / "images" / "stale.jpg"
    stale.write_bytes(b"old")

    assert untracked_image_paths(scene) == [stale]
