"""extract_frames.py の解析キャッシュ動作テスト。

video metadata と analysis_width によるキャッシュ有効性判定を検証する。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from extract_frames import (
    CACHE_VERSION,
    VideoInfo,
    cache_path_for,
    load_analysis_cache,
    save_analysis_cache,
    video_signature,
)


def _make_dummy_video(path: Path, content: bytes = b"fakevideo") -> None:
    path.write_bytes(content)


def _make_video_info(w: int = 3840, h: int = 1920, fps: float = 30.0,
                    duration: float = 10.0, total: int = 300) -> VideoInfo:
    return VideoInfo(width=w, height=h, fps=fps, duration=duration, total_frames=total)


# =============================================================================
# video_signature
# =============================================================================


def test_video_signature_returns_size_mtime(tmp_path: Path):
    p = tmp_path / "fake.mp4"
    _make_dummy_video(p, b"hello world")
    size, mtime_ns = video_signature(p)
    assert size == 11
    assert mtime_ns > 0


def test_video_signature_changes_when_file_modified(tmp_path: Path):
    p = tmp_path / "fake.mp4"
    _make_dummy_video(p, b"abc")
    sig1 = video_signature(p)
    # 内容変更 → 少なくとも size は変わる
    _make_dummy_video(p, b"abcdef")
    sig2 = video_signature(p)
    assert sig1 != sig2


# =============================================================================
# cache_path_for
# =============================================================================


def test_cache_path_for(tmp_path: Path):
    expected = tmp_path / "extract_cache.npz"
    assert cache_path_for(tmp_path) == expected


# =============================================================================
# save / load roundtrip
# =============================================================================


def test_save_load_roundtrip(tmp_path: Path):
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)

    blur = [10.0, 20.0, 30.0, 40.0]
    change = [0.0, 0.1, 0.2, 0.3]
    feature_motion = [0.0, 0.01, 0.02, 0.03]
    info = _make_video_info()

    quality = [0.1, 0.2, 0.3, 0.4]
    save_analysis_cache(cache, video, info, analysis_w=960, analysis_h=480,
                       blur_scores=blur, change_scores=change, quality_scores=quality,
                       feature_motion_scores=feature_motion,
                       quality_mode="sfm")

    loaded = load_analysis_cache(cache, video, info, analysis_width=960)
    assert loaded is not None
    blur_l, change_l, quality_l, feature_motion_l, aw_l, ah_l = loaded
    assert blur_l == blur
    assert change_l == change
    assert quality_l == quality
    assert feature_motion_l == feature_motion
    assert aw_l == 960
    assert ah_l == 480


def test_load_returns_none_when_cache_missing(tmp_path: Path):
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)
    info = _make_video_info()

    assert load_analysis_cache(cache, video, info, 960) is None


def test_load_returns_none_when_video_size_changed(tmp_path: Path):
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video, b"abc")
    cache = cache_path_for(tmp_path)
    info = _make_video_info()

    save_analysis_cache(cache, video, info, 960, 480, [1.0], [0.0], [0.5], [0.0], "sfm")

    # 動画ファイル変更
    _make_dummy_video(video, b"abcdefghij")

    loaded = load_analysis_cache(cache, video, info, 960)
    assert loaded is None


def test_load_returns_none_when_video_mtime_changed(tmp_path: Path):
    """同サイズで mtime が違う動画でも無効化されること。"""
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video, b"abcde")
    cache = cache_path_for(tmp_path)
    info = _make_video_info()

    save_analysis_cache(cache, video, info, 960, 480, [1.0], [0.0], [0.5], [0.0], "sfm")

    # mtime を未来にずらす（同サイズ）
    future_ns = int((video.stat().st_mtime + 100.0) * 1e9)
    os.utime(video, ns=(future_ns, future_ns))

    loaded = load_analysis_cache(cache, video, info, 960)
    assert loaded is None


def test_load_returns_none_when_analysis_width_changed(tmp_path: Path):
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)
    info = _make_video_info(w=3840, h=1920)

    # analysis_width=960 でキャッシュ
    save_analysis_cache(cache, video, info, 960, 480, [1.0], [0.0], [0.5], [0.0], "sfm")

    # 異なる analysis_width で読み込み → None
    loaded = load_analysis_cache(cache, video, info, 1920)
    assert loaded is None


def test_load_returns_none_when_quality_mode_changed(tmp_path: Path):
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)
    info = _make_video_info(w=3840, h=1920)

    save_analysis_cache(cache, video, info, 960, 480, [1.0], [0.0], [0.5], [0.0], "sfm")

    loaded = load_analysis_cache(cache, video, info, 960, quality_mode="sharpness")
    assert loaded is None


def test_load_returns_none_when_feature_motion_required_but_not_cached(tmp_path: Path):
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)
    info = _make_video_info(w=3840, h=1920)

    save_analysis_cache(
        cache,
        video,
        info,
        960,
        480,
        [1.0, 2.0],
        [0.0, 0.1],
        [0.5, 0.6],
        [0.0, 0.0],
        "sfm",
        feature_motion_computed=False,
    )

    assert load_analysis_cache(cache, video, info, 960) is not None
    loaded = load_analysis_cache(cache, video, info, 960, require_feature_motion=True)
    assert loaded is None


def test_load_handles_zero_analysis_width(tmp_path: Path):
    """analysis_width=0 (= フルサイズ) でも一貫した動作: 動画幅でキャッシュされ、
    再度 0 を指定したら同じ値で hit する。"""
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)
    info = _make_video_info(w=3840, h=1920)

    # 0 指定 → 内部で video.width にスケール → 3840 でキャッシュ
    save_analysis_cache(cache, video, info, analysis_w=3840, analysis_h=1920,
                       blur_scores=[1.0], change_scores=[0.0], quality_scores=[0.5],
                       feature_motion_scores=[0.0],
                       quality_mode="sfm")
    # 同じ video で 0 を指定 → 内部で 3840 と判定 → hit
    loaded = load_analysis_cache(cache, video, info, analysis_width=0)
    assert loaded is not None
    _, _, _, _, aw, _ = loaded
    assert aw == 3840


def test_load_handles_corrupt_cache(tmp_path: Path):
    """壊れた npz ファイルは load_analysis_cache が None を返してエラーにならない。"""
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)
    cache.write_bytes(b"not a valid npz file")
    info = _make_video_info()

    loaded = load_analysis_cache(cache, video, info, 960)
    assert loaded is None


def test_load_returns_none_when_version_mismatch(tmp_path: Path):
    """version フィールドが想定と違うキャッシュは無効化。"""
    video = tmp_path / "fake.mp4"
    _make_dummy_video(video)
    cache = cache_path_for(tmp_path)

    size, mtime_ns = video_signature(video)
    np.savez_compressed(
        cache,
        version=np.int64(CACHE_VERSION + 99),  # 未来バージョン
        video_size=np.int64(size),
        video_mtime_ns=np.int64(mtime_ns),
        video_width=np.int32(3840),
        video_height=np.int32(1920),
        video_fps=np.float64(30.0),
        video_duration=np.float64(10.0),
        video_total_frames=np.int64(300),
        analysis_width=np.int32(960),
        analysis_height=np.int32(480),
        quality_mode=np.asarray("sfm"),
        blur_scores=np.asarray([1.0]),
        change_scores=np.asarray([0.0]),
        quality_scores=np.asarray([0.5]),
    )
    info = _make_video_info()
    loaded = load_analysis_cache(cache, video, info, 960)
    assert loaded is None
