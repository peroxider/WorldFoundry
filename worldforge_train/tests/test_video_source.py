import logging

import numpy as np
import pytest

from worldforge_train.data.video_source import (
    DecordVideoSource,
    DVPPVideoSource,
    FallbackVideoSource,
    VideoSource,
    build_video_source,
)


@pytest.fixture
def sample_mp4(tmp_path):
    """Create a 2-second 64x64 mp4 with decord for tests."""
    import decord
    import av

    out = tmp_path / "sample.mp4"
    container = av.open(str(out), mode="w")
    stream = container.add_stream("h264", rate=10)
    stream.width, stream.height = 64, 64
    for i in range(20):
        frame = av.VideoFrame.from_ndarray(
            np.full((64, 64, 3), i * 12, dtype=np.uint8)
        )
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out


def test_decord_source_returns_uint8_frames(sample_mp4):
    src = DecordVideoSource(sample_mp4)
    frames = src.read_all()
    assert frames.dtype == np.uint8
    assert frames.shape == (20, 64, 64, 3)
    # Pixels vary frame-to-frame (i*12)
    assert frames[0, 0, 0, 0] != frames[-1, 0, 0, 0]


def test_build_video_source_default_returns_decord(sample_mp4):
    src = build_video_source(sample_mp4, prefer_dvpp=False)
    assert isinstance(src, DecordVideoSource)


def test_build_video_source_dvpp_unavailable_falls_back(sample_mp4, monkeypatch):
    monkeypatch.setattr(
        "worldforge_train.data.video_source._dvpp_available",
        lambda: False,
    )
    src = build_video_source(sample_mp4, prefer_dvpp=True)
    assert isinstance(src, DecordVideoSource)


def test_build_video_source_raises_for_missing_file(tmp_path, monkeypatch):
    """The factory's own guard must fire before any backend is constructed."""

    def _never_construct(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("backend constructed despite missing file")

    monkeypatch.setattr(
        "worldforge_train.data.video_source.DecordVideoSource",
        _never_construct,
    )
    monkeypatch.setattr(
        "worldforge_train.data.video_source.DVPPVideoSource",
        _never_construct,
    )
    with pytest.raises(FileNotFoundError):
        build_video_source(tmp_path / "nope.mp4")


def test_decord_source_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DecordVideoSource(tmp_path / "nope.mp4")


def test_dvpp_source_read_all_raises_not_implemented(sample_mp4):
    src = DVPPVideoSource(sample_mp4)
    with pytest.raises(NotImplementedError):
        src.read_all()


def _force_dvpp_available(monkeypatch):
    """Pin the environment: pretend DVPP is present regardless of hardware."""
    monkeypatch.setattr(
        "worldforge_train.data.video_source._dvpp_available",
        lambda: True,
    )


def test_build_video_source_dvpp_available_returns_fallback(sample_mp4, monkeypatch):
    _force_dvpp_available(monkeypatch)
    src = build_video_source(sample_mp4, prefer_dvpp=True)
    assert isinstance(src, FallbackVideoSource)
    assert isinstance(src, VideoSource)
    assert isinstance(src.primary, DVPPVideoSource)
    assert isinstance(src.fallback, DecordVideoSource)


def test_dvpp_fallback_yields_decord_frames(sample_mp4, monkeypatch):
    _force_dvpp_available(monkeypatch)
    src = build_video_source(sample_mp4, prefer_dvpp=True)
    frames = src.read_all()
    assert frames.dtype == np.uint8
    assert frames.shape == (20, 64, 64, 3)
    assert frames[0, 0, 0, 0] != frames[-1, 0, 0, 0]


def test_dvpp_fallback_logs_warning(sample_mp4, monkeypatch, caplog):
    _force_dvpp_available(monkeypatch)
    src = build_video_source(sample_mp4, prefer_dvpp=True)
    with caplog.at_level(
        logging.WARNING, logger="worldforge_train.data.video_source"
    ):
        src.read_all()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "fallback must not be silent (G8)"
    message = warnings[0].getMessage()
    assert "DVPPVideoSource" in message
    assert "DecordVideoSource" in message
    assert "sample.mp4" in message


class _RuntimeErrorSource(VideoSource):
    def read_all(self):
        raise RuntimeError("acl_dvpp decode failed")


class _ValueErrorSource(VideoSource):
    def read_all(self):
        raise ValueError("not a fallback-worthy failure")


def test_fallback_catches_runtime_error(sample_mp4, caplog):
    src = FallbackVideoSource(_RuntimeErrorSource(), DecordVideoSource(sample_mp4))
    with caplog.at_level(
        logging.WARNING, logger="worldforge_train.data.video_source"
    ):
        frames = src.read_all()
    assert frames.shape == (20, 64, 64, 3)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_fallback_propagates_non_targeted_exception(sample_mp4):
    """A non-targeted error must propagate, not be swallowed by a broad catch."""
    src = FallbackVideoSource(_ValueErrorSource(), DecordVideoSource(sample_mp4))
    with pytest.raises(ValueError):
        src.read_all()
