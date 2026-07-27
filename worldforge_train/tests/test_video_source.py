import numpy as np
import pytest

from worldforge_train.data.video_source import (
    DecordVideoSource,
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


def test_build_video_source_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_video_source(tmp_path / "nope.mp4")
