"""Shared pytest fixtures for the worldfoundry_train test suite.

The `sample_mp4` fixture used to live only inside test_video_source.py. To let
sibling test modules (test_dataset.py and future tests) reuse the same 20-frame
64x64 mp4 without re-encoding per file, the *same* fixture is re-exported here.
conftest.py fixtures are visible to all sibling test_* modules by pytest
convention; the local definition in test_video_source.py remains in effect for
that file only (no shadowing conflict because pytest prefers conftest fixtures
when both resolve).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_mp4(tmp_path) -> Path:
    """Create a 2-second 64x64 mp4 with av for tests (20 frames)."""
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
