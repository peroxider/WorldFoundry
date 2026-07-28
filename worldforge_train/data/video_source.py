"""Video source abstraction with DVPP (hardware) and decord (software) backends.

Per spec §3.1:
- Primary: Ascend DVPP (CANN hardware decoder)
- Fallback: decord (CPU)

Per spec §4.7, a single-batch DVPP decode failure falls back to decord. That
fallback lives at the *decode* layer, not at construction: when DVPP is
available and requested, ``build_video_source`` returns a
``FallbackVideoSource`` wrapping a DVPP primary and a decord fallback. Its
``read_all()`` tries DVPP first and, on ``NotImplementedError`` /
``RuntimeError``, emits a ``logging`` WARNING naming the failure and the file,
then decodes with decord. Any other exception propagates.

Phase-1: ``_dvpp_available()`` always returns False and ``DVPPVideoSource``
is a stub; the real acl_dvpp decoder arrives in Phase-2 W11.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


def _dvpp_available() -> bool:
    """Probe for Ascend DVPP availability. Returns False on non-Ascend hosts."""
    try:
        import torch_npu  # noqa: F401
        # Real DVPP detection requires CANN acl_dvpp import; treat as opt-in.
        return False  # Phase-1: always fall back; DVPP wired in Phase-2
    except ImportError:
        return False


class VideoSource(ABC):
    """Abstract video source returning T×H×W×3 uint8 frames."""

    @abstractmethod
    def read_all(self) -> np.ndarray:
        """Load all frames into memory."""


class DecordVideoSource(VideoSource):
    """Software video decoder using decord (CPU)."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

    def read_all(self) -> np.ndarray:
        import decord  # local import: optional dep
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(self.path))
        frames = reader.get_batch(range(len(reader))).asnumpy()
        return frames  # (T, H, W, 3) uint8


class DVPPVideoSource(VideoSource):
    """Hardware-accelerated decoder via Ascend DVPP.

    Phase-1: stub; raises NotImplementedError so callers fall back.
    Phase-2 W11: implement via acl_dvpp.
    """

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

    def read_all(self) -> np.ndarray:
        raise NotImplementedError(
            "DVPP decoder is wired in Phase-2 W11. "
            "Use build_video_source(prefer_dvpp=False) for now."
        )


class FallbackVideoSource(VideoSource):
    """Compose a primary source with a fallback used on decode failure.

    Per spec §4.7: a DVPP decode failure falls back to decord. The fallback is
    never silent — it logs a WARNING naming the underlying exception and the
    file being decoded. Only ``NotImplementedError`` (the Phase-1 DVPP stub)
    and ``RuntimeError`` (the expected shape of an acl_dvpp decode failure) are
    caught; anything else propagates to the caller.
    """

    #: Narrow, deliberate catch set. Do not widen to ``Exception``.
    FALLBACK_ERRORS = (NotImplementedError, RuntimeError)

    def __init__(self, primary: VideoSource, fallback: VideoSource) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def path(self) -> Path:
        return getattr(self.primary, "path", getattr(self.fallback, "path", None))

    def read_all(self) -> np.ndarray:
        try:
            return self.primary.read_all()
        except self.FALLBACK_ERRORS as exc:
            logger.warning(
                "Primary video source %s failed to decode %s (%r); "
                "falling back to %s",
                type(self.primary).__name__,
                self.path,
                exc,
                type(self.fallback).__name__,
            )
            return self.fallback.read_all()


def build_video_source(path: PathLike, prefer_dvpp: bool = True) -> VideoSource:
    """Build a video source, preferring DVPP if available and requested."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    if prefer_dvpp and _dvpp_available():
        return FallbackVideoSource(DVPPVideoSource(p), DecordVideoSource(p))
    return DecordVideoSource(p)
