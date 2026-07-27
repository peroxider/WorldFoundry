"""Video source abstraction with DVPP (hardware) and decord (software) backends.

Per spec §3.1:
- Primary: Ascend DVPP (CANN hardware decoder)
- Fallback: decord (CPU)
- Single-batch DVPP failure must fall back to decord (per spec §4.7)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


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


def build_video_source(path: PathLike, prefer_dvpp: bool = True) -> VideoSource:
    """Build a video source, preferring DVPP if available and requested."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    if prefer_dvpp and _dvpp_available():
        try:
            return DVPPVideoSource(p)
        except Exception:
            pass  # fall through to decord per spec §4.7
    return DecordVideoSource(p)
