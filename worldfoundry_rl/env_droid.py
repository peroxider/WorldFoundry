"""DroidEnv stub — offline dataset replay only (Phase-1, no live simulator).

Per spec §5.3: Phase-1 RL is the AC finetuning phase only, run on a fixed
Droid validation set. Live simulator integration (MuJoCo / SAPIEN) lands in
Phase-2 W14.
"""
from __future__ import annotations

from pathlib import Path


class DroidEnv:
    """Reads a Droid manifest and yields (state, action) pairs sequentially."""

    def __init__(self, manifest: Path, max_steps: int = 50) -> None:
        self.manifest = Path(manifest)
        self.max_steps = max_steps
        if not self.manifest.exists():
            raise FileNotFoundError(f"Droid manifest not found: {self.manifest}")

    def __iter__(self):
        with self.manifest.open() as f:
            for i, line in enumerate(f):
                if i >= self.max_steps:
                    break
                if line.startswith("video_path"):
                    continue
                yield line.strip().split(",")