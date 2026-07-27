"""Video dataset wrapper for V-JEPA 2 training.

Reads a CSV manifest of (video_path, num_frames, height, width), decodes
each video via the project's `build_video_source` factory, normalizes
frames to float32 in [0, 1], transposes to (T, 3, H, W), and invokes a
caller-supplied `mask_fn` with the per-token patch grid derived from
`patch_size` (default 14, V-JEPA 2 ViT-L).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from worldforge_train.data.video_source import build_video_source

MaskFn = Callable[[tuple[int, int, int], np.random.Generator], np.ndarray]


class Vjepa2Dataset:
    """Iterates (video, mask) samples from a CSV manifest.

    Manifest schema: video_path, num_frames, height, width.
    """

    def __init__(
        self,
        manifest_path: Path,
        mask_fn: MaskFn,
        patch_size: int = 14,
    ) -> None:
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0; got {patch_size}")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.mask_fn = mask_fn
        self.patch_size = patch_size
        self.entries: list[dict[str, str]] = []
        with self.manifest_path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "video_path",
                "num_frames",
                "height",
                "width",
            ], (
                "manifest schema must be "
                "(video_path, num_frames, height, width); "
                f"got {reader.fieldnames}"
            )
            self.entries = list(reader)
        for i, row in enumerate(self.entries):
            h = int(row["height"])
            w = int(row["width"])
            if h < patch_size or w < patch_size:
                raise ValueError(
                    f"manifest row {i} has height={h}, width={w} which is "
                    f"smaller than patch_size={patch_size}; grid would be "
                    "non-positive on at least one spatial axis"
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        row = self.entries[idx]
        src = build_video_source(row["video_path"], prefer_dvpp=False)
        frames = src.read_all()  # (T, H, W, 3) uint8
        frames = frames.astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, 3, H, W)
        T, _, H, W = frames.shape
        grid = (T, H // self.patch_size, W // self.patch_size)
        mask = self.mask_fn(grid, np.random.default_rng(idx))
        return {"frames": frames, "mask": mask}
