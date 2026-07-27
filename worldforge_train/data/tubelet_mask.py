"""Tubelet masking for V-JEPA 2 (ported from facebookresearch/vjepa2/src/utils/tensors.py).

Adapted from /opt/vjepa2/src/utils/tensors.py under the original BSD-style license.
Original: Copyright (c) Meta Platforms, Inc. and affiliates.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from worldforge_common.exceptions import MaskingError


def multiblock_mask(
    shape: Tuple[int, int, int],
    num_blocks: int,
    spatial_only: bool = False,
    target_ratio: float = 0.5,
    max_attempts: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a multiblock mask following Bardes et al. 2024 (V-JEPA 2).

    Blocks accumulate across rounds of `num_blocks` samples until at least
    `target_ratio` of the volume is masked, or `max_attempts` is reached.

    Args:
        shape: (T, H, W) target tensor shape.
        num_blocks: number of rectangular blocks sampled per round.
        spatial_only: if True, blocks are sampled on (H, W) and broadcast across T.
        target_ratio: required minimum fraction of cells masked (default 0.5).
        max_attempts: hard cap on sampling rounds (default 100).
        rng: numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    base_shape = (shape[1], shape[2]) if spatial_only else shape
    mask = np.zeros(base_shape, dtype=bool)
    for _ in range(max_attempts):
        if spatial_only:
            _accumulate_blocks_2d(mask, num_blocks, rng)
        else:
            _accumulate_blocks_3d(mask, num_blocks, rng)
        if mask.mean() >= target_ratio:
            break
    else:
        raise MaskingError(
            f"multiblock_mask reached only {mask.mean():.3f} of target "
            f"{target_ratio} after {max_attempts} rounds; raise num_blocks, "
            f"increase max_attempts, or lower target_ratio."
        )
    if spatial_only:
        mask = np.broadcast_to(mask[None], shape).copy()
    return mask


def _accumulate_blocks_2d(mask_hw: np.ndarray, num_blocks: int, rng: np.random.Generator) -> None:
    H, W = mask_hw.shape
    for _ in range(num_blocks):
        h = rng.integers(1, max(1, H // 2) + 1)
        w = rng.integers(1, max(1, W // 2) + 1)
        top = rng.integers(0, H - h + 1)
        left = rng.integers(0, W - w + 1)
        mask_hw[top:top + h, left:left + w] = True


def _accumulate_blocks_3d(mask: np.ndarray, num_blocks: int, rng: np.random.Generator) -> None:
    T, H, W = mask.shape
    for _ in range(num_blocks):
        t = rng.integers(1, max(1, T // 2) + 1)
        h = rng.integers(1, max(1, H // 2) + 1)
        w = rng.integers(1, max(1, W // 2) + 1)
        t0 = rng.integers(0, T - t + 1)
        h0 = rng.integers(0, H - h + 1)
        w0 = rng.integers(0, W - w + 1)
        mask[t0:t0 + t, h0:h0 + h, w0:w0 + w] = True


def tubelet_mask(
    shape: Tuple[int, int, int],
    tubelet_size: Tuple[int, int, int],
    mask_ratio: float = 0.75,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a tubelet-aligned binary mask by selecting whole tubelets at random.

    A tubelet of size (t,h,w) covers one cell in the grid (T/t, H/h, W/w).
    Each input axis must be an integer multiple of the corresponding tubelet
    axis; non-divisible shapes raise ``MaskingError`` (G8 — no silent
    truncation). The returned mask always has shape exactly equal to ``shape``.
    """
    if any(s % t != 0 for s, t in zip(shape, tubelet_size)):
        raise MaskingError(
            f"shape {shape} not divisible by tubelet_size {tubelet_size}; "
            f"vjepa2 requires aligned grid."
        )
    if rng is None:
        rng = np.random.default_rng()
    grid = tuple(s // t for s, t in zip(shape, tubelet_size))
    total = int(np.prod(grid))
    n_masked = int(round(mask_ratio * total))
    grid_mask = np.zeros(total, dtype=bool)
    grid_mask[rng.choice(total, size=n_masked, replace=False)] = True
    grid_mask = grid_mask.reshape(grid)
    # Broadcast each grid cell to a tubelet
    mask = np.kron(grid_mask, np.ones(tubelet_size, dtype=bool))
    return mask