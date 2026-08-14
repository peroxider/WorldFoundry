"""In-memory FrankaKitchen RGB-action transition dataset (W2.16).

Pairs every frame with the action that produced it, so the
action-conditioned predictor can train on real env transitions instead
of orchestrator-sampled random actions. Mirrors the ``__getitem__``
shape contract of :class:`worldfoundry_train.data.dataset.Vjepa2Dataset`
plus an extra ``actions`` field:

    item["frames"]  -> (T, 3, H, W)        float32 in [0, 1]
    item["mask"]    -> (T, Hg, Wg)         bool
    item["actions"] -> (T, action_dim)     float32

The action at index ``t`` is the action executed in the env at step
``t`` (i.e. the action that produced ``item["frames"][t]``). The
orchestrator's train loop picks ``actions[t]`` per training step so
the world-model sees the same action sequence that generated the
video clip.

Public surface
--------------

- :class:`FrankaTransitionDataset` — in-memory dataset fed by
  :func:`worldfoundry_rl.data.franka_video_collector.collect_franka_video_clips`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class FrankaTransitionDataset:
    """Indexed access over aligned ``(frames, actions)`` arrays.

    Args:
        frames: ``(N_clips, T, H, W, 3) uint8`` aligned RGB frames from
            a real env rollout.
        actions: ``(N_clips, T, action_dim) float32`` action sequence
            aligned with ``frames`` along the first two axes.
        frame_size: spatial side length; equal to ``H == W``.
        patch_size: encoder patch size; must divide ``frame_size``.
        mask_fn: ``(grid=(T, Hg, Wg), rng=np.random.Generator) -> bool
            mask`` closure. Mirrors the
            :class:`worldfoundry_train.data.dataset.Vjepa2Dataset`
            interface so callers can reuse the same
            ``tubelet_mask(...)`` closure.

    Raises:
        ValueError: if shape constraints are violated.
    """

    def __init__(
        self,
        *,
        frames: np.ndarray,
        actions: np.ndarray,
        frame_size: int,
        patch_size: int,
        mask_fn: Callable[[tuple, np.random.Generator], np.ndarray],
        tubelet_size: int = 1,
    ) -> None:
        if frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError(
                f"frames must be (N_clips, T, H, W, 3); got {frames.shape}"
            )
        if frames.dtype != np.uint8:
            raise ValueError(
                f"frames dtype must be uint8; got {frames.dtype}"
            )
        if actions.ndim != 3:
            raise ValueError(
                f"actions must be (N_clips, T, action_dim); got {actions.shape}"
            )
        if frames.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"frames.shape[:2] ({frames.shape[:2]}) must equal "
                f"actions.shape[:2] ({actions.shape[:2]})"
            )
        if frames.shape[2] != frame_size or frames.shape[3] != frame_size:
            raise ValueError(
                f"frames H/W must equal frame_size={frame_size}; got "
                f"H={frames.shape[2]}, W={frames.shape[3]}"
            )
        if frame_size % patch_size != 0:
            raise ValueError(
                f"frame_size={frame_size} must be divisible by "
                f"patch_size={patch_size}"
            )
        if tubelet_size < 1:
            raise ValueError(f"tubelet_size must be >= 1; got {tubelet_size}")
        self.frames = frames
        self.actions = actions
        self.frame_size = int(frame_size)
        self.patch_size = int(patch_size)
        self.mask_fn = mask_fn
        self.tubelet_size = int(tubelet_size)
        self.num_clips = int(frames.shape[0])
        self.frames_per_clip = int(frames.shape[1])
        self.action_dim = int(actions.shape[2])
        # W2.25+: align mask grid with encoder's post-tubelet token grid.
        # With ``tubelet_size=1`` (default) ``T_post == frames_per_clip``.
        self._grid = (
            max(1, self.frames_per_clip // self.tubelet_size),
            frame_size // patch_size,
            frame_size // patch_size,
        )

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < self.num_clips:
            raise IndexError(
                f"clip index {idx} out of range [0, {self.num_clips})"
            )
        # (T, H, W, 3) uint8 -> (T, 3, H, W) float32 in [0, 1].
        clip_uint8 = self.frames[idx]
        clip = np.transpose(clip_uint8, (0, 3, 1, 2)).astype(np.float32) / 255.0
        actions = self.actions[idx].astype(np.float32, copy=True)
        rng = np.random.default_rng(int(idx) + 1)
        mask = np.asarray(self.mask_fn(self._grid, rng), dtype=bool)
        return {
            "frames": clip,
            "mask": mask,
            "actions": actions,
        }


__all__ = ["FrankaTransitionDataset"]