"""Batch collator: Vjepa2Dataset -> VJEPA2Model batch naming."""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["collate_vjepa2_batch"]


def collate_vjepa2_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack B samples into a single VJEPA2Model batch.

    Returns: {"pixel_values_videos": (B,T,C,H,W) float32,
              "bool_masked_pos":     (B, T*Hg*Wg) bool}

    Raises:
        ValueError: empty samples list, or mismatched (T,H,W) across samples.
        TypeError: sample["frames"] is neither ndarray nor Tensor.
        KeyError: missing "frames" or "mask" key.
    """
    if not samples:
        raise ValueError("collate_vjepa2_batch received empty list")
    frames_t = [torch.as_tensor(s["frames"]) for s in samples]
    if not all(isinstance(t, torch.Tensor) for t in frames_t):
        raise TypeError("sample['frames'] must be ndarray or Tensor")
    ref_shape = tuple(frames_t[0].shape)
    for i, t in enumerate(frames_t):
        if tuple(t.shape) != ref_shape:
            raise ValueError(
                f"sample {i} frames shape {tuple(t.shape)} != reference {ref_shape}"
            )
    frames = torch.stack([t.float() for t in frames_t], dim=0)

    masks_t = [torch.as_tensor(s["mask"]).bool() for s in samples]
    bool_masked_pos = torch.stack([m.reshape(-1) for m in masks_t], dim=0)
    return {
        "pixel_values_videos": frames,
        "bool_masked_pos": bool_masked_pos,
    }