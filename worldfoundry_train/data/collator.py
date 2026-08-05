"""Batch collator: Vjepa2Dataset -> VJEPA2Model batch naming."""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["collate_vjepa2_batch"]


def collate_vjepa2_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack B samples into a single VJEPA2Model batch.

    Returns: {"pixel_values_videos": (B,T,C,H,W) float32,
              "bool_masked_pos":     (B, T*Hg*Wg) bool,
              "bool_masked_pos_k":   (B, K, T*Hg*Wg) bool}

    ``bool_masked_pos_k`` is emitted only when the samples carry a ``masks``
    list (``Vjepa2Dataset(mask_count=K)`` with K > 1, W22.8 design §3.2).
    ``bool_masked_pos`` always holds mask 0, so K=1 consumers are unaffected.

    Raises:
        ValueError: empty samples list, mismatched (T,H,W) across samples,
            inconsistent K across samples, or a batch that mixes multi-mask
            and single-mask samples.
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
    batch = {
        "pixel_values_videos": frames,
        "bool_masked_pos": bool_masked_pos,
    }

    # Multi-mask path (W22.8). A batch must be all-multi or all-single:
    # silently dropping to K=1 would make a num_masks>1 config a no-op.
    n_multi = sum(1 for s in samples if "masks" in s)
    if n_multi and n_multi != len(samples):
        raise ValueError(
            f"batch mixes multi-mask and single-mask samples: {n_multi} of "
            f"{len(samples)} carry a 'masks' key; all samples must come from "
            "the same Vjepa2Dataset mask_count setting"
        )
    if n_multi:
        ref_k = len(samples[0]["masks"])
        per_sample = []
        for i, s in enumerate(samples):
            k = len(s["masks"])
            if k != ref_k:
                raise ValueError(
                    f"sample {i} has K={k} masks != reference K={ref_k}"
                )
            per_sample.append(
                torch.stack(
                    [torch.as_tensor(m).bool().reshape(-1) for m in s["masks"]],
                    dim=0,
                )  # (K, N)
            )
        batch["bool_masked_pos_k"] = torch.stack(per_sample, dim=0)  # (B, K, N)
    return batch