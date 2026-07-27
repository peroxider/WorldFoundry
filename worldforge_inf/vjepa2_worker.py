"""VJEPA2Worker — minimal vLLM-Ascend plugin skeleton for V-JEPA 2 inference.

Per spec §4.1 + §4.2:
- V-JEPA 2 is non-autoregressive ViT; we DO NOT use vLLM's PagedAttention scheduler.
- We DO reuse vLLM-Ascend's Hardware Pluggable mechanism: register a worker class
  that exposes encode/predict/plan methods, sharing the operator library
  (FlashAttention, RMSNorm, quant).
- Phase-1: minimal skeleton; Phase-2 W3 wires the full Hardware Pluggable registration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def _load_hf_model(path: Path, dtype: torch.dtype = torch.bfloat16):
    """Lazy import of HF transformers; isolated to allow test mocking."""
    from transformers import VJEPA2Model
    return VJEPA2Model.from_pretrained(str(path), torch_dtype=dtype).eval()


class VJEPA2Worker:
    """Wraps a V-JEPA 2 model for batch inference."""

    def __init__(
        self,
        model_path: Path,
        adapter_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = Path(model_path)
        self.adapter_name = adapter_name
        self.dtype = dtype
        self.model = _load_hf_model(self.model_path, dtype=self.dtype)

    @torch.no_grad()
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos into dense features.

        Args:
            video_tensor: (B, T, C, H, W) float tensor (normalized).
        Returns:
            (B, N_tokens, hidden_dim) features.
        """
        out = self.model(pixel_values_videos=video_tensor)
        return out.last_hidden_state

    @torch.no_grad()
    def predict(self, masked_video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict masked tokens (V-JEPA 2 native objective)."""
        out = self.model(pixel_values_videos=masked_video, bool_masked_pos=mask)
        return out.last_hidden_state

    @torch.no_grad()
    def plan(self, video_tensor: torch.Tensor, actions: list[int]) -> tuple[torch.Tensor, list[int]]:
        """V-JEPA 2-AC: given (video, actions), return next-state features.

        Phase-1: passthrough that returns the encoded features and the input
        actions. Phase-2 W4 implements the action-conditioned predictor from
        facebookresearch/vjepa2.
        """
        feats = self.encode(video_tensor)
        return feats, list(actions)

