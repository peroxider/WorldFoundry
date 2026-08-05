"""VJEPAActor — wraps V-JEPA 2 for action-conditioned prediction (Phase-1 W7).

Per spec §5.1: Phase-1 covers only the AC finetuning phase (not full 80% repro).
- This actor reads the same VJEPA2Model backbone used in training.
- predict_latent_action is a Phase-1 passthrough; full latent-action head lands in Phase-2 W7.
"""
from __future__ import annotations

from pathlib import Path

import torch


def _load_hf_model(path: Path, dtype: torch.dtype = torch.bfloat16):
    from transformers import VJEPA2Model
    return VJEPA2Model.from_pretrained(str(path), torch_dtype=dtype).eval()


class VJEPAActor:
    """V-JEPA 2 backbone with a stub latent-action predictor."""

    def __init__(self, encoder) -> None:
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "VJEPAActor":
        return cls(_load_hf_model(Path(model_path)))

    @torch.no_grad()
    def predict_latent_action(
        self,
        state: torch.Tensor,
        action_history: list[int],
    ) -> torch.Tensor:
        """Phase-1 stub: returns encoded state features.

        Phase-2 W7: add the action-conditioned head from facebookresearch/vjepa2.
        """
        out = self.encoder(pixel_values_videos=state)
        return out.last_hidden_state