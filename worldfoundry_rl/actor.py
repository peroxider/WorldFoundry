"""VJEPAActor — wraps V-JEPA 2 for action-conditioned prediction (W7 / W32).

W7 baseline: passthrough latent-action prediction.
W32: ``predict_latent_action`` now runs a real ``LatentActionHead`` MLP
that maps (current encoder latent, current action) -> next action. The
head is zero-initialized at construction, so day-0 behavior is
byte-for-byte equivalent to the pre-W32 passthrough (head outputs 0);
training warmup then brings it online.

Per spec §5.1: Phase-1 covers only the AC finetuning phase (not full 80% repro).
"""
from __future__ import annotations

from pathlib import Path

import torch

from worldfoundry_train.model.latent_action_head import LatentActionHead


def _load_hf_model(path: Path, dtype: torch.dtype = torch.bfloat16):
    from transformers import VJEPA2Model
    return VJEPA2Model.from_pretrained(str(path), torch_dtype=dtype).eval()


def _encoder_hidden_size(encoder) -> int:
    """Resolve encoder hidden_size from common HF / test-double shapes."""
    cfg = getattr(encoder, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    raise ValueError(
        "encoder must expose `config.hidden_size` (HF VJEPA2Model contract); "
        f"got encoder={type(encoder).__name__}"
    )


class VJEPAActor:
    """V-JEPA 2 backbone + learnable latent-action head (W32)."""

    def __init__(
        self,
        encoder,
        *,
        action_dim: int = 7,
        head_hidden_dim: int | None = None,
    ) -> None:
        self.encoder = encoder
        self.action_dim = int(action_dim)
        encoder_dim = _encoder_hidden_size(encoder)
        self.head = LatentActionHead(
            encoder_dim=encoder_dim,
            action_dim=self.action_dim,
            hidden_dim=head_hidden_dim,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        action_dim: int = 7,
        head_hidden_dim: int | None = None,
    ) -> "VJEPAActor":
        return cls(
            _load_hf_model(Path(model_path)),
            action_dim=action_dim,
            head_hidden_dim=head_hidden_dim,
        )

    @torch.no_grad()
    def predict_latent_action(
        self,
        state: torch.Tensor,
        current_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next action from current state + current action.

        Args:
            state: video tensor shaped like an HF ``VJEPA2Model`` input
                (typically ``(B, T, C, H, W)``).
            current_action: ``(B, action_dim)`` per-frame action vector
                observed at the current step.

        Returns:
            ``(B, action_dim)`` predicted next action. At init the head
            is zero-initialized, so this returns the zero vector and
            downstream loss values match the pre-W32 passthrough.
        """
        out = self.encoder(pixel_values_videos=state)
        # Mean-pool over the patch-token sequence -> (B, encoder_dim).
        # The head is shape-agnostic; any pooling strategy that
        # preserves the batch dim is fine.
        pooled = out.last_hidden_state.mean(dim=1)
        return self.head(pooled, current_action)