"""W32: LatentActionHead — small MLP that predicts the next action from the
current encoder latent + current action.

This is the **training-side** counterpart to the W27 v2 inference wiring
(action-conditioned prediction). It replaces the Phase-1 passthrough in
``worldfoundry_rl.actor.VJEPAActor.predict_latent_action`` with a real
learnable head, so the RL actor can be supervised against ground-truth
next-action targets (smooth_l1) without the encoder having to memorize
the action mapping.

The head is intentionally small (one hidden GELU MLP). It is **not** a
world model: the encoder still owns the visual representation, and the
head only learns the action delta on top of that representation.

The class lives under ``worldfoundry_train`` (not ``worldfoundry_rl``)
because the head is a pure ``nn.Module`` with no RL-specific deps. The
RL actor wires it in; the trainer/optimizer can use it directly.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["LatentActionHead"]


def _init_trunc_normal(module: nn.Module, std: float = 0.02) -> None:
    """trunc_normal init on >=2-D params (mirrors JEPAPredictor's house style)."""
    for p in module.parameters():
        if p.ndim >= 2:
            nn.init.trunc_normal_(p, std=std)


class LatentActionHead(nn.Module):
    """Predict the next action from (current encoder latent, current action).

    Architecture:
        input  = concat(encoder_features, current_action)  # (B, encoder_dim+action_dim)
        hidden = GELU(Linear(input -> hidden_dim))
        out    = Linear(hidden -> action_dim)

    The final ``Linear`` is **zero-initialized** (both weight and bias)
    so that at init the head is exactly the zero mapping and the
    actor's behavior is identical to the pre-W32 passthrough. This
    matches the W22 warmup-stability convention used in
    ``JEPAPredictor`` (last block's linear2 zero-init) and guarantees
    that swapping the passthrough for the head on day-0 is a no-op for
    downstream loss values.

    Args:
        encoder_dim: hidden size of the encoder features fed in
            (e.g. 1024 for ViT-L).
        action_dim: per-frame action vector size (e.g. 7 for the
            default 6-DoF + gripper).
        hidden_dim: width of the GELU hidden layer. Defaults to
            ``max(64, 2 * action_dim)`` so the head is non-degenerate
            for the common action_dim=7 case.
    """

    def __init__(
        self,
        encoder_dim: int,
        action_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if encoder_dim <= 0:
            raise ValueError(f"encoder_dim must be > 0; got {encoder_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0; got {action_dim}")
        if hidden_dim is None:
            hidden_dim = max(64, 2 * action_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0; got {hidden_dim}")

        self.encoder_dim = encoder_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(encoder_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # trunc_normal(0.02) on every >=2-D param, then zero the final
        # linear's weight+bias so the head is a zero function at init.
        _init_trunc_normal(self)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        encoder_features: torch.Tensor,
        current_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next action.

        Args:
            encoder_features: ``(B, encoder_dim)`` pooled encoder output.
                The actor is responsible for pooling (e.g. mean over
                patch tokens) before calling the head; the head does
                not assume any spatial structure.
            current_action: ``(B, action_dim)`` per-frame action vector
                observed at the current step.

        Returns:
            ``(B, action_dim)`` predicted next action. At init this is
            the zero vector (head is a no-op), so loss values match
            the pre-W32 passthrough baseline.
        """
        if encoder_features.ndim != 2:
            raise ValueError(
                f"encoder_features must be (B, encoder_dim); got shape "
                f"{tuple(encoder_features.shape)}"
            )
        if current_action.ndim != 2:
            raise ValueError(
                f"current_action must be (B, action_dim); got shape "
                f"{tuple(current_action.shape)}"
            )
        if encoder_features.shape[0] != current_action.shape[0]:
            raise ValueError(
                f"batch mismatch: encoder_features B={encoder_features.shape[0]} "
                f"vs current_action B={current_action.shape[0]}"
            )
        if encoder_features.shape[1] != self.encoder_dim:
            raise ValueError(
                f"encoder_features last dim={encoder_features.shape[1]} "
                f"!= head.encoder_dim={self.encoder_dim}"
            )
        if current_action.shape[1] != self.action_dim:
            raise ValueError(
                f"current_action last dim={current_action.shape[1]} "
                f"!= head.action_dim={self.action_dim}"
            )
        return self.net(torch.cat([encoder_features, current_action], dim=-1))
