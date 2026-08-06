"""W32: RL training losses.

Currently a single loss: ``action_prediction_loss`` — the supervised
objective for the ``LatentActionHead``. The head predicts the next
action from the current encoder latent + current action, and we
regress against the ground-truth next action with smooth L1.

Scope: this module is **head-only** supervised pretraining. PPO / GRPO
losses land in W34; env-side losses land in W31.

The loss is dtype-agnostic (works in float32 / float16 / bfloat16) and
returns a scalar ``Tensor`` with ``requires_grad=True`` whenever
``pred`` is part of a grad-tracked graph.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["action_prediction_loss"]


def action_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 between predicted and target next action.

    Wraps ``torch.nn.functional.smooth_l1_loss`` with explicit shape
    checks and a default ``beta=1.0`` (PyTorch's default, but spelled
    out so future tuning is a single kwarg flip). The ``reduction``
    kwarg is forwarded unchanged.

    Args:
        pred: ``(B, action_dim)`` predicted next action from
            ``LatentActionHead.forward``.
        target: ``(B, action_dim)`` ground-truth next action.
        reduction: ``"mean"`` (default) | ``"sum"`` | ``"none"`` —
            forwarded to ``F.smooth_l1_loss``.
        beta: smooth-L1 transition threshold; forwarded unchanged.

    Returns:
        Scalar loss tensor (or ``(B, action_dim)`` if
        ``reduction="none"``).

    Raises:
        ValueError: on shape mismatch or unknown reduction.
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"pred and target must be (B, action_dim); got "
            f"pred.shape={tuple(pred.shape)} target.shape={tuple(target.shape)}"
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"unknown reduction: {reduction!r}; expected 'mean'|'sum'|'none'"
        )
    if beta <= 0:
        raise ValueError(f"beta must be > 0; got {beta}")
    return nn.functional.smooth_l1_loss(pred, target, reduction=reduction, beta=beta)
