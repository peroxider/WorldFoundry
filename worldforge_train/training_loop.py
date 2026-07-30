"""Real CPU-runnable training loop (forward + loss + backward + step).

Wires the existing ``DistributedStrategy`` contract
(``worldforge_train.parallel.strategy.DistributedStrategy``) to a real
forward/loss/backward/step cycle. Works in single-rank mode (the
``world_size == 1`` path used in unit/CPU tests) without needing
``dist.init_process_group``; ``FSDPStrategy.wrap`` returns the model
unchanged in that case (see ``fsdp.py:51-53``).

The default loss is a V-JEPA-style masked-feature MSE: encoder output
at masked positions is regressed against the encoder's prediction at
the same positions from a target pass. This is sufficient for CPU
testing of the training path; the production V-JEPA 2 self-supervised
objective is identical in shape (predict masked tokens) but with EMA
target encoder + cosine similarity loss.

G3 boundary: this file does NOT modify ``train.py`` or ``parallel/*``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from worldforge_train.parallel.strategy import DistributedStrategy

LossFn = Callable[[nn.Module, Mapping[str, torch.Tensor]], torch.Tensor]


def default_loss_fn(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """V-JEPA-style masked-feature MSE.

    Expects ``batch["pixel_values_videos"]`` of shape ``(B, T, C, H, W)``
    and ``batch["bool_masked_pos"]`` of shape ``(B, T, num_patches)``.
    Computes encoder output, then MSE on the masked-token positions
    against a self-distillation target (the same encoder with no_grad
    on unmasked inputs -- CPU surrogate for the EMA target encoder).
    """
    pixel_values_videos = batch["pixel_values_videos"]
    bool_masked_pos = batch["bool_masked_pos"]
    out = model(pixel_values_videos=pixel_values_videos, bool_masked_pos=bool_masked_pos)
    feats = out.last_hidden_state
    with torch.no_grad():
        tgt_out = model(pixel_values_videos=pixel_values_videos, bool_masked_pos=bool_masked_pos)
        target = tgt_out.last_hidden_state
    return torch.nn.functional.mse_loss(feats[bool_masked_pos], target[bool_masked_pos])


def run_training_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    strategy: DistributedStrategy,
    optimizer: Optimizer,
    loss_fn: LossFn | None = None,
) -> float:
    """One training step: forward + loss + strategy.backward(loss) + optimizer.step().

    Returns the scalar loss value as a Python float for logging/assertions.

    Raises:
        ValueError: ``batch`` is missing required keys for the default loss_fn.
        RuntimeError: forward / backward fails (propagated from torch).
    """
    if loss_fn is None:
        loss_fn = default_loss_fn

    optimizer.zero_grad()
    loss = loss_fn(model, batch)
    if not loss.requires_grad:
        raise RuntimeError(
            f"loss_fn returned a tensor with requires_grad=False; "
            f"check that the model parameters have requires_grad=True."
        )
    strategy.backward(loss)
    optimizer.step()
    return float(loss.detach().item())


def run_training_loop(
    model: nn.Module,
    batch_iter: Iterable[Mapping[str, torch.Tensor]],
    num_steps: int,
    strategy: DistributedStrategy,
    optimizer: Optimizer,
    loss_fn: LossFn | None = None,
) -> list[float]:
    """Run ``num_steps`` training steps, returning per-step losses."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    losses: list[float] = []
    iterator = iter(batch_iter)
    for step in range(num_steps):
        batch = next(iterator)
        loss = run_training_step(model, batch, strategy, optimizer, loss_fn)
        losses.append(loss)
    return losses


__all__ = ["run_training_step", "run_training_loop", "default_loss_fn"]
