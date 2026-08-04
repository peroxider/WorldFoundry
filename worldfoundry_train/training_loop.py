"""Real CPU-runnable training loop (forward + loss + backward + step).

Wires the existing ``DistributedStrategy`` contract
(``worldfoundry_train.parallel.strategy.DistributedStrategy``) to a real
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

from contextlib import nullcontext
from typing import Callable, Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from worldfoundry_train.parallel.strategy import DistributedStrategy

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
    post_step_hook: Callable[[], None] | None = None,
) -> float:
    """One training step: forward + loss + strategy.backward(loss) + optimizer.step().

    ``post_step_hook`` (if given) is invoked immediately after
    ``strategy.step(optimizer)`` — i.e. once per optimizer step, after the
    weights have moved. This is where the JEPA EMA update hooks in (W22).

    Returns the scalar loss value as a Python float for logging/assertions.

    Raises:
        ValueError: ``batch`` is missing required keys for the default loss_fn.
        RuntimeError: forward / backward fails (propagated from torch).
    """
    if loss_fn is None:
        loss_fn = default_loss_fn

    loss = loss_fn(model, batch)
    if not loss.requires_grad:
        raise RuntimeError(
            f"loss_fn returned a tensor with requires_grad=False; "
            f"check that the model parameters have requires_grad=True."
        )
    strategy.backward(loss)
    strategy.step(optimizer)
    if post_step_hook is not None:
        post_step_hook()
    return float(loss.detach().item())


def _rescale_grads(optimizer: Optimizer, factor: float) -> None:
    """Multiply all accumulated grads by ``factor`` in place.

    Used to flush a partial accumulation window: micro-batch losses are
    uniformly scaled by ``1 / grad_accum``, so a tail window of
    ``window_size < grad_accum`` micro-batches must be rescaled by
    ``grad_accum / window_size`` to recover the window-mean gradient.

    In-place ``mul_`` is safe for plain params, DDP-managed grads (DDP has
    already all-reduced into ``p.grad`` during the final backward), and
    FSDP with ``use_orig_params=True`` (``p.grad`` is a view into the
    sharded flat grad; elementwise scaling preserves the sharded layout).
    """
    for param_group in optimizer.param_groups:
        for p in param_group["params"]:
            if p.grad is not None:
                p.grad.mul_(factor)


def run_training_loop(
    model: nn.Module,
    batch_iter: Iterable[Mapping[str, torch.Tensor]],
    num_steps: int,
    strategy: DistributedStrategy,
    optimizer: Optimizer,
    loss_fn: LossFn | None = None,
    step_callback: Callable[[int, float], None] | None = None,
    grad_accum: int = 1,
    post_step_hook: Callable[[], None] | None = None,
) -> list[float]:
    """Run ``num_steps`` micro-batch steps with gradient accumulation.

    ``num_steps`` counts micro-batches (unchanged from the pre-accumulation
    contract). Every ``grad_accum`` micro-batches form one accumulation
    window: intermediate micro-batches only backward-accumulate scaled
    gradients (``loss / grad_accum``); the window-closing micro-batch also
    triggers ``strategy.step(optimizer)`` (optimizer.step + zero_grad).
    A trailing partial window (``num_steps % grad_accum != 0``) is flushed
    at the end with grads rescaled to the window mean.

    Intermediate micro-batches run under ``strategy.no_sync(model)`` so
    DDP/FSDP skip cross-rank gradient sync until the window closes.
    Strategies without sync suppression fall back to a null context
    (correct, just extra sync traffic).

    ``post_step_hook`` semantics: invoked immediately after every
    ``strategy.step(optimizer)`` — exactly once per optimizer step (never
    per micro-batch), including the flushed tail window — and BEFORE
    ``step_callback``, so checkpointing inside ``step_callback`` observes
    post-hook state (W22: this is where the JEPA EMA update lives, and EMA
    state must already be advanced when a checkpoint is written).

    ``step_callback`` semantics: called only on optimizer steps as
    ``step_callback(optimizer_step_no, window_mean_loss)`` with a 1-indexed
    optimizer-step counter. Checkpoint steps derived from this callback
    therefore count optimizer steps, not micro-batches. With
    ``grad_accum == 1`` every micro-batch is an optimizer step, so behavior
    is byte-for-byte identical to the pre-accumulation loop.

    Returns:
        Per-optimizer-step mean losses. Length is
        ``ceil(num_steps / grad_accum)`` (== ``num_steps`` when
        ``grad_accum == 1``).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")
    if loss_fn is None:
        loss_fn = default_loss_fn

    losses: list[float] = []
    window_losses: list[float] = []
    optimizer_step = 0
    iterator = iter(batch_iter)
    for step in range(num_steps):
        batch = next(iterator)
        # Sync grads on the window-closing micro-batch, and always on the
        # final micro-batch so a partial tail window is reduced + flushed.
        is_sync_step = (step + 1) % grad_accum == 0 or step == num_steps - 1
        ctx = nullcontext() if is_sync_step else strategy.no_sync(model)
        with ctx:
            loss = loss_fn(model, batch)
            if not loss.requires_grad:
                raise RuntimeError(
                    f"loss_fn returned a tensor with requires_grad=False; "
                    f"check that the model parameters have requires_grad=True."
                )
            strategy.backward(loss / grad_accum)
        window_losses.append(float(loss.detach().item()))

        if is_sync_step:
            window_size = len(window_losses)
            if window_size < grad_accum:
                _rescale_grads(optimizer, grad_accum / window_size)
            strategy.step(optimizer)
            if post_step_hook is not None:
                post_step_hook()
            optimizer_step += 1
            window_mean = sum(window_losses) / window_size
            losses.append(window_mean)
            window_losses.clear()
            if step_callback is not None:
                step_callback(optimizer_step, window_mean)
    return losses


__all__ = ["run_training_step", "run_training_loop", "default_loss_fn"]
