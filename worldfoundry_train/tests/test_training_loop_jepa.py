"""Unit tests for post_step_hook wiring in training_loop (W22.4).

Design §9.1 test_training_loop_jepa ①: the hook must fire exactly once per
optimizer step, immediately AFTER strategy.step(optimizer), before
step_callback, including the flushed tail window under grad_accum.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import SGD

from worldfoundry_train.parallel.strategy import DistributedStrategy
from worldfoundry_train.training_loop import run_training_loop, run_training_step


class _OrderRecordingStrategy(DistributedStrategy):
    """No-op strategy that appends "step" to a shared event log."""

    def __init__(self, events: list[str]):
        self.events = events

    def wrap(self, model: nn.Module) -> nn.Module:
        return model

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self, optimizer) -> None:
        optimizer.step()
        optimizer.zero_grad()
        self.events.append("step")


def _linear() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(1, 1, bias=False)


def _squared_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return (model(batch["x"]) ** 2).sum()


def _batches(n: int) -> list[dict[str, torch.Tensor]]:
    return [{"x": torch.tensor([[1.0]])} for _ in range(n)]


def test_post_step_hook_called_after_optimizer_step():
    events: list[str] = []
    model = _linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _OrderRecordingStrategy(events)
    run_training_step(
        model, _batches(1)[0], strategy, optimizer,
        loss_fn=_squared_loss, post_step_hook=lambda: events.append("hook"),
    )
    assert events == ["step", "hook"]


def test_post_step_hook_none_by_default():
    model = _linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _OrderRecordingStrategy([])
    loss = run_training_step(
        model, _batches(1)[0], strategy, optimizer, loss_fn=_squared_loss
    )
    assert loss > 0  # simply must not error


def test_hook_fires_once_per_optimizer_step_under_grad_accum():
    events: list[str] = []
    model = _linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _OrderRecordingStrategy(events)
    losses = run_training_loop(
        model, iter(_batches(6)), num_steps=6,
        strategy=strategy, optimizer=optimizer, loss_fn=_squared_loss,
        grad_accum=2, post_step_hook=lambda: events.append("hook"),
    )
    assert len(losses) == 3
    assert events == ["step", "hook"] * 3


def test_hook_fires_on_tail_window_flush_and_precedes_step_callback():
    events: list[str] = []
    model = _linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _OrderRecordingStrategy(events)
    losses = run_training_loop(
        model, iter(_batches(3)), num_steps=3,
        strategy=strategy, optimizer=optimizer, loss_fn=_squared_loss,
        grad_accum=2,
        post_step_hook=lambda: events.append("hook"),
        step_callback=lambda s, l: events.append(f"cb{s}"),
    )
    # windows: [m1, m2] -> step+hook+cb1 ; [m3 flush] -> step+hook+cb2
    assert len(losses) == 2
    assert events == ["step", "hook", "cb1", "step", "hook", "cb2"]
