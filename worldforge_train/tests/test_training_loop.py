"""Unit tests for worldforge_train.training_loop (mock-based, fast)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from worldforge_train.parallel.strategy import DistributedStrategy
from worldforge_train.training_loop import run_training_step, run_training_loop


class _PassthroughStrategy(DistributedStrategy):
    """A no-op DistributedStrategy for tests."""

    def __init__(self):
        self.backward_calls: list[float] = []

    def wrap(self, model: nn.Module) -> nn.Module:
        return model

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls.append(float(loss.item()))
        loss.backward()

    def step(self, optimizer) -> None:
        optimizer.step()


def _fake_batch() -> dict[str, torch.Tensor]:
    return {
        "pixel_values_videos": torch.zeros(1, 1, 3, 16, 16),
        "bool_masked_pos": torch.zeros(1, 1, 4, dtype=torch.bool),
    }


def test_run_training_step_returns_scalar_loss():
    model = nn.Linear(3 * 16 * 16, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()

    def loss_fn(model, batch):
        x = batch["pixel_values_videos"]
        return model(x.flatten(1)).sum()

    initial_bias = model.bias.detach().clone()
    loss_val = run_training_step(model, _fake_batch(), strategy, optimizer, loss_fn=loss_fn)
    assert isinstance(loss_val, float)
    assert strategy.backward_calls == [loss_val]
    assert not torch.equal(initial_bias, model.bias.detach()), "bias must change"


def test_run_training_step_raises_when_loss_does_not_require_grad():
    model = nn.Linear(10, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()

    def bad_loss_fn(model, batch):
        return torch.tensor(0.0)

    with pytest.raises(RuntimeError, match="requires_grad=False"):
        run_training_step(model, _fake_batch(), strategy, optimizer, loss_fn=bad_loss_fn)


def test_run_training_loop_runs_num_steps_and_returns_losses():
    model = nn.Linear(3 * 16 * 16, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()

    def loss_fn(model, batch):
        x = batch["pixel_values_videos"]
        return (model(x.flatten(1)) ** 2).sum()

    batches = [_fake_batch() for _ in range(5)]
    losses = run_training_loop(model, iter(batches), num_steps=3, strategy=strategy,
                                optimizer=optimizer, loss_fn=loss_fn)
    assert len(losses) == 3
    for v in losses:
        assert isinstance(v, float)
        assert v == v


def test_run_training_loop_rejects_zero_or_negative_num_steps():
    model = nn.Linear(10, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()
    with pytest.raises(ValueError, match="num_steps must be positive"):
        run_training_loop(model, iter([_fake_batch()]), num_steps=0,
                          strategy=strategy, optimizer=optimizer)
