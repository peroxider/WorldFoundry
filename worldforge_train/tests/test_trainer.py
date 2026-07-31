"""Unit tests for worldforge_train.trainer (mock-free, fast)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from worldforge_train.parallel.strategy import DistributedStrategy
from worldforge_train.trainer import run_training


class _PassthroughStrategy(DistributedStrategy):
    """Inline no-op strategy for trainer tests."""

    def __init__(self):
        self.backward_calls: list[float] = []
        self.step_calls: list[int] = []
        self.wrapped_models: list = []

    def wrap(self, model):
        self.wrapped_models.append(model)
        return model

    def backward(self, loss):
        self.backward_calls.append(float(loss.item()))
        loss.backward()

    def step(self, optimizer):
        self.step_calls.append(len(optimizer.param_groups))
        optimizer.step()


class _TinyLinearModel(nn.Module):
    """Mimics VJEPA2Model's forward(pixel_values_videos, bool_masked_pos) API."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3 * 16 * 16, 4)
        self.linear = nn.Linear(3 * 16 * 16, 4)

    def forward(self, *, pixel_values_videos, bool_masked_pos):
        return SimpleNamespace(
            last_hidden_state=self.q_proj(pixel_values_videos.flatten(2))
        )


def _fake_batch():
    return {
        "pixel_values_videos": torch.zeros(1, 1, 3, 16, 16),
        "bool_masked_pos": torch.zeros(1, 1, 4, dtype=torch.bool),
    }


class _FakeDataset:
    """Fake dataset: returns synthetic numpy samples without disk/av decoding."""

    def __init__(self, n=10, T=1, H=16, W=16, seed=0):
        self.n = n
        self.T = T
        self.H = H
        self.W = W
        np.random.seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1)
        return {
            "frames": rng.random((self.T, 3, self.H, self.W), dtype=np.float32),
            "mask": rng.integers(0, 2, size=(self.T, 1, 1), dtype=bool),
        }


def _cfg(**overrides):
    base = {
        "data": {"manifest": "/nonexistent", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 2, "epochs": 1, "lr": 1e-4,
                     "log_every_n": 2},
    }
    base.update(overrides)
    return base


def _loss_fn(model, batch):
    return model(
        pixel_values_videos=batch["pixel_values_videos"],
        bool_masked_pos=batch["bool_masked_pos"],
    ).last_hidden_state.sum()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    np.random.seed(42)


def test_run_training_returns_per_step_losses():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    counter = {"n": 0}

    def batch_fn(_i):
        counter["n"] += 1
        return _fake_batch()

    import worldforge_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        losses = run_training(
            cfg, strategy, base_model=model, num_steps=3,
            batch_fn=batch_fn, loss_fn=_loss_fn,
            logger=lambda s, l: None,
        )
        assert len(losses) == 3
        for v in losses:
            assert isinstance(v, float)
            assert v == v
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_num_steps_derived_from_yaml_when_unset():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    counter = {"n": 0}

    def batch_fn(_i):
        counter["n"] += 1
        return _fake_batch()

    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model,
            batch_fn=batch_fn, loss_fn=_loss_fn,
            logger=lambda s, l: None,
        )
        assert counter["n"] == 2
    finally:
        trainer_mod._build_dataset = original


def test_run_training_cli_steps_override_beats_yaml_derivation():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    counter = {"n": 0}

    def batch_fn(_i):
        counter["n"] += 1
        return _fake_batch()

    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=8)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=batch_fn, loss_fn=_loss_fn,
            logger=lambda s, l: None,
        )
        assert counter["n"] == 1
    finally:
        trainer_mod._build_dataset = original


def test_run_training_empty_dataset_raises_value_error():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=0)
    try:
        with pytest.raises(ValueError, match="Vjepa2Dataset is empty"):
            run_training(
                cfg, strategy, base_model=model, num_steps=1,
                batch_fn=lambda i: _fake_batch(),
                logger=lambda s, l: None,
            )
    finally:
        trainer_mod._build_dataset = original


def test_run_training_zero_num_steps_raises_value_error():
    cfg = _cfg()
    cfg["training"]["epochs"] = 0
    cfg["training"]["per_device_batch"] = 99
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        with pytest.raises(ValueError, match="num_steps="):
            run_training(
                cfg, strategy, base_model=model,
                batch_fn=lambda i: _fake_batch(),
                logger=lambda s, l: None,
            )
    finally:
        trainer_mod._build_dataset = original


def test_run_training_negative_num_steps_kwarg_raises():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        with pytest.raises(ValueError, match="num_steps="):
            run_training(
                cfg, strategy, base_model=model, num_steps=-1,
                batch_fn=lambda i: _fake_batch(),
                logger=lambda s, l: None,
            )
    finally:
        trainer_mod._build_dataset = original


def test_run_training_optimizer_filters_to_trainable_params_only():
    cfg = _cfg()
    model = _TinyLinearModel()
    for n, p in model.named_parameters():
        if "q_proj" not in n:
            p.requires_grad_(False)
    strategy = _PassthroughStrategy()
    captured = {}
    import worldforge_train.trainer as trainer_mod
    original_opt = trainer_mod._build_optimizer

    def capturing(model, lr):
        opt = original_opt(model, lr)
        captured["params"] = [tuple(p.shape) for p in opt.param_groups[0]["params"]]
        return opt

    original_ds = trainer_mod._build_dataset
    trainer_mod._build_optimizer = capturing
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, l: None,
        )
        # Only q_proj.weight + q_proj.bias (2 tensors)
        assert len(captured["params"]) == 2
    finally:
        trainer_mod._build_optimizer = original_opt
        trainer_mod._build_dataset = original_ds


def test_run_training_default_logger_emits_at_log_every_n():
    cfg = _cfg()
    cfg["training"]["log_every_n"] = 2
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    log_calls = []
    import worldforge_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=8)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=4,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, l: log_calls.append((s, l)),
        )
        assert [s for s, _ in log_calls] == [2, 4]
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_injected_logger_overrides_default():
    cfg = _cfg()
    cfg["training"]["log_every_n"] = 1
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    log_calls = []
    import worldforge_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=2,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, l: log_calls.append((s, l)),
        )
        assert len(log_calls) == 2
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_injected_loss_fn_overrides_default():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    sentinel = {"called": False}

    def sentinel_loss(model, batch):
        sentinel["called"] = True
        return torch.tensor(0.0, requires_grad=True)

    import worldforge_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=lambda i: _fake_batch(),
            loss_fn=sentinel_loss,
            logger=lambda s, l: None,
        )
        assert sentinel["called"]
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_strategy_wrap_called_with_lora_wrapped_model():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldforge_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, l: None,
        )
        assert len(strategy.wrapped_models) == 1
        wrapped = strategy.wrapped_models[0]
        # After LoRA wrap, the trainer hands strategy.wrap a peft.PeftModel
        # (which is NOT a bare _TinyLinearModel).
        assert not isinstance(wrapped, _TinyLinearModel)
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_strategy_backward_called_once_per_step():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldforge_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=3,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, l: None,
        )
        assert len(strategy.backward_calls) == 3
    finally:
        trainer_mod._build_dataset = original_ds