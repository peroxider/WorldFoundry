"""Unit tests for gradient accumulation in worldforge_train.training_loop.

Covers (CPU, no process group):
- optimizer-step semantics: strategy.step fires once per accumulation window
- loss scaling: accumulated update equals window-mean-gradient update
- tail flush: partial final window is stepped with rescaled gradients
- no_sync: intermediate micro-batches run under strategy.no_sync, window
  closers and the final micro-batch do not
- step_callback: fires on optimizer steps (1-indexed), not micro-batches
- grad_accum=1 regression: identical to the legacy per-micro-batch loop
- trainer wiring: cfg["training"]["grad_accum"] reaches the loop and
  checkpoint steps count optimizer steps
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from worldforge_train.parallel.strategy import DistributedStrategy
from worldforge_train.training_loop import run_training_loop, run_training_step


class _RecordingStrategy(DistributedStrategy):
    """No-op strategy that records backward sync-state and step calls.

    ``step`` performs a real optimizer.step + zero_grad so accumulation
    semantics match production strategies (FSDPStrategy/DDPStrategy both
    zero grads inside step).
    """

    def __init__(self):
        self.backward_calls: list[float] = []
        self.backward_under_no_sync: list[bool] = []
        self.step_calls = 0
        self._in_no_sync = False

    def wrap(self, model: nn.Module) -> nn.Module:
        return model

    def no_sync(self, model: nn.Module):
        strategy = self

        class _Ctx:
            def __enter__(self_):
                strategy._in_no_sync = True

            def __exit__(self_, *exc):
                strategy._in_no_sync = False
                return False

        return _Ctx()

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls.append(float(loss.item()))
        self.backward_under_no_sync.append(self._in_no_sync)
        loss.backward()

    def step(self, optimizer) -> None:
        self.step_calls += 1
        optimizer.step()
        optimizer.zero_grad()


def _scalar_batches(values: list[float]) -> list[dict[str, torch.Tensor]]:
    return [{"x": torch.tensor([[v]])} for v in values]


def _squared_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return (model(batch["x"]) ** 2).sum()


def _fresh_linear() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(1, 1, bias=False)


# --- optimizer-step semantics ---

def test_grad_accum_steps_once_per_window():
    model = _fresh_linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _RecordingStrategy()

    losses = run_training_loop(
        model, iter(_scalar_batches([1.0] * 8)), num_steps=8,
        strategy=strategy, optimizer=optimizer, loss_fn=_squared_loss,
        grad_accum=4,
    )
    assert strategy.step_calls == 2
    assert len(strategy.backward_calls) == 8
    assert len(losses) == 2


def test_grad_accum_rejects_less_than_one():
    model = _fresh_linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _RecordingStrategy()
    with pytest.raises(ValueError, match="grad_accum must be >= 1"):
        run_training_loop(
            model, iter(_scalar_batches([1.0])), num_steps=1,
            strategy=strategy, optimizer=optimizer, loss_fn=_squared_loss,
            grad_accum=0,
        )


# --- loss scaling correctness ---

def test_grad_accum_update_equals_window_mean_gradient():
    """2 micro-batches with grad_accum=2 == one step on the mean loss."""
    batches = _scalar_batches([1.0, 3.0])

    model_ref = _fresh_linear()
    opt_ref = SGD(model_ref.parameters(), lr=0.1)
    mean_loss = sum(_squared_loss(model_ref, b) for b in batches) / 2
    mean_loss.backward()
    opt_ref.step()

    model_acc = _fresh_linear()
    opt_acc = SGD(model_acc.parameters(), lr=0.1)
    strategy = _RecordingStrategy()
    losses = run_training_loop(
        model_acc, iter(batches), num_steps=2,
        strategy=strategy, optimizer=opt_acc, loss_fn=_squared_loss,
        grad_accum=2,
    )

    assert len(losses) == 1
    assert torch.allclose(model_acc.weight, model_ref.weight, atol=1e-7), (
        "accumulated update must equal the window-mean-gradient update"
    )


# --- tail flush ---

def test_grad_accum_tail_window_flushed_with_rescaled_grads():
    """num_steps=3, grad_accum=2 -> windows [b1,b2] + [b3]; the tail window
    of 1 must apply the gradient of b3 alone (rescaled, not halved)."""
    batches = _scalar_batches([1.0, 3.0, 5.0])

    model_ref = _fresh_linear()
    opt_ref = SGD(model_ref.parameters(), lr=0.1)
    ((_squared_loss(model_ref, batches[0]) + _squared_loss(model_ref, batches[1])) / 2).backward()
    opt_ref.step()
    opt_ref.zero_grad()
    _squared_loss(model_ref, batches[2]).backward()
    opt_ref.step()

    model_acc = _fresh_linear()
    opt_acc = SGD(model_acc.parameters(), lr=0.1)
    strategy = _RecordingStrategy()
    cb_calls: list[tuple[int, float]] = []
    losses = run_training_loop(
        model_acc, iter(batches), num_steps=3,
        strategy=strategy, optimizer=opt_acc, loss_fn=_squared_loss,
        grad_accum=2,
        step_callback=lambda s, l: cb_calls.append((s, l)),
    )

    assert strategy.step_calls == 2
    assert [s for s, _ in cb_calls] == [1, 2]
    assert len(losses) == 2
    assert torch.allclose(model_acc.weight, model_ref.weight, atol=1e-7), (
        "tail window gradient must be rescaled to the window mean"
    )


def test_grad_accum_larger_than_num_steps_flushes_single_window():
    model = _fresh_linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _RecordingStrategy()
    losses = run_training_loop(
        model, iter(_scalar_batches([1.0, 2.0, 3.0])), num_steps=3,
        strategy=strategy, optimizer=optimizer, loss_fn=_squared_loss,
        grad_accum=8,
    )
    assert strategy.step_calls == 1
    assert len(losses) == 1


# --- no_sync usage pattern ---

def test_no_sync_used_only_for_non_closing_micro_batches():
    model = _fresh_linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _RecordingStrategy()
    run_training_loop(
        model, iter(_scalar_batches([1.0] * 5)), num_steps=5,
        strategy=strategy, optimizer=optimizer, loss_fn=_squared_loss,
        grad_accum=2,
    )
    # windows: [m1*, m2], [m3*, m4], [m5 flush]  (* = under no_sync)
    # the final micro-batch always syncs so the tail window is reduced.
    assert strategy.backward_under_no_sync == [True, False, True, False, False]


# --- step_callback semantics ---

def test_step_callback_counts_optimizer_steps_with_window_mean_loss():
    model = _fresh_linear()
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _RecordingStrategy()
    batches = _scalar_batches([1.0, 1.0, 3.0, 3.0])
    expected_means = [
        float((_squared_loss(_fresh_linear(), batches[0]).item()
               + _squared_loss(_fresh_linear(), batches[1]).item()) / 2),
    ]
    cb_calls: list[tuple[int, float]] = []
    losses = run_training_loop(
        model, iter(batches), num_steps=4,
        strategy=strategy, optimizer=optimizer, loss_fn=_squared_loss,
        grad_accum=2,
        step_callback=lambda s, l: cb_calls.append((s, l)),
    )
    assert [s for s, _ in cb_calls] == [1, 2]
    assert len(losses) == 2
    # first window has identical batches -> window mean == micro loss
    assert cb_calls[0][1] == pytest.approx(expected_means[0])


# --- grad_accum=1 regression equivalence ---

def test_grad_accum_one_matches_legacy_step_loop():
    """grad_accum=1 must reproduce the pre-accumulation loop exactly:
    one optimizer step per micro-batch, per-micro-batch losses, and the
    same final weights as a manual run_training_step loop."""
    batches = _scalar_batches([1.0, 2.0, 3.0, 4.0])

    model_legacy = _fresh_linear()
    opt_legacy = SGD(model_legacy.parameters(), lr=0.1)
    strategy_legacy = _RecordingStrategy()
    legacy_losses = [
        run_training_step(model_legacy, b, strategy_legacy, opt_legacy,
                          loss_fn=_squared_loss)
        for b in batches
    ]

    model_acc = _fresh_linear()
    opt_acc = SGD(model_acc.parameters(), lr=0.1)
    strategy_acc = _RecordingStrategy()
    cb_calls: list[tuple[int, float]] = []
    losses = run_training_loop(
        model_acc, iter(batches), num_steps=4,
        strategy=strategy_acc, optimizer=opt_acc, loss_fn=_squared_loss,
        grad_accum=1,
        step_callback=lambda s, l: cb_calls.append((s, l)),
    )

    assert losses == pytest.approx(legacy_losses)
    assert [s for s, _ in cb_calls] == [1, 2, 3, 4]
    assert strategy_acc.step_calls == 4
    assert strategy_acc.backward_under_no_sync == [False] * 4
    assert torch.allclose(model_acc.weight, model_legacy.weight, atol=1e-7)


# --- trainer wiring (cfg["training"]["grad_accum"]) ---

from types import SimpleNamespace  # noqa: E402

from worldforge_train.trainer import run_training  # noqa: E402


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


class _FakeDataset:
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1)
        return {
            "frames": rng.random((1, 3, 16, 16), dtype=np.float32),
            "mask": rng.integers(0, 2, size=(1, 1, 1), dtype=bool),
        }


def _cfg(**training_overrides):
    training = {"per_device_batch": 2, "epochs": 1, "lr": 1e-4, "log_every_n": 2}
    training.update(training_overrides)
    return {
        "data": {"manifest": "/nonexistent", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": training,
    }


def _fake_batch():
    return {
        "pixel_values_videos": torch.zeros(1, 1, 3, 16, 16),
        "bool_masked_pos": torch.zeros(1, 1, 4, dtype=torch.bool),
    }


def _loss_fn(model, batch):
    return model(
        pixel_values_videos=batch["pixel_values_videos"],
        bool_masked_pos=batch["bool_masked_pos"],
    ).last_hidden_state.sum()


@pytest.fixture
def _patch_dataset(monkeypatch):
    import worldforge_train.trainer as trainer_mod
    monkeypatch.setattr(
        trainer_mod, "_build_dataset", lambda cfg: _FakeDataset(n=8)
    )


@pytest.fixture
def _patch_dist_uninitialized(monkeypatch):
    import torch.distributed as _dist
    monkeypatch.setattr(_dist, "is_initialized", lambda: False)


def test_trainer_consumes_grad_accum_from_cfg(_patch_dataset):
    cfg = _cfg(grad_accum=2)
    strategy = _RecordingStrategy()
    losses = run_training(
        cfg, strategy, base_model=_TinyLinearModel(), num_steps=4,
        batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
        logger=lambda s, l: None,
    )
    assert len(strategy.backward_calls) == 4  # all micro-batches backward
    assert strategy.step_calls == 2           # one optimizer step per window
    assert len(losses) == 2                   # per-optimizer-step losses


def test_trainer_rejects_grad_accum_below_one(_patch_dataset):
    cfg = _cfg(grad_accum=0)
    with pytest.raises(ValueError, match="grad_accum must be >= 1"):
        run_training(
            cfg, _RecordingStrategy(), base_model=_TinyLinearModel(),
            num_steps=2, batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, l: None,
        )


def test_trainer_checkpoint_step_counts_optimizer_steps(
    tmp_path, _patch_dataset, _patch_dist_uninitialized
):
    """4 micro-batches with grad_accum=2 -> final checkpoint step_2.pt."""
    cfg = _cfg(grad_accum=2)
    run_training(
        cfg, _RecordingStrategy(), base_model=_TinyLinearModel(), num_steps=4,
        batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
        logger=lambda s, l: None,
        checkpoint_dir=tmp_path,
    )
    saved = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert saved == ["step_2.pt"]


def test_trainer_resume_converts_optimizer_step_to_micro_batches(
    tmp_path, _patch_dataset, _patch_dist_uninitialized
):
    """Resume at optimizer step 2 with grad_accum=2 and target 6
    micro-batches -> 2 remaining micro-batches -> 1 more optimizer step
    -> final checkpoint step_3.pt."""
    cfg = _cfg(grad_accum=2)
    run_training(
        cfg, _RecordingStrategy(), base_model=_TinyLinearModel(), num_steps=4,
        batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
        logger=lambda s, l: None,
        checkpoint_dir=tmp_path,
    )
    saved = tmp_path / "step_2.pt"
    assert saved.is_file()

    losses = run_training(
        cfg, _RecordingStrategy(), base_model=_TinyLinearModel(), num_steps=6,
        batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
        logger=lambda s, l: None,
        resume_from=saved,
        checkpoint_dir=tmp_path,
    )
    assert len(losses) == 1
    assert (tmp_path / "step_3.pt").is_file()
