"""Unit tests for trainer.make_strategy profile mapping + DDPStrategy (CPU).

Covers:
- "ddp" -> DDPStrategy, "fsdp" -> FSDPStrategy
- legacy "train" alias -> FSDPStrategy (backward compat)
- unknown profile -> ValueError (fail-fast, no silent default)
- DDPStrategy single-rank contract: wrap identity, step + zero_grad,
  no_sync null-context for unwrapped models, real no_sync for DDP modules
- train.py parallel resolution: CLI > cfg[training][parallel] > "ddp",
  unknown cfg value -> ValueError
- run_ddp_training / run_fsdp_training consume cfg[training][parallel]
"""
from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import SGD

from worldforge_train.parallel.ddp import DDPStrategy
from worldforge_train.parallel.fsdp import FSDPStrategy
from worldforge_train.trainer import make_strategy


# --- make_strategy mapping ---

def test_ddp_profile_returns_ddp_strategy():
    assert isinstance(make_strategy("ddp"), DDPStrategy)


def test_fsdp_profile_returns_fsdp_strategy():
    assert isinstance(make_strategy("fsdp"), FSDPStrategy)


def test_train_legacy_alias_returns_fsdp_strategy():
    """Backward compat: the Phase-1 "train" profile still maps to FSDP."""
    assert isinstance(make_strategy("train"), FSDPStrategy)


def test_unknown_profile_raises_value_error():
    with pytest.raises(ValueError, match="bogus"):
        make_strategy("bogus")


# --- DDPStrategy single-rank contract ---

def test_ddp_strategy_wrap_identity_on_single_rank():
    model = nn.Linear(4, 2)
    strategy = DDPStrategy()
    assert strategy.wrap(model) is model


def test_ddp_strategy_step_steps_and_zeroes_grad():
    model = nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = DDPStrategy()
    strategy.wrap(model)

    loss = (model(torch.ones(1, 1)) ** 2).sum()
    strategy.backward(loss)
    assert model.weight.grad is not None

    before = model.weight.detach().clone()
    strategy.step(optimizer)
    assert not torch.equal(before, model.weight.detach())
    assert model.weight.grad is None or torch.all(model.weight.grad == 0)


def test_ddp_strategy_no_sync_is_null_context_for_unwrapped_model():
    strategy = DDPStrategy()
    model = nn.Linear(2, 1)
    # Unwrapped model: no cross-rank sync exists, context must be a no-op.
    with strategy.no_sync(model):
        pass


def test_ddp_strategy_no_sync_delegates_to_real_ddp_module():
    """A DDP-wrapped module gets its native no_sync context."""
    from torch.nn.parallel import DistributedDataParallel

    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        ddp_model = DistributedDataParallel(nn.Linear(2, 1))
        strategy = DDPStrategy()
        with strategy.no_sync(ddp_model):
            pass  # must not raise; real DDP.no_sync entered + exited
    finally:
        dist.destroy_process_group()


# --- train.py parallel resolution ---

def test_resolve_parallel_cli_overrides_cfg():
    from types import SimpleNamespace
    from worldforge_train.train import _resolve_parallel
    args = SimpleNamespace(parallel="fsdp")
    cfg = {"training": {"parallel": "ddp"}}
    assert _resolve_parallel(args, cfg) == "fsdp"


def test_resolve_parallel_uses_cfg_when_cli_absent():
    from types import SimpleNamespace
    from worldforge_train.train import _resolve_parallel
    args = SimpleNamespace(parallel=None)
    cfg = {"training": {"parallel": "fsdp"}}
    assert _resolve_parallel(args, cfg) == "fsdp"


def test_resolve_parallel_defaults_to_ddp():
    from types import SimpleNamespace
    from worldforge_train.train import _resolve_parallel
    args = SimpleNamespace(parallel=None)
    assert _resolve_parallel(args, {}) == "ddp"
    assert _resolve_parallel(args, {"training": {}}) == "ddp"


def test_resolve_parallel_unknown_cfg_value_raises():
    from types import SimpleNamespace
    from worldforge_train.train import _resolve_parallel
    args = SimpleNamespace(parallel=None)
    with pytest.raises(ValueError, match="tp"):
        _resolve_parallel(args, {"training": {"parallel": "tp"}})


def test_parse_args_parallel_defaults_to_none():
    from worldforge_train.train import parse_args
    assert parse_args(["--config", "x.yaml"]).parallel is None


# --- run_*_training consume cfg[training][parallel] ---

@pytest.mark.parametrize(
    "fn_name,cfg_parallel,expected",
    [
        ("run_ddp_training", "fsdp", "fsdp"),
        ("run_ddp_training", "ddp", "ddp"),
        ("run_ddp_training", None, "ddp"),
        ("run_fsdp_training", "ddp", "ddp"),
        ("run_fsdp_training", None, "fsdp"),
    ],
)
def test_run_training_consumes_cfg_parallel(monkeypatch, fn_name, cfg_parallel, expected):
    import worldforge_train.train as train_mod

    captured: dict[str, str] = {}

    def fake_make_strategy(profile):
        captured["profile"] = profile
        raise RuntimeError("sentinel: stop before training")

    monkeypatch.setattr(
        "worldforge_train.trainer.make_strategy", fake_make_strategy
    )

    cfg = {"training": {}}
    if cfg_parallel is not None:
        cfg["training"]["parallel"] = cfg_parallel

    fn = getattr(train_mod, fn_name)
    with pytest.raises(RuntimeError, match="sentinel"):
        fn(cfg, local_rank=0, global_rank=0, world_size=1)
    assert captured["profile"] == expected
