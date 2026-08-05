"""Integration tests for the W22.4 trainer JEPA wiring.

Covers: jepa.enabled=true end-to-end (predictor built, EMA advances per
optimizer step, ema_schedule_steps=null resolution with grad_accum), env
override WORLDFORGE_JEPA=off, feature-flag equivalence (jepa.enabled=false
== no jepa key, byte-for-byte), cpu_offload mutual exclusion, loss_fn
conflict, and num_masks<1 rejection.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

import worldfoundry_train.trainer as trainer_mod
from worldfoundry_train.jepa_objective import JEPAObjective
from worldfoundry_train.parallel.strategy import DistributedStrategy
from worldfoundry_train.trainer import run_training


def _tiny_vjepa2() -> VJEPA2Model:
    cfg = VJEPA2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=224, patch_size=16, tubelet_size=1, in_chans=3,
        frames_per_clip=1, drop_path_rate=0.0,
    )
    return VJEPA2Model(cfg)


class _FakeGridDataset:
    """Mask grid (1, 14, 14) matching the tiny model's 196 patches."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1)
        return {
            "frames": rng.random((1, 3, 224, 224), dtype=np.float32),
            "mask": rng.integers(0, 2, size=(1, 14, 14), dtype=bool),
        }


class _Passthrough(DistributedStrategy):
    cpu_offload = False

    def wrap(self, model):
        return model

    def backward(self, loss):
        loss.backward()

    def step(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()


def _cfg(jepa: dict | None = None, grad_accum: int = 1) -> dict:
    cfg = {
        "data": {"manifest": "/x", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-4,
                     "log_every_n": 100, "grad_accum": grad_accum},
    }
    if jepa is not None:
        cfg["jepa"] = jepa
    return cfg


def _jepa_cfg(**overrides) -> dict:
    jepa = {
        "enabled": True,
        "predictor": {"embed_dim": 32, "depth": 2, "num_heads": 2,
                      "mlp_ratio": 4.0, "lr_scale": 2.0},
        "ema": {"momentum_start": 0.9, "momentum_end": 0.99,
                "schedule": "cosine", "schedule_steps": None,
                "update_every": 1},
        "loss": {"type": "smooth_l1", "beta": 1.0, "num_masks": 1},
    }
    jepa.update(overrides)
    return jepa


def _batch(i: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(i)
    return {
        "pixel_values_videos": torch.from_numpy(
            rng.random((1, 1, 3, 224, 224), dtype=np.float32)
        ),
        "bool_masked_pos": torch.from_numpy(
            rng.integers(0, 2, size=(1, 196), dtype=bool)
        ),
    }


@pytest.fixture
def _patch_dataset(monkeypatch):
    monkeypatch.setattr(
        trainer_mod, "_build_dataset", lambda cfg: _FakeGridDataset(n=4)
    )


@pytest.fixture
def _capture_objective(monkeypatch):
    """Capture every JEPAObjective built by the trainer."""
    captured: list[JEPAObjective] = []
    original = JEPAObjective.from_online_model

    def _spy(cls, *args, **kwargs):
        obj = original(*args, **kwargs)
        captured.append(obj)
        return obj

    monkeypatch.setattr(
        trainer_mod.JEPAObjective, "from_online_model",
        classmethod(_spy),
    )
    return captured


def test_trainer_jepa_end_to_end(_patch_dataset, _capture_objective):
    torch.manual_seed(42)
    losses = run_training(
        _cfg(jepa=_jepa_cfg()), _Passthrough(),
        base_model=_tiny_vjepa2(), num_steps=3, batch_fn=_batch,
        logger=lambda s, l: None,
    )
    assert len(losses) == 3
    assert all(v == v and v >= 0.0 for v in losses)

    assert len(_capture_objective) == 1
    obj = _capture_objective[0]
    # EMA advanced once per optimizer step; schedule_steps=null resolved to
    # the total optimizer-step count (3 micro / accum 1 = 3).
    assert obj.ema.step == 3
    assert obj.ema.schedule_steps == 3
    # Predictor got the configured lr_scale group.
    assert obj.predictor.predictor_dim == 32
    assert obj.last_target_norm is not None


def test_trainer_jepa_schedule_steps_resolve_with_grad_accum(
    _patch_dataset, _capture_objective
):
    torch.manual_seed(42)
    losses = run_training(
        _cfg(jepa=_jepa_cfg(), grad_accum=2), _Passthrough(),
        base_model=_tiny_vjepa2(), num_steps=5, batch_fn=_batch,
        logger=lambda s, l: None,
    )
    # 5 micro-batches, accum 2 -> windows [2, 2, 1flush] = 3 optimizer steps.
    assert len(losses) == 3
    obj = _capture_objective[0]
    assert obj.ema.step == 3           # EMA per optimizer step, not micro-batch
    assert obj.ema.schedule_steps == 3  # ceil(5/2)


def test_env_override_forces_legacy_path(_patch_dataset, _capture_objective,
                                         monkeypatch):
    monkeypatch.setenv("WORLDFORGE_JEPA", "off")
    torch.manual_seed(42)
    losses = run_training(
        _cfg(jepa=_jepa_cfg()), _Passthrough(),
        base_model=_tiny_vjepa2(), num_steps=2, batch_fn=_batch,
        logger=lambda s, l: None,
    )
    assert len(losses) == 2
    assert _capture_objective == []  # JEPA never constructed


def test_feature_flag_disabled_is_byte_equivalent_to_no_jepa_key(
    _patch_dataset,
):
    """jepa.enabled=false must behave exactly like a pre-W22 cfg (§9.3)."""
    torch.manual_seed(123)
    losses_plain = run_training(
        _cfg(jepa=None), _Passthrough(),
        base_model=_tiny_vjepa2(), num_steps=3, batch_fn=_batch,
        logger=lambda s, l: None,
    )
    torch.manual_seed(123)
    losses_flag_off = run_training(
        _cfg(jepa={"enabled": False}), _Passthrough(),
        base_model=_tiny_vjepa2(), num_steps=3, batch_fn=_batch,
        logger=lambda s, l: None,
    )
    assert losses_plain == losses_flag_off


def test_jepa_rejects_injected_loss_fn(_patch_dataset):
    with pytest.raises(ValueError, match="loss_fn"):
        run_training(
            _cfg(jepa=_jepa_cfg()), _Passthrough(),
            base_model=_tiny_vjepa2(), num_steps=1, batch_fn=_batch,
            loss_fn=lambda m, b: torch.tensor(0.0, requires_grad=True),
            logger=lambda s, l: None,
        )


def test_jepa_rejects_cpu_offload(_patch_dataset):
    class _OffloadStrategy(_Passthrough):
        cpu_offload = True

    with pytest.raises(ValueError, match="cpu_offload"):
        run_training(
            _cfg(jepa=_jepa_cfg()), _OffloadStrategy(),
            base_model=_tiny_vjepa2(), num_steps=1, batch_fn=_batch,
            logger=lambda s, l: None,
        )


def test_jepa_rejects_num_masks_below_one(_patch_dataset):
    jepa = _jepa_cfg()
    jepa["loss"] = {"type": "smooth_l1", "beta": 1.0, "num_masks": 0}
    with pytest.raises(ValueError, match="num_masks"):
        run_training(
            _cfg(jepa=jepa), _Passthrough(),
            base_model=_tiny_vjepa2(), num_steps=1, batch_fn=_batch,
            logger=lambda s, l: None,
        )


def test_predictor_lr_scale_reaches_optimizer(_patch_dataset,
                                              _capture_objective, monkeypatch):
    captured_opt = []
    original = trainer_mod._build_optimizer

    def _spy(*args, **kwargs):
        opt = original(*args, **kwargs)
        captured_opt.append(opt)
        return opt

    monkeypatch.setattr(trainer_mod, "_build_optimizer", _spy)
    torch.manual_seed(42)
    run_training(
        _cfg(jepa=_jepa_cfg()), _Passthrough(),
        base_model=_tiny_vjepa2(), num_steps=1, batch_fn=_batch,
        logger=lambda s, l: None,
    )
    opt = captured_opt[0]
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-4, 2e-4]  # base lr and lr * lr_scale=2.0
