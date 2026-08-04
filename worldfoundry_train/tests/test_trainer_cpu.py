"""CPU e2e: real tiny V-JEPA 2 + real LoRA + trainer.run_training.

NOT NPU-gated -- runs by default. Validates the W14 orchestrator end-to-end:
1. ``trainer.run_training`` accepts ``base_model`` + ``batch_fn`` injections
   and runs 3 real forward+backward+step passes through the real loop
2. Trainable weights change (real gradient flow through LoRA -> base)
3. ``_build_dataset`` monkey-patch path with fake Vjepa2Dataset-compatible
   dataset (collator round-trip; no manual batch_fn injection needed)
4. ``_default_num_steps`` derivation drops the partial last batch when
   ``ds_len % per_device_batch != 0``
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from transformers import VJEPA2Config, VJEPA2Model

from worldforge_train.model.lora_wrapper import build_lora_model
from worldforge_train.parallel.fsdp import FSDPStrategy
from worldforge_train.parallel.strategy import DistributedStrategy
from worldforge_train.trainer import run_training


def _build_real_tiny_vjepa2() -> VJEPA2Model:
    cfg = VJEPA2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=224,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        frames_per_clip=1,
        drop_path_rate=0.0,
    )
    return VJEPA2Model(cfg).eval()


def _synthetic_batch(i: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(i)
    return {
        "pixel_values_videos": torch.from_numpy(
            rng.random((1, 1, 3, 224, 224), dtype=np.float32)
        ),
        # 224/16=14, 14*14=196 patches. Half masked so MSE has non-empty input
        # (all-False mask -> empty indexed tensor -> NaN loss).
        "bool_masked_pos": torch.from_numpy(
            rng.integers(0, 2, size=(1, 196), dtype=bool)
        ),
    }


def _cfg() -> dict:
    """Full cfg with data stub -- trainer calls _build_dataset before checking
    batch_fn, so cfg['data'] must exist even when batch_fn is injected."""
    return {
        "data": {"manifest": "/x", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-4,
                     "log_every_n": 1},
    }


class _FakeVJ2Dataset:
    """Fake Vjepa2Dataset-compatible dataset for collator round-trip tests.

    Returns numpy ``{frames, mask}`` dicts without disk/video decoding.
    Trainer wraps with collate_vjepa2_batch; we never construct a real
    Vjepa2Dataset (avoids pyav and the disk-I/O path).
    """

    def __init__(self, n: int, T: int = 1, H: int = 16, W: int = 16):
        self.n, self.T, self.H, self.W = n, T, H, W

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1)
        return {
            "frames": rng.random((self.T, 3, self.H, self.W), dtype=np.float32),
            "mask": rng.integers(0, 2, size=(self.T, 1, 1), dtype=bool),
        }


class _Passthrough(DistributedStrategy):
    """Inline no-op strategy (mirrors test_training_loop.py pattern)."""

    def wrap(self, model):
        return model

    def backward(self, loss):
        loss.backward()

    def step(self, optimizer):
        optimizer.step()


class _Tiny(nn.Module):
    """Tiny model with q_proj discoverable by leaf-name match in
    discover_lora_targets (suffix_only=True)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3 * 16 * 16, 4)
        self.linear = nn.Linear(3 * 16 * 16, 4)

    def forward(self, *, pixel_values_videos, bool_masked_pos):
        return type("O", (), {
            "last_hidden_state": self.q_proj(pixel_values_videos.flatten(2))
        })()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    np.random.seed(42)


def test_real_vjepa2_three_step_training_via_trainer():
    base = _build_real_tiny_vjepa2()
    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeVJ2Dataset(n=4)
    try:
        losses = run_training(
            cfg=_cfg(),
            strategy=FSDPStrategy(),
            base_model=base,
            num_steps=3,
            batch_fn=_synthetic_batch,
            loss_fn=None,  # default_loss_fn
            logger=lambda s, l: None,
        )
    finally:
        trainer_mod._build_dataset = original
    assert len(losses) == 3
    for v in losses:
        assert v == v
        assert v >= 0.0


def test_trainer_loss_decreases_or_weights_change_on_real_vjepa2():
    base = _build_real_tiny_vjepa2()
    lora_result = build_lora_model(base, r=4, alpha=8)
    model = lora_result.peft_model
    initial = {
        n: p.detach().clone()
        for n, p in model.named_parameters() if p.requires_grad
    }
    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeVJ2Dataset(n=4)
    try:
        run_training(
            cfg=_cfg(),
            strategy=FSDPStrategy(),
            base_model=base,
            num_steps=3,
            batch_fn=_synthetic_batch,
            loss_fn=None,
            logger=lambda s, l: None,
        )
    finally:
        trainer_mod._build_dataset = original
    changed = sum(
        1 for n, p in model.named_parameters()
        if p.requires_grad and not torch.equal(initial[n], p.detach())
    )
    assert changed >= 1


def test_trainer_collator_round_trip_with_fake_dataset():
    """No VJEPA 2 weights; uses _Tiny model + _FakeVJ2Dataset +
    real collator_vjepa2_batch via trainer.run_training (no batch_fn)."""
    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeVJ2Dataset(n=8)
    try:
        model = _Tiny()

        def loss_fn(m, batch):
            return m(
                pixel_values_videos=batch["pixel_values_videos"],
                bool_masked_pos=batch["bool_masked_pos"],
            ).last_hidden_state.sum()

        losses = run_training(
            cfg=_cfg(),
            strategy=_Passthrough(),
            base_model=model,
            loss_fn=loss_fn,
            logger=lambda s, l: None,
        )
        # epochs=1, ds_size=8, batch=2 -> 4 steps (cfg overrides via _tiny_cfg
        # is unnecessary -- _cfg uses per_device_batch=1: 8 // 1 = 8 steps)
    finally:
        trainer_mod._build_dataset = original
    # _cfg uses per_device_batch=1 -> 8 steps
    assert len(losses) == 8
    for v in losses:
        assert v == v


def test_trainer_uses_real_ds_len_for_num_steps_when_no_override():
    """When ds_len not divisible by batch_size, drops partial last batch."""
    cfg = _cfg()
    cfg["training"]["per_device_batch"] = 2
    import worldforge_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeVJ2Dataset(n=7)
    try:
        model = _Tiny()

        def loss_fn(m, batch):
            return m(
                pixel_values_videos=batch["pixel_values_videos"],
                bool_masked_pos=batch["bool_masked_pos"],
            ).last_hidden_state.sum()

        losses = run_training(
            cfg=cfg,
            strategy=_Passthrough(),
            base_model=model,
            loss_fn=loss_fn,
            logger=lambda s, l: None,
        )
    finally:
        trainer_mod._build_dataset = original
    # epochs=1, ds_size=7, batch=2 -> 7 // 2 = 3 steps (drops last 1)
    assert len(losses) == 3

