"""Unit tests for EMAUpdater (W22.2, design §9.1 test_ema_updater).

Covers: single-step numeric exactness, cosine schedule endpoints/monotonicity,
``update_every=k`` momentum compensation, fp32-master/bf16-writeback
consistency, target isolation from online edits, and state_dict roundtrip.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from worldforge_train.model.ema import EMAUpdater


def _pair(shape=(3, 4), dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    online = nn.Parameter(torch.randn(shape, generator=g).to(dtype))
    target = nn.Parameter(torch.randn(shape, generator=g).to(dtype))
    return online, target


def test_single_step_update_numeric_exactness_m05():
    online, target = _pair()
    online0 = online.detach().clone()
    target0 = target.detach().clone()
    ema = EMAUpdater(
        [online], [target],
        momentum_start=0.5, momentum_end=0.5, schedule="constant",
    )
    ema.update()
    expected = 0.5 * target0 + 0.5 * online0
    assert torch.allclose(target.detach(), expected, atol=1e-7)
    # Online params are never written by the updater.
    assert torch.equal(online.detach(), online0)


def test_cosine_schedule_endpoints_and_monotonicity():
    online, target = _pair()
    ema = EMAUpdater(
        [online], [target],
        momentum_start=0.996, momentum_end=1.0,
        schedule="cosine", schedule_steps=100,
    )
    assert ema.momentum_at(0) == pytest.approx(0.996, abs=1e-12)
    assert ema.momentum_at(100) == pytest.approx(1.0, abs=1e-12)
    seq = [ema.momentum_at(s) for s in range(101)]
    assert all(b >= a for a, b in zip(seq, seq[1:])), "schedule must be monotonic"
    # Midpoint of a cosine from 0.996 -> 1.0 is the mean of the endpoints.
    assert ema.momentum_at(50) == pytest.approx((0.996 + 1.0) / 2, abs=1e-9)


def test_update_every_k_equivalent_to_m_pow_k():
    # Two consecutive m=0.5 updates == one update with m_eff = 0.5**2 = 0.25
    # (online held constant), i.e. target -> 0.25*t0 + 0.75*online.
    online, target = _pair()
    online0 = online.detach().clone()
    target0 = target.detach().clone()
    ema = EMAUpdater(
        [online], [target],
        momentum_start=0.5, momentum_end=0.5, schedule="constant",
        update_every=2,
    )
    assert ema.update() is None  # step 1: cadence not hit, no write
    assert torch.equal(target.detach(), target0)
    m_eff = ema.update()  # step 2: applies
    assert m_eff == pytest.approx(0.25)
    expected = 0.25 * target0 + 0.75 * online0
    assert torch.allclose(target.detach(), expected, atol=1e-7)


def test_fp32_master_with_bf16_writeback():
    online, target = _pair(dtype=torch.bfloat16)
    online0 = online.detach().clone()
    target0 = target.detach().clone()
    ema = EMAUpdater(
        [online], [target],
        momentum_start=0.9, momentum_end=0.9, schedule="constant",
    )
    ema.update()
    master = ema.state_dict()["master"][0]
    assert master.dtype == torch.float32
    expected_f32 = 0.9 * target0.float() + 0.1 * online0.float()
    assert torch.allclose(master, expected_f32, atol=1e-6)
    # bf16 target holds the fp32 master rounded to bf16.
    assert target.dtype == torch.bfloat16
    assert torch.equal(target.detach(), master.to(torch.bfloat16))


def test_target_isolated_from_online_edits_until_update():
    online, target = _pair()
    target0 = target.detach().clone()
    ema = EMAUpdater(
        [online], [target],
        momentum_start=0.996, momentum_end=1.0,
        schedule="cosine", schedule_steps=10,
    )
    with torch.no_grad():
        online.add_(1.0)  # simulate an optimizer step on the online encoder
    assert torch.equal(target.detach(), target0), "target changed before EMA update"
    ema.update()
    assert not torch.equal(target.detach(), target0)


def test_cosine_schedule_requires_steps():
    online, target = _pair()
    with pytest.raises(ValueError, match="schedule_steps"):
        EMAUpdater([online], [target], schedule="cosine", schedule_steps=None)


def test_state_dict_roundtrip_continues_schedule():
    online, target = _pair()
    ema = EMAUpdater(
        [online], [target],
        momentum_start=0.5, momentum_end=0.9,
        schedule="cosine", schedule_steps=10, update_every=1,
    )
    ema.update()
    ema.update()
    state = ema.state_dict()

    online2, target2 = _pair(seed=1)  # fresh params, will be overwritten
    ema2 = EMAUpdater(
        [online2], [target2],
        momentum_start=0.1, momentum_end=0.2,
        schedule="constant", schedule_steps=1,
    )
    ema2.load_state_dict(state)
    assert ema2.step == 2
    assert ema2.schedule == "cosine"
    assert ema2.schedule_steps == 10
    # Target params were re-synced from the restored fp32 master.
    assert torch.allclose(target2.detach(), state["master"][0], atol=1e-7)
    # Momentum continues where it left off (step 2 of 10 on the same curve).
    assert ema2.momentum == pytest.approx(ema.momentum_at(2))
