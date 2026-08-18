"""Unit tests for the M3-T1 schedulers (DDIM + Flow Matching).

Covers training-time noising (forward process), the descending
timestep schedule, the full sampling loop with a mock denoiser, and
shape/determinism contracts.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_inf.diffusion.scheduler import (
    DDIMScheduler,
    FlowMatchingScheduler,
    SchedulerConfig,
    SchedulerBase,
)


@pytest.fixture()
def cfg() -> SchedulerConfig:
    return SchedulerConfig(num_train_steps=1000, beta_start=0.0001, beta_end=0.02)


@pytest.fixture()
def shape() -> tuple[int, ...]:
    return (2, 4, 4, 4)


def test_scheduler_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        SchedulerBase()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# DDIM
# --------------------------------------------------------------------------- #


def test_ddim_add_noise_shape(cfg: SchedulerConfig, shape: tuple[int, ...]) -> None:
    sched = DDIMScheduler(cfg)
    x0 = torch.randn(*shape)
    noise = torch.randn(*shape)
    t = torch.tensor([0.1, 0.9])
    xt = sched.add_noise(x0, noise, t)
    assert xt.shape == shape
    assert torch.isfinite(xt).all()


def test_ddim_add_noise_interpolates(cfg: SchedulerConfig) -> None:
    """t=0 -> ~x0 (alpha_cumprod[0] = 1-beta_start, not exactly 1); t->1 -> noise."""
    sched = DDIMScheduler(cfg)
    x0 = torch.randn(1, 4, 4, 4)
    noise = torch.randn(1, 4, 4, 4)
    xt0 = sched.add_noise(x0, noise, torch.tensor([0.0]))
    # At t=0, alpha_cumprod[0] = 1 - beta_start = 0.9999 -> xt0 is x0
    # plus a tiny noise leak (sqrt(1-0.9999) ~ 0.01); assert close, not exact.
    assert torch.allclose(xt0, x0, atol=0.05)
    xt1 = sched.add_noise(x0, noise, torch.tensor([1.0]))
    # At t=1, alpha_cumprod -> beta_end-clamped small -> xt ~ noise.
    assert torch.allclose(xt1, noise, atol=0.05)


def test_ddim_timesteps_descending(cfg: SchedulerConfig) -> None:
    ts = DDIMScheduler(cfg).timesteps(5)
    assert ts.shape == (5,)
    assert torch.all(ts[1:] <= ts[:-1])
    assert ts[0] == pytest.approx(1.0)


def test_ddim_sample_shape(cfg: SchedulerConfig, shape: tuple[int, ...]) -> None:
    sched = DDIMScheduler(cfg)
    out = sched.sample(lambda x, t: torch.zeros_like(x), shape, num_steps=4)
    assert out.shape == shape
    assert torch.isfinite(out).all()


def test_ddim_sample_deterministic(cfg: SchedulerConfig, shape: tuple[int, ...]) -> None:
    sched = DDIMScheduler(cfg)
    torch.manual_seed(0)
    noise = torch.randn(*shape)
    out1 = sched.sample(lambda x, t: torch.zeros_like(x), shape, num_steps=6, noise=noise)
    out2 = sched.sample(lambda x, t: torch.zeros_like(x), shape, num_steps=6, noise=noise)
    assert torch.equal(out1, out2)


def test_ddim_sample_denoiser_called_with_float_t(cfg: SchedulerConfig, shape: tuple[int, ...]) -> None:
    """The denoiser receives a float timestep tensor (B,) matching the batch."""
    sched = DDIMScheduler(cfg)
    seen: list[torch.Tensor] = []

    def denoiser(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        seen.append(t.detach().clone())
        return torch.zeros_like(x)

    sched.sample(denoiser, (2, 4, 4, 4), num_steps=3)
    assert seen
    assert all(t.shape == (2,) for t in seen)
    assert all(torch.isfinite(t).all() for t in seen)


# --------------------------------------------------------------------------- #
# Flow Matching
# --------------------------------------------------------------------------- #


def test_flow_add_noise_shape(cfg: SchedulerConfig, shape: tuple[int, ...]) -> None:
    sched = FlowMatchingScheduler(cfg)
    x0 = torch.randn(*shape)
    noise = torch.randn(*shape)
    t = torch.tensor([0.0, 1.0])
    xt = sched.add_noise(x0, noise, t)
    assert xt.shape == shape


def test_flow_add_noise_interpolates(cfg: SchedulerConfig) -> None:
    """Flow matching: x_t = (1-t)*x0 + t*noise."""
    sched = FlowMatchingScheduler(cfg)
    x0 = torch.randn(1, 4, 4, 4)
    noise = torch.randn(1, 4, 4, 4)
    assert torch.allclose(
        sched.add_noise(x0, noise, torch.tensor([0.0])), x0, atol=1e-5
    )
    assert torch.allclose(
        sched.add_noise(x0, noise, torch.tensor([1.0])), noise, atol=1e-5
    )
    mid = sched.add_noise(x0, noise, torch.tensor([0.5]))
    assert torch.allclose(mid, 0.5 * x0 + 0.5 * noise, atol=1e-5)


def test_flow_sample_shape(cfg: SchedulerConfig, shape: tuple[int, ...]) -> None:
    sched = FlowMatchingScheduler(cfg)
    out = sched.sample(lambda x, t: torch.zeros_like(x), shape, num_steps=5)
    assert out.shape == shape


def test_flow_sample_velocity_field_straightens(cfg: SchedulerConfig) -> None:
    """A constant velocity field v = x0 moves x_t linearly to x0.

    With num_steps steps over t in [1, 0), x_final = x_start - v (dt=1),
    i.e. one Euler step covers the whole path for a constant v.
    """
    sched = FlowMatchingScheduler(cfg)
    torch.manual_seed(1)
    x_start = torch.randn(2, 4, 4, 4)

    def const_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x  # v = x_t means flow pulls straight to 0

    out = sched.sample(const_velocity, x_start.shape, num_steps=20, noise=x_start)
    # v=x_t: x_{t-dt} = x_t - dt*x_t = (1-dt)*x_t -> geometric decay.
    assert torch.allclose(out, x_start * (1.0 - 1.0 / 20) ** 20, atol=1e-4)


def test_flow_sample_deterministic(cfg: SchedulerConfig, shape: tuple[int, ...]) -> None:
    sched = FlowMatchingScheduler(cfg)
    torch.manual_seed(2)
    noise = torch.randn(*shape)
    out1 = sched.sample(lambda x, t: torch.zeros_like(x), shape, num_steps=5, noise=noise)
    out2 = sched.sample(lambda x, t: torch.zeros_like(x), shape, num_steps=5, noise=noise)
    assert torch.equal(out1, out2)


# --------------------------------------------------------------------------- #
# Shared surface
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sched_cls", [DDIMScheduler, FlowMatchingScheduler])
def test_schedulers_share_surface(
    sched_cls, cfg: SchedulerConfig, shape: tuple[int, ...]
) -> None:
    sched = sched_cls(cfg)
    assert isinstance(sched, SchedulerBase)
    x0 = torch.randn(*shape)
    noise = torch.randn(*shape)
    xt = sched.add_noise(x0, noise, torch.tensor([0.3, 0.6]))
    assert xt.shape == shape
    ts = sched.timesteps(4)
    assert ts.shape == (4,)
    out = sched.sample(lambda x, t: torch.zeros_like(x), shape, num_steps=4)
    assert out.shape == shape
