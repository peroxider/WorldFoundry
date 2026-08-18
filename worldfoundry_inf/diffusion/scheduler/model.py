"""Schedulers for the latent→frame diffusion pipeline (M3-T1).

Why this exists
---------------
M3-T1 replaces the v1 CPU random-projection stub with real diffusion
building blocks. This module is the **scheduler** half: the discrete
time-step bookkeeping + sampling loops that turn the DiT denoiser
(:mod:`worldfoundry_inf.diffusion.transformer`) into a generative
model. Two schedulers are shipped:

* :class:`DDIMScheduler` — deterministic DDIM (Song et al. 2021)
  sampling over a user-specified number of steps.
* :class:`FlowMatchingScheduler` — rectified-flow (Lipman et al. 2023)
  sampling with a linear noise schedule, the current default for
  DiT-class text/video models (SD3 / SVD style).

Both expose the same surface so the pipeline
(:mod:`worldfoundry_inf.diffusion.backends.vae_diffusion`) can swap
between them without changing call sites:

* :meth:`SchedulerBase.sample` — run the full denoising loop with a
  callable denoiser: ``denoiser(noisy_latent, t) -> predicted_clean``.
* :meth:`SchedulerBase.add_noise` — forward noising for training
  (``(x0, noise, t) -> x_t``).
* :meth:`SchedulerBase.timesteps` — the discrete step schedule.

All ops are pure PyTorch (CPU/Atlas portable); no external scheduler
dependency (diffusers is optional and only used for weight loading).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class SchedulerConfig:
    """Shared scheduler hyperparameters.

    Attributes:
        num_train_steps: continuous-time training steps (DDIM uses
            this to derive the beta schedule).
        beta_start / beta_end: linear beta schedule endpoints.
    """

    num_train_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02


class SchedulerBase(ABC):
    """Common scheduler surface (sampling + training noising)."""

    name: str = "base"

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    @staticmethod
    def _broadcast_t(t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B,)`` t to broadcast against ``x0``'s spatial rank."""
        return t.view(t.shape[0], *([1] * (x0.ndim - 1)))

    # ------------------------------------------------------------------ #
    # training-time noising
    # ------------------------------------------------------------------ #
    @abstractmethod
    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process: ``x_t = f(x0, noise, t)`` (continuous t in [0,1))."""

    # ------------------------------------------------------------------ #
    # sampling schedule
    # ------------------------------------------------------------------ #
    @abstractmethod
    def timesteps(self, num_steps: int) -> torch.Tensor:
        """Return the discrete sampling schedule (``(num_steps,)`` floats)."""

    # ------------------------------------------------------------------ #
    # full sampling loop
    # ------------------------------------------------------------------ #
    @abstractmethod
    def sample(
        self,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        *,
        num_steps: int = 20,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the denoising loop.

        Args:
            denoiser: ``(noisy, t_float) -> predicted_clean`` (the DiT
                transformer forward, batched over ``t``).
            shape: output latent shape ``(B, *latent_dims)``.
            num_steps: number of sampling steps.
            noise: initial noise ``(shape,)``; default ``randn``.

        Returns:
            ``(shape,)`` denoised latent.
        """


# ---------------------------------------------------------------------------
# DDIM (deterministic)
# ---------------------------------------------------------------------------


class DDIMScheduler(SchedulerBase):
    """Deterministic DDIM sampler (Song et al. 2021).

    Uses a standard linear beta schedule to define
    ``alpha_cumprod``, then the DDIM update with ``eta=0``
    (fully deterministic given the initial noise).
    """

    name = "ddim"

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        super().__init__(config)
        n = self.config.num_train_steps
        betas = torch.linspace(
            self.config.beta_start, self.config.beta_end, n
        )
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        # Small epsilon so log/sqrt are stable at the last step.
        self.register_buffer("alpha_cumprod", alpha_cumprod.clamp_min(1e-8))

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    # ------------------------------------------------------------------ #
    def _alpha_cumprod_at(self, t_float: torch.Tensor) -> torch.Tensor:
        """Continuous alpha_cumprod via linear interpolation in step space.

        Returns ``(B,)`` (flat per-sample values); callers reshape to
        match their tensor's spatial rank.
        """
        n = self.config.num_train_steps
        pos = t_float * (n - 1)  # 0..n-1
        lo = pos.floor().long().clamp(0, n - 1)
        hi = (lo + 1).clamp(0, n - 1)
        frac = pos - lo.float()
        acp = self.alpha_cumprod[lo]
        acp_next = self.alpha_cumprod[hi]
        return acp + frac * (acp_next - acp)

    @staticmethod
    def _broadcast_t(t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B,)`` t to broadcast against ``x0``'s spatial rank."""
        return t.view(t.shape[0], *([1] * (x0.ndim - 1)))

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_b = self._broadcast_t(t, x0)
        acp = self._alpha_cumprod_at(t).view_as(t_b)  # (B, 1, ...)
        sqrt_acp = acp.sqrt()
        sqrt_one_minus = (1.0 - acp).sqrt()
        return sqrt_acp * x0 + sqrt_one_minus * noise

    def timesteps(self, num_steps: int) -> torch.Tensor:
        # Descending schedule: 1 -> 0 (exclusive end for stability).
        return torch.linspace(1.0, 0.0, num_steps + 1)[:-1]

    def sample(
        self,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        *,
        num_steps: int = 20,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = noise if noise is not None else torch.randn(*shape)
        batch = x.shape[0]
        ts = self.timesteps(num_steps)
        for i in range(num_steps):
            t_cur = ts[i].expand(batch)
            t_next = (
                ts[i + 1].expand(batch)
                if i + 1 < num_steps
                else ts[i].expand(batch)
            )
            pred = denoiser(x, t_cur)  # predicted x0
            acp_cur = self._alpha_cumprod_at(t_cur).view(
                batch, *([1] * (x.ndim - 1))
            )
            acp_next = self._alpha_cumprod_at(t_next).view(
                batch, *([1] * (x.ndim - 1))
            )
            sqrt_acp_cur = acp_cur.sqrt()
            sqrt_acp_next = acp_next.sqrt()
            eps = (x - sqrt_acp_cur * pred) / (1.0 - acp_cur).sqrt().clamp_min(1e-8)
            # DDIM update (eta=0): x_{t-1} = sqrt(acp_next) * x0_pred
            #                        + sqrt(1-acp_next) * eps
            x = sqrt_acp_next * pred + (1.0 - acp_next).sqrt() * eps
        return x


# ---------------------------------------------------------------------------
# Flow Matching (rectified flow)
# ---------------------------------------------------------------------------


class FlowMatchingScheduler(SchedulerBase):
    """Rectified-flow sampler with a linear noise schedule.

    Forward: ``x_t = (1-t) * x0 + t * noise`` for ``t in [0,1)``.
    Reverse: Euler steps over the velocity field predicted by the
    denoiser: ``x_{t-dt} = x_t - dt * v(x_t, t)``.
    """

    name = "flow_matching"

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_b = self._broadcast_t(t, x0)
        return (1.0 - t_b) * x0 + t_b * noise

    def timesteps(self, num_steps: int) -> torch.Tensor:
        # Descending: 1 -> 0 (exclusive), matching the DDIM schedule
        # convention so sampling loops can share the loop skeleton.
        return torch.linspace(1.0, 0.0, num_steps + 1)[:-1]

    def sample(
        self,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        *,
        num_steps: int = 20,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = noise if noise is not None else torch.randn(*shape)
        batch = x.shape[0]
        ts = self.timesteps(num_steps)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_cur = ts[i].expand(batch)
            v = denoiser(x, t_cur)  # predicted velocity
            x = x - dt * v
        return x


__all__ = [
    "SchedulerConfig",
    "SchedulerBase",
    "DDIMScheduler",
    "FlowMatchingScheduler",
]
