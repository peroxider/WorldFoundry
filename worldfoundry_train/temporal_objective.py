"""Temporal JEPA training objective: past K -> future H world-model loss (W26).

Sister to :class:`worldfoundry_train.jepa_objective.JEPAObjective`, but for
the temporal "predict next frames" task instead of the masked-reconstruction
task. Reuses the W22 EMA target-encoder machinery and the W24
:class:`TemporalPredictor`; the only new code is the data slicing + loss
composition.

Per-step dataflow
-----------------
- ``video`` is ``(B, T, C, H, W)`` with ``T = K + H`` (the trainer asserts).
- ``h_past = online_encoder(video[:, :K])`` -> ``(B, N_past, encoder_dim)``
- ``h_future_target = target_encoder(video[:, K:]).detach()``
- ``pred = temporal_predictor(h_past, future_grid)`` -> ``(B, N_future, D)``
- ``loss = smooth_l1(pred, h_future_target)`` (or l1 / cosine; same
  :func:`jepa_loss` as :class:`JEPAObjective`).

EMA cadence: :meth:`after_step` must be called once per *optimizer step*
(post ``optimizer.step()``), matching JEPAObjective.

Out of scope (W26 v1)
---------------------
- Multi-mask K>1 (the W22.8 multi-mask pipeline is for the masked task; the
  temporal task has only one valid (K, H) per request).
- Multi-step roll-out (predict past K, predict next H, use that as input to
  predict the next H, ...). W26 v1 is single-step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from worldfoundry_train.jepa_objective import LOSS_TYPES, jepa_loss
from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.temporal_predictor import TemporalPredictor

__all__ = ["TemporalObjective"]


@dataclass(frozen=True)
class _PeakMemoryAPI:
    """NPU/CUDA peak-memory reset+peak pair (mirrors JEPAObjective)."""

    name: str
    reset: Callable[[], None]
    peak: Callable[[], int]


def _peak_memory_api() -> _PeakMemoryAPI | None:
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return _PeakMemoryAPI(
                name="npu",
                reset=torch.npu.reset_peak_memory_stats,
                peak=torch.npu.max_memory_allocated,
            )
    except (ImportError, AttributeError):
        pass
    if torch.cuda.is_available():
        return _PeakMemoryAPI(
            name="cuda",
            reset=torch.cuda.reset_peak_memory_stats,
            peak=torch.cuda.max_memory_allocated,
        )
    return None


class TemporalObjective(nn.Module):
    """Past K -> future H world-model training objective (W26).

    Args:
        online_encoder: trainable encoder (grads flow).
        target_encoder: EMA-tracked twin of ``online_encoder`` (no grads).
        temporal_predictor: W24 :class:`TemporalPredictor`.
        past_frames: K (past-frame count).
        future_frames: H (future-frame count).
        past_grid: ``(K, Hg, Wg)`` of past frame grid.
        future_grid: ``(H, Hg', Wg')`` of future frame grid.
        momentum_start: EMA momentum at step 0.
        momentum_end: EMA momentum at schedule horizon.
        schedule_steps: cosine schedule horizon.
        update_every: EMA update cadence.
        loss_type: ``"smooth_l1"`` | ``"l1"`` | ``"cosine"``.
        beta: smooth-L1 Huber threshold (ignored for non-smooth_l1).
        timing: optional dict for NPU/CUDA peak-memory instrumentation
            (same key as :class:`JEPAObjective`: ``ema_gather_peak_mb``).
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        temporal_predictor: TemporalPredictor,
        *,
        past_frames: int,
        future_frames: int,
        past_grid: tuple[int, int, int],
        future_grid: tuple[int, int, int],
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
        schedule_steps: int = 10000,
        update_every: int = 1,
        loss_type: str = "smooth_l1",
        beta: float = 1.0,
        timing: dict | None = None,
    ) -> None:
        super().__init__()
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.predictor = temporal_predictor
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.past_grid = past_grid
        self.future_grid = future_grid
        self.loss_type = loss_type
        self.beta = float(beta)
        self.timing = timing

        # Mirror EMA construction in JEPAObjective: online -> trainable
        # param list; target -> EMA-tracked twin.
        self.ema = EMAUpdater(
            online_params=list(online_encoder.parameters()),
            target_params=list(target_encoder.parameters()),
            momentum_start=momentum_start,
            momentum_end=momentum_end,
            schedule="cosine",
            schedule_steps=schedule_steps,
            update_every=update_every,
        )
        # Freeze the target encoder (grads come from EMA, not optimizer).
        for p in target_encoder.parameters():
            p.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Compute the temporal-prediction loss for a ``(B, T, C, H, W)`` video.

        Returns a scalar loss tensor (graph-attached for backward).
        """
        if video.ndim != 5:
            raise ValueError(
                f"video must be (B, T, C, H, W), got {tuple(video.shape)}"
            )
        if video.shape[1] != self.past_frames + self.future_frames:
            raise ValueError(
                f"video has {video.shape[1]} frames; expected "
                f"past_frames + future_frames = "
                f"{self.past_frames + self.future_frames}"
            )
        past_video = video[:, : self.past_frames]
        future_video = video[:, self.past_frames:]

        h_past = self.online_encoder(pixel_values_videos=past_video).last_hidden_state
        pred = self.predictor(h_past, future_grid=self.future_grid)
        with torch.no_grad():
            h_future_target = self.target_encoder(
                pixel_values_videos=future_video
            ).last_hidden_state
        return jepa_loss(pred, h_future_target, self.loss_type, self.beta)

    def after_step(self) -> None:
        """EMA update; call once per optimizer step."""
        peak_api = _peak_memory_api() if self.timing is not None else None
        if peak_api is not None:
            peak_api.reset()
        self.ema.update()
        if peak_api is not None:
            mb = float(peak_api.peak()) / (1 << 20)
            prev = self.timing.get("ema_gather_peak_mb", 0.0)  # type: ignore[union-attr]
            if mb > prev:
                self.timing["ema_gather_peak_mb"] = mb  # type: ignore[index]

    # -- checkpoint plumbing (consumed by inference-side loader, W24.2) -----

    def state_dict(self) -> dict:
        return {
            "predictor_state": self.predictor.state_dict(),
            "ema_state": self.ema.state_dict(),
            "ema_step": self.ema.step,
            "temporal_signature": self.signature(),
            "temporal_hyperparams": self.hyperparams(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.predictor.load_state_dict(state["predictor_state"])
        self.ema.load_state_dict(state["ema_state"])

    def signature(self) -> dict:
        return {
            "pred_dim": self.predictor.predictor_dim,
            "pred_depth": len(self.predictor.blocks.layers),
            "pred_heads": self.predictor.num_heads,
            "encoder_dim": self.predictor.encoder_dim,
            "ema_schedule": self.ema.schedule,
            "loss_type": self.loss_type,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
            "past_grid": list(self.past_grid),
            "future_grid": list(self.future_grid),
        }

    def hyperparams(self) -> dict:
        return {
            "momentum_end": float(self.ema.momentum_end),
            "loss_beta": float(self.beta),
        }
