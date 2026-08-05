"""EMA updater for the JEPA target encoder (W22.2).

Design source: docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md §5.

EMA formula (per optimizer step ``s``)::

    theta_tgt <- m_s * theta_tgt + (1 - m_s) * theta_online

Key properties (design §5.1–§5.5):

- Momentum follows a cosine warmup ``m_start -> m_end`` over
  ``schedule_steps`` optimizer steps (I-JEPA/V-JEPA convention, §5.2).
- The updater operates on an explicit *paired* parameter list (online /
  target). In the LoRA fine-tuning regime this is exactly the set of
  trainable encoder params (= LoRA adapters); it never assumes LoRA
  structure, so a future full-FT profile reuses the same code path (§5.3).
- EMA state is held as **fp32 master copies**; each update lerps in fp32 and
  writes back to the (possibly bf16) target params, avoiding bf16 rounding
  drift when ``m -> 1.0`` (§5.5).
- The update counter advances on **optimizer steps**, not micro-batches — the
  caller (``JEPAObjective.after_step``) is wired to the training loop's
  post-optimizer-step hook, so gradient accumulation windows do not inflate
  the schedule (§5.2; aligns with ``training_loop.run_training_loop``
  ``step_callback`` semantics).

FSDP gather-then-update (§5.4) is W22.6 scope; this class implements the
single-rank / already-gathered lerp path.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn

__all__ = ["EMAUpdater"]


class EMAUpdater:
    """Cosine-scheduled EMA over paired (online, target) parameter lists.

    Args:
        online_params: parameters being trained (read-only here).
        target_params: same-length parameters of the target encoder; updated
            in place by :meth:`update`.
        momentum_start: ``m`` at schedule step 0 (0.996, I-JEPA/V-JEPA).
        momentum_end: ``m`` at the schedule horizon (1.0 -> frozen target).
        schedule: ``"cosine"`` | ``"constant"`` (constant pins ``m_start``).
        schedule_steps: horizon S. Required for ``"cosine"``; the trainer
            resolves ``None`` in config to the total optimizer-step count.
        update_every: apply EMA every k optimizer steps; the effective
            momentum is compensated as ``m_eff = m_s ** k`` (§5.4 note).
    """

    def __init__(
        self,
        online_params: list[nn.Parameter],
        target_params: list[nn.Parameter],
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
        schedule: str = "cosine",
        schedule_steps: int | None = None,
        update_every: int = 1,
        names: list[str] | None = None,
    ) -> None:
        if len(online_params) != len(target_params):
            raise ValueError(
                f"online/target param lists differ in length: "
                f"{len(online_params)} vs {len(target_params)}"
            )
        if not online_params:
            raise ValueError("EMAUpdater received an empty parameter list")
        for i, (on, tgt) in enumerate(zip(online_params, target_params)):
            if on.shape != tgt.shape:
                raise ValueError(
                    f"param {i}: online shape {tuple(on.shape)} != "
                    f"target shape {tuple(tgt.shape)}"
                )
        if schedule not in ("cosine", "constant"):
            raise ValueError(f"unknown EMA schedule {schedule!r}")
        if schedule == "cosine" and (schedule_steps is None or schedule_steps < 1):
            raise ValueError(
                "cosine EMA schedule requires schedule_steps >= 1 "
                "(trainer resolves config `null` to total optimizer steps)"
            )
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        if names is not None and len(names) != len(online_params):
            raise ValueError(
                f"names length {len(names)} != online params {len(online_params)}"
            )

        self.online_params = list(online_params)
        self.target_params = list(target_params)
        self.names = list(names) if names is not None else None
        self.momentum_start = float(momentum_start)
        self.momentum_end = float(momentum_end)
        self.schedule = schedule
        self.schedule_steps = schedule_steps
        self.update_every = update_every
        # fp32 EMA master copies (§5.5), initialized from the target params.
        self._master: list[torch.Tensor] = [
            p.detach().to(torch.float32).clone() for p in self.target_params
        ]
        self.step = 0  # optimizer-step counter
        # Optional FSDP gather source (W22.6, design §5.4): the wrapped
        # online model, bound post-wrap via bind_fsdp_source().
        self._fsdp_source: nn.Module | None = None
        # W22.7 throughput instrumentation. DEFAULT OFF (None = zero
        # overhead); JEPAObjective shares its env-gated timing dict here.
        # Keys written when set: ema_gather_s, ema_update_s, ema_updates.
        self.timing: dict[str, float] | None = None

    def bind_fsdp_source(self, wrapped_model: nn.Module) -> None:
        """Bind the FSDP-wrapped online model for gather-then-update EMA.

        Requires ``names`` (the state_dict FQNs of ``online_params``). The
        gather only activates when the bound model is truly FSDP-wrapped on
        an initialized multi-rank process group; single-rank / DDP / test
        paths keep reading the (already full) local params.
        """
        if self.names is None:
            raise ValueError(
                "bind_fsdp_source requires EMAUpdater(names=...) so gathered "
                "full params can be matched to the EMA pairs"
            )
        self._fsdp_source = wrapped_model

    def momentum_at(self, step: int) -> float:
        """Momentum ``m_s`` at schedule step ``s`` (clamped to [0, S])."""
        if self.schedule == "constant":
            return self.momentum_start
        s = min(max(step, 0), self.schedule_steps)
        frac = 0.5 * (1.0 + math.cos(math.pi * s / self.schedule_steps))
        return self.momentum_end - (self.momentum_end - self.momentum_start) * frac

    @property
    def momentum(self) -> float:
        """Momentum used at the current step counter (for logging)."""
        return self.momentum_at(self.step)

    def _online_f32(self) -> list[torch.Tensor]:
        """Full fp32 online params on every rank.

        Under true FSDP (FULL_SHARD, ``use_orig_params=True``) the local
        ``online_params`` are sharded views; reading them directly would lerp
        the target encoder toward garbage. We therefore gather the full
        trainable weights via FULL_STATE_DICT with ``rank0_only=False,
        offload_to_cpu=False`` (design §5.4 gather-then-update; the dispatch
        check mirrors ``checkpoint._is_fsdp_module``), then filter to the
        EMA-tracked names. All ranks gather identical input and lerp
        identically, so target copies stay bitwise consistent with no extra
        broadcast.

        NOTE: FSDP1 FULL_STATE_DICT materializes the FULL model (frozen
        backbone included) transiently on every rank — for ViT-L bf16 that
        is ~2.4GB per rank per EMA step, not the ~10MB the design estimated
        for a LoRA-only allgather. Accepted for v1 correctness; the lever is
        ``ema_update_every > 1``. Single-rank / DDP / unwrapped paths read
        the (already full) local params directly.
        """
        model = self._fsdp_source
        if model is not None:
            import torch.distributed as dist

            from worldfoundry_train.checkpoint import _is_fsdp_module

            if (
                dist.is_initialized()
                and dist.get_world_size() > 1
                and _is_fsdp_module(model)
            ):
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                )
                from torch.distributed.fsdp.api import (
                    FullStateDictConfig, StateDictType,
                )

                t0 = time.perf_counter() if self.timing is not None else None
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                ):
                    full = model.state_dict()
                if self.timing is not None:
                    self.timing["ema_gather_s"] = (
                        self.timing.get("ema_gather_s", 0.0)
                        + (time.perf_counter() - t0)
                    )
                return [full[n].detach().to(torch.float32) for n in self.names]
        return [p.detach().to(torch.float32) for p in self.online_params]

    def update(self) -> float | None:
        """Advance one optimizer step; apply EMA when the cadence hits.

        Returns the effective momentum when an update was applied, else
        ``None``. Must be called once per *optimizer step* (post
        ``optimizer.step()``), never per micro-batch.
        """
        self.step += 1
        if self.step % self.update_every != 0:
            return None
        m_eff = self.momentum_at(self.step) ** self.update_every
        weight = 1.0 - m_eff
        t0 = time.perf_counter() if self.timing is not None else None
        with torch.no_grad():
            online_f32 = self._online_f32()
            # Shape drift guard (review M3, 2026-08-05): if someone rebinds
            # EMAUpdater to a different online param subset (e.g. LoRA rank
            # changed at resume, or a different module was mistakenly attached),
            # _master and online_f32 may silently have different shapes and
            # _foreach_lerp_ will produce garbage on the target. Assert
            # up-front so the misconfiguration surfaces immediately.
            for i, (master, on) in enumerate(zip(self._master, online_f32)):
                if master.shape != on.shape:
                    name = self.names[i] if self.names else f"param[{i}]"
                    raise RuntimeError(
                        f"EMA shape mismatch on {name}: master {tuple(master.shape)} "
                        f"vs online {tuple(on.shape)}; refusing to lerp. This "
                        f"usually means the EMA was bound to a stale online "
                        f"subset (e.g. LoRA config changed since construction)."
                    )
            try:
                torch._foreach_lerp_(self._master, online_f32, weight)
            except (RuntimeError, TypeError) as exc:  # NPU fallback, §8.3
                import warnings

                warnings.warn(
                    f"torch._foreach_lerp_ failed ({exc!r}); falling back to "
                    f"per-tensor lerp_",
                    stacklevel=2,
                )
                for buf, on in zip(self._master, online_f32):
                    buf.lerp_(on, weight)
            for tgt, buf in zip(self.target_params, self._master):
                tgt.copy_(buf)  # copy_ casts fp32 master -> target dtype
        if self.timing is not None:
            self.timing["ema_update_s"] = (
                self.timing.get("ema_update_s", 0.0) + (time.perf_counter() - t0)
            )
            self.timing["ema_updates"] = self.timing.get("ema_updates", 0.0) + 1
        return m_eff

    # -- checkpoint plumbing (consumed by W22.5) ---------------------------

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "master": [buf.clone() for buf in self._master],
            "momentum_start": self.momentum_start,
            "momentum_end": self.momentum_end,
            "schedule": self.schedule,
            "schedule_steps": self.schedule_steps,
            "update_every": self.update_every,
        }

    def load_state_dict(self, state: dict) -> None:
        if len(state["master"]) != len(self._master):
            raise ValueError(
                f"EMA state has {len(state['master'])} params, expected "
                f"{len(self._master)}"
            )
        for buf, saved in zip(self._master, state["master"]):
            if buf.shape != saved.shape:
                raise ValueError(
                    f"EMA master shape mismatch: {tuple(saved.shape)} vs "
                    f"{tuple(buf.shape)}"
                )
            buf.copy_(saved.to(torch.float32))
        self.step = int(state["step"])
        # Restore schedule config so a resumed run continues the same curve.
        self.momentum_start = float(state["momentum_start"])
        self.momentum_end = float(state["momentum_end"])
        self.schedule = state["schedule"]
        self.schedule_steps = state["schedule_steps"]
        self.update_every = int(state["update_every"])
        # Re-sync target params from the restored master copies.
        with torch.no_grad():
            for tgt, buf in zip(self.target_params, self._master):
                tgt.copy_(buf)
