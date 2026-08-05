"""True JEPA training objective: predictor + EMA target encoder (W22.3).

Design source: docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md
(§3 architecture, §4.3 dataflow, §5 EMA, §6 loss, §7.1 integration contract).

Replaces the self-distillation proxy loss in ``training_loop.default_loss_fn``
with the real V-JEPA 2 structure:

    h_ctx = online_encoder(full video)          # has grad
    p     = predictor(h_ctx, bool_masked_pos)   # mask-token replacement inside
    y     = ema_target_encoder(full video)      # no_grad, stop-gradient
    loss  = smooth_l1(p, y[mask]) / M           # or l1 / cosine (config)

bool_masked_pos semantics — empirically resolved (design §11 risk #1):
    In transformers 5.1.0, ``VJEPA2Model.forward`` does NOT consume
    ``bool_masked_pos``; the kwarg is swallowed by ``**kwargs`` and the
    encoder output is bit-identical for any mask value (verified by
    ``tests/test_jepa_objective.py::test_hf_encoder_ignores_bool_masked_pos``
    against a random-init VJEPA2Model: outputs with/without the kwarg and with
    two different masks are all ``torch.equal``). The design-doc worry "True
    means masked vs True means predict" is therefore moot at the HF level:
    masking happens exclusively inside :class:`JEPAPredictor` (late masking,
    mask-token replacement), and the online/target forwards differ only in
    weights (online vs EMA) and grad mode. Corollary: the current
    ``default_loss_fn`` mask never alters either forward — it only selects
    regression positions, making the proxy loss an even purer identity
    regression than design §1.2 describes.

    Both forwards also pass ``skip_predictor=True`` so HF's built-in (frozen)
    predictor submodule is not executed — it is dead compute for this
    objective.

EMA cadence: :meth:`JEPAObjective.after_step` must be invoked once per
**optimizer step** (post ``optimizer.step()``), matching the training loop's
``step_callback`` / grad-accum semantics — never per micro-batch (§5.2).
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.predictor import JEPAPredictor
from worldfoundry_train.parallel.strategy import DistributedStrategy

__all__ = ["JEPAObjective", "PredictorGradSyncStrategy", "jepa_loss"]

LOSS_TYPES = ("smooth_l1", "l1", "cosine")


# ---------------------------------------------------------------------------
# Peak-memory API detection (review §2.3, 2026-08-05)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PeakMemoryAPI:
    """A device's reset+peak-memory pair, returned by ``_peak_memory_api``.

    The W22.7 NPU profile rule (addendum §2.3) asks for a per-EMA-step peak
    so we can decide whether to raise ``ema_update_every`` (>4GB => action).
    Holding the reset/peak callables as data lets the EMA wrapper
    (``JEPAObjective.after_step``) call them in pairs without re-importing
    torch_npu on every step.

    NPU is preferred because Atlas always has ``torch_npu`` available
    (lazy bindings, even on CPU build); without an explicit
    ``is_available()`` guard the helpers raise. CUDA is the standard
    fallback for local dev/test rigs.
    """
    name: str
    reset: Callable[[], None]
    peak: Callable[[], int]


def _peak_memory_api() -> _PeakMemoryAPI | None:
    """Detect a peak-memory API, NPU first then CUDA; None on pure CPU."""
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


def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "smooth_l1",
    beta: float = 1.0,
) -> torch.Tensor:
    """JEPA regression loss on masked-position features (design §6).

    ``smooth_l1``/``l1`` are normalized as ``sum / M`` (M = number of masked
    tokens on this rank, §6.2); the target side is NOT normalized. ``cosine``
    is ``mean(1 - cos_sim)`` over tokens — a bounded alignment readout kept
    as a config option, not the default (§6.1).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}"
        )
    m = pred.shape[0]
    if m == 0:
        raise ValueError("jepa_loss received zero masked tokens (M=0)")
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=beta, reduction="sum") / m
    if loss_type == "l1":
        return F.l1_loss(pred, target, reduction="sum") / m
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    raise ValueError(f"unknown loss_type {loss_type!r}; expected one of {LOSS_TYPES}")


class JEPAObjective:
    """Aggregates predictor + EMA target encoder + EMA updater (design §7.1).

    Single aggregation point so the training loop only needs two handles:
    ``loss_fn`` (conforms to ``training_loop.LossFn``) and ``after_step``
    (EMA + momentum advance, wired to the post-optimizer-step hook).
    """

    def __init__(
        self,
        predictor: JEPAPredictor,
        target_encoder: nn.Module,
        ema: EMAUpdater,
        loss_type: str = "smooth_l1",
        beta: float = 1.0,
    ) -> None:
        if loss_type not in LOSS_TYPES:
            raise ValueError(
                f"unknown loss_type {loss_type!r}; expected one of {LOSS_TYPES}"
            )
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.ema = ema
        self.loss_type = loss_type
        self.beta = beta
        # Last target-feature L2 norm, for feature-scale monitoring (§6.2).
        self.last_target_norm: float | None = None
        # W22.7 throughput instrumentation (design §12-② measurement hooks).
        # DEFAULT OFF: ``None`` means zero timing overhead and zero behavior
        # change. Set WORLDFORGE_JEPA_TIMING=1 to accumulate wall-clock
        # seconds into this dict; the EMA updater shares the same dict so
        # gather/update costs land alongside forward costs. Keys:
        #   loss_fn_s, context_forward_s, target_forward_s,
        #   ema_update_s, ema_gather_s, ema_updates
        self.timing: dict[str, float] | None = (
            {} if os.environ.get("WORLDFORGE_JEPA_TIMING", "") == "1" else None
        )
        self.ema.timing = self.timing

    @classmethod
    def from_online_model(
        cls,
        online_model: nn.Module,
        predictor: JEPAPredictor,
        loss_type: str = "smooth_l1",
        beta: float = 1.0,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
        ema_schedule: str = "cosine",
        ema_schedule_steps: int | None = None,
        ema_update_every: int = 1,
    ) -> "JEPAObjective":
        """Build the target encoder + EMA from the online (PEFT-wrapped) model.

        Must be called **before** any FSDP wrap (design §5.5: the target
        encoder is a deepcopy held outside FSDP). EMA pairs every trainable
        online param (= LoRA adapters under the LoRA profile) with its
        same-named target copy; the updater never assumes LoRA structure, so
        a future full-FT profile is covered by the same path (§5.3-4).
        """
        target_encoder = copy.deepcopy(online_model)
        target_encoder.requires_grad_(False)
        target_encoder.eval()  # deterministic target: kills LoRA dropout (§5.5)

        target_by_name = dict(target_encoder.named_parameters())
        online_params: list[nn.Parameter] = []
        target_params: list[nn.Parameter] = []
        names: list[str] = []
        for name, p in online_model.named_parameters():
            if p.requires_grad:
                names.append(name)
                online_params.append(p)
                target_params.append(target_by_name[name])
        if not online_params:
            raise ValueError(
                "online model has no trainable parameters to EMA; check the "
                "LoRA build (G4) before constructing JEPAObjective"
            )

        ema = EMAUpdater(
            online_params,
            target_params,
            momentum_start=ema_momentum_start,
            momentum_end=ema_momentum_end,
            schedule=ema_schedule,
            schedule_steps=ema_schedule_steps,
            update_every=ema_update_every,
            names=names,
        )
        return cls(predictor, target_encoder, ema, loss_type=loss_type, beta=beta)

    def bind_distributed(self, wrapped_model: nn.Module) -> None:
        """Bind the wrapped (possibly FSDP) online model for the EMA gather.

        Call after ``strategy.wrap``. On single-rank / DDP this is a harmless
        no-op dispatch; under true multi-rank FSDP the EMA update switches to
        gather-then-update (design §5.4, see ``EMAUpdater._online_f32``).
        """
        self.ema.bind_fsdp_source(wrapped_model)

    # -- training-loop contract ---------------------------------------------

    def loss_fn(
        self, model: nn.Module, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """JEPA loss; conforms to ``training_loop.LossFn`` (model, batch).

        ``model`` is the online (PEFT-wrapped) encoder. See the module
        docstring for why ``bool_masked_pos`` is NOT forwarded to the HF
        encoder and why both forwards pass ``skip_predictor=True``.

        Multi-mask (W22.8 design §4.1): when the collator supplies
        ``bool_masked_pos_k`` of shape ``(B, K, N)`` the loss averages over
        all K masks. Both encoder forwards are mask-independent, so K scales
        only the predictor and loss work, never the backbone.
        """
        x = batch["pixel_values_videos"]
        if "bool_masked_pos_k" in batch:
            mk = batch["bool_masked_pos_k"]  # (B, K, N)
            masks = [mk[:, i] for i in range(mk.shape[1])]  # K x (B, N)
        else:
            masks = [batch["bool_masked_pos"]]  # K=1 path (W22 contract)

        t0 = time.perf_counter() if self.timing is not None else None
        h_ctx = model(
            pixel_values_videos=x, skip_predictor=True
        ).last_hidden_state  # (B, N, D), has grad
        t1 = time.perf_counter() if self.timing is not None else None
        with torch.no_grad():
            h_tgt = self.target_encoder(
                pixel_values_videos=x, skip_predictor=True
            ).last_hidden_state  # (B, N, D), stop-gradient
        loss = self._loss_for_masks(h_ctx, h_tgt, masks)
        if self.timing is not None:
            t2 = time.perf_counter()
            self.timing["context_forward_s"] = (
                self.timing.get("context_forward_s", 0.0) + (t1 - t0)
            )
            self.timing["target_forward_s"] = (
                self.timing.get("target_forward_s", 0.0) + (t2 - t1)
            )
            self.timing["loss_fn_s"] = (
                self.timing.get("loss_fn_s", 0.0) + (t2 - t0)
            )
        return loss

    def _loss_for_masks(
        self,
        h_ctx: torch.Tensor,
        h_tgt: torch.Tensor,
        masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Mean loss over K independently sampled masks (design §6.2).

        K comes from ``jepa.loss.num_masks``: K=1 uses the batch's
        ``bool_masked_pos``, K>1 the collator's ``bool_masked_pos_k``
        (W22.8 multi-mask pipeline). An all-False mask is skipped; if every
        mask is empty the loss is a graph-attached zero so backward stays
        well-defined.
        """
        if h_tgt.requires_grad:
            raise RuntimeError("target features must be stop-gradient")
        self.last_target_norm = float(h_tgt.norm(dim=-1).mean().item())

        losses: list[torch.Tensor] = []
        for mask in masks:
            if int(mask.sum()) == 0:
                continue
            pred = self.predictor(h_ctx, mask)
            target = h_tgt[mask.bool()]
            losses.append(jepa_loss(pred, target, self.loss_type, self.beta))
        if not losses:
            # Graph-attached zero: anchor on mask_token (always trainable) so
            # backward stays well-defined even if h_ctx carries no grad.
            return h_ctx.sum() * 0.0 + self.predictor.mask_token.sum() * 0.0
        return torch.stack(losses).mean()

    def after_step(self) -> None:
        """EMA update + momentum advance. Call once per optimizer step.

        When ``self.timing`` is enabled (``WORLDFORGE_JEPA_TIMING=1``), the
        EMA gather/update window is wrapped in a device-side peak-memory
        capture (review §2.3 NPU profile) so the experiment driver can
        report ``timing.ema_gather_peak_mb`` and decide whether to raise
        ``jepa.ema.update_every`` when the peak exceeds the 4 GB budget.
        No-op cost when timing is None (default).
        """
        peak_api = _peak_memory_api() if self.timing is not None else None
        if peak_api is not None:
            peak_api.reset()
        self.ema.update()
        if peak_api is not None:
            mb = float(peak_api.peak()) / (1 << 20)
            prev = self.timing.get("ema_gather_peak_mb", 0.0)  # type: ignore[union-attr]
            if mb > prev:
                self.timing["ema_gather_peak_mb"] = mb  # type: ignore[index]

    # -- checkpoint plumbing (consumed by W22.5) -----------------------------

    def state_dict(self) -> dict:
        return {
            "predictor_state": self.predictor.state_dict(),
            "ema_state": self.ema.state_dict(),
            "ema_step": self.ema.step,
            "jepa_signature": self.signature(),
            "jepa_hyperparams": self.hyperparams(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.predictor.load_state_dict(state["predictor_state"])
        self.ema.load_state_dict(state["ema_state"])

    def signature(self) -> dict:
        """Architecture-identifying fields for JEPASignature checks (§7.4)."""
        return {
            "pred_dim": self.predictor.predictor_dim,
            "pred_depth": len(self.predictor.blocks.layers),
            "pred_heads": self.predictor.num_heads,
            "encoder_dim": self.predictor.encoder_dim,
            "ema_schedule": self.ema.schedule,
            "loss_type": self.loss_type,
        }

    def hyperparams(self) -> dict:
        """Numerically-stability-sensitive hyperparams (W22 addendum).

        Fields: ``momentum_end`` (EMA cosine schedule endpoint) and
        ``loss_beta`` (smooth-L1 Huber threshold). These are validated with
        ERROR-level semantics across resume — see
        :func:`worldfoundry_train.checkpoint.validate_jepa_hyperparams`.
        """
        return {
            "momentum_end": float(self.ema.momentum_end),
            "loss_beta": float(self.beta),
        }


class PredictorGradSyncStrategy(DistributedStrategy):
    """Strategy decorator: explicit all_reduce of predictor grads (design §8.1).

    The predictor lives OUTSIDE the DDP/FSDP wrap (~21M params, full copy per
    rank), so neither DDP's backward hooks nor FSDP's reduce-scatter touches
    its gradients. This decorator all_reduces ``p.grad`` of the predictor
    params (SUM / world_size, mirroring the manual branch in
    ``FSDPStrategy.step``) immediately before delegating ``step`` to the
    inner strategy — i.e. after backward, before ``optimizer.step()``.

    With gradient accumulation the sync fires only on window-closing steps
    (the training loop calls ``strategy.step`` only there), and accumulated
    predictor grads are scaled identically to the encoder's (``loss /
    grad_accum`` + tail rescale), so the reduced gradient is the window mean.
    On single-rank / uninitialized process groups the all_reduce is skipped
    and the decorator is a pure passthrough.

    Why a decorator, not a new ``DistributedStrategy`` subclass?

    Review M6 (2026-08-05): project convention places one concrete strategy
    per file under ``worldfoundry_train/parallel/`` (DDP, FSDP). Adding a
    third concrete class there for the predictor-only all_reduce would
    duplicate the ``wrap / backward / no_sync`` forwarding logic and require
    the trainer to pick between ``FSDP`` and ``FSDP+Predictor`` at every
    call site — fragile if a third sync need lands later (e.g. EMA gather
    timing). A decorator wraps any concrete strategy and is the minimal
    surface that touches the training loop (``strategy = PredictorGradSyncStrategy(
    strategy, predictor)`` in trainer.py:298). The cost is that ``inner`` is
    public for tests; treat it as the seam, not the API.
    """

    def __init__(self, inner: DistributedStrategy, predictor: nn.Module) -> None:
        self._inner = inner
        self._predictor_params = list(predictor.parameters())

    @property
    def inner(self) -> DistributedStrategy:
        return self._inner

    def wrap(self, model: nn.Module) -> nn.Module:
        return self._inner.wrap(model)

    def backward(self, loss: torch.Tensor) -> None:
        self._inner.backward(loss)

    def no_sync(self, model: nn.Module):
        return self._inner.no_sync(model)

    def step(self, optimizer) -> None:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            world = dist.get_world_size()
            for p in self._predictor_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world
        self._inner.step(optimizer)
