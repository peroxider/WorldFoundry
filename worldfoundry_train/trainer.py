"""CPU-runnable training orchestration: dataset -> LoRA -> AdamW -> loop.

Wires Vjepa2Dataset -> collate_vjepa2_batch -> build_lora_model ->
strategy.wrap -> AdamW (trainable params only) -> run_training_loop.

CPU single-process + multi-rank (gloo) by construction. Multi-rank
all-reduce handled by FSDPStrategy.step (see
worldfoundry_train/parallel/fsdp.py:62). Checkpoint save/load handled by
worldfoundry_train.checkpoint (W16).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import torch
from torch import nn

from worldfoundry_train.checkpoint import (
    LoraSignature, load_checkpoint, save_checkpoint, validate_lora_signature,
)
from worldfoundry_train.data.collator import collate_vjepa2_batch
from worldfoundry_train.data.dataset import Vjepa2Dataset
from worldfoundry_train.data.tubelet_mask import tubelet_mask
from worldfoundry_train.device import resolve_device, to_device
from worldfoundry_train.jepa_objective import JEPAObjective, PredictorGradSyncStrategy
from worldfoundry_train.model.lora_wrapper import build_lora_model
from worldfoundry_train.model.predictor import JEPAPredictor

if TYPE_CHECKING:
    from worldfoundry_train.parallel.strategy import DistributedStrategy
    from worldfoundry_train.training_loop import LossFn

__all__ = ["run_training", "make_strategy", "default_logger"]


def default_logger(step: int, loss: float) -> None:
    print(f"[TRAIN] step={step} loss={loss:.6f}")


def make_strategy(profile: str = "train") -> "DistributedStrategy":
    """Map a parallel profile name to a strategy instance.

    - ``"ddp"``  -> :class:`~worldfoundry_train.parallel.ddp.DDPStrategy`
      (Phase-1 default for LoRA per spec §3.3)
    - ``"fsdp"`` -> :class:`~worldfoundry_train.parallel.fsdp.FSDPStrategy`
    - ``"train"`` -> FSDPStrategy (legacy Phase-1 alias, kept for backward
      compatibility; new configs should use ``"ddp"`` / ``"fsdp"``)
    - ``"rl"``   -> DDPStrategy (W37 v1 CPU skeleton; single-process mirror
      of ``"ddp"`` — the profile name aligns with the RL workload but no
      DDP wrapping of PPO is performed; multi-rank = W37+).

    Unknown names raise ValueError (fail-fast per G4) instead of silently
    returning a default.
    """
    if profile == "ddp":
        from worldfoundry_train.parallel.ddp import DDPStrategy
        return DDPStrategy()
    if profile in ("train", "fsdp"):
        from worldfoundry_train.parallel.fsdp import FSDPStrategy
        return FSDPStrategy()
    if profile == "rl":
        from worldfoundry_train.parallel.ddp import DDPStrategy
        return DDPStrategy()
    raise ValueError(
        f"unknown profile: {profile!r}; expected 'ddp', 'fsdp', 'rl' or legacy 'train'"
    )


def _default_num_steps(cfg: dict, ds_len: int, batch: int) -> int:
    epochs = int(cfg["training"]["epochs"])
    return epochs * (ds_len // batch)


def _build_dataset(cfg: dict) -> Vjepa2Dataset:
    tubelet = tuple(cfg["data"]["mask_tubelet_size"])
    ratio = float(cfg["data"]["mask_ratio"])
    def mask_fn(shape, rng):
        return tubelet_mask(shape, tubelet, mask_ratio=ratio, rng=rng)
    # W22.8 multi-mask: K>1 drives the dataset to sample that many masks
    # per item. The full .get chain keeps cfgs that lack a jepa/loss
    # section (most of the legacy test cfg dicts) buildable with K=1.
    num_masks = int(
        cfg.get("jepa", {}).get("loss", {}).get("num_masks", 1)
    )
    return Vjepa2Dataset(
        manifest_path=cfg["data"]["manifest"],
        mask_fn=mask_fn,
        mask_count=num_masks,
    )


def _build_optimizer(
    model: nn.Module,
    lr: float,
    predictor: nn.Module | None = None,
    predictor_lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    """AdamW over trainable encoder params, plus an optional predictor group.

    The predictor group (W22 JEPA) gets ``lr * predictor_lr_scale``; when
    ``predictor is None`` the param-group layout is identical to the
    pre-JEPA single-group optimizer (feature-flag equivalence, §9.3).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    kwargs = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    if predictor is None:
        return torch.optim.AdamW(trainable, **kwargs)
    groups = [
        {"params": trainable, "lr": lr},
        {"params": list(predictor.parameters()), "lr": lr * predictor_lr_scale},
    ]
    return torch.optim.AdamW(groups, **kwargs)


def _jepa_enabled(cfg: dict) -> bool:
    """Feature flag with env override (design §9.3): env > yaml.

    ``WORLDFORGE_JEPA=off`` (also ``0``/``false``) forces the legacy proxy
    loss regardless of YAML — the production fast-rollback path. ``on`` /
    ``1`` / ``true`` force-enable. Unset env -> YAML ``jepa.enabled``
    (default False, so pre-W22 cfgs are byte-for-byte unaffected).
    """
    env = os.environ.get("WORLDFORGE_JEPA", "").strip().lower()
    if env in ("off", "0", "false"):
        return False
    if env in ("on", "1", "true"):
        return True
    return bool(cfg.get("jepa", {}).get("enabled", False))


def _build_jepa_objective(
    cfg: dict,
    peft_model: nn.Module,
    grid_size: tuple[int, int, int],
    total_optimizer_steps: int,
) -> JEPAObjective:
    """Construct predictor + EMA target encoder from cfg["jepa"] (W22).

    MUST be called before ``strategy.wrap`` so the EMA target encoder is
    deepcopied outside any FSDP wrapping (design §5.5). ``ema_schedule_steps:
    null`` in YAML resolves to ``total_optimizer_steps`` here.
    """
    jepa_cfg = cfg["jepa"]
    pred_cfg = jepa_cfg.get("predictor", {})
    ema_cfg = jepa_cfg.get("ema", {})
    loss_cfg = jepa_cfg.get("loss", {})

    encoder_dim = int(peft_model.config.hidden_size)
    num_masks = int(loss_cfg.get("num_masks", 1))
    if num_masks < 1:
        raise ValueError(
            f"jepa.loss.num_masks={num_masks} must be >= 1."
        )

    predictor = JEPAPredictor(
        num_patches=math.prod(grid_size),
        grid_size=grid_size,
        encoder_dim=encoder_dim,
        predictor_dim=int(pred_cfg.get("embed_dim", 384)),
        depth=int(pred_cfg.get("depth", 12)),
        num_heads=int(pred_cfg.get("num_heads", 12)),
        mlp_ratio=float(pred_cfg.get("mlp_ratio", 4.0)),
    )
    schedule_steps = ema_cfg.get("schedule_steps", None)
    if schedule_steps is None:
        schedule_steps = total_optimizer_steps
    return JEPAObjective.from_online_model(
        peft_model,
        predictor,
        loss_type=str(loss_cfg.get("type", "smooth_l1")),
        beta=float(loss_cfg.get("beta", 1.0)),
        ema_momentum_start=float(ema_cfg.get("momentum_start", 0.996)),
        ema_momentum_end=float(ema_cfg.get("momentum_end", 1.0)),
        ema_schedule=str(ema_cfg.get("schedule", "cosine")),
        ema_schedule_steps=int(schedule_steps),
        ema_update_every=int(ema_cfg.get("update_every", 1)),
    )


def _make_signature_from_cfg(cfg: dict, lora_result) -> LoraSignature:
    return LoraSignature(
        r=int(cfg["lora"]["r"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        target_modules=tuple(lora_result.target_modules),
        trainable_params=int(lora_result.trainable_params),
    )


def _enable_full_finetune(model: nn.Module) -> nn.Module:
    """Enable gradients for every parameter in full fine-tuning mode."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def run_training(
    cfg: dict,
    strategy: "DistributedStrategy",
    *,
    mode: str = "lora",
    base_model: nn.Module | None = None,
    num_steps: int | None = None,
    loss_fn: "LossFn | None" = None,
    batch_fn: Callable[[int], dict[str, torch.Tensor]] | None = None,
    logger: Callable[[int, float], None] | None = None,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    save_every_n: int = 0,
    step_callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Run end-to-end training. Returns per-optimizer-step losses.

    G4: empty dataset -> ValueError; num_steps<=0 -> ValueError;
    save_every_n>0 without checkpoint_dir -> ValueError;
    resume_from nonexistent -> FileNotFoundError; corrupted .pt -> IOError;
    LoRA signature mismatch -> ValueError; grad_accum<1 -> ValueError.

    ``num_steps`` (and the YAML-derived default) counts micro-batches.
    ``cfg["training"]["grad_accum"]`` (default 1) controls gradient
    accumulation: every ``grad_accum`` micro-batches trigger one
    optimizer step. Checkpoint ``step`` values, ``save_every_n`` and the
    returned loss list therefore count optimizer steps
    (``ceil(micro_steps / grad_accum)``); with ``grad_accum == 1`` all
    semantics are identical to the pre-accumulation behavior.

    Args:
        checkpoint_dir: optional Path. If set, final checkpoint (and intermediate
            ones at every save_every_n optimizer steps) is written here. Parent dirs created.
        resume_from: optional Path. If set, load this checkpoint before training
            and continue from its recorded (optimizer) step. LoRA signature is validated.
        save_every_n: int. 0 = save only at end. >0 = save every N optimizer steps.
            Raises ValueError if >0 and checkpoint_dir is None.
        step_callback: optional Callable[[int, float], None] invoked after the
            internal checkpointing callback on every optimizer step (1-indexed
            step number, window-mean loss). Default None = zero behavior
            change. W22.7 pilot mode uses it for per-step timing/memory
            instrumentation.

    Returns:
        Per-optimizer-step mean losses for the steps actually executed
        (post-resume if applicable).
    """
    from worldfoundry_train.training_loop import default_loss_fn, run_training_loop

    injected_loss_fn = loss_fn  # None means "use default" (or JEPA, if enabled)
    loss_fn = loss_fn or default_loss_fn
    logger = logger or default_logger
    log_every_n = int(cfg["training"].get("log_every_n", 10))
    grad_accum = int(cfg["training"].get("grad_accum", 1))
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")

    ds = _build_dataset(cfg)
    if len(ds) == 0:
        raise ValueError("Vjepa2Dataset is empty (manifest has 0 rows); cannot train.")

    B = int(cfg["training"]["per_device_batch"])
    if num_steps is None:
        num_steps = _default_num_steps(cfg, len(ds), B)
    if num_steps <= 0:
        raise ValueError(
            f"num_steps={num_steps} (epochs={cfg['training']['epochs']} "
            f"ds_size={len(ds)} batch={B}); pass --steps > 0 to override."
        )
    print(f"[TRAIN] ds_size={len(ds)} batch={B} num_steps={num_steps}")

    if mode not in ("lora", "full"):
        raise ValueError(f"unknown training mode: {mode!r}; expected 'lora' or 'full'")

    if base_model is None:
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(cfg["model"]["hf_repo"])
    if mode == "lora":
        lora_result = build_lora_model(
            base_model,
            r=int(cfg["lora"]["r"]),
            alpha=int(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
            bias=cfg["lora"]["bias"],
        )
        peft_model = lora_result.peft_model
        ckpt_sig = _make_signature_from_cfg(cfg, lora_result)
    else:
        peft_model = _enable_full_finetune(base_model)
        lora_result = None
        ckpt_sig = None

    # --- W22 JEPA objective (feature-flagged, design §7.2) ---
    # Built BEFORE strategy.wrap so the EMA target encoder is deepcopied
    # outside any FSDP wrapping (design §5.5).
    jepa_objective: JEPAObjective | None = None
    if _jepa_enabled(cfg):
        if injected_loss_fn is not None:
            raise ValueError(
                "jepa.enabled=true replaces loss_fn with JEPAObjective.loss_fn; "
                "do not inject a custom loss_fn at the same time."
            )
        if getattr(strategy, "cpu_offload", False):
            raise ValueError(
                "fsdp cpu_offload=true is mutually exclusive with "
                "jepa.enabled=true in v1 (design §8.3); disable one of them."
            )
        # Grid of the flattened bool_masked_pos, from a dataset sample's mask
        # (T, Hg, Wg). v1 assumes a fixed grid across samples (design §4.2).
        grid_size = tuple(int(s) for s in ds[0]["mask"].shape)
        total_optimizer_steps = -(-num_steps // grad_accum)  # ceil
        jepa_objective = _build_jepa_objective(
            cfg, peft_model, grid_size, total_optimizer_steps
        )
        loss_fn = jepa_objective.loss_fn
        n_pred = sum(p.numel() for p in jepa_objective.predictor.parameters())
        print(
            f"[TRAIN] JEPA enabled: predictor_params={n_pred} "
            f"grid={grid_size} loss={jepa_objective.loss_type} "
            f"ema_schedule_steps={jepa_objective.ema.schedule_steps}"
        )

    model = strategy.wrap(peft_model)
    # --- W20 device migration ---
    # Resolve the training device once and migrate the wrapped model
    # (and the JEPA predictor, which lives outside the DDP/FSDP wrap)
    # onto it. On a CPU-only host `resolve_device()` returns "cpu" and
    # `to_device` is a no-op aside from a single `.to("cpu")` call; the
    # surrounding test suite continues to pass unchanged. On an Atlas
    # box the same code path migrates onto `torch.npu`. Visibility of
    # the resolved device in the log is part of the contract: a missing
    # `[TRAIN] device=...` line means the migration was bypassed.
    device = resolve_device()
    model = to_device(model, device)
    if jepa_objective is not None:
        to_device(jepa_objective.predictor, device)
    from worldfoundry_train.device import is_npu_available as _is_npu
    print(f"[TRAIN] device={device} npu_available={_is_npu()}")
    if jepa_objective is not None:
        # W22.6: EMA gather-then-update needs the wrapped (maybe FSDP) model
        # (design §5.4); predictor grads live outside the DDP/FSDP wrap and
        # need an explicit all_reduce before optimizer.step (design §8.1).
        jepa_objective.bind_distributed(model)
        strategy = PredictorGradSyncStrategy(strategy, jepa_objective.predictor)
    if mode == "lora":
        print(f"[TRAIN] LoRA trainable={lora_result.trainable_params} "
              f"total={lora_result.total_params} targets={lora_result.target_modules}")
    else:
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"[TRAIN] full trainable={trainable_params} total={total_params}")

    if jepa_objective is not None:
        optimizer = _build_optimizer(
            model,
            lr=float(cfg["training"]["lr"]),
            predictor=jepa_objective.predictor,
            predictor_lr_scale=float(
                cfg.get("jepa", {}).get("predictor", {}).get("lr_scale", 1.0)
            ),
        )
    else:
        # Legacy path: identical call shape as pre-W22 (feature-flag
        # equivalence, §9.3 — including for test doubles of _build_optimizer).
        optimizer = _build_optimizer(model, lr=float(cfg["training"]["lr"]))

    # --- W16 Checkpoint orchestration ---
    if save_every_n > 0 and checkpoint_dir is None:
        raise ValueError(
            f"save_every_n={save_every_n} but checkpoint_dir is None. "
            f"Either set checkpoint_dir or set save_every_n=0."
        )

    start_step = 0
    if resume_from is not None:
        ckpt_result = load_checkpoint(
            resume_from, model, optimizer, jepa_objective=jepa_objective
        )
        start_step = int(ckpt_result["step"])
        validate_lora_signature(ckpt_result["lora_signature"], ckpt_sig)
        print(f"[TRAIN] resumed from {resume_from} at step={start_step}")

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if mode == "lora":
            ckpt_sig = _make_signature_from_cfg(cfg, lora_result)
    else:
        ckpt_sig = None

    def _step_cb(s: int, loss: float) -> None:
        if checkpoint_dir is not None and save_every_n > 0 and s % save_every_n == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{s}.pt",
                model, optimizer, s, lora_signature=ckpt_sig,
                jepa_objective=jepa_objective,
            )
        if step_callback is not None:
            step_callback(s, loss)

    # --- end W16 block ---

    # Checkpoint `step` counts OPTIMIZER steps (see run_training_loop
    # step_callback semantics), while num_steps counts micro-batches.
    # Convert the resumed optimizer-step counter back to micro-batches;
    # exact when all completed accumulation windows were full (checkpoints
    # are only written at window boundaries or at the final flush).
    # grad_accum == 1 -> start_step * 1 == start_step (unchanged behavior).
    effective_num_steps = num_steps - start_step * grad_accum
    if effective_num_steps <= 0:
        raise ValueError(
            f"effective_num_steps={effective_num_steps} (num_steps={num_steps}, "
            f"start_step={start_step}, grad_accum={grad_accum}); checkpoint "
            f"step already at/over target. "
            f"Raise --steps or omit --resume-from."
        )
    print(f"[TRAIN] start_step={start_step} target={num_steps} "
          f"remaining={effective_num_steps}")

    if batch_fn is not None:
        batches = (batch_fn(i) for i in range(effective_num_steps))
    else:
        def batch_iter():
            n = len(ds)
            while True:
                for start in range(0, n, B):
                    yield collate_vjepa2_batch(
                        [ds[i] for i in range(start, min(start + B, n))]
                    )
        batches = batch_iter()

    losses = run_training_loop(
        model, batches, num_steps=effective_num_steps,
        strategy=strategy, optimizer=optimizer, loss_fn=loss_fn,
        step_callback=_step_cb, grad_accum=grad_accum,
        post_step_hook=(
            jepa_objective.after_step if jepa_objective is not None else None
        ),
    )

    if checkpoint_dir is not None:
        # len(losses) == number of optimizer steps executed this run
        # (== effective_num_steps when grad_accum == 1).
        final_step = start_step + len(losses)
        save_checkpoint(
            checkpoint_dir / f"step_{final_step}.pt",
            model, optimizer, final_step, lora_signature=ckpt_sig,
            jepa_objective=jepa_objective,
        )
        print(f"[TRAIN] saved final checkpoint step={final_step}")

    for i, loss in enumerate(losses):
        actual_step = start_step + i + 1
        if actual_step % log_every_n == 0 or i == len(losses) - 1:
            logger(actual_step, loss)
    return losses