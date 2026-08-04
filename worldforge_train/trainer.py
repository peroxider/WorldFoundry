"""CPU-runnable training orchestration: dataset -> LoRA -> AdamW -> loop.

Wires Vjepa2Dataset -> collate_vjepa2_batch -> build_lora_model ->
strategy.wrap -> AdamW (trainable params only) -> run_training_loop.

CPU single-process + multi-rank (gloo) by construction. Multi-rank
all-reduce handled by FSDPStrategy.step (see
worldforge_train/parallel/fsdp.py:62). Checkpoint save/load handled by
worldforge_train.checkpoint (W16).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, TYPE_CHECKING

import torch
from torch import nn

from worldforge_train.checkpoint import (
    LoraSignature, load_checkpoint, save_checkpoint, validate_lora_signature,
)
from worldforge_train.data.collator import collate_vjepa2_batch
from worldforge_train.data.dataset import Vjepa2Dataset
from worldforge_train.data.tubelet_mask import tubelet_mask
from worldforge_train.model.lora_wrapper import build_lora_model

if TYPE_CHECKING:
    from worldforge_train.parallel.strategy import DistributedStrategy
    from worldforge_train.training_loop import LossFn

__all__ = ["run_training", "make_strategy", "default_logger"]


def default_logger(step: int, loss: float) -> None:
    print(f"[TRAIN] step={step} loss={loss:.6f}")


def make_strategy(profile: str = "train") -> "DistributedStrategy":
    """Map a parallel profile name to a strategy instance.

    - ``"ddp"``  -> :class:`~worldforge_train.parallel.ddp.DDPStrategy`
      (Phase-1 default for LoRA per spec §3.3)
    - ``"fsdp"`` -> :class:`~worldforge_train.parallel.fsdp.FSDPStrategy`
    - ``"train"`` -> FSDPStrategy (legacy Phase-1 alias, kept for backward
      compatibility; new configs should use ``"ddp"`` / ``"fsdp"``)

    Unknown names raise ValueError (fail-fast per G4) instead of silently
    returning a default.
    """
    if profile == "ddp":
        from worldforge_train.parallel.ddp import DDPStrategy
        return DDPStrategy()
    if profile in ("train", "fsdp"):
        from worldforge_train.parallel.fsdp import FSDPStrategy
        return FSDPStrategy()
    raise ValueError(
        f"unknown profile: {profile!r}; expected 'ddp', 'fsdp' or legacy 'train'"
    )


def _default_num_steps(cfg: dict, ds_len: int, batch: int) -> int:
    epochs = int(cfg["training"]["epochs"])
    return epochs * (ds_len // batch)


def _build_dataset(cfg: dict) -> Vjepa2Dataset:
    tubelet = tuple(cfg["data"]["mask_tubelet_size"])
    ratio = float(cfg["data"]["mask_ratio"])
    def mask_fn(shape, rng):
        return tubelet_mask(shape, tubelet, mask_ratio=ratio, rng=rng)
    return Vjepa2Dataset(
        manifest_path=cfg["data"]["manifest"],
        mask_fn=mask_fn,
    )


def _build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )


def _make_signature_from_cfg(cfg: dict, lora_result) -> LoraSignature:
    return LoraSignature(
        r=int(cfg["lora"]["r"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        target_modules=tuple(lora_result.target_modules),
        trainable_params=int(lora_result.trainable_params),
    )


def run_training(
    cfg: dict,
    strategy: "DistributedStrategy",
    *,
    base_model: nn.Module | None = None,
    num_steps: int | None = None,
    loss_fn: "LossFn | None" = None,
    batch_fn: Callable[[int], dict[str, torch.Tensor]] | None = None,
    logger: Callable[[int, float], None] | None = None,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    save_every_n: int = 0,
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

    Returns:
        Per-optimizer-step mean losses for the steps actually executed
        (post-resume if applicable).
    """
    from worldforge_train.training_loop import default_loss_fn, run_training_loop

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

    if base_model is None:
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(cfg["model"]["hf_repo"])
    lora_result = build_lora_model(
        base_model,
        r=int(cfg["lora"]["r"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        bias=cfg["lora"]["bias"],
    )
    peft_model = lora_result.peft_model
    model = strategy.wrap(peft_model)
    print(f"[TRAIN] LoRA trainable={lora_result.trainable_params} "
          f"total={lora_result.total_params} targets={lora_result.target_modules}")

    optimizer = _build_optimizer(model, lr=float(cfg["training"]["lr"]))

    # --- W16 Checkpoint orchestration ---
    if save_every_n > 0 and checkpoint_dir is None:
        raise ValueError(
            f"save_every_n={save_every_n} but checkpoint_dir is None. "
            f"Either set checkpoint_dir or set save_every_n=0."
        )

    start_step = 0
    if resume_from is not None:
        ckpt_result = load_checkpoint(resume_from, model, optimizer)
        start_step = int(ckpt_result["step"])
        sig = _make_signature_from_cfg(cfg, lora_result)
        validate_lora_signature(ckpt_result["lora_signature"], sig)
        print(f"[TRAIN] resumed from {resume_from} at step={start_step}")

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_sig = _make_signature_from_cfg(cfg, lora_result)
    else:
        ckpt_sig = None

    def _step_cb(s: int, loss: float) -> None:
        if checkpoint_dir is not None and save_every_n > 0 and s % save_every_n == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{s}.pt",
                model, optimizer, s, lora_signature=ckpt_sig,
            )

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
    )

    if checkpoint_dir is not None:
        # len(losses) == number of optimizer steps executed this run
        # (== effective_num_steps when grad_accum == 1).
        final_step = start_step + len(losses)
        save_checkpoint(
            checkpoint_dir / f"step_{final_step}.pt",
            model, optimizer, final_step, lora_signature=ckpt_sig,
        )
        print(f"[TRAIN] saved final checkpoint step={final_step}")

    for i, loss in enumerate(losses):
        actual_step = start_step + i + 1
        if actual_step % log_every_n == 0 or i == len(losses) - 1:
            logger(actual_step, loss)
    return losses