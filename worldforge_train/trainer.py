"""CPU-runnable training orchestration: dataset -> LoRA -> AdamW -> loop.

Wires Vjepa2Dataset -> collate_vjepa2_batch -> build_lora_model ->
strategy.wrap -> AdamW (trainable params only) -> run_training_loop.

CPU single-process by construction. Multi-rank FSDP gradient all-reduce
in FSDPStrategy.step is a known gap deferred to a future shard-aware
sub-project (see .superpowers/sdd/progress.md W13 caveats).
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import torch
from torch import nn

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
    if profile in ("train", "fsdp"):
        from worldforge_train.parallel.fsdp import FSDPStrategy
        return FSDPStrategy()
    raise ValueError(f"unknown profile: {profile!r}; expected 'train' or 'fsdp'")


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


def run_training(
    cfg: dict,
    strategy: "DistributedStrategy",
    *,
    base_model: nn.Module | None = None,
    num_steps: int | None = None,
    loss_fn: "LossFn | None" = None,
    batch_fn: Callable[[int], dict[str, torch.Tensor]] | None = None,
    logger: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Run end-to-end training. Returns per-step losses.

    G4: empty dataset -> ValueError; num_steps<=0 -> ValueError.
    """
    from worldforge_train.training_loop import default_loss_fn, run_training_loop

    loss_fn = loss_fn or default_loss_fn
    logger = logger or default_logger
    log_every_n = int(cfg["training"].get("log_every_n", 10))

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

    if batch_fn is not None:
        batches = (batch_fn(i) for i in range(num_steps))
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
        model, batches, num_steps=num_steps,
        strategy=strategy, optimizer=optimizer, loss_fn=loss_fn,
    )
    for step, loss in enumerate(losses):
        if (step + 1) % log_every_n == 0 or step == len(losses) - 1:
            logger(step + 1, loss)
    return losses