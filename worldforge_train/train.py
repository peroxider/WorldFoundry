"""WorldForge Phase-1 training entry.

Usage:
    torchrun --nproc_per_node=8 -m worldforge_train.train \\
        --config worldforge_train/configs/vjepa2_vitl_1.2b.yaml \\
        --parallel ddp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from worldforge_common.npu_env import check_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldForge V-JEPA 2 LoRA training")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--parallel", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--strict-versions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_versions(strict=args.strict_versions)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    if args.parallel == "ddp":
        from worldforge_train.parallel.ddp import cleanup_ddp, setup_ddp
        world_size, local_rank, global_rank = setup_ddp()
        print(f"[DDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")
    else:
        from worldforge_train.parallel.fsdp import cleanup_fsdp, setup_fsdp
        world_size, local_rank, global_rank = setup_fsdp()
        print(f"[FSDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")

    try:
        if args.parallel == "ddp":
            run_ddp_training(cfg, local_rank, global_rank, world_size)
        else:
            run_fsdp_training(cfg, local_rank, global_rank, world_size)
    finally:
        if args.parallel == "ddp":
            cleanup_ddp()
        else:
            cleanup_fsdp()
    return 0


def run_ddp_training(cfg: dict, local_rank: int, global_rank: int, world_size: int) -> None:
    """Build dataset, LoRA-wrapped model, and run training loop.

    Implementation lives in Task 7 main code path; tests cover the entry only.
    """
    from worldforge_train.data.dataset import Vjepa2Dataset
    from worldforge_train.data.tubelet_mask import tubelet_mask
    from worldforge_train.model.lora_wrapper import build_lora_model

    manifest = cfg["data"]["manifest"]
    ds = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: tubelet_mask(
            shape, tuple(cfg["data"]["mask_tubelet_size"]), cfg["data"]["mask_ratio"], rng=rng
        ),
    )

    # On CPU tests we skip actually loading VJEPA2Model (heavy); the real
    # download happens at first run on an NPU host.
    print(f"[TRAIN] dataset size={len(ds)}")
    print(f"[TRAIN] LoRA build (deferred to NPU host) cfg={cfg['lora']}")

    # Stub training loop: in Phase-1 W2 the real loop is implemented in
    # worldforge_train/trainer.py once NPU host is available.
    for step, batch in enumerate(_fake_batch_iter(ds)):
        if step >= 2:
            break
        print(f"[TRAIN] rank={global_rank} step={step} batch_keys={list(batch.keys())}")


def _fake_batch_iter(ds):
    """Yield a single batch by stacking samples from one rank's view of the dataset."""
    if len(ds) == 0:
        return
    yield {k: v for k, v in ds[0].items()}


def run_fsdp_training(cfg: dict, local_rank: int, global_rank: int, world_size: int) -> None:
    raise NotImplementedError(
        "FSDP path is implemented in Task 8. Use --parallel ddp for Phase-1."
    )


if __name__ == "__main__":
    raise SystemExit(main())
