"""WorldForge training entry.

Usage:
    python -m worldforge_train.train --config <yaml> [--steps N] [--parallel ddp|fsdp]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from worldforge_common.npu_env import check_versions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldForge V-JEPA 2 LoRA training")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--parallel", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--strict-versions", action="store_true")
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Number of training steps. >0 overrides YAML derivation. <=0 derives from YAML.",
    )
    return parser.parse_args(argv)


def _resolve_num_steps(args: argparse.Namespace) -> int | None:
    return args.steps if args.steps > 0 else None


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

    num_steps = _resolve_num_steps(args)
    try:
        if args.parallel == "ddp":
            run_ddp_training(cfg, local_rank, global_rank, world_size,
                             num_steps=num_steps)
        else:
            run_fsdp_training(cfg, local_rank, global_rank, world_size,
                              num_steps=num_steps)
    finally:
        if args.parallel == "ddp":
            cleanup_ddp()
        else:
            cleanup_fsdp()
    return 0


def run_ddp_training(cfg: dict, local_rank: int, global_rank: int,
                     world_size: int, *, num_steps: int | None = None) -> list[float]:
    """Delegates to trainer.run_training. Returns per-step losses."""
    from worldforge_train.trainer import make_strategy, run_training
    return run_training(cfg, make_strategy("train"), num_steps=num_steps)


def run_fsdp_training(cfg: dict, local_rank: int, global_rank: int,
                      world_size: int, *, num_steps: int | None = None) -> list[float]:
    """Delegates to trainer.run_training with FSDPStrategy.

    FSDPStrategy.wrap is identity on single-rank (proven in
    tests/test_training_loop_cpu.py) and FullyShardedDataParallel on
    multi-rank (proven in tests/test_fsdp_strategy.py +
    tests/test_fsdp_allreduce_cpu.py).
    """
    from worldforge_train.trainer import make_strategy, run_training
    return run_training(cfg, make_strategy("train"), num_steps=num_steps)


if __name__ == "__main__":
    raise SystemExit(main())

