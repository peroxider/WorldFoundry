"""WorldFoundry training entry.

Usage:
    python -m worldfoundry_train.train --config <yaml> [--steps N] [--parallel ddp|fsdp]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from worldfoundry_common.npu_env import check_versions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry V-JEPA 2 LoRA training")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--parallel", choices=["ddp", "fsdp"], default=None,
        help="Parallel strategy override. Default: cfg[training][parallel], "
             "falling back to 'ddp'.",
    )
    parser.add_argument("--strict-versions", action="store_true")
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Number of training steps. >0 overrides YAML derivation. <=0 derives from YAML.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Directory to write checkpoint .pt files. Parent dirs created.",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to a .pt checkpoint to resume training from.",
    )
    parser.add_argument(
        "--save-every-n", type=int, default=None,
        help="Save checkpoint every N steps. 0=disable. "
             "Default: use cfg[training][save_every_n].",
    )
    parser.add_argument(
        "--mode", choices=["lora", "full"], default=None,
        help="Training mode. 'lora' wraps the encoder with PEFT/LoRA; "
             "'full' fine-tunes every encoder parameter. "
             "Default: cfg[training][mode] or 'lora'.",
    )
    return parser.parse_args(argv)


def _resolve_num_steps(args: argparse.Namespace) -> int | None:
    return args.steps if args.steps > 0 else None


def _resolve_parallel(args: argparse.Namespace, cfg: dict) -> str:
    """Resolve the parallel profile: CLI flag > cfg[training][parallel] > "ddp".

    Fail-fast (ValueError) on unknown names so a typo'd config cannot
    silently launch the wrong strategy.
    """
    profile = args.parallel or str(cfg.get("training", {}).get("parallel") or "ddp")
    if profile not in ("ddp", "fsdp"):
        raise ValueError(
            f"unknown parallel profile: {profile!r}; expected 'ddp' or 'fsdp'"
        )
    return profile


def _resolve_save_every_n(args: argparse.Namespace, cfg: dict) -> int:
    if args.save_every_n is not None:
        return args.save_every_n
    return int(cfg.get("training", {}).get("save_every_n", 0))


def _resolve_mode(args: argparse.Namespace, cfg: dict) -> str:
    profile = args.mode or str(cfg.get("training", {}).get("mode", "lora"))
    if profile not in ("lora", "full"):
        raise ValueError(
            f"unknown training mode: {profile!r}; expected 'lora' or 'full'"
        )
    return profile


def _apply_backend_device(cfg: dict) -> None:
    """Honor ``backends.device`` (``"cpu"`` / ``"npu"``) by pinning the env.

    The trainer + parallel setup resolve the device via
    ``WORLDFORGE_DEVICE`` + NPU auto-detection, so pinning that env var here
    is the single seam that lets a ``backends.device`` entry drive the whole
    run without threading a device through every signature. ``"auto"`` (or
    absence) leaves the env var untouched and keeps auto-detection.
    """
    block = cfg.get("backends")
    dev = block.get("device") if isinstance(block, dict) else None
    if dev in ("cpu", "npu"):
        os.environ["WORLDFORGE_DEVICE"] = dev


def _report_backends(cfg: dict) -> None:
    """Validate + log the resolved backend stack (honors the ``backends:`` map)."""
    from worldfoundry_common.backends import resolve_config

    stack = resolve_config(cfg.get("backends"))
    print(
        "[TRAIN] backends: "
        + json.dumps({a: stack.backends[a].name for a in sorted(stack.backends)})
    )


def main() -> int:
    args = parse_args()
    check_versions(strict=args.strict_versions)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    from worldfoundry_train.quantize_bridge import maybe_quantize_first
    cfg = maybe_quantize_first(cfg)

    # Wire the `backends:` map: pin device + report the resolved stack.
    _apply_backend_device(cfg)
    _report_backends(cfg)

    profile = _resolve_parallel(args, cfg)

    if profile == "ddp":
        from worldfoundry_train.parallel.ddp import cleanup_ddp, setup_ddp
        world_size, local_rank, global_rank = setup_ddp()
        print(f"[DDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")
    else:
        from worldfoundry_train.parallel.fsdp import cleanup_fsdp, setup_fsdp
        world_size, local_rank, global_rank = setup_fsdp()
        print(f"[FSDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")

    num_steps = _resolve_num_steps(args)
    save_every_n = _resolve_save_every_n(args, cfg)
    mode = _resolve_mode(args, cfg)
    try:
        if profile == "ddp":
            run_ddp_training(cfg, local_rank, global_rank, world_size,
                             num_steps=num_steps,
                             checkpoint_dir=args.checkpoint_dir,
                             resume_from=args.resume_from,
                             save_every_n=save_every_n,
                             profile=profile, mode=mode)
        else:
            run_fsdp_training(cfg, local_rank, global_rank, world_size,
                              num_steps=num_steps,
                              checkpoint_dir=args.checkpoint_dir,
                              resume_from=args.resume_from,
                              save_every_n=save_every_n,
                              profile=profile, mode=mode)
    finally:
        if profile == "ddp":
            cleanup_ddp()
        else:
            cleanup_fsdp()
    return 0


def run_ddp_training(cfg: dict, local_rank: int, global_rank: int,
                     world_size: int, *, num_steps: int | None = None,
                     checkpoint_dir: Path | None = None,
                     resume_from: Path | None = None,
                     save_every_n: int = 0,
                     profile: str | None = None,
                     mode: str = "lora") -> list[float]:
    """Delegates to trainer.run_training. Returns per-optimizer-step losses.

    Strategy selection consumes ``cfg["training"]["parallel"]`` unless an
    explicit ``profile`` override is passed; defaults to ``"ddp"``.
    """
    from worldfoundry_train.trainer import make_strategy, run_training
    profile = profile or str(cfg.get("training", {}).get("parallel", "ddp"))
    return run_training(cfg, make_strategy(profile), mode=mode,
                        num_steps=num_steps,
                        checkpoint_dir=checkpoint_dir, resume_from=resume_from,
                        save_every_n=save_every_n)


def run_fsdp_training(cfg: dict, local_rank: int, global_rank: int,
                      world_size: int, *, num_steps: int | None = None,
                      checkpoint_dir: Path | None = None,
                      resume_from: Path | None = None,
                      save_every_n: int = 0,
                      profile: str | None = None,
                      mode: str = "lora") -> list[float]:
    """Delegates to trainer.run_training. Returns per-optimizer-step losses.

    Strategy selection consumes ``cfg["training"]["parallel"]`` unless an
    explicit ``profile`` override is passed; defaults to ``"fsdp"``.

    FSDPStrategy.wrap is identity on single-rank (proven in
    tests/test_training_loop_cpu.py) and FullyShardedDataParallel on
    multi-rank (proven in tests/test_fsdp_strategy.py +
    tests/test_fsdp_allreduce_cpu.py).
    """
    from worldfoundry_train.trainer import make_strategy, run_training
    profile = profile or str(cfg.get("training", {}).get("parallel", "fsdp"))
    return run_training(cfg, make_strategy(profile), mode=mode,
                        num_steps=num_steps,
                        checkpoint_dir=checkpoint_dir, resume_from=resume_from,
                        save_every_n=save_every_n)


if __name__ == "__main__":
    raise SystemExit(main())

