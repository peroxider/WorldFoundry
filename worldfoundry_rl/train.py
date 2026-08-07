"""RL training CLI runner (W37 v1 CPU skeleton).

`python -m worldfoundry_rl.train --config <yaml>` drives a single-process
on-policy RL loop using :class:`worldfoundry_rl.env_pool.SyncEnvPool` +
:class:`worldfoundry_rl.vision_policy.VisionPolicy` +
:class:`worldfoundry_rl.ppo.PPO`.

Mirrors the CLI surface of :mod:`worldfoundry_train.train` (YAML config +
``--steps`` / ``--parallel`` / ``--mode`` flags + ``--checkpoint-dir``) so the
two trainers read like siblings. Reuses the W20 device seam
(:func:`worldfoundry_train.device.resolve_device` +
:func:`worldfoundry_train.device.to_device`) and the W37-extended
``rl_payload`` kwargs on :func:`worldfoundry_train.checkpoint.save_checkpoint`.

v1 CPU scope: ``--mode rl`` only, ``--parallel {ddp,fsdp,rl}`` all collapsed
to single-process (no DDP wrap of PPO). Multi-rank + async env workers + the
``actor_pool`` / ``learner_pool`` / ``scheduler`` deliverables are W37+
and require W35 framework choice + Atlas cluster.

Per W37 plan §A.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from worldfoundry_train.checkpoint import save_checkpoint
from worldfoundry_train.device import resolve_device, to_device


__all__ = ["main", "parse_args", "build_pool", "build_policy"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Mirror the surface of :mod:`worldfoundry_train.train` argparse."""
    parser = argparse.ArgumentParser(
        description="WorldFoundry RL training CLI runner (W37 v1 CPU)",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML config path (see worldfoundry_rl/configs/rl_manipulator_cpu.yaml)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override training.total_pool_steps from the YAML.",
    )
    parser.add_argument(
        "--parallel", choices=("ddp", "fsdp", "rl"), default=None,
        help="Override training.parallel from the YAML. v1 single-process; "
             "'rl' aligns the name with the workload but mirrors 'ddp'.",
    )
    parser.add_argument(
        "--mode", choices=("rl",), default="rl",
        help="v1 only ships 'rl'. Placeholder for parity with train.py.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="If set + save_every_n>0, PPO final state is saved here.",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Optional .pt to restore policy weights from.",
    )
    parser.add_argument(
        "--save-every-n", type=int, default=None,
        help="Override training.save_every_n from the YAML.",
    )
    return parser.parse_args(argv)


def _import_class(dotted: str) -> type:
    """Import ``"package.module:ClassName"`` and return the class object.

    Splits on the rightmost ``:`` to separate module path from class name —
    mirrors the traceback repr so an ImportError points the user at the right
    line of their YAML.
    """
    if ":" not in dotted:
        raise ValueError(
            f"env.type must be of the form 'package.module:ClassName', got {dotted!r}"
        )
    module_path, class_name = dotted.rsplit(":", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"{module_path!r} has no attribute {class_name!r} (env.type)"
        ) from exc


def build_pool(cfg: dict) -> tuple[Any, int, int]:
    """Construct the env pool from YAML ``env`` block.

    Returns ``(pool, obs_dim, action_dim)``. ``obs_dim`` is the flattened
    width of a single env observation (sum of all per-key arrays' sizes).
    """
    env_type = cfg.get("env", {}).get("type")
    if not env_type:
        raise ValueError("config missing required `env.type`")
    env_cls = _import_class(env_type)
    env_kwargs = dict(cfg.get("env", {}).get("kwargs", {}))
    num_envs = int(cfg.get("env", {}).get("num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be >= 1, got {num_envs}")

    from worldfoundry_rl.env_pool import SyncEnvPool
    seed = cfg.get("seed")
    pool = SyncEnvPool(
        [lambda: env_cls(**env_kwargs) for _ in range(num_envs)],
        seed=seed,
    )

    sample = pool.obs[0]
    if isinstance(sample, dict):
        obs_dim = sum(int(np.asarray(v).ravel().size) for v in sample.values())
    else:
        obs_dim = int(np.asarray(sample).ravel().size)
    action_dim = int(pool.single_action_space.shape[0])
    return pool, obs_dim, action_dim


def build_policy(
    cfg: dict, obs_dim: int, action_dim: int, device: str,
) -> Any:
    """Construct the policy (``VisionPolicy`` for v1) and move it to ``device``.

    Only ``encoder: state_mlp`` is implemented in v1; cnn / vjepa2 raise.
    """
    from worldfoundry_rl.vision_policy import VisionPolicy

    policy_cfg = cfg.get("policy", {})
    hidden_dim = int(policy_cfg.get("hidden_dim", 64))
    encoder_kind = str(policy_cfg.get("encoder", "state_mlp"))
    if encoder_kind != "state_mlp":
        raise ValueError(
            f"policy.encoder {encoder_kind!r} not implemented in W37 v1; "
            "use 'state_mlp' (drives VisionPolicy.from_state_dim)."
        )
    policy = VisionPolicy.from_state_dim(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
    )
    return to_device(policy, device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.config.is_file():
        print(f"[W37] FATAL: config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        print(f"[W37] FATAL: config root must be a mapping, got {type(cfg).__name__}",
              file=sys.stderr)
        return 2

    # CLI overrides onto YAML.
    steps_yaml = int(cfg.get("training", {}).get("total_pool_steps", 0))
    total_pool_steps = int(args.steps) if args.steps is not None else steps_yaml
    if total_pool_steps < 1:
        print(f"[W37] FATAL: training.total_pool_steps must be >= 1, got "
              f"{total_pool_steps}", file=sys.stderr)
        return 2
    cfg["training"]["total_pool_steps"] = total_pool_steps

    if args.parallel is not None:
        cfg.setdefault("training", {})["parallel"] = args.parallel
    if args.save_every_n is not None:
        cfg.setdefault("training", {})["save_every_n"] = int(args.save_every_n)

    log_every_n = int(cfg.get("training", {}).get("log_every_n", 1))

    # W20 seam.
    device = resolve_device()

    pool, obs_dim, action_dim = build_pool(cfg)
    try:
        policy = build_policy(cfg, obs_dim, action_dim, device)

        # Lazy import — keep torch import-light on parser-only paths.
        from worldfoundry_rl.ppo import PPO, PPOConfig

        ppo_cfg_dict = dict(cfg.get("ppo", {}))
        ppo = PPO(policy, PPOConfig(**ppo_cfg_dict))

        if args.resume_from is not None:
            from worldfoundry_train.checkpoint import load_checkpoint
            payload = {"total_env_steps": 0}
            load_checkpoint(
                args.resume_from, policy, ppo.optimizer,
                rl_payload=payload,
            )
            print(f"[W37] resumed from {args.resume_from} (rl_payload={payload})")

        t0 = time.perf_counter()
        history = ppo.train(pool, total_steps=total_pool_steps)
        t_train = time.perf_counter() - t0
        total_env_steps = total_pool_steps * pool.num_envs
    finally:
        pool.close()

    # Per-update report log.
    for i, h in enumerate(history):
        if (i % log_every_n) == 0 or i == len(history) - 1:
            print(
                f"[W37] update={i} "
                f"policy_loss={h['policy_loss']:+.4f} "
                f"value_loss={h['value_loss']:.4f} "
                f"entropy={h['entropy']:.4f} "
                f"mean_episode_reward={h['mean_episode_reward']:+.4f}"
            )

    last = history[-1] if history else {}
    print(
        f"[W37] done updates={len(history)} "
        f"total_pool_steps={total_pool_steps} total_env_steps={total_env_steps} "
        f"device={device} wall={t_train:.2f}s "
        f"final_policy_loss={last.get('policy_loss', 0.0):+.4f} "
        f"final_value_loss={last.get('value_loss', 0.0):.4f}"
    )

    save_every_n = int(cfg.get("training", {}).get("save_every_n", 0))
    if args.checkpoint_dir is not None and save_every_n > 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.checkpoint_dir / "final.pt"
        save_checkpoint(
            ckpt_path, policy, ppo.optimizer, step=len(history),
            rl_payload={"total_env_steps": total_env_steps},
        )
        print(f"[W37] checkpoint saved: {ckpt_path}")

    summary = json.dumps({
        "label": "W37 v1 CPU RL training",
        "config": str(args.config),
        "updates": len(history),
        "total_pool_steps": total_pool_steps,
        "total_env_steps": total_env_steps,
        "device": device,
        "wall_seconds": round(t_train, 3),
        "final_policy_loss": last.get("policy_loss", 0.0),
        "final_value_loss": last.get("value_loss", 0.0),
        "final_mean_episode_reward": last.get("mean_episode_reward", 0.0),
    })
    print(f"[W37] summary: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
