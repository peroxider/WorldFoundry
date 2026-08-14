"""Distributed RL orchestration entry (W37 CPU multi-process skeleton).

``python -m worldfoundry_rl.distributed_train --config <yaml>`` drives the
W28 topology on a single CPU box without Ray / verl / Atlas:

    LearnerPool (main process)  ◄── RolloutChunk ──  ActorPool (N processes)
        │                                                   │
        └────────── versioned state_dict ───────────────────┘

The orchestrator:

1. builds the *master* policy + :class:`~worldfoundry_rl.learner_pool.LearnerPool`
   in the main process,
2. spawns :class:`~worldfoundry_rl.actor_pool.ActorPool` (N actor subprocesses,
   each running an :class:`~worldfoundry_rl.env_pool.AsyncEnvPool`),
3. loops: publish weights → collect one rollout per actor → PPO update → publish
   updated weights, and
4. optionally saves a checkpoint via the W37 ``rl_payload`` seam.

This is the CPU-faithful stand-in for the cluster orchestrator; the message
contract (:class:`~worldfoundry_rl.actor_pool.RolloutChunk` + versioned weights)
is what the W35 framework must reproduce. Per W37 plan §Deliverable 1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from worldfoundry_rl.actor_pool import ActorPool
from worldfoundry_rl.env_pool import SyncEnvPool, make_single_env
from worldfoundry_rl.learner_pool import LearnerPool
from worldfoundry_rl.ppo import PPOConfig, flatten_obs
from worldfoundry_common.backends import resolve_config
from worldfoundry_common.backends.sim_env import resolve_env_spec
from worldfoundry_train.device import resolve_device, to_device


__all__ = ["main", "parse_args", "derive_dims", "run_distributed", "_resolve_env_type"]


def _resolve_env_type(cfg: dict) -> str:
    """Pick the env spec: ``env.type`` first, else ``backends.sim_env`` alias."""
    raw = cfg.get("env", {}).get("type")
    if not raw:
        block = cfg.get("backends")
        raw = block.get("sim_env") if isinstance(block, dict) else None
    if not raw:
        raise ValueError("config missing required `env.type`")
    return resolve_env_spec(raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WorldFoundry distributed RL (W37 CPU multi-process skeleton)",
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config path (see worldfoundry_rl/configs/rl_manipulator_cpu.yaml)")
    parser.add_argument("--actors", type=int, default=2,
                        help="Number of actor subprocesses.")
    parser.add_argument("--updates", type=int, default=2,
                        help="Number of PPO updates (rollout rounds).")
    parser.add_argument("--envs-per-actor", type=int, default=2,
                        help="Envs per actor subprocess.")
    parser.add_argument("--worker-count", type=int, default=1,
                        help="Env worker subprocesses per actor (AsyncEnvPool).")
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="Override ppo.rollout_steps from the YAML.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from the YAML.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="If set, final master checkpoint is saved here.")
    parser.add_argument("--log-every-n", type=int, default=1,
                        help="Print an update line every N updates.")
    return parser.parse_args(argv)


def derive_dims(env_type: str, env_kwargs: dict) -> tuple[int, int]:
    """Compute ``(obs_dim, action_dim)`` from a 1-env probe pool."""
    pool = SyncEnvPool([lambda: make_single_env(env_type, dict(env_kwargs))], seed=0)
    try:
        obs_dim = int(flatten_obs(pool.obs[0]).shape[0])
        action_dim = int(pool.single_action_space.shape[0])
        return obs_dim, action_dim
    finally:
        pool.close()


def run_distributed(
    cfg: dict,
    *,
    num_actors: int = 2,
    envs_per_actor: int = 2,
    worker_count: int = 1,
    num_updates: int = 2,
    rollout_steps: int | None = None,
    seed: int | None = None,
    log_every_n: int = 1,
) -> tuple[list[dict[str, float]], LearnerPool, dict[str, int]]:
    """Run the distributed ActorPool + LearnerPool loop and return ``(history, learner, dims)``.

    ``cfg`` mirrors the top-level keys of the RL YAML (``env`` / ``policy`` /
    ``ppo`` / ``seed``). ``rollout_steps`` / ``seed`` override the YAML values
    when given. ``history`` is one metrics dict per update.
    """
    env_block = cfg.get("env", {})
    # Accept a friendly alias (e.g. "mujoco") or a full "pkg:Cls" spec, and
    # fall back to the `backends.sim_env` alias when `env.type` is absent.
    env_type = _resolve_env_type(cfg)
    env_kwargs = dict(env_block.get("kwargs", {}) or {})
    effective_seed = int(cfg.get("seed")) if seed is None else int(seed)

    # Resolve the compute device. Precedence:
    #   cfg["device"]  -> cfg["backends"]["device"] -> WORLDFORGE_DEVICE -> auto.
    # The configs document the switch under `backends.device` (e.g. "auto"),
    # so honor both that and a top-level `device` key. This is the free
    # CPU/NPU switch point — no code path change needed.
    device = resolve_device(
        override=cfg.get("device") or (cfg.get("backends") or {}).get("device")
    )

    obs_dim, action_dim = derive_dims(env_type, env_kwargs)

    from worldfoundry_rl.vision_policy import VisionPolicy

    hidden_dim = int(cfg.get("policy", {}).get("hidden_dim", 64))
    policy = VisionPolicy.from_state_dim(obs_dim, action_dim, hidden_dim=hidden_dim)
    # Master policy lives on the compute device (learner side). Actors keep
    # their CPU copy for mujoco rollout (envs stay on CPU).
    policy = to_device(policy, device)

    ppo_cfg = PPOConfig(**dict(cfg.get("ppo", {})))
    if rollout_steps is not None:
        ppo_cfg.rollout_steps = int(rollout_steps)
    learner = LearnerPool(policy, ppo_cfg, device=device)

    actor_cfg = {
        "env_type": env_type,
        "env_kwargs": env_kwargs,
        "num_envs": int(envs_per_actor),
        "worker_count": int(worker_count),
        "rollout_steps": int(ppo_cfg.rollout_steps),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "seed": effective_seed,
    }
    actors = ActorPool(int(num_actors), actor_cfg)
    history: list[dict[str, float]] = []
    try:
        actors.publish_weights(learner.version, learner.state_dict())
        for upd in range(int(num_updates)):
            chunks = actors.collect()
            metrics = learner.update(chunks)
            history.append(metrics)
            # Publish updated weights for the *next* round only. After the
            # last update we deliberately do not publish: actors stay parked
            # on `recv`, so `close()` can hand them a clean "close" and they
            # shut down their env workers without an extra wasted rollout.
            if upd != int(num_updates) - 1:
                actors.publish_weights(learner.version, learner.state_dict())
            if (upd % int(log_every_n)) == 0 or upd == int(num_updates) - 1:
                print(
                    f"[WF-RL] update={upd} version={learner.version} "
                    f"policy_loss={metrics['policy_loss']:+.4f} "
                    f"value_loss={metrics['value_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f} "
                    f"mean_episode_reward={metrics['mean_episode_reward']:+.4f}"
                )
    finally:
        actors.close()

    dims = {"obs_dim": obs_dim, "action_dim": action_dim, "device": device}
    return history, learner, dims


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.config.is_file():
        print(f"[WF-RL] FATAL: config not found: {args.config}", file=__import__("sys").stderr)
        return 2
    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        print("[WF-RL] FATAL: config root must be a mapping", file=__import__("sys").stderr)
        return 2

    # Validate + report the backend stack (honors the `backends:` map).
    stack = resolve_config(cfg.get("backends"))
    print(
        "[WF-RL] backends: "
        + json.dumps({a: stack.backends[a].name for a in sorted(stack.backends)})
    )

    t0 = time.perf_counter()
    history, learner, dims = run_distributed(
        cfg,
        num_actors=args.actors,
        envs_per_actor=args.envs_per_actor,
        worker_count=args.worker_count,
        num_updates=args.updates,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        log_every_n=args.log_every_n,
    )
    wall = time.perf_counter() - t0

    total_env_steps = args.updates * args.actors * args.envs_per_actor * learner.config.rollout_steps
    last = history[-1] if history else {}

    if args.checkpoint_dir is not None:
        from worldfoundry_train.checkpoint import save_checkpoint

        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt = args.checkpoint_dir / "final.pt"
        save_checkpoint(
            ckpt,
            learner.policy,
            learner.ppo.optimizer,
            step=len(history),
            rl_payload={
                "distributed": True,
                "num_actors": args.actors,
                "total_env_steps": total_env_steps,
            },
        )
        print(f"[WF-RL] checkpoint saved: {ckpt}")

    summary = json.dumps({
        "label": "W37 CPU distributed RL skeleton",
        "mode": dims["device"],
        "num_actors": args.actors,
        "envs_per_actor": args.envs_per_actor,
        "worker_count": args.worker_count,
        "updates": len(history),
        "total_env_steps": total_env_steps,
        "obs_dim": dims["obs_dim"],
        "action_dim": dims["action_dim"],
        "wall_seconds": round(wall, 3),
        "final_policy_loss": last.get("policy_loss", 0.0),
        "final_value_loss": last.get("value_loss", 0.0),
        "final_mean_episode_reward": last.get("mean_episode_reward", 0.0),
    })
    print(f"[WF-RL] summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
