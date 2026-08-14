"""Distributed RL orchestration smoke tests (W37 CPU multi-process skeleton).

These tests exercise the real subprocess path on CPU (Windows ``spawn``-only):

1. :class:`~worldfoundry_rl.env_pool.AsyncEnvPool` — subprocess env workers step
   a vector of :class:`~worldfoundry_rl.env_manipulator.ManipulatorEnv` and
   return correct shapes.
2. :class:`~worldfoundry_rl.actor_pool.ActorPool` — spawns actors that collect a
   :class:`~worldfoundry_rl.actor_pool.RolloutChunk` under published weights.
3. :func:`~worldfoundry_rl.distributed_train.run_distributed` — the full
   ActorPool + LearnerPool loop closes end-to-end (rollout → GAE → PPO update →
   weight broadcast) and returns one metrics dict per update.
4. CLI :func:`~worldfoundry_rl.distributed_train.main` — exit code 0 + summary.

Env spec: the importable ``ManipulatorEnv`` is used rather than a module-local
fake, because subprocesses import it by dotted path — a test-module class is not
reliably importable in the spawned child on Windows.

All processes are daemonized and joined/terminated in ``close()``, so a failure
mid-test cannot leak orphan children.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from worldfoundry_rl.actor_pool import ActorPool
from worldfoundry_rl.distributed_train import main as distributed_main
from worldfoundry_rl.distributed_train import run_distributed
from worldfoundry_rl.env_pool import AsyncEnvPool

MANIPULATOR = "worldfoundry_rl.env_manipulator:ManipulatorEnv"


def _tiny_cfg(**overrides: Any) -> dict:
    cfg = {
        "env": {
            "type": MANIPULATOR,
            "kwargs": {"task_kwargs": {"random": 42}},
        },
        "policy": {"hidden_dim": 16, "encoder": "state_mlp"},
        "ppo": {
            "rollout_steps": 8,
            "epochs_per_update": 1,
            "minibatch_size": 8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "learning_rate": 3.0e-4,
            "normalize_advantages": True,
        },
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Test 1: AsyncEnvPool subprocess vector stepping.
# ---------------------------------------------------------------------------


def test_async_env_pool_steps_multiple_workers() -> None:
    pool = AsyncEnvPool(
        MANIPULATOR,
        {"task_kwargs": {"random": 42}},
        num_envs=4,
        worker_count=2,
        seed=42,
    )
    try:
        assert pool.num_envs == 4
        assert len(pool.obs) == 4
        action_dim = 5  # dm_control manipulator 5-DoF
        actions = [np.zeros(action_dim, dtype=np.float32)] * 4
        obs, rewards, term, trunc, infos = pool.step(actions)
        assert len(obs) == 4
        assert rewards.shape == (4,)
        assert term.shape == (4,) and term.dtype == bool
        assert trunc.shape == (4,) and trunc.dtype == bool
        assert len(infos) == 4
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Test 2: ActorPool collects a rollout chunk under published weights.
# ---------------------------------------------------------------------------


def test_actor_pool_collects_rollout_chunk() -> None:
    from worldfoundry_rl.distributed_train import derive_dims

    obs_dim, action_dim = derive_dims(MANIPULATOR, {"task_kwargs": {"random": 42}})
    actor_cfg = {
        "env_type": MANIPULATOR,
        "env_kwargs": {"task_kwargs": {"random": 42}},
        "num_envs": 2,
        "worker_count": 1,
        "rollout_steps": 6,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": 16,
        "seed": 42,
    }
    actors = ActorPool(2, actor_cfg)
    try:
        from worldfoundry_rl.vision_policy import VisionPolicy

        policy = VisionPolicy.from_state_dim(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=16
        )
        actors.publish_weights(0, policy.state_dict())
        chunks = actors.collect()
        assert len(chunks) == 2
        for c in chunks:
            assert c.version == 0
            assert c.observations.shape == (6, 2, obs_dim)
            assert c.actions.shape == (6, 2, action_dim)
            assert c.log_probs.shape == (6, 2)
            assert c.values.shape == (6, 2)
            assert c.rewards.shape == (6, 2)
            assert c.dones.shape == (6, 2)
            assert c.last_values.shape == (2,)
    finally:
        actors.close()


# ---------------------------------------------------------------------------
# Test 3: full distributed loop.
# ---------------------------------------------------------------------------


def test_run_distributed_end_to_end() -> None:
    cfg = _tiny_cfg()
    history, learner, dims = run_distributed(
        cfg,
        num_actors=2,
        envs_per_actor=2,
        worker_count=1,
        num_updates=2,
    )
    assert len(history) == 2
    required = {"policy_loss", "value_loss", "entropy", "approx_kl", "mean_episode_reward"}
    for metrics in history:
        assert required <= set(metrics)
        for k in required:
            assert np.isfinite(metrics[k]), f"{k} not finite: {metrics[k]}"
    assert learner.version == 2
    assert dims["action_dim"] == 5


# ---------------------------------------------------------------------------
# Test 4: CLI runner.
# ---------------------------------------------------------------------------


def test_distributed_main_runs(tmp_path: Path) -> None:
    yaml_path = tmp_path / "rl_dist.yaml"
    cfg = _tiny_cfg()
    lines = [
        "env:",
        "  type: worldfoundry_rl.env_manipulator:ManipulatorEnv",
        "  kwargs:",
        "    task_kwargs: {random: 42}",
        "  num_envs: 2",
        "policy:",
        "  hidden_dim: 16",
        "  encoder: state_mlp",
        "ppo:",
        "  rollout_steps: 8",
        "  epochs_per_update: 1",
        "  minibatch_size: 8",
        "  gamma: 0.99",
        "  gae_lambda: 0.95",
        "  clip_ratio: 0.2",
        "  value_loss_coef: 0.5",
        "  entropy_coef: 0.01",
        "  max_grad_norm: 0.5",
        "  learning_rate: 3.0e-4",
        "  normalize_advantages: true",
        "seed: 42",
    ]
    yaml_path.write_text("\n".join(lines) + "\n")
    rc = distributed_main([
        "--config", str(yaml_path),
        "--actors", "2",
        "--updates", "1",
        "--envs-per-actor", "2",
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# Arg sanity.
# ---------------------------------------------------------------------------


def test_parse_args_requires_config() -> None:
    from worldfoundry_rl.distributed_train import parse_args

    with pytest.raises(SystemExit):
        parse_args([])
