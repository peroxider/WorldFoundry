"""Tests for the mock RLinf runner factory (M1-T4 P0 host smoke path).

The factory lives in :mod:`worldfoundry_rl.rlinf_mock_factory` and is
wired through the orchestrator's ``--rl-backend rlinf`` seam (M1-T3).
On Atlas the same seam is fed a real factory that constructs an
``rlinf.runners.embodied_runner.EmbodiedRunner``; here we only assert the
mock's contract -- same shape, same return dict keys.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from worldfoundry_rl.env_manipulator import ManipulatorEnv
from worldfoundry_rl.env_pool import SyncEnvPool
from worldfoundry_rl.vision_policy import VisionPolicy
from worldfoundry_rl.rlinf_mock_factory import (
    MockRLinfRunner,
    MockWorkerGroup,
    build_runner,
)


def _make_pool_and_policy(
    num_envs: int = 2,
    hidden_dim: int = 32,
    seed: int = 0,
) -> tuple[SyncEnvPool, VisionPolicy]:
    """Construct a tiny env pool + matching :class:`VisionPolicy`."""
    pool = SyncEnvPool(
        [
            lambda i=i: ManipulatorEnv(task_kwargs={"random": seed + i})
            for i in range(num_envs)
        ]
    )
    obs = pool.reset(seed=seed)
    sample = obs[0]
    obs_dim = sum(int(np.asarray(v).ravel().size) for v in sample.values())
    action_dim = int(pool.single_action_space.shape[0])
    policy = VisionPolicy.from_state_dim(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
    )
    return pool, policy


@dataclass
class _Cfg:
    n_updates: int = 1
    rollout_steps: int = 4
    latent_dim: int = 16


# ---------------------------------------------------------------------------
# Factory contract
# ---------------------------------------------------------------------------


def test_build_runner_returns_mock_rlinf_runner() -> None:
    """``build_runner(config)`` returns a :class:`MockRLinfRunner`."""
    pool, policy = _make_pool_and_policy()
    runner = build_runner({"env_pool": pool, "policy": policy, "cfg": _Cfg()})
    assert isinstance(runner, MockRLinfRunner)
    pool.close()


def test_build_runner_requires_env_pool() -> None:
    """Missing ``env_pool`` raises ``ValueError`` with a clear message."""
    with pytest.raises(ValueError, match="env_pool"):
        build_runner({"policy": object(), "cfg": _Cfg()})


def test_build_runner_requires_policy() -> None:
    """Missing ``policy`` raises ``ValueError`` with a clear message."""
    pool, _ = _make_pool_and_policy()
    try:
        with pytest.raises(ValueError, match="policy"):
            build_runner({"env_pool": pool, "cfg": _Cfg()})
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Worker groups
# ---------------------------------------------------------------------------


def test_build_runner_creates_four_worker_groups() -> None:
    """The mock stands up actor / rollout / env / reward worker groups."""
    pool, policy = _make_pool_and_policy()
    try:
        runner = build_runner(
            {"env_pool": pool, "policy": policy, "cfg": _Cfg()}
        )
        names = set(runner.groups.keys())
        assert names == {"actor", "rollout", "env", "reward"}
        for name, group in runner.groups.items():
            assert isinstance(group, MockWorkerGroup)
            assert group.name == name
            assert group.initialized is False  # lazy-init in .run()
            assert group.module is not None
    finally:
        pool.close()


def test_reward_worker_group_has_learnable_module() -> None:
    """The reward worker group holds a small ``nn.Module`` reward head."""
    pool, policy = _make_pool_and_policy()
    try:
        runner = build_runner(
            {"env_pool": pool, "policy": policy, "cfg": _Cfg(latent_dim=16)}
        )
        reward_module = runner.groups["reward"].module
        assert isinstance(reward_module, torch.nn.Module)
        n_params = sum(p.numel() for p in reward_module.parameters())
        assert n_params > 0
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def test_runner_run_completes_summary_shape() -> None:
    """``MockRLinfRunner.run()`` returns the documented summary dict."""
    pool, policy = _make_pool_and_policy()
    try:
        runner = build_runner(
            {"env_pool": pool, "policy": policy, "cfg": _Cfg(n_updates=2)}
        )
        out = runner.run()
    finally:
        pool.close()
    assert out["framework"] == "rlinf"
    assert out["status"] == "completed"
    assert out["kind"] == "mock"
    assert out["n_updates"] == 2
    assert out["rollout_steps"] == 4
    assert out["latent_dim"] == 16
    assert set(out["groups"].keys()) == {"actor", "rollout", "env", "reward"}
    assert isinstance(out["update_history"], list)
    assert len(out["update_history"]) == 2
    assert "wall_seconds" in out
    # Every update must carry a numeric loss dict.
    for u in out["update_history"]:
        assert {"policy_loss", "reward_loss", "total_loss"} <= u.keys()
        assert isinstance(u["mean_reward"], float)


def test_runner_runs_with_zero_updates() -> None:
    """``n_updates=0`` is allowed and produces an empty update_history."""
    pool, policy = _make_pool_and_policy()
    try:
        runner = build_runner(
            {
                "env_pool": pool,
                "policy": policy,
                "cfg": _Cfg(n_updates=0, rollout_steps=2),
            }
        )
        out = runner.run()
    finally:
        pool.close()
    assert out["status"] == "completed"
    assert out["update_history"] == []


def test_runner_initializes_groups_during_run() -> None:
    """``run()`` flips ``initialized`` on every worker group."""
    pool, policy = _make_pool_and_policy()
    runner = build_runner(
        {"env_pool": pool, "policy": policy, "cfg": _Cfg()}
    )
    runner.run()
    for group in runner.groups.values():
        assert group.initialized is True
    # Env pool closed by the runner's finally block.


# ---------------------------------------------------------------------------
# Integration with the orchestrator adapter
# ---------------------------------------------------------------------------


def test_orchestrator_rlinf_runner_dispatches_to_mock_factory() -> None:
    """The M1-T3 adapter wires the mock factory through ``RLinfRunner``."""
    from worldfoundry_common.adapters.rlinf import RLinfRunner

    pool, policy = _make_pool_and_policy()
    try:
        runner = RLinfRunner(runner_factory=build_runner)
        result = runner.run(
            {"env_pool": pool, "policy": policy, "cfg": _Cfg()}
        )
    finally:
        pool.close()
    assert result["framework"] == "rlinf"
    assert result["status"] == "completed"
    assert result["kind"] == "mock"
    assert isinstance(result["result"]["update_history"], list)


def test_dotted_path_resolves_to_real_factory() -> None:
    """The CLI flag's ``--rlinf-runner-factory`` dotted path resolves."""
    import importlib

    module = importlib.import_module("worldfoundry_rl.rlinf_mock_factory")
    attr = getattr(module, "build_runner")
    assert callable(attr)
    pool, policy = _make_pool_and_policy()
    try:
        runner = attr(
            {"env_pool": pool, "policy": policy, "cfg": _Cfg(n_updates=1)}
        )
        assert isinstance(runner, MockRLinfRunner)
        runner.run()
    finally:
        pool.close()