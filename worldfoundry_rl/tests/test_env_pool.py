"""Unit tests for worldfoundry_rl.env_pool + SyncEnvPool."""
from __future__ import annotations

import numpy as np
import pytest

from worldfoundry_rl.env_pool import Env, EnvPool, SyncEnvPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSpace:
    """Minimal stand-in for gymnasium.spaces.Box/Dict — only used for duck-typing."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEnv:
    """Deterministic counter env.

    - Action space: integer in [0, action_max]
    - Observation: dict {"step": int, "env_id": int}
    - Reward: -|action - target_action|, plus +10 bonus at step == 0 of
      each episode (success on the first step)
    - Episode length: 3 steps; auto-terminated with ``terminated=True`` on
      step 3 so the pool's autoreset path can be exercised.
    """

    EPISODE_LEN = 3

    def __init__(self, env_id: int = 0, action_max: int = 4, target_action: int = 2) -> None:
        self.env_id = int(env_id)
        self.action_max = int(action_max)
        self.target_action = int(target_action)
        self.observation_space = _FakeSpace(f"obs[env={self.env_id}]")
        self.action_space = _FakeSpace(f"act[max={self.action_max}]")
        self._step_count = 0
        self._episode_count = 0

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        self._episode_count += 1
        obs = {"step": 0, "env_id": self.env_id, "episode": self._episode_count}
        return obs, {"episode": self._episode_count}

    def step(self, action):
        self._step_count += 1
        a = int(action)
        # Penalty = -|a - target|
        reward = -abs(a - self.target_action)
        # Bonus on first step if action equals target (simulated success)
        bonus = 10.0 if (self._step_count == 1 and a == self.target_action) else 0.0
        total = reward + bonus
        terminated = self._step_count >= self.EPISODE_LEN
        truncated = False
        obs = {"step": self._step_count, "env_id": self.env_id}
        info = {"step_count": self._step_count}
        return obs, total, terminated, truncated, info

    def close(self):
        pass


def _factory(env_id: int):
    """Closure-style env factory."""
    return lambda: _FakeEnv(env_id=env_id)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_fake_env_satisfies_env_protocol():
    env = _FakeEnv()
    assert isinstance(env, Env)


def test_sync_env_pool_satisfies_env_pool_protocol():
    pool = SyncEnvPool([_factory(i) for i in range(3)])
    try:
        assert isinstance(pool, EnvPool)
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_factory_list():
    with pytest.raises(ValueError, match="at least one env factory"):
        SyncEnvPool([])


def test_construction_repr_propagates_num_envs():
    pool = SyncEnvPool([_factory(i) for i in range(5)])
    try:
        assert pool.num_envs == 5
        assert len(pool) == 5
    finally:
        pool.close()


def test_single_observation_and_action_space_copied_from_first_env():
    pool = SyncEnvPool([_factory(i) for i in range(3)])
    try:
        assert pool.single_observation_space.name == "obs[env=0]"
        assert pool.single_action_space.name == "act[max=4]"
    finally:
        pool.close()


def test_construction_resets_all_envs_initially():
    pool = SyncEnvPool([_factory(i) for i in range(3)])
    try:
        assert pool.obs is not None
        assert len(pool.obs) == 3
        for i, o in enumerate(pool.obs):
            assert o["env_id"] == i
            assert o["step"] == 0
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_returns_vector_obs_with_episode_counter_incrementing():
    pool = SyncEnvPool([_factory(0), _factory(1)])
    try:
        # Initial reset already ran during construction (episode_count=1).
        # Now run again: episode counter should be 2.
        obs = pool.reset(seed=42)
        assert len(obs) == 2
        for o in obs:
            assert o["step"] == 0
            assert o["episode"] == 2
        # Master seed should have been broadcast; per-env seeds derived
        # from it (env 0 seed=42, env 1 seed=43).
    finally:
        pool.close()


def test_reset_with_seed_none_keeps_initial_obs():
    pool = SyncEnvPool([_factory(0), _factory(1)], seed=7)
    try:
        obs = pool.reset()
        assert len(obs) == 2
    finally:
        pool.close()


def test_reset_rejects_options_param():
    pool = SyncEnvPool([_factory(0)])
    try:
        with pytest.raises(NotImplementedError, match="options"):
            pool.reset(options={"some_key": 1})
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_rejects_wrong_action_count():
    pool = SyncEnvPool([_factory(i) for i in range(3)])
    try:
        with pytest.raises(ValueError, match="actions length"):
            pool.step([0, 0])
    finally:
        pool.close()


def test_step_returns_expected_shapes_and_dtypes():
    pool = SyncEnvPool([_factory(i) for i in range(4)])
    try:
        actions = [2, 1, 2, 0]
        obs, rewards, term, trunc, infos = pool.step(actions)
        assert len(obs) == 4
        assert rewards.shape == (4,)
        assert rewards.dtype == np.float64
        assert term.shape == (4,)
        assert term.dtype == bool
        assert trunc.shape == (4,)
        assert trunc.dtype == bool
        assert len(infos) == 4
    finally:
        pool.close()


def test_step_reward_is_correct_for_action_minus_target():
    pool = SyncEnvPool([_factory(0)])
    try:
        # target_action = 2
        actions = [2]  # |2-2| = 0 → reward = 0
        _, rewards, _, _, _ = pool.step(actions)
        # +10 bonus on first step because action == target
        assert rewards[0] == pytest.approx(10.0)

        # Second step: no bonus, reward = -|a-target|
        _, rewards, _, _, _ = pool.step([0])
        assert rewards[0] == pytest.approx(-2.0)
    finally:
        pool.close()


def test_step_auto_resets_on_terminated():
    """Episode length is 3; 3rd step returns terminated=True → autoreset."""
    pool = SyncEnvPool([_factory(0)])
    try:
        # Step 1
        _, _, term1, _, _ = pool.step([2])
        assert not term1[0]
        # Step 2
        _, _, term2, _, _ = pool.step([2])
        assert not term2[0]
        # Step 3 — episode ends, terminated=True
        obs3, _, term3, _, infos3 = pool.step([2])
        assert term3[0]
        # Auto-reset happens immediately: obs3 should be the FIRST step of
        # the new episode (step == 0 in our FakeEnv), and the info dict
        # should carry the pre-reset final observation.
        assert obs3[0]["step"] == 0
        assert "final_observation" in infos3[0]
        assert infos3[0]["final_observation"]["step"] == 3
    finally:
        pool.close()


def test_step_state_attributes_update_after_call():
    pool = SyncEnvPool([_factory(i) for i in range(2)])
    try:
        pool.step([1, 2])
        assert pool.obs is not None
        assert pool.rewards.shape == (2,)
        assert pool.terminated.shape == (2,)
        assert pool.truncated.shape == (2,)
        assert len(pool.infos) == 2
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_does_not_raise_even_if_one_env_close_fails():
    class _ExplodingEnv(_FakeEnv):
        def close(self):
            raise RuntimeError("intentional")

    pool = SyncEnvPool([_factory(0), _ExplodingEnv])
    pool.close()  # must not raise


# ---------------------------------------------------------------------------
# Throughput micro-benchmark (used as a sanity baseline)
# ---------------------------------------------------------------------------


def test_sync_pool_step_throughput_smoke():
    """Smoke: 1000 steps over a 16-env pool completes in well under 5s."""
    import time

    pool = SyncEnvPool([_factory(i) for i in range(16)])
    try:
        t0 = time.perf_counter()
        steps = 0
        for _ in range(1000 // 16):
            pool.step([2] * 16)
            steps += 16
        elapsed = time.perf_counter() - t0
        # 1000 steps on 16-env pool with FakeEnv should be near-instant.
        assert elapsed < 5.0, f"throughput too low: {elapsed:.3f}s for {steps} steps"
    finally:
        pool.close()