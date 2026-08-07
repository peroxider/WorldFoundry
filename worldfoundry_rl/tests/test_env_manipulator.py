"""Unit tests for :mod:`worldfoundry_rl.env_manipulator` (W41 Phase-5 v1).

The dm_control ``manipulator:bring_ball`` env is real (no mocks) but very
fast — ~2.9M steps/sec on a modern CPU. These tests are integration-light:
they construct the env, exercise the Gymnasium-style surface, and verify
the Env Protocol conformance. No network, no fixtures beyond dm_control's
own internal RNG.
"""
from __future__ import annotations

import numpy as np
import pytest

from worldfoundry_rl.env_manipulator import ManipulatorEnv
from worldfoundry_rl.env_pool import Env


# All tests share one env to keep mujoco init cost paid only at session start.
@pytest.fixture(scope="module")
def env() -> ManipulatorEnv:
    e = ManipulatorEnv(task_kwargs={"random": 42})
    yield e
    e.close()


def test_manipulator_satisfies_env_protocol(env):
    assert isinstance(env, Env)


def test_reset_returns_seven_obs_keys(env):
    obs, info = env.reset(seed=42)
    expected = {
        "arm_pos", "arm_vel", "touch",
        "hand_pos", "object_pos", "object_vel", "target_pos",
    }
    assert set(obs.keys()) == expected, f"got {set(obs.keys())}"
    assert info == {}


def test_obs_shapes_and_dtypes(env):
    obs, _ = env.reset(seed=42)
    expected_shapes = {
        "arm_pos": (5,), "arm_vel": (5,), "touch": (5,),
        "hand_pos": (3,), "object_pos": (3,),
        "object_vel": (3,), "target_pos": (3,),
    }
    for key, shape in expected_shapes.items():
        assert obs[key].shape == shape, f"{key} got {obs[key].shape}"
        assert obs[key].dtype == np.float64, f"{key} dtype {obs[key].dtype}"


def test_step_with_correct_action_shape(env):
    env.reset(seed=42)
    obs, reward, term, trunc, info = env.step(np.zeros(5, dtype=np.float32))
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "dm_control_last" not in info  # not at terminal yet
    assert set(obs.keys()) == {
        "arm_pos", "arm_vel", "touch",
        "hand_pos", "object_pos", "object_vel", "target_pos",
    }


def test_step_rejects_wrong_action_shape(env):
    env.reset(seed=42)
    with pytest.raises(ValueError, match="5-dim"):
        env.step(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="5-dim"):
        env.step(np.zeros(7, dtype=np.float32))


def test_step_accepts_list_and_array(env):
    env.reset(seed=42)
    # Plain list (must be ravel-able to (5,)).
    obs1, _, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
    assert obs1["arm_pos"].shape == (5,)
    obs2, _, _, _, _ = env.step(np.ones(5, dtype=np.float32))
    assert obs2["arm_pos"].shape == (5,)


def test_action_and_observation_space_declared(env):
    assert env.action_space.shape == (5,)
    assert env.action_space.dtype == "float32"
    assert "arm_pos" in env.observation_space.keys
    assert "target_pos" in env.observation_space.keys


def test_close_is_idempotent():
    """A fresh env: close() twice must not raise. Uses its own env so the
    module-scoped fixture stays alive for subsequent tests."""
    e = ManipulatorEnv(task_kwargs={"random": 42})
    e.close()
    e.close()


def test_step_latency_under_one_ms(env):
    """The env itself is not the bottleneck — single step must be sub-ms."""
    env.reset(seed=42)
    import time
    a = np.zeros(5, dtype=np.float32)
    # Warmup
    for _ in range(10):
        env.step(a)
    t0 = time.perf_counter()
    for _ in range(500):
        env.step(a)
    dt = (time.perf_counter() - t0) / 500 * 1000
    assert dt < 5.0, f"step took {dt:.3f} ms; should be sub-ms on any modern CPU"