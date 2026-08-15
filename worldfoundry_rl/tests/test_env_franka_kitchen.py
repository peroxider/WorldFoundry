"""Tests for the FrankaKitchenEnv wrapper (W31, v2.15 backend).

Tests are gated on ``dm_control`` being importable; skipped cleanly on
machines where the package isn't installed. The wrapper now uses
``dm_control.manipulation`` (Kinova Jaco arm, 9-DoF) because canonical
``franka_kitchen`` PyPI package does not exist — see
``worldfoundry_rl/env_franka_kitchen.py`` module docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

dm_control = pytest.importorskip("dm_control")


@pytest.mark.host_skip(reason="C-T9: dm_control GL context; Atlas only")
def test_env_franka_kitchen_instantiates_and_resets() -> None:
    """Env can construct; reset returns dict with ee_pos/target_pos/task_success."""
    from worldfoundry_rl.env_franka_kitchen import FrankaKitchenEnv

    env = FrankaKitchenEnv(task_name="lift_brick_features", seed=42)

    obs, info = env.reset()
    assert isinstance(obs, dict), f"obs must be dict, got {type(obs)}"
    assert set(obs.keys()) == {"ee_pos", "target_pos", "task_success"}, (
        f"obs keys must be exactly the 3-key contract, got {sorted(obs.keys())}"
    )
    assert obs["ee_pos"].shape == (3,), f"ee_pos must be (3,), got {obs['ee_pos'].shape}"
    assert obs["target_pos"].shape == (3,), f"target_pos must be (3,), got {obs['target_pos'].shape}"
    assert isinstance(obs["task_success"], bool), "task_success must be bool"

    # Spaces reflect the contract; observation_space is the dict stand-in,
    # action_space is the box stand-in.
    assert env.observation_space.keys == ("ee_pos", "target_pos", "task_success")
    assert env.action_space.shape == (9,), (
        f"FrankaKitchen action_space must be (9,), got {env.action_space.shape}"
    )

    env.close()


@pytest.mark.host_skip(reason="C-T9: dm_control GL context; Atlas only")
def test_env_franka_kitchen_step_returns_obs_dict() -> None:
    """step(action) returns (obs_dict, reward, term, trunc, info) with correct shapes."""
    from worldfoundry_rl.env_franka_kitchen import FrankaKitchenEnv

    env = FrankaKitchenEnv(task_name="lift_brick_features", seed=42)
    env.reset(seed=42)

    # Zero action: dm_control.manipulation accepts a (9,) float action.
    action = np.zeros(9, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, dict)
    assert obs["ee_pos"].shape == (3,)
    assert obs["target_pos"].shape == (3,)
    assert isinstance(reward, float), f"reward must be float, got {type(reward)}"
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Second step with non-zero action: positions must change (real MuJoCo
    # dynamics — the env actually simulates).
    action2 = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.5, -0.5, 0.0], dtype=np.float32)
    obs2, reward2, _, _, _ = env.step(action2)
    diff = np.linalg.norm(obs2["ee_pos"] - obs["ee_pos"])
    assert diff > 1e-6, (
        f"EE position must move under non-zero action (real MuJoCo); "
        f"||ee_pos_after - ee_pos_before||={diff:.3e}"
    )

    env.close()


@pytest.mark.host_skip(reason="C-T9: dm_control GL context; Atlas only")
def test_franka_kitchen_rejects_wrong_action_shape() -> None:
    """step() raises ValueError when action is not 9-dim (defends the contract)."""
    from worldfoundry_rl.env_franka_kitchen import FrankaKitchenEnv

    env = FrankaKitchenEnv(task_name="lift_brick_features", seed=42)
    env.reset(seed=42)

    with pytest.raises(ValueError, match="action must be 9-dim"):
        env.step(np.zeros(4, dtype=np.float32))  # mock env's default action_dim

    env.close()


@pytest.mark.host_skip(reason="C-T9: dm_control GL context; Atlas only")
def test_franka_kitchen_unknown_task_raises_with_actionable_hint() -> None:
    """Unknown task_name raises ValueError listing available dm_control.manipulation tasks."""
    from worldfoundry_rl.env_franka_kitchen import FrankaKitchenEnv

    with pytest.raises(ValueError, match="dm_control.manipulation has no task"):
        FrankaKitchenEnv(task_name="nonexistent_task_xyz", seed=42)
