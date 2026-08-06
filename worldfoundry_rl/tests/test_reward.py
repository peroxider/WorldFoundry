"""Unit tests for worldfoundry_rl.reward + worldfoundry_rl.reward_impl."""
from __future__ import annotations

import numpy as np
import pytest

from worldfoundry_rl.reward import (
    RewardFn,
    compose,
    ensure_reward_fn,
)
from worldfoundry_rl.reward_impl import (
    FrankaKitchenReward,
    LatentDistanceReward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstReward:
    """Minimal RewardFn for compose / ensure_reward_fn tests."""

    def __init__(self, value: float, raises_on_reset: bool = False) -> None:
        self.value = float(value)
        self.reset_calls = 0
        self._raises_on_reset = raises_on_reset

    def __call__(self, state, action, next_state) -> float:
        return self.value

    def reset(self) -> None:
        self.reset_calls += 1
        if self._raises_on_reset:
            raise RuntimeError("reset exploded")


class _FakeEncoder:
    """Maps an input object to a deterministic feature vector.

    Uses ``id(state)`` (object identity) so different state objects get
    different features — sufficient to drive LatentDistanceReward tests
    without depending on torch / HF.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = int(dim)

    def __call__(self, state: object) -> np.ndarray:
        seed = id(state) % 1024
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim)


# ---------------------------------------------------------------------------
# Protocol + ensure_reward_fn
# ---------------------------------------------------------------------------


def test_reward_fn_protocol_runtime_checkable_accepts_conforming_class():
    obj = _ConstReward(0.5)
    assert isinstance(obj, RewardFn)


def test_ensure_reward_fn_returns_object_unchanged():
    obj = _ConstReward(1.0)
    out = ensure_reward_fn(obj)
    assert out is obj


def test_ensure_reward_fn_rejects_object_missing_reset():
    class _NoReset:
        def __call__(self, s, a, ns):
            return 0.0

    with pytest.raises(TypeError, match="RewardFn"):
        ensure_reward_fn(_NoReset())


def test_ensure_reward_fn_rejects_object_missing_call():
    class _NoCall:
        def reset(self):
            pass

    with pytest.raises(TypeError, match="RewardFn"):
        ensure_reward_fn(_NoCall())


def test_ensure_reward_fn_rejects_plain_object():
    with pytest.raises(TypeError, match="RewardFn"):
        ensure_reward_fn(object())


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def test_compose_default_weights_are_ones():
    r1 = _ConstReward(0.5)
    r2 = _ConstReward(0.3)
    composed = compose(r1, r2)
    assert composed(None, None, None) == pytest.approx(0.8)


def test_compose_with_custom_weights():
    r1 = _ConstReward(1.0)
    r2 = _ConstReward(2.0)
    composed = compose(r1, r2, weights=[0.5, 0.25])
    assert composed(None, None, None) == pytest.approx(0.5 + 0.5)


def test_compose_rejects_empty_rewards():
    with pytest.raises(ValueError, match="at least one reward"):
        compose()


def test_compose_rejects_mismatched_weights_length():
    with pytest.raises(ValueError, match="weights length"):
        compose(_ConstReward(0.0), _ConstReward(0.0), weights=[1.0])


def test_compose_reset_fans_out_to_all_components():
    r1 = _ConstReward(0.0)
    r2 = _ConstReward(0.0)
    composed = compose(r1, r2)
    composed.reset()
    assert r1.reset_calls == 1
    assert r2.reset_calls == 1


def test_compose_returned_object_is_reward_fn():
    composed = compose(_ConstReward(0.0))
    assert isinstance(composed, RewardFn)


def test_compose_is_not_a_passthrough_to_first_reward():
    """Sanity: composite must compute the weighted sum, not just the first component."""
    r1 = _ConstReward(1.0)
    r2 = _ConstReward(-0.5)
    composed = compose(r1, r2)
    assert composed(None, None, None) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# FrankaKitchenReward
# ---------------------------------------------------------------------------


def test_franka_kitchen_distance_penalty_zero_when_at_target():
    r = FrankaKitchenReward(distance_coef=2.0)
    next_state = {"ee_pos": np.zeros(3), "target_pos": np.zeros(3), "task_success": False}
    val = r(None, np.zeros(7), next_state)
    # distance = 0, action = 0, no bonus, no step penalty
    assert val == pytest.approx(0.0)


def test_franka_kitchen_distance_penalty_scales_with_distance_squared():
    r = FrankaKitchenReward(distance_coef=1.0, action_coef=0.0, step_penalty=0.0)
    next_state = {
        "ee_pos": np.array([3.0, 4.0, 0.0]),
        "target_pos": np.zeros(3),
        "task_success": False,
    }
    # ||[3,4,0]||^2 = 25, distance_coef=1 → reward = -25
    assert r(None, np.zeros(7), next_state) == pytest.approx(-25.0)


def test_franka_kitchen_action_regularization():
    r = FrankaKitchenReward(distance_coef=0.0, action_coef=0.1, step_penalty=0.0)
    next_state = {"ee_pos": np.zeros(3), "target_pos": np.zeros(3), "task_success": False}
    action = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # ||action||^2 = 5 → reward = -0.5
    assert r(None, action, next_state) == pytest.approx(-0.5)


def test_franka_kitchen_step_penalty_is_constant():
    r = FrankaKitchenReward(distance_coef=0.0, action_coef=0.0, step_penalty=0.1)
    next_state = {"ee_pos": np.zeros(3), "target_pos": np.zeros(3), "task_success": False}
    assert r(None, np.zeros(7), next_state) == pytest.approx(-0.1)


def test_franka_kitchen_success_bonus_first_time_only():
    r = FrankaKitchenReward(distance_coef=0.0, action_coef=0.0, step_penalty=0.0, success_bonus=2.0)
    next_state = {"ee_pos": np.zeros(3), "target_pos": np.zeros(3), "task_success": True}
    # First success → +2
    assert r(None, np.zeros(7), next_state) == pytest.approx(2.0)
    # Second success call → 0 (no double bonus)
    assert r(None, np.zeros(7), next_state) == pytest.approx(0.0)


def test_franka_kitchen_reset_clears_success_flag():
    r = FrankaKitchenReward(success_bonus=2.0)
    next_state = {"ee_pos": np.zeros(3), "target_pos": np.zeros(3), "task_success": True}
    r(None, np.zeros(7), next_state)  # consume the bonus
    r.reset()
    # After reset, next success triggers the bonus again.
    assert r(None, np.zeros(7), next_state) == pytest.approx(2.0)


def test_franka_kitchen_negative_coef_rejected():
    with pytest.raises(ValueError, match="distance_coef"):
        FrankaKitchenReward(distance_coef=-0.1)
    with pytest.raises(ValueError, match="action_coef"):
        FrankaKitchenReward(action_coef=-0.1)
    with pytest.raises(ValueError, match="step_penalty"):
        FrankaKitchenReward(step_penalty=-0.1)
    with pytest.raises(ValueError, match="success_bonus"):
        FrankaKitchenReward(success_bonus=-0.1)


def test_franka_kitchen_accepts_torch_action_tensor():
    """Protocol is array-like agnostic — torch tensors should also work."""
    torch = pytest.importorskip("torch")
    r = FrankaKitchenReward(distance_coef=0.0, action_coef=0.0)
    next_state = {"ee_pos": np.zeros(3), "target_pos": np.zeros(3), "task_success": False}
    action = torch.zeros(7)
    val = r(None, action, next_state)
    assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LatentDistanceReward
# ---------------------------------------------------------------------------


def test_latent_distance_reward_zero_when_no_motion():
    encoder = _FakeEncoder()
    r = LatentDistanceReward(encoder, beta=0.1)
    state = np.zeros(4)
    # encoder is deterministic in `id(state)` → same state twice yields same latent
    assert r(state, None, state) == pytest.approx(0.0)


def test_latent_distance_reward_negative_for_different_states():
    encoder = _FakeEncoder()
    r = LatentDistanceReward(encoder, beta=0.5)
    s = np.zeros(4)
    s_prime = np.ones(4)
    val = r(s, None, s_prime)
    assert val < 0.0


def test_latent_distance_reward_scales_with_beta():
    encoder = _FakeEncoder()
    s = np.zeros(4)
    s_prime = np.ones(4)
    r1 = LatentDistanceReward(encoder, beta=0.1)
    r2 = LatentDistanceReward(encoder, beta=0.5)
    v1 = r1(s, None, s_prime)
    v2 = r2(s, None, s_prime)
    # Higher beta → more negative reward (proportional).
    assert v2 < v1 < 0.0
    assert v2 == pytest.approx(5.0 * v1)


def test_latent_distance_reward_non_positive_beta_rejected():
    encoder = _FakeEncoder()
    with pytest.raises(ValueError, match="beta must be > 0"):
        LatentDistanceReward(encoder, beta=0.0)
    with pytest.raises(ValueError, match="beta must be > 0"):
        LatentDistanceReward(encoder, beta=-0.1)


def test_latent_distance_reward_satisfies_protocol():
    r = LatentDistanceReward(_FakeEncoder(), beta=0.1)
    assert isinstance(r, RewardFn)