"""Unit + smoke tests for worldfoundry_rl.ppo (W34)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from worldfoundry_rl.env_pool import SyncEnvPool
from worldfoundry_rl.ppo import (
    PPO,
    PPOActor,
    PPOConfig,
    RolloutBuffer,
    compute_gae,
    flatten_obs,
    infer_obs_dim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSpace:
    def __init__(self, name: str) -> None:
        self.name = name


class _ReachEnv:
    """Continuous 1-D "reach the target action" task.

    Observation is the constant vector ``[1.0]``; reward is
    ``-(action - target) ** 2``. The optimal policy is a deterministic
    ``action == target``, so a working PPO implementation must drive the
    episode return up toward 0. Episodes are fixed-length.
    """

    def __init__(self, target: float = 0.7, episode_len: int = 16) -> None:
        self.target = float(target)
        self.episode_len = int(episode_len)
        self.observation_space = _FakeSpace("obs[1]")
        self.action_space = _FakeSpace("act[1]")
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        a = float(np.asarray(action).reshape(-1)[0])
        reward = -((a - self.target) ** 2)
        terminated = self._t >= self.episode_len
        obs = np.array([1.0], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def close(self):
        pass


def _reach_pool(num_envs: int = 8, **kwargs) -> SyncEnvPool:
    return SyncEnvPool([lambda: _ReachEnv(**kwargs) for _ in range(num_envs)])


# ---------------------------------------------------------------------------
# compute_gae
# ---------------------------------------------------------------------------


def test_compute_gae_returns_expected_shapes():
    T, N = 5, 3
    adv, ret = compute_gae(
        torch.zeros(T, N),
        torch.zeros(T, N),
        torch.zeros(T, N),
        last_value=torch.zeros(N),
    )
    assert adv.shape == (T, N)
    assert ret.shape == (T, N)


def test_compute_gae_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must share shape"):
        compute_gae(
            torch.zeros(4, 2),
            torch.zeros(4, 3),
            torch.zeros(4, 2),
            last_value=torch.zeros(2),
        )


def test_compute_gae_rejects_bad_last_value_shape():
    with pytest.raises(ValueError, match="last_value"):
        compute_gae(
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            last_value=torch.zeros(5),
        )


def test_compute_gae_with_zero_gamma_lambda_is_td_residual():
    """gamma=0 collapses GAE to ``reward - value`` at every step."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.zeros(3, 1)
    adv, ret = compute_gae(
        rewards, values, dones, last_value=torch.zeros(1), gamma=0.0, lam=0.0
    )
    assert torch.allclose(adv, rewards - values)
    assert torch.allclose(ret, rewards)


def test_compute_gae_masks_bootstrap_at_episode_end():
    """A done step must not propagate the successor value into its advantage."""
    rewards = torch.tensor([[0.0], [0.0]])
    values = torch.tensor([[0.0], [0.0]])
    # Step 0 ends the episode; step 1 does not.
    dones = torch.tensor([[1.0], [0.0]])
    adv, _ = compute_gae(
        rewards, values, dones, last_value=torch.tensor([100.0]), gamma=1.0, lam=1.0
    )
    # Step 1 bootstraps off last_value=100 -> advantage 100.
    assert adv[1].item() == pytest.approx(100.0)
    # Step 0 is masked, so it sees neither step-1 value (0) nor last_value.
    assert adv[0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PPOActor
# ---------------------------------------------------------------------------


def test_actor_rejects_non_positive_dims():
    with pytest.raises(ValueError, match="obs_dim"):
        PPOActor(0, 2)
    with pytest.raises(ValueError, match="action_dim"):
        PPOActor(4, 0)


def test_actor_rejects_backbone_without_feature_dim():
    with pytest.raises(ValueError, match="feature_dim"):
        PPOActor(4, 2, backbone=nn.Linear(4, 8))


def test_actor_accepts_custom_backbone():
    actor = PPOActor(4, 2, backbone=nn.Linear(4, 8), feature_dim=8)
    mean, value = actor(torch.zeros(3, 4))
    assert mean.shape == (3, 2)
    assert value.shape == (3,)


def test_actor_forward_and_act_shapes():
    actor = PPOActor(5, 3)
    obs = torch.randn(7, 5)
    mean, value = actor(obs)
    assert mean.shape == (7, 3)
    assert value.shape == (7,)
    action, log_prob, v = actor.act(obs)
    assert action.shape == (7, 3)
    assert log_prob.shape == (7,)
    assert v.shape == (7,)


def test_actor_evaluate_actions_matches_act_log_prob():
    torch.manual_seed(0)
    actor = PPOActor(4, 2)
    obs = torch.randn(6, 4)
    action, log_prob, _ = actor.act(obs)
    recomputed, entropy, value = actor.evaluate_actions(obs, action)
    assert torch.allclose(recomputed, log_prob, atol=1e-6)
    assert entropy.shape == (6,)
    assert value.shape == (6,)


def test_actor_log_std_is_a_trainable_parameter():
    actor = PPOActor(4, 3)
    assert actor.log_std.requires_grad
    assert actor.log_std.shape == (3,)
    assert any(p is actor.log_std for p in actor.parameters())


def test_actor_act_is_deterministic_under_a_fixed_seed():
    actor = PPOActor(4, 2)
    obs = torch.ones(2, 4)
    torch.manual_seed(1234)
    a1, _, _ = actor.act(obs)
    torch.manual_seed(1234)
    a2, _, _ = actor.act(obs)
    assert torch.equal(a1, a2)


def test_actor_rejects_inverted_log_std_bounds():
    with pytest.raises(ValueError, match="log_std_min"):
        PPOActor(4, 2, log_std_min=1.0, log_std_max=1.0)


def test_actor_clamps_log_std_into_the_configured_band():
    """A runaway log_std must not translate into a runaway sampling sigma."""
    actor = PPOActor(2, 1, log_std_min=-1.0, log_std_max=0.5)
    with torch.no_grad():
        actor.log_std.fill_(50.0)
    dist, _ = actor.distribution(torch.zeros(3, 2))
    assert torch.allclose(dist.scale, torch.full_like(dist.scale, math.exp(0.5)))
    with torch.no_grad():
        actor.log_std.fill_(-50.0)
    dist, _ = actor.distribution(torch.zeros(3, 2))
    assert torch.allclose(dist.scale, torch.full_like(dist.scale, math.exp(-1.0)))


# ---------------------------------------------------------------------------
# PPOConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rollout_steps": 0}, "rollout_steps"),
        ({"minibatch_size": 0}, "minibatch_size"),
        ({"epochs_per_update": 0}, "epochs_per_update"),
        ({"clip_ratio": 0.0}, "clip_ratio"),
        ({"clip_ratio": 1.0}, "clip_ratio"),
        ({"gamma": 1.5}, "gamma"),
        ({"gae_lambda": -0.1}, "gae_lambda"),
    ],
)
def test_config_validation_rejects_bad_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PPOConfig(**kwargs)


# ---------------------------------------------------------------------------
# obs flattening
# ---------------------------------------------------------------------------


def test_flatten_obs_handles_array_scalar_bool_and_tensor():
    assert flatten_obs(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert flatten_obs(3.5).tolist() == [3.5]
    assert flatten_obs(True).tolist() == [1.0]
    assert flatten_obs(torch.tensor([[1.0, 2.0]])).tolist() == [1.0, 2.0]


def test_flatten_obs_dict_uses_sorted_key_order():
    """FrankaKitchen-shaped obs must flatten deterministically."""
    obs = {
        "target_pos": np.array([4.0, 5.0, 6.0]),
        "ee_pos": np.array([1.0, 2.0, 3.0]),
        "task_success": False,
    }
    flat = flatten_obs(obs)
    # sorted keys -> ee_pos, target_pos, task_success
    assert flat.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]
    assert flat.dtype == np.float32


def test_infer_obs_dim_from_pool():
    pool = _reach_pool(num_envs=2)
    try:
        assert infer_obs_dim(pool) == 1
    finally:
        pool.close()


def test_infer_obs_dim_rejects_pool_without_obs():
    class _Empty:
        obs: list = []

    with pytest.raises(ValueError, match="cannot infer obs_dim"):
        infer_obs_dim(_Empty())


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def test_collect_rollout_returns_correctly_shaped_buffer():
    pool = _reach_pool(num_envs=4)
    try:
        actor = PPOActor(infer_obs_dim(pool), 1)
        trainer = PPO(actor, PPOConfig(rollout_steps=10))
        buf = trainer.collect_rollout(pool)
        assert isinstance(buf, RolloutBuffer)
        assert buf.num_steps == 10
        assert buf.num_envs == 4
        assert buf.observations.shape == (10, 4, 1)
        assert buf.actions.shape == (10, 4, 1)
        for field in (buf.log_probs, buf.values, buf.rewards, buf.dones,
                      buf.advantages, buf.returns):
            assert field.shape == (10, 4)
        assert torch.isfinite(buf.advantages).all()
    finally:
        pool.close()


def test_collect_rollout_marks_dones_at_episode_boundaries():
    """Episode length 4 over an 8-step rollout -> exactly 2 dones per env."""
    pool = _reach_pool(num_envs=2, episode_len=4)
    try:
        actor = PPOActor(1, 1)
        trainer = PPO(actor, PPOConfig(rollout_steps=8))
        buf = trainer.collect_rollout(pool)
        assert buf.dones.sum(dim=0).tolist() == [2.0, 2.0]
    finally:
        pool.close()


def test_rollout_buffer_flatten_collapses_step_and_env_axes():
    pool = _reach_pool(num_envs=3)
    try:
        actor = PPOActor(1, 1)
        trainer = PPO(actor, PPOConfig(rollout_steps=5))
        obs, actions, logp, adv, ret = trainer.collect_rollout(pool).flatten()
        assert obs.shape == (15, 1)
        assert actions.shape == (15, 1)
        assert logp.shape == (15,)
        assert adv.shape == (15,)
        assert ret.shape == (15,)
    finally:
        pool.close()


def test_action_transform_is_applied_before_env_step():
    seen: list[float] = []

    class _RecordingEnv(_ReachEnv):
        def step(self, action):
            seen.append(float(np.asarray(action).reshape(-1)[0]))
            return super().step(action)

    pool = SyncEnvPool([lambda: _RecordingEnv() for _ in range(2)])
    try:
        actor = PPOActor(1, 1)
        trainer = PPO(
            actor,
            PPOConfig(rollout_steps=3),
            action_transform=lambda a: np.clip(a, -0.01, 0.01),
        )
        trainer.collect_rollout(pool)
        assert seen, "env.step was never called"
        assert all(abs(a) <= 0.01 + 1e-9 for a in seen)
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_update_returns_finite_metrics_and_changes_parameters():
    torch.manual_seed(0)
    pool = _reach_pool(num_envs=4)
    try:
        actor = PPOActor(1, 1)
        trainer = PPO(actor, PPOConfig(rollout_steps=16, minibatch_size=8))
        before = actor.policy_head.weight.detach().clone()
        metrics = trainer.update(trainer.collect_rollout(pool))
        assert set(metrics) == {"policy_loss", "value_loss", "entropy", "approx_kl"}
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"
        assert not torch.equal(before, actor.policy_head.weight)
    finally:
        pool.close()


def test_update_handles_minibatch_larger_than_batch():
    torch.manual_seed(0)
    pool = _reach_pool(num_envs=2)
    try:
        actor = PPOActor(1, 1)
        # 4 steps * 2 envs = 8 samples but minibatch_size=512.
        trainer = PPO(actor, PPOConfig(rollout_steps=4, minibatch_size=512))
        metrics = trainer.update(trainer.collect_rollout(pool))
        assert np.isfinite(metrics["policy_loss"])
    finally:
        pool.close()


def test_train_rejects_non_positive_total_steps():
    pool = _reach_pool(num_envs=2)
    try:
        trainer = PPO(PPOActor(1, 1), PPOConfig(rollout_steps=4))
        with pytest.raises(ValueError, match="total_steps"):
            trainer.train(pool, total_steps=0)
    finally:
        pool.close()


def test_train_reports_one_metrics_dict_per_update():
    pool = _reach_pool(num_envs=2)
    try:
        trainer = PPO(PPOActor(1, 1), PPOConfig(rollout_steps=4, minibatch_size=4))
        history = trainer.train(pool, total_steps=12)
        assert len(history) == 3  # ceil(12 / 4)
        assert "mean_episode_reward" in history[0]
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# G7 acceptance smoke: reward must go up
# ---------------------------------------------------------------------------


def test_ppo_reward_curve_rises_on_reach_task():
    """W34 acceptance gate: reward curve rises on a synthetic task (CPU).

    ``_ReachEnv`` rewards ``-(action - 0.7) ** 2``; a correct PPO must move
    the policy mean toward 0.7 and shrink sigma, so the mean episode return
    at the end of training must be materially better than at the start.
    """
    torch.manual_seed(7)
    np.random.seed(7)
    pool = _reach_pool(num_envs=8, episode_len=16)
    try:
        actor = PPOActor(infer_obs_dim(pool), 1)
        config = PPOConfig(
            rollout_steps=32,
            minibatch_size=64,
            epochs_per_update=4,
            learning_rate=1e-2,
        )
        trainer = PPO(actor, config)
        history = trainer.train(pool, total_steps=32 * 20)

        rewards = [h["mean_episode_reward"] for h in history]
        first = float(np.mean(rewards[:2]))
        last = float(np.mean(rewards[-2:]))
        # Measured over seeds {0,1,2,7,42,123}: ~-21 -> between -0.98 and
        # -5.42. Require at least a 2x improvement so the gate fails loudly
        # on a broken advantage/ratio/update path but tolerates RL noise.
        assert last > first / 2.0, (
            f"reward did not improve enough: first={first:.3f} last={last:.3f}"
        )
        # The policy mean should have moved toward the 0.7 target.
        with torch.no_grad():
            learned_mean = float(actor(torch.ones(1, 1))[0].item())
        assert abs(learned_mean - 0.7) < abs(0.0 - 0.7), (
            f"policy mean {learned_mean:.3f} is no closer to target 0.7 than init"
        )
    finally:
        pool.close()
