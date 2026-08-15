"""Unit tests for :mod:`worldfoundry_rl.vision_policy` (W41 Phase-5 v1).

Pure PyTorch unit tests — no env, no NPU. Proves the pluggable-encoder
seam is contract-correct: forward/act/evaluate_actions shapes match
:class:`worldfoundry_rl.ppo.PPOActor` so the policy drops into the
existing PPO trainer unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from worldfoundry_rl.vision_policy import VisionPolicy


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_none_encoder():
    with pytest.raises(ValueError, match="encoder is required"):
        VisionPolicy(encoder=None, feature_dim=10, action_dim=4)


def test_constructor_rejects_bad_feature_dim():
    encoder = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="feature_dim must be >= 1"):
        VisionPolicy(encoder=encoder, feature_dim=0, action_dim=4)
    with pytest.raises(ValueError, match="feature_dim must be >= 1"):
        VisionPolicy(encoder=encoder, feature_dim=-1, action_dim=4)


def test_constructor_rejects_bad_action_dim():
    encoder = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="action_dim must be >= 1"):
        VisionPolicy(encoder=encoder, feature_dim=8, action_dim=0)


def test_constructor_rejects_inverted_log_std_clamp():
    encoder = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="log_std_min"):
        VisionPolicy(
            encoder=encoder, feature_dim=8, action_dim=4,
            log_std_min=0.0, log_std_max=-1.0,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_from_state_dim_builds_default_mlp():
    p = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5, hidden_dim=64)
    assert isinstance(p.encoder, nn.Sequential)
    assert p.feature_dim == 64
    assert p.action_dim == 5
    assert p.log_std.shape == (5,)


def test_from_state_dim_rejects_bad_obs_dim():
    with pytest.raises(ValueError, match="obs_dim"):
        VisionPolicy.from_state_dim(obs_dim=0, action_dim=4)


# ---------------------------------------------------------------------------
# Forward / distribution / act / evaluate_actions shapes
# ---------------------------------------------------------------------------


def test_forward_batch_returns_correct_shapes():
    p = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5)
    obs = torch.randn(4, 27)
    mean, value = p.forward(obs)
    assert mean.shape == (4, 5)
    assert value.shape == (4,)


def test_forward_single_dict_obs_adds_batch_dim():
    """Dict observations from the env (one sample) are flattened + batched."""
    p = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5)
    single = {
        "arm_pos": np.zeros(5, dtype=np.float64),
        "arm_vel": np.zeros(5, dtype=np.float64),
        "touch": np.zeros(5, dtype=np.float64),
        "hand_pos": np.zeros(3, dtype=np.float64),
        "object_pos": np.zeros(3, dtype=np.float64),
        "object_vel": np.zeros(3, dtype=np.float64),
        "target_pos": np.zeros(3, dtype=np.float64),
    }
    mean, value = p.forward(single)
    assert mean.shape == (1, 5)
    assert value.shape == (1,)


def test_act_returns_action_log_prob_value():
    p = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5)
    obs = torch.randn(3, 27)
    action, log_prob, value = p.act(obs)
    assert action.shape == (3, 5)
    assert log_prob.shape == (3,)
    assert value.shape == (3,)


def test_evaluate_actions_returns_log_prob_entropy_value():
    p = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5)
    obs = torch.randn(3, 27)
    actions = torch.randn(3, 5)
    log_prob, entropy, value = p.evaluate_actions(obs, actions)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3,)
    assert value.shape == (3,)


def test_distribution_normal_params_correct():
    p = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5)
    obs = torch.randn(2, 27)
    dist, value = p.distribution(obs)
    assert dist.loc.shape == (2, 5)
    assert dist.scale.shape == (2, 5)
    # log_std clamp: -5 <= log_std <= 2, so all std > 0
    assert (dist.scale > 0).all()
    assert value.shape == (2,)


# ---------------------------------------------------------------------------
# Pluggable encoder seam (the NPU port story)
# ---------------------------------------------------------------------------


class _TinyCNN(nn.Module):
    """Stand-in for a vision backbone. Maps (B, 3, 16, 16) -> (B, 32)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=4, stride=4)  # -> (B, 8, 4, 4)
        self.fc = nn.Linear(8 * 4 * 4, 32)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def test_pluggable_cnn_encoder_works():
    p = VisionPolicy(encoder=_TinyCNN(), feature_dim=32, action_dim=4)
    img = torch.randn(2, 3, 16, 16)
    mean, value = p.forward(img)
    assert mean.shape == (2, 4)
    assert value.shape == (2,)


def test_to_device_seam_no_op_on_cpu():
    """The .to(device) call site is the NPU port story; verify it doesn't
    crash on the platform we're on now (cpu)."""
    p = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5)
    p.to("cpu")
    obs = torch.randn(2, 27)
    mean, _ = p.forward(obs)
    assert mean.shape == (2, 5)


# ---------------------------------------------------------------------------
# Compatibility with the existing PPO trainer
# ---------------------------------------------------------------------------


@pytest.mark.host_skip(reason="C-T9: ManipulatorEnv triggers dm_control GL; Atlas only")
def test_vision_policy_works_with_ppo_trainer():
    """Round-trip with PPO + SyncEnvPool + a real env — proves the
    VisionPolicy surface is PPO-compatible end-to-end on CPU."""
    from worldfoundry_rl.env_manipulator import ManipulatorEnv
    from worldfoundry_rl.env_pool import SyncEnvPool
    from worldfoundry_rl.ppo import PPO, PPOConfig

    pool = SyncEnvPool([lambda: ManipulatorEnv(task_kwargs={"random": 42})])
    try:
        # 27-dim flattened obs from manipulator: 5+5+5+3+3+3+3.
        policy = VisionPolicy.from_state_dim(obs_dim=27, action_dim=5)
        cfg = PPOConfig(rollout_steps=16, epochs_per_update=2, minibatch_size=8)
        ppo = PPO(policy, cfg)
        history = ppo.train(pool, total_steps=16)
        assert len(history) == 1
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            assert k in history[0]
    finally:
        pool.close()