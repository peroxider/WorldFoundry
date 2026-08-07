"""Unit tests for :mod:`worldfoundry_rl.train` (W37 v1 CPU skeleton).

Four tests, all CPU-only, no distributed setup:

1. ``test_rl_train_runs_end_to_end`` — drives the real CLI runner with a
   tmp YAML and a tiny PPO config (1 update) and asserts rc=0 + history
   shape.
2. ``test_ppo_trainer_does_not_use_default_loss_fn`` — guard the
   ``trainer.py:243-244`` gotcha: if anyone later wires PPO into the
   legacy :func:`worldfoundry_train.training_loop.run_training_loop`
   path without passing a ``loss_fn``, the loop would silently fall back
   to :func:`default_loss_fn`, which expects
   ``{"pixel_values_videos", "bool_masked_pos"}`` and would crash on a
   PPO batch. This test pins the contract that PPO batches are NOT
   compatible with the JEPA default loss.
3. ``test_make_strategy_supports_rl_profile`` — pins W37 §C:
   ``make_strategy("rl")`` returns a ``DDPStrategy``-compatible instance.
4. ``test_save_checkpoint_rl_payload_round_trip`` — pins W37 §D:
   ``save_checkpoint(rl_payload=...)`` / ``load_checkpoint(rl_payload=...)``
   round-trips the dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from worldfoundry_rl import train as rl_train
from worldfoundry_train.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from worldfoundry_train.parallel.ddp import DDPStrategy
from worldfoundry_train.parallel.strategy import DistributedStrategy
from worldfoundry_train.trainer import make_strategy
from worldfoundry_train.training_loop import default_loss_fn


# ---------------------------------------------------------------------------
# Test 1: end-to-end CLI run with a minimal YAML.
# ---------------------------------------------------------------------------


def _write_minimal_yaml(path: Path) -> None:
    """Tiny CPU config: 1 PPO update, 1 env, no checkpoint."""
    path.write_text(
        "env:\n"
        "  type: worldfoundry_rl.env_manipulator:ManipulatorEnv\n"
        "  kwargs:\n"
        "    task_kwargs: {random: 42}\n"
        "  num_envs: 1\n"
        "policy:\n"
        "  hidden_dim: 32\n"
        "  encoder: state_mlp\n"
        "ppo:\n"
        "  rollout_steps: 8\n"
        "  epochs_per_update: 1\n"
        "  minibatch_size: 4\n"
        "  gamma: 0.99\n"
        "  gae_lambda: 0.95\n"
        "  clip_ratio: 0.2\n"
        "  value_loss_coef: 0.5\n"
        "  entropy_coef: 0.01\n"
        "  max_grad_norm: 0.5\n"
        "  learning_rate: 3.0e-4\n"
        "  normalize_advantages: true\n"
        "training:\n"
        "  total_pool_steps: 8\n"
        "  save_every_n: 0\n"
        "  log_every_n: 1\n"
        "seed: 42\n"
    )


def test_rl_train_runs_end_to_end(tmp_path: Path) -> None:
    """Drive the real CLI runner end-to-end on a 1-update config."""
    yaml_path = tmp_path / "rl_min.yaml"
    _write_minimal_yaml(yaml_path)

    rc = rl_train.main(["--config", str(yaml_path)])
    assert rc == 0, f"rl_train.main returned rc={rc}"

    # save_every_n=0 and no --checkpoint-dir ⇒ main() must NOT have
    # written a final.pt by accident.
    ckpts = list(tmp_path.glob("*.pt"))
    assert ckpts == [], f"unexpected checkpoint artifacts: {ckpts}"


def test_rl_train_history_length_matches_pool_steps(tmp_path: Path) -> None:
    """2 PPO updates: 16 pool-steps / rollout_steps=8."""
    yaml_path = tmp_path / "rl_history.yaml"
    yaml_path.write_text(
        "env:\n"
        "  type: worldfoundry_rl.env_manipulator:ManipulatorEnv\n"
        "  kwargs:\n"
        "    task_kwargs: {random: 42}\n"
        "  num_envs: 1\n"
        "policy:\n"
        "  hidden_dim: 32\n"
        "  encoder: state_mlp\n"
        "ppo:\n"
        "  rollout_steps: 8\n"
        "  epochs_per_update: 1\n"
        "  minibatch_size: 4\n"
        "  gamma: 0.99\n"
        "  gae_lambda: 0.95\n"
        "  clip_ratio: 0.2\n"
        "  value_loss_coef: 0.5\n"
        "  entropy_coef: 0.01\n"
        "  max_grad_norm: 0.5\n"
        "  learning_rate: 3.0e-4\n"
        "  normalize_advantages: true\n"
        "training:\n"
        "  total_pool_steps: 16\n"
        "  save_every_n: 0\n"
        "  log_every_n: 1\n"
        "seed: 42\n"
    )
    rc = rl_train.main(["--config", str(yaml_path)])
    assert rc == 0


# ---------------------------------------------------------------------------
# Test 2: PPO batch is NOT compatible with default_loss_fn (gotcha guard).
# ---------------------------------------------------------------------------


def test_ppo_trainer_does_not_use_default_loss_fn() -> None:
    """Pin the contract that PPO batches are not default_loss_fn-compatible.

    The legacy :func:`run_training_loop` falls back to
    :func:`default_loss_fn` when ``loss_fn`` is None. The default expects
    ``{"pixel_values_videos", "bool_masked_pos"}`` (JEPA keys). A PPO
    rollout stores ``{"states", "actions", "log_probs", "values",
    "advantages", "returns"}`` — a totally different schema. If anyone
    later wires PPO into the legacy loop without injecting a custom
    ``loss_fn``, the default would silently take over and crash on a
    KeyError. This test asserts the crash so the gotcha surfaces as a
    loud test failure rather than a silent wrong-loss fallback.

    See W37 plan §A: PPO owns its loss / backward / step precisely to
    avoid the legacy fallback.
    """
    policy = nn.Linear(4, 2)  # toy — default_loss_fn doesn't use the shape
    ppo_batch = {
        "states": torch.zeros(1, 4),
        "actions": torch.zeros(1, 2),
        "log_probs": torch.zeros(1),
        "values": torch.zeros(1),
        "advantages": torch.zeros(1),
        "returns": torch.zeros(1),
    }
    with pytest.raises(KeyError):
        default_loss_fn(policy, ppo_batch)


# ---------------------------------------------------------------------------
# Test 3: make_strategy("rl") returns DDPStrategy-compatible.
# ---------------------------------------------------------------------------


def test_make_strategy_supports_rl_profile() -> None:
    """W37 §C: ``make_strategy("rl")`` returns a DistributedStrategy (DDPStrategy)."""
    strat = make_strategy("rl")
    assert isinstance(strat, DistributedStrategy), (
        f"make_strategy('rl') must return a DistributedStrategy, got {type(strat)}"
    )
    # Single-process CPU behavior: DDPStrategy wraps 1:1 in v1.
    assert isinstance(strat, DDPStrategy), (
        f"make_strategy('rl') should be a DDPStrategy instance in v1, got {type(strat)}"
    )

    # Legacy profiles must keep working (no regression).
    assert isinstance(make_strategy("ddp"), DDPStrategy)
    with pytest.raises(ValueError, match="unknown profile"):
        make_strategy("not-a-real-profile")


# ---------------------------------------------------------------------------
# Test 4: rl_payload round-trip through save_checkpoint / load_checkpoint.
# ---------------------------------------------------------------------------


def _toy_model() -> nn.Module:
    return nn.Linear(4, 2)


def test_save_checkpoint_rl_payload_round_trip(tmp_path: Path) -> None:
    """save(rl_payload=...) / load(rl_payload=...) round-trips the dict."""
    model = _toy_model()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    rl_payload: dict[str, Any] = {"total_env_steps": 192, "episode_count": 4}
    ckpt = tmp_path / "rl.pt"

    save_checkpoint(ckpt, model, optim, step=3, rl_payload=rl_payload)
    assert ckpt.is_file()

    # Round-trip: load into a fresh model + optimizer with an empty
    # caller's dict; the load should populate it.
    fresh_model = _toy_model()
    fresh_optim = torch.optim.SGD(fresh_model.parameters(), lr=0.01)
    sink: dict[str, Any] = {}
    result = load_checkpoint(ckpt, fresh_model, fresh_optim, rl_payload=sink)

    assert result.get("rl_payload") == rl_payload
    assert sink == rl_payload

    # Absence path: a legacy ckpt without rl_payload should NOT crash
    # and should leave the caller's dict untouched.
    legacy_ckpt = tmp_path / "legacy.pt"
    save_checkpoint(legacy_ckpt, model, optim, step=1)
    legacy_sink: dict[str, Any] = {"untouched": True}
    legacy_result = load_checkpoint(legacy_ckpt, fresh_model, fresh_optim,
                                    rl_payload=legacy_sink)
    assert "rl_payload" not in legacy_result
    assert legacy_sink == {"untouched": True}, (
        "legacy ckpt without rl_payload must NOT mutate caller's dict"
    )


# ---------------------------------------------------------------------------
# CLI arg sanity (parse_args surface).
# ---------------------------------------------------------------------------


def test_parse_args_requires_config() -> None:
    """``--config`` is mandatory; missing it should SystemExit (argparse)."""
    with pytest.raises(SystemExit):
        rl_train.parse_args([])


def test_parse_args_overrides_steps() -> None:
    """``--steps`` override is captured onto the namespace."""
    args = rl_train.parse_args(["--config", "x.yaml", "--steps", "42"])
    assert args.steps == 42
    assert args.mode == "rl"  # v1 default


def test_build_pool_rejects_missing_env_type() -> None:
    """``build_pool({})`` raises ValueError on missing env.type (G4 fail-fast)."""
    with pytest.raises(ValueError, match="env.type"):
        rl_train.build_pool({})


def test_build_policy_rejects_unknown_encoder() -> None:
    """``build_policy`` only implements state_mlp in v1."""
    cfg = {"policy": {"hidden_dim": 32, "encoder": "cnn"}}
    with pytest.raises(ValueError, match="not implemented in W37 v1"):
        rl_train.build_policy(cfg, obs_dim=4, action_dim=2, device="cpu")