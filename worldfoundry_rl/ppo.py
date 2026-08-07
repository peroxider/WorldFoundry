"""Minimal self-built PPO trainer (W34) — rollout → GAE → clipped update.

Closes roadmap G7 ("no PPO/GRPO trainer"). This is deliberately a *small*
reference implementation, not a competitor to verl / RLinf: its job is to

1. prove the Phase-4 RL loop closes end-to-end on CPU
   (:class:`worldfoundry_rl.env_pool.SyncEnvPool` → rollout → advantage →
   gradient step → reward goes up), and
2. serve as the **correctness baseline** the W35 framework bake-off is
   scored against (per
   ``docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`` §6, the
   framework must reproduce this trainer's learning curve on the same
   task before it is adopted).

Scope boundaries (intentional):

* **Synchronous, single-process.** Distributed rollout / learner sharding
  is W37 and rides on the chosen framework's scheduler, not on this file.
* **No NPU code path.** Everything runs on whatever device the caller's
  ``backbone`` lives on; there is no ``.to("npu")`` here (that is W20).
* **Continuous actions only.** The policy is a state-independent
  diagonal Gaussian (learnable ``log_std``), which is the standard
  minimal PPO policy and matches FrankaKitchen's 9-DoF continuous
  action space (``worldfoundry_rl.env_franka_kitchen``). Discrete-action
  support is not needed by the W30-selected first env.

Algorithm follows Schulman et al. 2017 (PPO-clip) with GAE-λ
(Schulman et al. 2015): clipped surrogate policy loss + clipped value
loss + entropy bonus, minibatched over several epochs per rollout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Policy / value network
# ---------------------------------------------------------------------------


class PPOActor(nn.Module):
    """Actor-critic head pair over an arbitrary observation encoder.

    Args:
        obs_dim: flattened observation dimensionality.
        action_dim: continuous action dimensionality.
        hidden_dim: width of the default MLP trunk (ignored when
            ``backbone`` is supplied).
        backbone: optional caller-supplied feature extractor mapping
            ``(B, obs_dim) -> (B, feature_dim)``. When omitted a 2-layer
            tanh MLP is built. Supplying a backbone is how W37 will
            attach the V-JEPA 2 encoder without changing this file.
        feature_dim: output width of ``backbone``; required when
            ``backbone`` is supplied.
        log_std_init: initial value of the state-independent log-σ.
        log_std_min: lower clamp on log-σ. Prevents the policy from
            collapsing to a zero-variance delta (which makes the
            importance ratio explode and the run unrecoverable).
        log_std_max: upper clamp on log-σ. Prevents the well-known
            failure mode where early positive advantages sit far from the
            mean, push σ up, and the policy diverges instead of
            concentrating.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 64,
        backbone: nn.Module | None = None,
        feature_dim: int | None = None,
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if backbone is not None and feature_dim is None:
            raise ValueError("feature_dim is required when a backbone is supplied")
        if log_std_min >= log_std_max:
            raise ValueError(
                f"log_std_min ({log_std_min}) must be < log_std_max ({log_std_max})"
            )

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        if backbone is None:
            self.backbone = nn.Sequential(
                nn.Linear(self.obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            feat = hidden_dim
        else:
            self.backbone = backbone
            feat = int(feature_dim)  # type: ignore[arg-type]

        self.policy_head = nn.Linear(feat, self.action_dim)
        self.value_head = nn.Linear(feat, 1)
        # State-independent diagonal Gaussian: σ is a free parameter, not
        # a network output. Simpler and markedly more stable at this scale.
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

        # Small-gain init on the policy head keeps the initial policy close
        # to deterministic-mean behaviour (standard PPO practice); the value
        # head gets a unit-gain init.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(action_mean, value)`` — shapes ``(B, A)`` and ``(B,)``."""
        features = self.backbone(obs)
        mean = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return mean, value

    def distribution(self, obs: Tensor) -> tuple[torch.distributions.Normal, Tensor]:
        """Returns ``(Normal(mean, std), value)``."""
        mean, value = self.forward(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    @torch.no_grad()
    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action. Returns ``(action, log_prob, value)``."""
        dist, value = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Re-score stored actions. Returns ``(log_prob, entropy, value)``."""
        dist, value = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Advantage estimation
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    last_value: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation over a ``(T, N)`` rollout.

    Args:
        rewards: ``(T, N)`` per-step reward.
        values: ``(T, N)`` value estimate at the state the action was
            taken from.
        dones: ``(T, N)`` 1.0 when the episode ended at that step (the
            successor value is masked out), 0.0 otherwise.
        last_value: ``(N,)`` bootstrap value of the state reached after
            the final rollout step.
        gamma: discount factor.
        lam: GAE-λ trace decay.

    Returns:
        ``(advantages, returns)``, both ``(T, N)``. ``returns`` is
        ``advantages + values``, i.e. the λ-return used as the value
        regression target.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            "rewards/values/dones must share shape (T, N); got "
            f"{tuple(rewards.shape)} / {tuple(values.shape)} / {tuple(dones.shape)}"
        )
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be 2-D (T, N); got {rewards.ndim}-D")
    if last_value.shape != rewards.shape[1:]:
        raise ValueError(
            f"last_value must have shape (N,) = {tuple(rewards.shape[1:])}; "
            f"got {tuple(last_value.shape)}"
        )

    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(num_steps)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


@dataclass
class RolloutBuffer:
    """One on-policy rollout, laid out ``(T, N)`` = (steps, parallel envs)."""

    observations: Tensor  # (T, N, obs_dim)
    actions: Tensor  # (T, N, action_dim)
    log_probs: Tensor  # (T, N)
    values: Tensor  # (T, N)
    rewards: Tensor  # (T, N)
    dones: Tensor  # (T, N)
    advantages: Tensor  # (T, N)
    returns: Tensor  # (T, N)

    @property
    def num_steps(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.observations.shape[1])

    def flatten(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Collapse the ``(T, N)`` layout to ``(T*N, ...)`` for minibatching."""
        batch = self.num_steps * self.num_envs
        return (
            self.observations.reshape(batch, -1),
            self.actions.reshape(batch, -1),
            self.log_probs.reshape(batch),
            self.advantages.reshape(batch),
            self.returns.reshape(batch),
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """PPO hyperparameters. Defaults follow the original paper's Mujoco set."""

    rollout_steps: int = 128
    epochs_per_update: int = 4
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {self.rollout_steps}")
        if self.minibatch_size < 1:
            raise ValueError(f"minibatch_size must be >= 1, got {self.minibatch_size}")
        if self.epochs_per_update < 1:
            raise ValueError(
                f"epochs_per_update must be >= 1, got {self.epochs_per_update}"
            )
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError(f"clip_ratio must be in (0, 1), got {self.clip_ratio}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")


def flatten_obs(obs: Any) -> np.ndarray:
    """Flatten one env observation into a 1-D ``float32`` vector.

    Handles the three shapes the Phase-4 envs emit: a numpy array, a plain
    scalar, and the ``dict`` observation contract used by
    :class:`worldfoundry_rl.env_franka_kitchen.FrankaKitchenEnv`
    (``{"ee_pos": (3,), "target_pos": (3,), "task_success": bool}``).
    Dict keys are visited in **sorted order** so the flattened layout is
    stable across steps and processes.
    """
    if isinstance(obs, dict):
        parts = [flatten_obs(obs[k]) for k in sorted(obs)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if isinstance(obs, Tensor):
        return obs.detach().cpu().reshape(-1).to(torch.float32).numpy()
    if isinstance(obs, (bool, np.bool_)):
        return np.asarray([float(obs)], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def infer_obs_dim(pool: Any) -> int:
    """Derive the flattened obs width from a pool's current observation."""
    obs = getattr(pool, "obs", None)
    if not obs:
        raise ValueError(
            "cannot infer obs_dim: pool has no current observation; "
            "call pool.reset() first or pass obs_dim explicitly"
        )
    return int(flatten_obs(obs[0]).shape[0])


class PPO:
    """PPO-clip trainer driving a vectorized :class:`~worldfoundry_rl.env_pool.EnvPool`.

    Args:
        actor: the policy/value network.
        config: hyperparameters (defaults to :class:`PPOConfig`).
        optimizer: optional pre-built optimizer; defaults to Adam over
            ``actor.parameters()`` at ``config.learning_rate``.
        action_transform: optional callable applied to each sampled action
            before it is handed to the env (e.g. clipping to the env's
            action bounds). Receives and returns a ``(action_dim,)``
            numpy array.
    """

    def __init__(
        self,
        actor: PPOActor,
        config: PPOConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.actor = actor
        self.config = config or PPOConfig()
        self.optimizer = optimizer or torch.optim.Adam(
            actor.parameters(), lr=self.config.learning_rate
        )
        self.action_transform = action_transform
        # Running per-env episode return, used to report mean_episode_reward.
        self._episode_returns: list[float] = []

    # -- rollout ---------------------------------------------------------

    def _stack_obs(self, obs_list: Sequence[Any]) -> Tensor:
        return torch.as_tensor(
            np.stack([flatten_obs(o) for o in obs_list]), dtype=torch.float32
        )

    @torch.no_grad()
    def collect_rollout(self, pool: Any) -> RolloutBuffer:
        """Run ``config.rollout_steps`` env steps and return the filled buffer.

        The pool is *not* reset here — rollouts continue from wherever the
        pool currently is, which is what on-policy training wants across
        successive updates. ``SyncEnvPool`` auto-resets terminated envs, so
        episode boundaries are handled by the pool, and ``dones`` records
        them for GAE masking.
        """
        cfg = self.config
        num_envs = int(pool.num_envs)
        obs_t = self._stack_obs(pool.obs)

        obs_buf = torch.zeros(cfg.rollout_steps, num_envs, obs_t.shape[1])
        act_buf = torch.zeros(cfg.rollout_steps, num_envs, self.actor.action_dim)
        logp_buf = torch.zeros(cfg.rollout_steps, num_envs)
        val_buf = torch.zeros(cfg.rollout_steps, num_envs)
        rew_buf = torch.zeros(cfg.rollout_steps, num_envs)
        done_buf = torch.zeros(cfg.rollout_steps, num_envs)

        running = np.zeros(num_envs, dtype=np.float64)
        completed: list[float] = []

        for t in range(cfg.rollout_steps):
            action, log_prob, value = self.actor.act(obs_t)
            obs_buf[t] = obs_t
            act_buf[t] = action
            logp_buf[t] = log_prob
            val_buf[t] = value

            actions_np = action.cpu().numpy()
            env_actions = [
                self.action_transform(a) if self.action_transform else a
                for a in actions_np
            ]
            next_obs, rewards, terminated, truncated, _ = pool.step(env_actions)

            rew_buf[t] = torch.as_tensor(np.asarray(rewards), dtype=torch.float32)
            done = np.logical_or(np.asarray(terminated), np.asarray(truncated))
            done_buf[t] = torch.as_tensor(done.astype(np.float32))

            running += np.asarray(rewards, dtype=np.float64)
            for i, d in enumerate(done):
                if d:
                    completed.append(float(running[i]))
                    running[i] = 0.0

            obs_t = self._stack_obs(next_obs)

        _, last_value = self.actor.forward(obs_t)
        advantages, returns = compute_gae(
            rew_buf,
            val_buf,
            done_buf,
            last_value=last_value,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        # Envs that never terminated during the rollout still contribute a
        # partial return; report those only if nothing finished, so the
        # metric is never empty on short/long-episode tasks.
        self._episode_returns = completed if completed else [float(r) for r in running]

        return RolloutBuffer(
            observations=obs_buf,
            actions=act_buf,
            log_probs=logp_buf,
            values=val_buf,
            rewards=rew_buf,
            dones=done_buf,
            advantages=advantages,
            returns=returns,
        )

    # -- update ----------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run ``epochs_per_update`` minibatched PPO epochs over one rollout.

        Returns mean ``policy_loss`` / ``value_loss`` / ``entropy`` /
        ``approx_kl`` across every minibatch gradient step.
        """
        cfg = self.config
        obs, actions, old_log_probs, advantages, returns = buffer.flatten()
        batch = obs.shape[0]
        minibatch = min(cfg.minibatch_size, batch)

        if cfg.normalize_advantages and batch > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        num_updates = 0

        for _ in range(cfg.epochs_per_update):
            order = torch.randperm(batch)
            for start in range(0, batch, minibatch):
                idx = order[start : start + minibatch]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_logp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                new_logp, entropy, values = self.actor.evaluate_actions(mb_obs, mb_actions)
                ratio = (new_logp - mb_old_logp).exp()

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = torch.nn.functional.mse_loss(values, mb_ret)
                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator.
                    log_ratio = new_logp - mb_old_logp
                    approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()

                totals["policy_loss"] += float(policy_loss.detach())
                totals["value_loss"] += float(value_loss.detach())
                totals["entropy"] += float(entropy_mean.detach())
                totals["approx_kl"] += float(approx_kl)
                num_updates += 1

        if num_updates == 0:  # pragma: no cover - guarded by PPOConfig validation
            raise RuntimeError("PPO.update performed zero gradient steps")
        return {k: v / num_updates for k, v in totals.items()}

    # -- driver ----------------------------------------------------------

    def train(self, pool: Any, *, total_steps: int) -> list[dict[str, float]]:
        """Alternate rollout collection and updates for ``total_steps`` env steps.

        ``total_steps`` counts *pool* steps (each advancing ``num_envs``
        environments), so the number of updates is
        ``ceil(total_steps / rollout_steps)``.

        Returns one metrics dict per update, each carrying the four loss
        terms plus ``mean_episode_reward`` from the rollout that produced it.
        """
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        num_updates = math.ceil(total_steps / self.config.rollout_steps)
        history: list[dict[str, float]] = []
        for _ in range(num_updates):
            buffer = self.collect_rollout(pool)
            metrics = self.update(buffer)
            metrics["mean_episode_reward"] = (
                float(np.mean(self._episode_returns)) if self._episode_returns else 0.0
            )
            history.append(metrics)
        return history


__all__ = [
    "PPOActor",
    "PPOConfig",
    "PPO",
    "RolloutBuffer",
    "compute_gae",
    "flatten_obs",
    "infer_obs_dim",
]
