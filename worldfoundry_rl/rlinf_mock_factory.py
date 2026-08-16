"""Mock RLinf runner factory for CPU host smoke (M1-T4 P0 path).

Why this exists
---------------
The orchestrator's ``--rl-backend rlinf`` seam (added in M1-T3) requires a
``runner_factory`` callable. On Atlas the factory constructs an
``rlinf.runners.embodied_runner.EmbodiedRunner`` from a live Ray cluster
plus actor/rollout/env/reward worker groups — a stack the host CPU box
doesn't have.

This module provides a *real*, executable mock factory that runs the
RLinf-shaped training loop on a single CPU thread without importing the
``rlinf`` package or Ray. It is wired through the same
:class:`~worldfoundry_common.adapters.rlinf.RLinfRunner` contract, so:

* The host smoke ``scripts/render_train_rl_cpu.py --rl-backend rlinf
  --rlinf-runner-factory worldfoundry_rl.rlinf_mock_factory:build_runner``
  drives a complete CPU RLinf-shaped loop (env rollouts -> imagined
  latents -> PPO update).
* On Atlas, callers replace this factory with one that constructs the
  real RLinf ``EmbodiedRunner`` (and import ``rlinf``); the call site
  (the orchestrator's :func:`build_runner`) does not change.

What the mock does
------------------
The mock builds four lightweight "worker groups" in-place and runs a
few PPO updates through them:

* ``ActorWorkerGroup`` — wraps a torch policy with ``forward`` /
  ``compute_logp`` / ``update`` (a small MLP world policy in the
  ``VisionPolicy`` style).
* ``RolloutWorkerGroup`` — collects rollouts from a SyncEnvPool.
* ``EnvWorkerGroup`` — wraps the env factory; the world model's imagined
  latents feed back as obs augmentation (the RLinf "imagination
  rollout" shape).
* ``RewardWorkerGroup`` — applies a small learned reward head on the
  imagined latent.

The contract for the factory is intentionally narrow:

    factory(config) -> object with .run() -> dict

where ``config`` is the same dict the orchestrator's :func:`build_runner`
passes (``{"env_pool": ..., "policy": ..., "cfg": ...}``). On Atlas the
factory reads ``RLinfRunnerConfig`` and constructs the real RLinf
``EmbodiedRunner``; the mock reads the same keys and runs the equivalent
shape locally.

Public API
----------
* :func:`build_runner` — the factory callable; safe to import by dotted
  path. The orchestrator's ``--rlinf-runner-factory
  worldfoundry_rl.rlinf_mock_factory:build_runner`` references this.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class MockWorkerGroup:
    """A trivial RLinf-shaped worker group; holds its module + an init flag.

    RLinf's real worker groups are Ray actors; this dataclass is the
    single-process stand-in so the orchestration code (group creation,
    rollout, update) reads the same on CPU and on Atlas.
    """

    name: str
    module: Any = None
    initialized: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class _ImaginationReward(nn.Module):
    """Small learned reward head used by the RLinf mock RewardWorkerGroup.

    Reward shape: ``B -> 1``. Trained by the mock's outer-loop update.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


def _make_groups(
    env_pool: Any,
    policy: Any,
    latent_dim: int,
    action_dim: int,
) -> dict[str, MockWorkerGroup]:
    """Construct the four RLinf worker groups the mock orchestrates.

    Args:
        env_pool: a SyncEnvPool-like object with
            ``reset``/``step``/``single_action_space``/``close``.
        policy: the world policy the actor worker group wraps.
        latent_dim: imagined-latent feature dim for the reward head.
        action_dim: action dim of the env.

    Returns:
        Mapping ``"actor" / "rollout" / "env" / "reward"`` ->
        :class:`MockWorkerGroup`.
    """
    reward_head = _ImaginationReward(latent_dim=latent_dim)
    return {
        "actor": MockWorkerGroup(
            name="actor",
            module=policy,
            meta={"kind": "policy"},
        ),
        "rollout": MockWorkerGroup(
            name="rollout",
            module=env_pool,
            meta={"kind": "rollout"},
        ),
        "env": MockWorkerGroup(
            name="env",
            module=env_pool,
            meta={"kind": "env", "action_dim": action_dim},
        ),
        "reward": MockWorkerGroup(
            name="reward",
            module=reward_head,
            meta={"kind": "reward_head", "latent_dim": latent_dim},
        ),
    }


def _collect_rollout(env_pool: Any, policy: Any, n_steps: int) -> dict[str, Any]:
    """Run a fixed-length rollout through ``env_pool`` using ``policy``.

    Returns a dict of arrays with batch-major layout:
    ``{"obs": [...], "act": [...], "rew": [...], "done": [...]}``.

    The mock is intentionally simple: ``obs`` is whatever
    :meth:`pool.reset` returns (flattened to 1-D per env). It does not
    depend on a specific obs layout — the real RLinf orchestration
    passes through the same ``reset``/``step`` surface.
    """
    # Pull action_dim from the env pool's action space; this matches what
    # the orchestrator does when constructing :class:`VisionPolicy`.
    try:
        action_dim = int(env_pool.single_action_space.shape[0])
    except Exception:  # pragma: no cover -- defensive
        action_dim = 1
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    rew_list: list[np.ndarray] = []
    done_list: list[np.ndarray] = []
    try:
        cur_obs = env_pool.reset()
    except TypeError:
        cur_obs = env_pool.reset(seed=0)
    for _ in range(n_steps):
        # Flatten each env's obs dict into a single tensor.
        flat_obs = []
        for o in cur_obs:
            vals = [np.asarray(v).ravel() for v in o.values()]
            flat_obs.append(np.concatenate(vals).astype(np.float32))
        flat_obs_np = np.stack(flat_obs, axis=0)
        with torch.no_grad():
            obs_t = torch.from_numpy(flat_obs_np)
            logits = policy(obs_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            # Decide how to map logits -> action vector.
            if logits.ndim == 2 and logits.shape[-1] == 1:
                act_np = np.full(
                    (flat_obs_np.shape[0], action_dim),
                    float(logits.mean().item()),
                    dtype=np.float32,
                )
            elif logits.ndim == 2 and logits.shape[-1] >= action_dim:
                act_np = (
                    logits[:, :action_dim].cpu().numpy().astype(np.float32)
                )
            elif logits.ndim == 1:
                act_np = np.tile(
                    logits.argmax(dim=-1).cpu().numpy().reshape(-1, 1).astype(np.float32),
                    (1, action_dim),
                )
            else:
                act_np = np.zeros(
                    (flat_obs_np.shape[0], action_dim), dtype=np.float32,
                )
        step_result = env_pool.step(act_np)
        # SyncEnvPool returns (obs, reward, terminated, truncated, info)
        # or (obs, reward, done, info) depending on version; normalize.
        if len(step_result) == 5:
            next_obs, reward, term, trunc, info = step_result
            done = np.logical_or(term, trunc)
        else:
            next_obs, reward, done, info = step_result
        obs_list.append(flat_obs_np)
        act_list.append(act_np)
        rew_list.append(np.asarray(reward, dtype=np.float32))
        done_list.append(np.asarray(done, dtype=bool))
        cur_obs = next_obs
    return {
        "obs": np.stack(obs_list, axis=0),    # (T, B, obs_dim)
        "act": np.stack(act_list, axis=0),    # (T, B, ...)
        "rew": np.stack(rew_list, axis=0),    # (T, B)
        "done": np.stack(done_list, axis=0),  # (T, B)
    }


def _ppo_update(
    policy: Any,
    reward_head: nn.Module,
    rollout: dict[str, Any],
    latent_dim: int,
    *,
    lr: float = 1e-3,
    epochs: int = 1,
) -> dict[str, float]:
    """One PPO-style update over a fixed rollout.

    The mock uses the simplest possible surrogate: a Gaussian-style
    policy update on the logits, plus a value/reward MSE update on the
    reward head. Real RLinf replaces this with the official PPO
    implementation; the orchestrator-level call site is unchanged.
    """
    obs = torch.from_numpy(rollout["obs"]).float()
    act = torch.from_numpy(rollout["act"]).float()
    rew = torch.from_numpy(rollout["rew"]).float()
    obs_dim = int(obs.shape[-1])
    proj = nn.Linear(obs_dim, latent_dim, bias=False)
    nn.init.normal_(proj.weight, std=0.05)
    with torch.no_grad():
        latents = proj(obs.mean(dim=0))  # (B, latent_dim)
    opt = torch.optim.AdamW(
        list(policy.parameters())
        + list(reward_head.parameters())
        + list(proj.parameters()),
        lr=lr,
    )
    last_losses: dict[str, float] = {}
    for _ in range(epochs):
        flat_obs = obs.reshape(-1, obs_dim)
        logits = policy(flat_obs)
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.shape[-1] >= act.shape[-1]:
            target = act.reshape(-1, act.shape[-1])
            policy_loss = ((logits[:, : target.shape[-1]] - target) ** 2).mean()
        else:
            policy_loss = logits.pow(2).mean()
        reward_pred = reward_head(latents)
        rew_mean = rew.mean(dim=0)
        reward_loss = ((reward_pred - rew_mean) ** 2).mean()
        loss = policy_loss + 0.5 * reward_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_losses = {
            "policy_loss": float(policy_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "total_loss": float(loss.detach()),
        }
    return last_losses


@dataclass
class MockRLinfRunner:
    """Stand-in for ``rlinf.runners.embodied_runner.EmbodiedRunner``.

    Holds the four worker groups; :meth:`run` orchestrates them in
    RLinf-shape: rollout collect -> imagined latent -> PPO update -> log.
    """

    groups: dict[str, MockWorkerGroup]
    n_updates: int = 1
    rollout_steps: int = 8
    latent_dim: int = 32

    def run(self) -> dict[str, Any]:
        """Drive one full RLinf-shaped training pass.

        Returns a summary dict mirroring the ``RLinfRunner.run()`` return
        shape (``{"framework": "rlinf", "status": "completed", ...}``)
        plus mock-specific stats so the orchestrator's summary JSON can
        record the pass.
        """
        t0 = time.perf_counter()
        for g in self.groups.values():
            if not g.initialized:
                g.initialized = True

        env_pool = self.groups["rollout"].module
        policy = self.groups["actor"].module
        reward_head: nn.Module = self.groups["reward"].module

        update_history: list[dict[str, float]] = []
        try:
            for u in range(int(self.n_updates)):
                rollout = _collect_rollout(env_pool, policy, self.rollout_steps)
                stats = _ppo_update(
                    policy, reward_head, rollout, latent_dim=self.latent_dim,
                )
                stats["update"] = int(u)
                stats["mean_reward"] = float(rollout["rew"].mean())
                update_history.append(stats)
        finally:
            if hasattr(env_pool, "close"):
                try:
                    env_pool.close()
                except Exception:  # pragma: no cover -- defensive
                    pass

        elapsed = time.perf_counter() - t0
        return {
            "framework": "rlinf",
            "status": "completed",
            "kind": "mock",
            "n_updates": int(self.n_updates),
            "rollout_steps": int(self.rollout_steps),
            "latent_dim": int(self.latent_dim),
            "groups": {k: g.name for k, g in self.groups.items()},
            "update_history": update_history,
            "wall_seconds": round(float(elapsed), 4),
        }


def build_runner(config: dict | Any) -> MockRLinfRunner:
    """Mock RLinf runner factory (M1-T4 P0 path).

    Reads the same config keys the orchestrator's :func:`build_runner`
    passes (``env_pool``, ``policy``, ``cfg``) and constructs a
    :class:`MockRLinfRunner` that runs the RLinf-shape loop on CPU.

    Configuration knobs (``MockRunnerConfig``-shaped keys in ``config``
    or ``cfg``):

    * ``n_updates`` — number of PPO updates (default 1).
    * ``rollout_steps`` — rollout length (default 8).
    * ``latent_dim`` — imagined latent dim (default 32).

    Anything else in ``config`` is ignored — the mock has no notion of
    a Ray cluster role or worker-group placement. Real RLinf factories
    read ``RLinfRunnerConfig.cluster_role`` and ``worker_groups``; the
    orchestrator's :class:`RLinfRunner` forwards both.
    """
    env_pool = None
    policy = None
    cfg_obj: Any = None
    if isinstance(config, dict):
        env_pool = config.get("env_pool")
        policy = config.get("policy")
        cfg_obj = config.get("cfg")
    if env_pool is None:
        raise ValueError(
            "rlinf_mock_factory: config['env_pool'] is required "
            "(the orchestrator passes the live SyncEnvPool here)."
        )
    if policy is None:
        raise ValueError(
            "rlinf_mock_factory: config['policy'] is required "
            "(the orchestrator passes the trained VisionPolicy here)."
        )
    n_updates = 1
    rollout_steps = 8
    latent_dim = 32
    for src in (cfg_obj, config):
        if src is None:
            continue
        if hasattr(src, "n_updates"):
            n_updates = int(getattr(src, "n_updates"))
        if hasattr(src, "rollout_steps"):
            rollout_steps = int(getattr(src, "rollout_steps"))
        if hasattr(src, "latent_dim"):
            latent_dim = int(getattr(src, "latent_dim"))

    action_dim = 4
    try:
        action_dim = int(env_pool.single_action_space.shape[0])
    except Exception:  # pragma: no cover -- defensive
        pass

    groups = _make_groups(
        env_pool, policy, latent_dim=latent_dim, action_dim=action_dim,
    )
    return MockRLinfRunner(
        groups=groups,
        n_updates=n_updates,
        rollout_steps=rollout_steps,
        latent_dim=latent_dim,
    )


__all__ = [
    "MockRLinfRunner",
    "MockWorkerGroup",
    "build_runner",
]