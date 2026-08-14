"""LearnerPool — gradient-update side of distributed RL (W37).

The learner half of the W28 RL topology (§5.2 of the design doc): owns the
*master* policy + optimizer, receives raw :class:`~worldfoundry_rl.actor_pool.RolloutChunk`
batches from the :class:`~worldfoundry_rl.actor_pool.ActorPool`, merges them
along the env axis, computes GAE + PPO-clip updates, and exposes the updated
``state_dict`` for the orchestrator to publish back to the actors.

Design:

* **Single process.** In this CPU skeleton the learner lives in the main
  process (the orchestrator thread). On Atlas + W35 the same
  :meth:`LearnerPool.update` contract is the seam the framework shards over;
  the merge/GAE/update math is identical.
* **Reuses the W34 baseline trainer.** :meth:`update` builds a
  :class:`~worldfoundry_rl.ppo.RolloutBuffer` and delegates the minibatched
  clipped update to :class:`~worldfoundry_rl.ppo.PPO.update`, so the
  distributed learner provably reproduces the single-process correctness
  baseline (per W35 ADR §6).
* **No envs here.** The learner never touches a simulator — it only consumes
  transitions. This is the NPU/CPU split the topology wants (learner on NPU,
  envs on CPU).

Per W37 plan §Deliverable 1.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from worldfoundry_rl.actor_pool import RolloutChunk
from worldfoundry_rl.ppo import PPO, PPOConfig, RolloutBuffer, compute_gae


def _episode_returns_from_rollout(
    rewards: np.ndarray, dones: np.ndarray
) -> list[float]:
    """Recover per-episode returns from a ``(T, N)`` reward/done rollout."""
    T, N = rewards.shape
    running = np.zeros(N, dtype=np.float64)
    completed: list[float] = []
    for t in range(T):
        running += rewards[t].astype(np.float64)
        for i in range(N):
            if dones[t, i]:
                completed.append(float(running[i]))
                running[i] = 0.0
    return completed if completed else [float(r) for r in running]


class LearnerPool:
    """Merges actor rollout chunks and performs PPO gradient updates.

    Args:
        policy: the master policy/value network (e.g. a
            :class:`~worldfoundry_rl.vision_policy.VisionPolicy`). The caller
            is expected to have already placed it on ``device`` (e.g. via
            ``to_device(policy, device)``).
        config: PPO hyperparameters (defaults to :class:`PPOConfig`).
        optimizer: optional pre-built optimizer; defaults to Adam over
            ``policy.parameters()`` at ``config.learning_rate``.
        device: device the learner computes on (``"cpu"`` or ``"npu"``). The
            actor/env side stays on CPU (mujoco sim), so this is the
            NPU/CPU split the topology wants. Gradient updates run on
            ``device``; the merged rollout is moved there from the CPU chunks.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        config: PPOConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu",
    ) -> None:
        self.ppo = PPO(policy, config or PPOConfig(), optimizer=optimizer)
        self._device = device
        self._version = 0

    @property
    def policy(self) -> torch.nn.Module:
        return self.ppo.actor

    @property
    def config(self) -> PPOConfig:
        return self.ppo.config

    @property
    def version(self) -> int:
        return self._version

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Snapshot the master policy weights for broadcast to actors.

        Always returns **CPU** tensors: the weights cross a process boundary
        to the actor subprocesses via multiprocessing, which cannot pickle
        NPU/CUDA tensors. Moving here is safe regardless of ``device``.
        """
        return {k: v.cpu() for k, v in self.policy.state_dict().items()}

    def _merge_chunks(self, chunks: list[RolloutChunk]) -> RolloutBuffer:
        """Stack raw chunks along the env axis and compute GAE."""
        if not chunks:
            raise ValueError("LearnerPool.update requires at least one rollout chunk")
        T = chunks[0].num_steps
        for c in chunks:
            if c.num_steps != T:
                raise ValueError(
                    "all rollout chunks must share rollout_steps; "
                    f"got {c.num_steps} and {T}"
                )

        obs = torch.from_numpy(
            np.concatenate([c.observations for c in chunks], axis=1)
        ).float()
        act = torch.from_numpy(
            np.concatenate([c.actions for c in chunks], axis=1)
        ).float()
        logp = torch.from_numpy(
            np.concatenate([c.log_probs for c in chunks], axis=1)
        ).float()
        val = torch.from_numpy(
            np.concatenate([c.values for c in chunks], axis=1)
        ).float()
        rew = torch.from_numpy(
            np.concatenate([c.rewards for c in chunks], axis=1)
        ).float()
        done = torch.from_numpy(
            np.concatenate([c.dones for c in chunks], axis=1)
        ).float()
        last_value = torch.from_numpy(
            np.concatenate([c.last_values for c in chunks])
        ).float()  # (N_total,)

        advantages, returns = compute_gae(
            rew,
            val,
            done,
            last_value=last_value,
            gamma=self.config.gamma,
            lam=self.config.gae_lambda,
        )
        return RolloutBuffer(
            observations=obs,
            actions=act,
            log_probs=logp,
            values=val,
            rewards=rew,
            dones=done,
            advantages=advantages,
            returns=returns,
        )

    def _to_device(self, buffer: RolloutBuffer) -> RolloutBuffer:
        """Return a copy of ``buffer`` with all tensors moved to ``self._device``."""
        if self._device == "cpu":
            return buffer
        return RolloutBuffer(
            observations=buffer.observations.to(self._device),
            actions=buffer.actions.to(self._device),
            log_probs=buffer.log_probs.to(self._device),
            values=buffer.values.to(self._device),
            rewards=buffer.rewards.to(self._device),
            dones=buffer.dones.to(self._device),
            advantages=buffer.advantages.to(self._device),
            returns=buffer.returns.to(self._device),
        )

    def update(self, chunks: list[RolloutChunk]) -> dict[str, float]:
        """Merge ``chunks``, run one PPO update, and return metrics.

        Increments the internal weight ``version``. Returns the four loss
        terms plus ``mean_episode_reward`` computed over the merged rollout.
        The merged rollout is built on CPU (raw numpy chunks) for the episode
        metric, then moved to ``self._device`` for the gradient update — so
        the learner runs on NPU while the actor/env side stays on CPU.
        """
        self._version += 1
        buffer = self._merge_chunks(chunks)

        mean_episode_reward = float(
            np.mean(_episode_returns_from_rollout(
                buffer.rewards.numpy(), buffer.dones.numpy()
            ))
        )

        metrics = self.ppo.update(self._to_device(buffer))
        metrics["mean_episode_reward"] = mean_episode_reward
        return metrics


__all__ = ["LearnerPool"]
