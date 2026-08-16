"""ActorPool — multi-process rollout collection for distributed RL (W37).

The rollout half of the W28 RL topology (see
``docs/superpowers/specs/2026-07-27-ascend-world-model-design.md`` §5.2):
a pool of *actor* subprocesses, each holding a policy snapshot (inference
only) plus an :class:`~worldfoundry_rl.env_pool.AsyncEnvPool`, collects a
:class:`RolloutChunk` of raw transitions, and ships it back to the learner.

Scope / design intent:

* **CPU-first, framework-free.** Deliberately no Ray / verl — this is the
  self-contained skeleton that closes the distributed loop on a single box
  before Atlas + W35 framework land. The process / message contract here is
  what the framework-backed version must reproduce.
* **Windows spawn-safe.** The only start method is ``spawn`` (no ``fork`` on
  Windows), so every object crossing a process boundary must be picklable.
  The actor target :func:`_actor_main` is a module-level function and its
  ``cfg`` is a dict of strings / ints / dicts. Policy weights travel as a
  plain ``state_dict`` (dict of tensors, picklable).
* **Learner-driven weight publication.** The parent owns the learner and
  publishes a versioned ``state_dict`` to every actor before each rollout
  round; actors block on ``recv`` for the next version, collect one rollout
  with it, and reply with the chunk. This keeps the on-policy guarantee:
  every transition was produced under exactly the weights the learner's
  update consumes.

A :class:`RolloutChunk` is intentionally *raw* (transitions + bootstrap
value), NOT advantage-annotated: GAE lives in the learner
(:mod:`worldfoundry_rl.learner_pool`) so advantage math runs once, on the
merged multi-actor buffer.

Per W37 plan §Deliverable 1.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from worldfoundry_rl.env_pool import AsyncEnvPool
from worldfoundry_rl.ppo import flatten_obs


@dataclass
class RolloutChunk:
    """Raw transitions from one actor over one rollout round, laid out ``(T, N)``.

    ``T`` = ``rollout_steps``, ``N`` = that actor's env count. ``last_values``
    is the value estimate of the observation reached *after* the final rollout
    step — the GAE bootstrap for envs that did not terminate.
    """

    version: int
    observations: np.ndarray  # (T, N, obs_dim) float32
    actions: np.ndarray  # (T, N, action_dim) float32
    log_probs: np.ndarray  # (T, N) float32
    values: np.ndarray  # (T, N) float32
    rewards: np.ndarray  # (T, N) float32
    dones: np.ndarray  # (T, N) float32
    last_values: np.ndarray  # (N,) float32

    @property
    def num_steps(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.observations.shape[1])


@torch.no_grad()
def collect_rollout_chunk(
    policy: torch.nn.Module,
    pool: Any,
    *,
    rollout_steps: int,
    version: int,
    obs_dim: int,
    action_dim: int,
) -> RolloutChunk:
    """Run ``rollout_steps`` env steps with ``policy`` and return a raw chunk.

    The pool is *not* reset here (on-policy continuation across rounds);
    terminated envs are auto-reset by the pool and recorded in ``dones``.
    """
    num_envs = int(pool.num_envs)

    def _stack(obs_list):
        return torch.as_tensor(
            np.stack([flatten_obs(o) for o in obs_list]), dtype=torch.float32
        )

    obs_t = _stack(pool.obs)
    obs_buf = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((rollout_steps, num_envs, action_dim), dtype=np.float32)
    logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    val_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    rew_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    done_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

    for t in range(rollout_steps):
        action, log_prob, value = policy.act(obs_t)
        obs_buf[t] = obs_t.numpy()
        act_buf[t] = action.cpu().numpy()
        logp_buf[t] = log_prob.cpu().numpy()
        val_buf[t] = value.cpu().numpy()

        actions_np = action.cpu().numpy()
        next_obs, rewards, terminated, truncated, _ = pool.step(list(actions_np))
        rew_buf[t] = np.asarray(rewards, dtype=np.float32)
        done = np.logical_or(np.asarray(terminated), np.asarray(truncated))
        done_buf[t] = done.astype(np.float32)

        obs_t = _stack(next_obs)

    _, last_value = policy.forward(obs_t)
    return RolloutChunk(
        version=version,
        observations=obs_buf,
        actions=act_buf,
        log_probs=logp_buf,
        values=val_buf,
        rewards=rew_buf,
        dones=done_buf,
        last_values=last_value.cpu().numpy(),
    )


def _build_actor_policy(cfg: dict) -> torch.nn.Module:
    """Build the actor's inference policy from a picklable ``cfg`` dict."""
    from worldfoundry_rl.vision_policy import VisionPolicy

    policy = VisionPolicy.from_state_dim(
        obs_dim=int(cfg["obs_dim"]),
        action_dim=int(cfg["action_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
    )
    policy.eval()
    return policy


def _actor_main(worker_id: int, conn: Any, cfg: dict) -> None:
    """Top-level actor-process target (spawn-safe). See :class:`ActorPool`."""
    policy = _build_actor_policy(cfg)
    pool = AsyncEnvPool(
        cfg["env_type"],
        dict(cfg.get("env_kwargs", {})),
        num_envs=int(cfg["num_envs"]),
        worker_count=int(cfg.get("worker_count", 1)),
        seed=cfg.get("seed"),
    )
    try:
        while True:
            msg = conn.recv()
            if msg == "close":
                return
            if isinstance(msg, tuple) and msg[0] == "weights":
                _, version, state_dict = msg
                policy.load_state_dict(state_dict)
                chunk = collect_rollout_chunk(
                    policy,
                    pool,
                    rollout_steps=int(cfg["rollout_steps"]),
                    version=int(version),
                    obs_dim=int(cfg["obs_dim"]),
                    action_dim=int(cfg["action_dim"]),
                )
                conn.send(("rollout", chunk))
    finally:
        pool.close()


class ActorPool:
    """Manage N actor subprocesses that collect rollouts for a learner.

    Args:
        num_actors: how many actor processes to spawn.
        cfg: picklable per-actor config dict with keys ``env_type``,
            ``env_kwargs``, ``num_envs``, ``rollout_steps``, ``obs_dim``,
            ``action_dim``, ``hidden_dim``, optional ``worker_count`` /
            ``seed``. Every actor is built from the same spec (on-policy
            workers are interchangeable).
        ctx: optional multiprocessing context (defaults to ``"spawn"``).
    """

    def __init__(self, num_actors: int, cfg: dict, *, ctx: Any = None) -> None:
        if num_actors < 1:
            raise ValueError(f"num_actors must be >= 1, got {num_actors}")
        self.num_actors = int(num_actors)
        self._ctx = ctx or mp.get_context("spawn")
        self._workers: list[mp.Process] = []
        self._conns: list[Any] = []
        for i in range(self.num_actors):
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(
                target=_actor_main,
                args=(i, child_conn, cfg),
                daemon=False,  # each actor is a supervisor of its own AsyncEnvPool
            )
            proc.start()
            child_conn.close()
            self._workers.append(proc)
            self._conns.append(parent_conn)

    def publish_weights(self, version: int, state_dict: dict[str, torch.Tensor]) -> None:
        """Broadcast ``state_dict`` tagged ``version`` to every actor.

        Each actor consumes it and replies with one rollout (see :meth:`collect`).
        """
        for conn in self._conns:
            conn.send(("weights", int(version), state_dict))

    def collect(self) -> list[RolloutChunk]:
        """Gather one rollout chunk from every actor, in actor order."""
        chunks: list[RolloutChunk] = []
        for conn in self._conns:
            kind, chunk = conn.recv()
            if kind != "rollout":
                raise RuntimeError(f"ActorPool worker sent unexpected message {kind!r}")
            chunks.append(chunk)
        return chunks

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send("close")
            except Exception:
                pass
        for proc in self._workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass


__all__ = ["RolloutChunk", "collect_rollout_chunk", "ActorPool"]
