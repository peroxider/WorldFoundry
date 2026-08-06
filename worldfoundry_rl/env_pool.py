"""Vectorized environment pool (W31) — Gymnasium-style sync vec-env API.

Provides a single-env ``Env`` Protocol and a concrete ``SyncEnvPool`` that
runs N copies of an env-factory synchronously. Mirrors the Gymnasium
``VectorEnv`` surface (``reset`` / ``step`` / ``close`` + vector-space
attributes) but only the synchronous subset — async/subprocess workers
are deferred to W37 (RL integration), where they ride on Ray/RLinf/verl's
existing scheduler.

Design contract:

* **Single-env Protocol** — :class:`Env` is structural-typed; any object
  with ``reset`` / ``step`` / ``close`` / ``observation_space`` /
  ``action_space`` attributes is acceptable. This decouples concrete envs
  (FrankaKitchen, SAPIEN, …) from the pool machinery.
* **Vector obs/infos** — returned as a list of per-env payloads; the pool
  does not assume the observation type (numpy / torch / dict). Callers
  in the training stack are responsible for batching (e.g. via
  ``torch.utils.data.default_collate``).
* **Auto-reset on episode end** — when ``terminated`` or ``truncated`` is
  True for an env, the pool resets that env on the next ``step`` call,
  matching Gymnasium's ``VectorEnv.autoreset_mode="on"`` default.

Per W30 ADR §4: env is CPU-resident (MuJoCo / SAPIEN / CARLA), learner is
NPU-resident; this pool is the env-side abstraction. Per W28 §3 topology
(``异构混部``), env workers may share a node with learner workers via Ray.

Unblocked by W30 (MuJoCo env selection); required by W34 (PPO trainer).
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Env(Protocol):
    """Single-env Protocol (Gymnasium-style)."""

    observation_space: Any
    action_space: Any

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Reset the env. Returns ``(observation, info)``."""
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Take one step. Returns ``(observation, reward, terminated, truncated, info)``."""
        ...

    def close(self) -> None:
        """Release env resources."""
        ...


@runtime_checkable
class EnvPool(Protocol):
    """Vector-env pool (Gymnasium ``VectorEnv`` subset)."""

    num_envs: int
    single_observation_space: Any
    single_action_space: Any
    obs: list[Any]
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict]

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> list[Any]:
        """Reset all envs; return vector observation list."""
        ...

    def step(self, actions: list[Any]) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all envs; return ``(obs, rewards, terminated, truncated, infos)``."""
        ...

    def close(self) -> None:
        """Close all envs."""
        ...


class SyncEnvPool:
    """Synchronous vec-env: N envs stepped in the calling thread.

    Args:
        env_factories: callable list, one per env; each returns a fresh
            single ``Env``. Using factories (not envs) ensures every env
            is constructed independently — required when envs hold RNG
            state, file handles, or simulator scenes.
        seed: optional master seed broadcast to each env's ``reset``.
    """

    def __init__(
        self,
        env_factories: list[Callable[[], Env]],
        *,
        seed: int | None = None,
    ) -> None:
        if not env_factories:
            raise ValueError("SyncEnvPool requires at least one env factory")
        self._factories = list(env_factories)
        self._envs: list[Env] = [f() for f in self._factories]
        self.num_envs = len(self._envs)
        if self.num_envs < 1:
            raise RuntimeError("SyncEnvPool constructed with zero live envs")
        # Space attributes copied from the first env; callers that need
        # per-env variation should consult the underlying envs directly.
        self.single_observation_space = self._envs[0].observation_space
        self.single_action_space = self._envs[0].action_space
        self._master_seed = seed
        obs, _ = self._reset_all(seed=seed)
        self.obs = obs
        self.rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.terminated = np.zeros(self.num_envs, dtype=bool)
        self.truncated = np.zeros(self.num_envs, dtype=bool)
        self.infos: list[dict] = [{} for _ in range(self.num_envs)]

    def _reset_all(self, *, seed: int | None) -> tuple[list[Any], list[dict]]:
        obs_list: list[Any] = []
        infos: list[dict] = []
        per_env_seed = None if seed is None else int(seed)
        for i, env in enumerate(self._envs):
            env_seed = None if per_env_seed is None else per_env_seed + i
            o, info = env.reset(seed=env_seed, options=None)
            obs_list.append(o)
            infos.append(info)
        return obs_list, infos

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> list[Any]:
        if options is not None:
            raise NotImplementedError(
                "SyncEnvPool.reset does not propagate per-env `options`; "
                "configure via the env factory instead."
            )
        effective_seed = self._master_seed if seed is None else seed
        obs, infos = self._reset_all(seed=effective_seed)
        self.obs = obs
        self.infos = infos
        self.rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.terminated = np.zeros(self.num_envs, dtype=bool)
        self.truncated = np.zeros(self.num_envs, dtype=bool)
        return obs

    def step(
        self,
        actions: list[Any],
    ) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions length ({len(actions)}) must match num_envs ({self.num_envs})"
            )
        obs_list: list[Any] = []
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        for i, (env, action) in enumerate(zip(self._envs, actions)):
            o, r, term, trunc, info = env.step(action)
            # Auto-reset on episode end (Gymnasium autoreset_mode="on").
            if term or trunc:
                final_o = o
                o, reset_info = env.reset(seed=None, options=None)
                # Surface the final observation under a known key so the
                # caller can bootstrap value targets if needed.
                info = dict(info)
                info["final_observation"] = final_o
                info.setdefault("episode", reset_info.get("episode", {}))
            obs_list.append(o)
            rewards[i] = float(r)
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            infos.append(info)
        self.obs = obs_list
        self.rewards = rewards
        self.terminated = terminated
        self.truncated = truncated
        self.infos = infos
        return obs_list, rewards, terminated, truncated, infos

    def close(self) -> None:
        for env in self._envs:
            try:
                env.close()
            except Exception:
                # Best-effort close: a failing close on one env must not
                # prevent closing the others.
                pass

    def __len__(self) -> int:
        return self.num_envs


__all__ = ["Env", "EnvPool", "SyncEnvPool"]