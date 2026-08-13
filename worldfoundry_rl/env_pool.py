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

import importlib
import multiprocessing as mp
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


# ---------------------------------------------------------------------------
# AsyncEnvPool — subprocess-vectorized env pool (W37 distributed RL).
# ---------------------------------------------------------------------------


def make_single_env(env_type: str, env_kwargs: dict) -> Env:
    """Build one env from a ``"package.module:ClassName"`` string spec.

    Mirrors :func:`worldfoundry_rl.train._import_class`. The class is looked
    up by dotted path so the spec is *picklable* — required when an
    ``AsyncEnvPool`` spawns workers on Windows / via the ``"spawn"`` context,
    where closures and lambdas cannot cross the process boundary.
    """
    if ":" not in env_type:
        raise ValueError(
            f"env_type must be of the form 'package.module:ClassName', got {env_type!r}"
        )
    module_path, class_name = env_type.rsplit(":", 1)
    module = importlib.import_module(module_path)
    env_cls = getattr(module, class_name)
    return env_cls(**dict(env_kwargs))


def _env_worker_main(
    worker_id: int,
    conn: Any,
    env_type: str,
    env_kwargs: dict,
    shard_count: int,
    seed: int | None,
) -> None:
    """Top-level target for an AsyncEnvPool worker subprocess (spawn-safe).

    Owns a :class:`SyncEnvPool` over ``shard_count`` envs and serves
    ``reset`` / ``step`` / ``close`` messages over a duplex Pipe. Running this
    as a module-level function (with only picklable args) is what makes the
    subprocess spawn on Windows without a ``fork`` start method.
    """
    factories = [lambda: make_single_env(env_type, env_kwargs) for _ in range(shard_count)]
    pool = SyncEnvPool(factories, seed=seed)
    try:
        while True:
            msg = conn.recv()
            if msg == "close":
                return
            if msg == "reset":
                conn.send(("ok", pool.reset(seed=seed)))
            elif isinstance(msg, tuple) and msg[0] == "step":
                _, actions = msg
                conn.send(("ok", pool.step(actions)))
    finally:
        pool.close()
        conn.close()


class AsyncEnvPool:
    """Subprocess-vectorized env pool: N envs sharded across worker processes.

    This is the W37 multi-process variant of :class:`SyncEnvPool`. Each
    worker subprocess owns a :class:`SyncEnvPool` over a shard and steps it
    remotely; the parent fans a vector action out to the workers and gathers
    the per-env results back. Envs therefore run in separate processes, so a
    slow simulator (MuJoCo / SAPIEN / CARLA) no longer blocks the caller's
    policy forward on the same thread.

    Unlike :class:`SyncEnvPool` (which takes factories), this variant takes an
    ``env_type`` **string** + ``env_kwargs``. The spec is picklable, which is
    mandatory for the ``"spawn"``-only start method on Windows and for use
    from the distributed actor processes (see :mod:`worldfoundry_rl.actor_pool`).

    Exposes the env-side subset of the :class:`EnvPool` surface used by the
    RL stack: ``num_envs`` / ``obs`` / ``step`` / ``reset`` / ``close``.

    Args:
        env_type: ``"package.module:ClassName"`` env spec.
        env_kwargs: kwargs forwarded to the env constructor.
        num_envs: total envs to run.
        worker_count: number of worker subprocesses to spawn (clamped to
            ``[1, num_envs]``). Each worker runs ``ceil(num_envs / n)`` envs.
        seed: optional master seed broadcast to every env.
        ctx: optional multiprocessing context (defaults to ``"spawn"``).
    """

    def __init__(
        self,
        env_type: str,
        env_kwargs: dict | None = None,
        num_envs: int = 1,
        *,
        worker_count: int = 1,
        seed: int | None = None,
        ctx: mp.context.BaseContext | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = int(num_envs)
        self._ctx = ctx or mp.get_context("spawn")

        self._worker_count = min(int(worker_count), self.num_envs)
        base, rem = divmod(self.num_envs, self._worker_count)
        self._shard_sizes = [
            base + (1 if i < rem else 0) for i in range(self._worker_count)
        ]

        self._workers: list[mp.Process] = []
        self._conns: list[Any] = []
        for wid, shard_count in enumerate(self._shard_sizes):
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(
                target=_env_worker_main,
                args=(wid, child_conn, env_type, dict(env_kwargs or {}), shard_count, seed),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent only keeps parent_conn
            self._workers.append(proc)
            self._conns.append(parent_conn)

        self.obs = self._call_all_reset()

    # -- internals ------------------------------------------------------

    def _call_all_reset(self) -> list[Any]:
        """Reset every worker and return the flattened obs list."""
        for conn in self._conns:
            conn.send("reset")
        obs_all: list[Any] = []
        for conn in self._conns:
            kind, obs = conn.recv()
            if kind != "ok":
                raise RuntimeError(f"AsyncEnvPool worker reset failed: {kind}")
            obs_all.extend(obs)
        return obs_all

    # -- EnvPool surface ------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> list[Any]:
        if options is not None:
            raise NotImplementedError(
                "AsyncEnvPool.reset does not propagate per-env `options`; "
                "configure via the env factory instead."
            )
        self.obs = self._call_all_reset()
        return self.obs

    def step(
        self,
        actions: list[Any],
    ) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions length ({len(actions)}) must match num_envs ({self.num_envs})"
            )
        # Fan actions out to each worker's shard.
        idx = 0
        for conn, shard in zip(self._conns, self._shard_sizes):
            conn.send(("step", actions[idx : idx + shard]))
            idx += shard

        obs_all: list[Any] = []
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        off = 0
        for conn in self._conns:
            kind, (o, r, term, trunc, info_list) = conn.recv()
            if kind != "ok":
                raise RuntimeError(f"AsyncEnvPool worker step failed: {kind}")
            n = len(o)
            obs_all.extend(o)
            rewards[off : off + n] = r
            terminated[off : off + n] = term
            truncated[off : off + n] = trunc
            infos.extend(info_list)
            off += n

        self.obs = obs_all
        self.rewards = rewards
        self.terminated = terminated
        self.truncated = truncated
        self.infos = infos
        return obs_all, rewards, terminated, truncated, infos

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

    def __len__(self) -> int:
        return self.num_envs


__all__ = ["Env", "EnvPool", "SyncEnvPool", "AsyncEnvPool", "make_single_env"]