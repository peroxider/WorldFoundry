"""RLinf framework runner/worker adapters.

Wraps RLinf's orchestration behind WorldFoundry's adapter contracts:

* :class:`RLinfRunner` — drives RLinf's :class:`EmbodiedRunner` behind the
  :class:`~worldfoundry_common.adapters.base.TrainingRunner` surface.
* :class:`RLinfEnvWorker` — adapts a WorldFoundry env factory to the env
  surface RLinf's ``EnvWorker`` expects (``reset`` / ``chunk_step``), so a
  WorldFoundry simulator (mock / mujoco / droid / …) can be rolled out by an
  RLinf worker.

RLinf lives under ``spikes/w35-3-rlinf`` (evaluated in W35) and is not yet a
declared dependency (verl superseded it). Both adapters import ``rlinf``
lazily so this module imports cleanly when RLinf is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from worldfoundry_common.adapters.base import TrainingRunner


def _require_rlinf(component: str) -> Any:
    """Import the rlinf package, raising an actionable error if absent."""
    try:
        import rlinf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            f"rlinf is required to build the {component}. "
            "It is vendored under spikes/w35-3-rlinf and not yet a declared "
            "dependency; install it first."
        ) from exc
    return rlinf


class RLinfRunner(TrainingRunner):
    """Adapter that drives an RLinf :class:`EmbodiedRunner`.

    RLinf's ``EmbodiedRunner`` is constructed from a live Ray ``Cluster`` plus
    pre-built worker groups (actor / rollout / env / reward), which cannot be
    safely defaulted inside this adapter. Supply ``runner_factory`` — a
    callable ``(config: dict) -> runner`` that builds the cluster + worker
    groups and returns an object with ``run()`` — to wire the real
    orchestration; otherwise :meth:`run` raises an actionable error.

    Args:
        config: optional default config dict (RLinf YAML content).
        runner_factory: optional ``(dict) -> runner`` used to build the
            RLinf runner. When omitted, the adapter still satisfies the
            :class:`TrainingRunner` contract but requires injection.
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        runner_factory: Callable[[dict], Any] | None = None,
    ) -> None:
        self._config: dict = config or {}
        self._runner_factory = runner_factory
        self._runner: Any = None

    def _ensure_runner(self, config: dict) -> Any:
        if self._runner is None:
            if self._runner_factory is not None:
                self._runner = self._runner_factory(config)
            else:
                _require_rlinf("rlinf runner")
                raise NotImplementedError(
                    "RLinfRunner needs a runner_factory to build worker groups "
                    "(actor/rollout/env/reward) from a live Ray Cluster. Inject "
                    "one that constructs rlinf.runners.embodied_runner.EmbodiedRunner."
                )
        return self._runner

    def run(self, config: dict | None = None, **kwargs: Any) -> dict[str, Any]:
        """Run the RLinf training loop and return a summary dict.

        If the inner runner returns a dict (e.g. the host-smoke mock
        factory in ``worldfoundry_rl.rlinf_mock_factory`` returns
        ``update_history`` / ``wall_seconds`` / ``kind``), the adapter
        merges it under ``result`` so downstream summary JSONs can
        surface the mock's per-update stats. Real RLinf
        ``EmbodiedRunner.run()`` returns ``None``; the adapter still
        returns the canonical ``{"framework", "status"}`` shape.
        """
        cfg = config if config is not None else self._config
        runner = self._ensure_runner(cfg)
        inner = runner.run()
        summary: dict[str, Any] = {"framework": "rlinf", "status": "completed"}
        if isinstance(inner, dict):
            summary["result"] = inner
            if inner.get("kind"):
                summary["kind"] = inner["kind"]
        return summary


class RLinfEnvWorker:
    """Adapt a WorldFoundry env factory to RLinf's env-worker surface.

    RLinf's ``EnvWorker`` interacts with envs through ``reset()`` and
    ``chunk_step(actions)`` (a chunk = ``num_action_chunks`` consecutive
    actions per env). This adapter wraps a WorldFoundry env factory and
    exposes that surface, so an RLinf env worker can roll out a WorldFoundry
    simulator.

    Args:
        env_factory: no-arg callable returning a WorldFoundry env (``reset`` /
            ``step`` / ``close``).
        num_action_chunks: actions per ``chunk_step`` call.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        *,
        num_action_chunks: int = 1,
    ) -> None:
        self._env_factory = env_factory
        self._num_action_chunks = int(num_action_chunks)
        self._env: Any = None

    def _ensure_env(self) -> Any:
        if self._env is None:
            self._env = self._env_factory()
        return self._env

    def reset(self) -> tuple[Any, Any]:
        """Reset the wrapped env; returns ``(obs, info)`` (RLinf EnvOutput obs)."""
        return self._ensure_env().reset()

    def chunk_step(
        self, chunk_actions: Any
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Apply ``num_action_chunks`` actions and return
        ``(obs_list, rewards, terminations, truncations, infos_list)``.

        Delegates to the wrapped env's ``step`` per action; lists are returned
        so RLinf's per-substep bookkeeping works as expected.
        """
        env = self._ensure_env()
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for i in range(self._num_action_chunks):
            act = chunk_actions[i] if self._num_action_chunks > 1 else chunk_actions
            obs, reward, terminated, truncated, info = env.step(list(act))
            obs_list.append(obs)
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            infos.append(info)
        return obs_list, rewards, terms, truncs, infos

    def close(self) -> None:
        env = self._env
        self._env = None
        if env is not None and hasattr(env, "close"):
            env.close()


__all__ = [
    "RLinfRunner",
    "RLinfRunnerConfig",
    "RLinfEnvWorker",
    "RunnerFactory",
]


@dataclass
class RLinfRunnerConfig:
    """Typed configuration passed to a :class:`RunnerFactory`.

    RLinf's :class:`EmbodiedRunner` is built from a live Ray ``Cluster``
    plus pre-built worker groups (``actor`` / ``rollout`` / ``env`` /
    ``reward``). The adapter cannot safely default any of these, so the
    factory receives this dataclass and is responsible for turning it
    into a real RLinf runner. A :class:`dict` form is also accepted for
    downstream call sites that haven't migrated to the dataclass yet.

    Attributes:
        cluster_role: role this process plays in the Ray cluster
            (``"head"`` for the orchestrating process, ``"worker"`` for a
            Ray worker node). Used by the factory to decide whether to
            start ``ray.init`` and where to attach worker groups.
        worker_groups: mapping from RLinf worker-group name (``"actor"``,
            ``"rollout"``, ``"env"``, ``"reward"``) to the actor class
            that backs that group. Empty dict means the factory chooses
            its own defaults.
        world_model_builder: optional zero-arg callable that returns a
            built world model (e.g. ``WorldModelActor`` from M1-T2). The
            factory attaches it to the actor worker group.
        extras: opaque dict for framework-specific overrides (e.g. RLinf
            Hydra config keys). The factory decides how to apply them.
    """

    cluster_role: str = "head"
    worker_groups: dict[str, str] = field(default_factory=dict)
    world_model_builder: Callable[[], Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render as the legacy plain-dict form for downstream callers."""
        return {
            "cluster_role": self.cluster_role,
            "worker_groups": dict(self.worker_groups),
            "world_model_builder": self.world_model_builder,
            "extras": dict(self.extras),
        }


@runtime_checkable
class RunnerFactory(Protocol):
    """Callable protocol for an RLinf runner builder.

    A factory takes either a :class:`RLinfRunnerConfig` or a legacy
    plain ``dict`` (kept for backward compatibility with call sites that
    haven't migrated) and returns an object with ``run()``. Marked
    ``runtime_checkable`` so unit tests can ``isinstance(factory,
    RunnerFactory)`` without subclassing.
    """

    def __call__(self, config: RLinfRunnerConfig | dict[str, Any]) -> Any: ...
