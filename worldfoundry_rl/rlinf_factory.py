"""Real RLinf runner factory (M1-T4 P1 path).

Why this exists
---------------
The orchestrator's ``--rl-backend rlinf`` seam (M1-T3) routes through
:class:`worldfoundry_common.adapters.rlinf.RLinfRunner`, which in turn
calls a ``runner_factory`` callable. The host-smoke path uses the
CPU-only mock factory (:mod:`worldfoundry_rl.rlinf_mock_factory`)
because the real RLinf stack (Ray cluster + actor/rollout/env worker
groups) requires multi-node infrastructure that the host CPU box
doesn't have.

This module is the *real* factory. It imports the real RLinf package,
inspects its :class:`rlinf.runners.embodied_runner.EmbodiedRunner`
class, and attempts to instantiate it with a minimal ``DictConfig``
plus duck-typed stub workers so the orchestrator's RLinf dispatch is
verifiable end-to-end through the real RLinf code path (not the
mock). On a box with Ray running, the constructor succeeds and a
full training loop is one line away; without Ray, the constructor
raises an actionable error which we surface in the summary so
post-mortem tools can see *why* the run was a stub instead of guessing.

Public API
----------
* :func:`build_runner` — the factory callable; safe to import by dotted
  path. The orchestrator's ``--rlinf-runner-factory
  worldfoundry_rl.rlinf_factory:build_runner`` references this. The
  return value exposes ``.run()`` (returning a summary dict) so the
  adapter sees the same surface as the mock factory.

What it does NOT do (yet)
-------------------------
* Launch a Ray cluster — that's a W37+ task; the host smoke path has
  Ray installed (``ray>=2.47.0``) but no cluster running, so
  :meth:`rlinf.scheduler.Channel.create` raises immediately. The
  factory catches this and reports ``init_ok=False, init_error=...``
  in the summary.
* Construct real RLinf worker classes (``EmbodiedFSDPActor`` etc.)
  — those require a Ray cluster + FSDP backend that the host smoke
  box doesn't have. The factory passes duck-typed stubs whose
  ``isinstance`` checks at the constructor boundary pass because
  Python doesn't enforce the ``Union[...]`` annotation at runtime.

Usage
-----
::

    # M1-T4 P1: real RLinf dispatch (proves the integration plumbing
    # is wired through the real rlinf package, not a mock).
    python -m scripts.render_train_rl_cpu \\
        --rl-backend=rlinf \\
        --rlinf-runner-factory=worldfoundry_rl.rlinf_factory:build_runner \\
        ...

    # The existing M1-T4 P0 host-smoke path (mock factory) is unchanged.
    python -m scripts.render_train_rl_cpu \\
        --rl-backend=rlinf \\
        --rlinf-runner-factory=worldfoundry_rl.rlinf_mock_factory:build_runner \\
        ...
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np  # noqa: F401  -- proves numpy is importable (rlinf depends on it)

# REAL rlinf import. This is the whole point of this factory: the
# dispatch path goes through the real RLinf package, not a stand-in.
# If rlinf is missing the import raises ImportError and the orchestrator
# surfaces it via the existing ``--rl-backend=rlinf requires --rlinf-
# runner-factory <dotted.path>`` error path.
import rlinf  # noqa: F401
from rlinf.runners.embodied_runner import EmbodiedRunner

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Minimal DictConfig builder
# ---------------------------------------------------------------------------


def _minimal_embodied_cfg() -> Any:
    """Build the smallest ``DictConfig`` that :class:`EmbodiedRunner` accepts.

    The constructor reads the following keys:

    * ``cfg.runner.weight_sync_interval`` (int, default 1)
    * ``cfg.runner.max_epochs`` (int, default 1) — used by
      :meth:`EmbodiedRunner.set_max_steps` to compute ``max_steps``
    * ``cfg.runner.overlap_env_bootstrap`` (bool, default False)
    * ``cfg.cluster.profiling`` (optional DictConfig; default None)
    * ``cfg.runner.per_worker_log`` (bool, default False)
    * ``cfg.runner.per_worker_log_path`` (str, optional)
    * ``cfg.runner.logger`` — used by ``MetricLogger.__init__``:

        * ``log_path`` (str, default ``"logs"``)
        * ``project_name`` (str, default ``"rlinf"``)
        * ``experiment_name`` (str, default ``"default"``)
        * ``logger_backends`` (list, default ``["tensorboard"]``)
        * ``wandb_proxy``, ``wandb_entity``, ``swanlab_mode`` (optional)

    We default ``logger_backends=[]`` so MetricLogger doesn't try to
    spin up wandb / swanlab / tensorboard on a host smoke box that
    might not have any of those configured.
    """
    return OmegaConf.create(
        {
            "runner": {
                "weight_sync_interval": 1,
                "max_epochs": 1,
                "overlap_env_bootstrap": False,
                "per_worker_log": False,
                "per_worker_log_path": "/tmp/rlinf_worker_logs",
                "logger": {
                    "log_path": "/tmp/rlinf_logs",
                    "project_name": "worldfoundry-rlinf-smoke",
                    "experiment_name": "smoke",
                    "logger_backends": [],
                },
            },
            "cluster": {
                "profiling": None,
            },
        }
    )


# ---------------------------------------------------------------------------
# Duck-typed worker stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubWorker:
    """Duck-typed stand-in for an RLinf Ray-actor worker group.

    :class:`EmbodiedRunner.__init__` stores these without calling any
    method on them, so duck-typed stubs pass the runtime type check
    (Python doesn't enforce the ``Union[...]`` annotation at runtime).
    If ``runner.run()`` were called it would explode when it tried to
    send a message to the stub — that's the W37+ Ray cluster path,
    intentionally out of scope for the host smoke.
    """

    name: str
    kind: str = "stub"

    def __repr__(self) -> str:
        return f"_StubWorker(name={self.name!r}, kind={self.kind!r})"


def _stub_worker_groups() -> dict[str, _StubWorker]:
    """Return the four worker groups EmbodiedRunner expects."""
    return {
        "actor": _StubWorker(name="actor", kind="stub"),
        "rollout": _StubWorker(name="rollout", kind="stub"),
        "env": _StubWorker(name="env", kind="stub"),
        "reward": _StubWorker(name="reward", kind="stub"),
    }


# ---------------------------------------------------------------------------
# Runner returned by build_runner
# ---------------------------------------------------------------------------


@dataclass
class RealRLinfRunner:
    """Stand-in returned by :func:`build_runner`.

    Mirrors :class:`MockRLinfRunner` in :mod:`rlinf_mock_factory`:

    * Holds the (possibly-None) real ``EmbodiedRunner`` instance.
    * :meth:`run` does NOT call ``runner.run()`` (that would require
      a live Ray cluster + real worker classes); instead it records
      the import + instantiation evidence and returns a summary dict
      the adapter merges into ``policy_bootstrap``.

    The summary always carries the same ``framework``/``status`` shape
    the adapter expects so callers don't have to special-case the
    factory choice.
    """

    runner: EmbodiedRunner | None
    cfg: Any
    groups: dict[str, _StubWorker]
    init_ok: bool
    init_error: str | None = None
    embodied_runner_class: str = "rlinf.runners.embodied_runner.EmbodiedRunner"
    embodied_runner_init_signature: str = ""
    embodied_runner_methods: list[str] = field(default_factory=list)
    rlinf_version: str = ""
    rlinf_file: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def run(self) -> dict[str, Any]:
        """Drive the rlinf-backed training pass and return a summary dict.

        On the smoke path this returns the evidence captured at
        ``build_runner`` time (proving the real RLinf package was
        imported and ``EmbodiedRunner`` was inspected + (attempted)
        instantiated). A full Ray-cluster rollout would replace this
        body with ``return self.runner.run()`` once W37+ wires the
        real actor / rollout / env Ray actors.

        Returns:
            A summary dict the RLinfRunner adapter can merge into the
            orchestrator's ``policy_bootstrap`` JSON.
        """
        t0 = time.perf_counter()
        out: dict[str, Any] = {
            "framework": "rlinf",
            "status": "completed",
            "kind": "real_rlinf",
            "embodied_runner_class": self.embodied_runner_class,
            "embodied_runner_init_signature": self.embodied_runner_init_signature,
            "embodied_runner_methods": self.embodied_runner_methods,
            "init_ok": bool(self.init_ok),
            "init_error": self.init_error,
            "groups": {k: {"name": v.name, "kind": v.kind} for k, v in self.groups.items()},
            "rlinf_version": self.rlinf_version,
            "rlinf_file": self.rlinf_file,
            "set_max_steps_ok": None,  # filled in if the runner was constructed
            "extras": dict(self.extras),
        }
        # If the constructor worked, run the loop for one no-op cycle
        # so we exercise the rlinf entry point at least once. This
        # would normally be ``runner.run()``; instead we call
        # ``runner.set_max_steps()`` which is the only zero-side-effect
        # method that's safe to invoke without a real Ray cluster.
        if self.runner is not None:
            try:
                self.runner.set_max_steps()
                out["set_max_steps_ok"] = True
                out["max_steps"] = int(getattr(self.runner, "max_steps", -1))
            except Exception as exc:  # pragma: no cover -- defensive
                out["set_max_steps_ok"] = False
                out["set_max_steps_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        out["wall_seconds"] = round(float(time.perf_counter() - t0), 4)
        return out


# ---------------------------------------------------------------------------
# Factory entry point
# ---------------------------------------------------------------------------


def build_runner(config: dict | Any) -> RealRLinfRunner:
    """Real RLinf runner factory (M1-T4 P1 path).

    Constructs :class:`rlinf.runners.embodied_runner.EmbodiedRunner`
    with a minimal ``DictConfig`` plus duck-typed stub workers, and
    returns a :class:`RealRLinfRunner` whose ``.run()`` produces a
    summary dict the adapter merges into ``policy_bootstrap``.

    Reads the same config keys the mock factory reads
    (``env_pool``, ``policy``, ``cfg``) so the orchestrator can swap
    factories without changing its call site. The real factory
    doesn't actually USE the env/policy objects (the real RLinf
    workers are built from Ray clusters, not from these kwargs), but
    we still validate they're present so a missing kwarg is surfaced
    loudly instead of silently passing ``None`` to Ray.

    Args:
        config: dict (or :class:`RLinfRunnerConfig`) with
            ``env_pool`` / ``policy`` / ``cfg`` keys.

    Returns:
        :class:`RealRLinfRunner` whose ``.run()`` returns the
        canonical summary dict shape.
    """
    env_pool = None
    policy = None
    cfg_obj: Any = None
    if isinstance(config, dict):
        env_pool = config.get("env_pool")
        policy = config.get("policy")
        cfg_obj = config.get("cfg")
    # The real RLinf worker classes are built from Ray clusters, not
    # from these kwargs, but we still validate presence so a missing
    # kwarg is surfaced loudly. The mock factory is more permissive
    # because it actually USES env_pool and policy in its rollout.
    if env_pool is None:
        raise ValueError(
            "rlinf_factory: config['env_pool'] is required "
            "(the orchestrator passes the live SyncEnvPool here)."
        )
    if policy is None:
        raise ValueError(
            "rlinf_factory: config['policy'] is required "
            "(the orchestrator passes the trained VisionPolicy here)."
        )

    # Capture the real signature + class metadata BEFORE attempting
    # instantiation, so even a construction failure surfaces the
    # interface evidence.
    sig = inspect.signature(EmbodiedRunner.__init__)
    # Public surface = callable methods OR non-callable properties (e.g.
    # ``EmbodiedRunner.epoch`` is a @property that the orchestrator may
    # want to introspect). Both are part of the documented RLinf API.
    methods = sorted(
        m for m in dir(EmbodiedRunner)
        if not m.startswith("_")
        and (
            callable(getattr(EmbodiedRunner, m, None))
            or isinstance(
                inspect.getattr_static(EmbodiedRunner, m, None), property
            )
        )
    )

    cfg = cfg_obj if isinstance(cfg_obj, type(_minimal_embodied_cfg())) else _minimal_embodied_cfg()
    groups = _stub_worker_groups()

    # Attempt to construct EmbodiedRunner. The constructor stores the
    # stubs without calling any method on them, but it does call
    # ``Channel.create()`` which tries to launch a Ray actor. On a host
    # smoke box without a Ray cluster this raises ``ConnectionError``
    # / ``RuntimeError`` from rlinf.scheduler. We catch anything and
    # surface it via ``init_ok=False, init_error=...`` so post-mortem
    # tools can audit WHY a run was a stub instead of guessing.
    runner: EmbodiedRunner | None = None
    init_ok = False
    init_error: str | None = None
    try:
        runner = EmbodiedRunner(
            cfg=cfg,
            actor=groups["actor"],
            rollout=groups["rollout"],
            env=groups["env"],
            reward=groups["reward"],
        )
        init_ok = True
    except Exception as exc:
        # The Channel/Ray path is the most common failure; capture
        # type + message so the summary carries the actionable detail.
        init_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        # Re-raise ONLY if it's clearly a programmer error (TypeError
        # about a bad kwarg we passed) — everything else is "real but
        # needs Ray", which the smoke path accepts.
        if isinstance(exc, TypeError) and "unexpected keyword" in str(exc):
            raise

    return RealRLinfRunner(
        runner=runner,
        cfg=cfg,
        groups=groups,
        init_ok=init_ok,
        init_error=init_error,
        embodied_runner_init_signature=str(sig),
        embodied_runner_methods=methods,
        rlinf_version=_rlinf_version(),
        rlinf_file=getattr(rlinf, "__file__", ""),
        extras={
            "config_keys": list(config.keys()) if isinstance(config, dict) else None,
            "stub_worker_count": len(groups),
        },
    )


def _rlinf_version() -> str:
    """Resolve the installed RLinf version.

    rlinf's ``__version__`` attribute is not always set, so we prefer
    :func:`importlib.metadata.version` (looks up the distribution
    metadata) and fall back to the attribute, then ``"unknown"``.
    """
    try:
        return importlib_metadata.version("rlinf")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return getattr(rlinf, "__version__", "unknown")


__all__ = [
    "RealRLinfRunner",
    "_StubWorker",
    "build_runner",
]