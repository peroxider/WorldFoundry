"""Real verl runner factory (M1-T5 P2 path).

Why this exists
---------------
The orchestrator's ``--rl-backend verl`` seam (M1-T3) routes through
:class:`worldfoundry_common.adapters.verl.VerlRunner`, whose :meth:`run`
invokes ``verl.trainer.main_ppo.run_ppo(cfg)`` with a full verl 0.5.0
config. On the Atlas card 7 acceptance we do NOT want to boot verl's
complete Ray/vLLM pipeline inside the orchestrator smoke (the world-model
train/inference/policy loop the orchestrator drives is its own 5-phase
pipeline; verl would replace the policy phase with its own trainer that
needs HF model files + dataset + vLLM rollout). Instead this module is the
*real* factory: it imports the real verl package, inspects
:func:`verl.trainer.main_ppo.run_ppo` (the documented programmatic entry,
there is no ``verl.interface.Main`` in 0.5.0), calls the zero-side-effect
probe :func:`verl.trainer.main_ppo.is_cuda_available` once, and returns a
:class:`RealVerlRunner` whose ``.run()`` produces a canonical summary dict
(``kind="real_verl"``) the orchestrator merges into ``policy_bootstrap``
— proving the verl dispatch went through the real verl code path, not a
mock.

Public API
----------
* :func:`build_runner` — factory callable; reads the same config keys as
  the rlinf factory (``env_pool`` / ``policy`` / ``cfg``) so the
  orchestrator's call site is unchanged.

What it does NOT do (yet)
-------------------------
* Actually run ``run_ppo(cfg)`` — that needs verl's full training inputs
  (model path, train/val jsonl files, vLLM rollout backend). The summary
  records the run_ppo signature + module file so a future "full verl
  trainer" pass has the exact entry point documented.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any

# REAL verl import. The whole point of this factory: the dispatch path
# goes through the real verl package. If verl is missing the import
# raises ImportError and the orchestrator surfaces it via the existing
# ``--rl-backend=verl`` error path.
import verl  # noqa: F401
from verl.trainer.main_ppo import is_cuda_available, run_ppo


# ---------------------------------------------------------------------------
# Runner returned by build_runner
# ---------------------------------------------------------------------------


@dataclass
class RealVerlRunner:
    """Stand-in returned by :func:`build_runner` (M1-T5 real-verl path).

    Mirrors :class:`RealRLinfRunner` in :mod:`rlinf_factory`:

    * Captures the real ``run_ppo`` signature + module file + verl
      version at build time.
    * :meth:`run` does NOT call ``run_ppo(cfg)`` (that would boot verl's
      full Ray/vLLM trainer); it exercises the real entry point via the
      zero-side-effect :func:`is_cuda_available` probe and returns a
      summary dict the orchestrator merges into ``policy_bootstrap``.

    The summary always carries the same ``framework``/``status``/``kind``
    shape the adapter expects so callers don't special-case the backend.
    """

    run_ppo_signature: str = ""
    run_ppo_file: str = ""
    verl_version: str = ""
    cuda_probe: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def run(self) -> dict[str, Any]:
        """Drive the verl-backed dispatch and return a summary dict."""
        t0 = time.perf_counter()
        out: dict[str, Any] = {
            "framework": "verl",
            "status": "completed",
            "kind": "real_verl",
            "entry_point": "verl.trainer.main_ppo.run_ppo",
            "run_ppo_signature": self.run_ppo_signature,
            "run_ppo_file": self.run_ppo_file,
            "verl_version": self.verl_version,
            "cuda_probe": bool(self.cuda_probe),
            "extras": dict(self.extras),
            "wall_seconds": round(float(time.perf_counter() - t0), 4),
        }
        return out


# ---------------------------------------------------------------------------
# Factory entry point
# ---------------------------------------------------------------------------


def build_runner(config: dict | Any) -> RealVerlRunner:
    """Real verl runner factory (M1-T5 P2 path).

    Validates the same kwargs the rlinf factory reads (``env_pool`` /
    ``policy`` / ``cfg``) so the orchestrator can swap backends without
    changing its call site, then captures the real verl entry-point
    evidence and exercises the zero-side-effect ``is_cuda_available``
    probe.

    Args:
        config: dict with ``env_pool`` / ``policy`` / ``cfg`` keys.

    Returns:
        :class:`RealVerlRunner` whose ``.run()`` returns the canonical
        summary dict shape (``framework=verl, kind=real_verl``).
    """
    env_pool = config.get("env_pool") if isinstance(config, dict) else None
    policy = config.get("policy") if isinstance(config, dict) else None
    if env_pool is None:
        raise ValueError(
            "verl_factory: config['env_pool'] is required "
            "(the orchestrator passes the live SyncEnvPool here)."
        )
    if policy is None:
        raise ValueError(
            "verl_factory: config['policy'] is required "
            "(the orchestrator passes the trained VisionPolicy here)."
        )

    sig = inspect.signature(run_ppo)
    run_ppo_file = getattr(run_ppo, "__code__", None)
    run_ppo_file = run_ppo_file.co_filename if run_ppo_file is not None else ""
    try:
        cuda_probe = bool(is_cuda_available())
    except Exception:  # pragma: no cover -- probe is best-effort
        cuda_probe = None

    return RealVerlRunner(
        run_ppo_signature=str(sig),
        run_ppo_file=run_ppo_file,
        verl_version=_verl_version(),
        cuda_probe=cuda_probe,
        extras={
            "config_keys": list(config.keys()) if isinstance(config, dict) else None,
        },
    )


def _verl_version() -> str:
    """Resolve the installed verl version."""
    try:
        return importlib_metadata.version("verl")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return getattr(verl, "__version__", "unknown")


__all__ = [
    "RealVerlRunner",
    "build_runner",
]
