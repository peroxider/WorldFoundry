"""Simulation-environment axis — which simulator backs the RL loop.

Backends:
* ``mock``         — in-process synthetic env (no external simulator dep).
* ``mujoco``       — MuJoCo via ``FrankaKitchenEnv`` (alias ``franka_kitchen``).
* ``droid``        — DROID env (``DroidEnv``).
* ``manipulator``  — SAPIEN manipulator env (``ManipulatorEnv``).

Every backend's ``build(**kwargs)`` returns an **env factory** (a no-arg
callable producing a single :class:`~worldfoundry_rl.env_pool.Env`), matching
the factory contract that :class:`~worldfoundry_rl.env_pool.SyncEnvPool`
consumes. Concrete envs stay decoupled from the pool machinery via the
structural ``Env`` Protocol in ``env_pool``.

The module also exposes :func:`resolve_env_spec`, which maps a friendly
alias (``"mock"`` / ``"mujoco"`` / ``"droid"`` / ``"manipulator"``) to the
picklable ``"package.module:ClassName"`` spec that the env pool machinery
consumes — so YAML ``env.type`` can be either a full spec or a short alias.
"""
from __future__ import annotations

from typing import Any, Callable

from worldfoundry_common.backends.registry import Backend, BackendRegistry

registry = BackendRegistry("sim_env")

# Friendly alias -> picklable env spec, the single source of truth used by
# both the registered backends and :func:`resolve_env_spec`.
_ENV_SPECS: dict[str, str] = {
    "mock": "worldfoundry_rl.env_mock:MockEnv",
    "mujoco": "worldfoundry_rl.env_franka_kitchen:FrankaKitchenEnv",
    "franka_kitchen": "worldfoundry_rl.env_franka_kitchen:FrankaKitchenEnv",
    "droid": "worldfoundry_rl.env_droid:DroidEnv",
    "manipulator": "worldfoundry_rl.env_manipulator:ManipulatorEnv",
}


def resolve_env_spec(env_type: str) -> str:
    """Map a friendly alias to a picklable ``"package.module:ClassName"`` spec.

    A value that already contains ``":"`` is returned unchanged (it is
    assumed to be a full spec). Unknown aliases raise ``KeyError`` listing
    the available aliases, so a typo in a YAML ``env.type`` is surfaced
    eagerly instead of failing later at import time.
    """
    if ":" in env_type:
        return env_type
    if env_type in _ENV_SPECS:
        return _ENV_SPECS[env_type]
    raise KeyError(
        f"unknown sim_env alias {env_type!r}; available: {sorted(_ENV_SPECS)}"
    )


def _build_from_spec(spec: str, kwargs: dict[str, Any]) -> Callable[[], Any]:
    from worldfoundry_rl.env_pool import make_single_env

    return lambda: make_single_env(spec, dict(kwargs))


def _build_mock(**kwargs: Any) -> Callable[[], Any]:
    return _build_from_spec(_ENV_SPECS["mock"], kwargs)


def _mujoco_available() -> bool:
    try:
        import dm_control  # noqa: F401
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


def _build_mujoco(**kwargs: Any) -> Callable[[], Any]:
    return _build_from_spec(_ENV_SPECS["mujoco"], kwargs)


def _build_droid(**kwargs: Any) -> Callable[[], Any]:
    return _build_from_spec(_ENV_SPECS["droid"], kwargs)


def _build_manipulator(**kwargs: Any) -> Callable[[], Any]:
    return _build_from_spec(_ENV_SPECS["manipulator"], kwargs)


_MUJOCO_HINT = (
    "pip install dm_control mujoco  (transitive deps of the CPU loop, "
    "bundled in the NPU base image)."
)

registry.register(
    "mock",
    Backend(
        "mock",
        "In-process synthetic env (no simulator dep).",
        build=_build_mock,
    ),
)
registry.register(
    "mujoco",
    Backend(
        "mujoco",
        "MuJoCo manipulation env (FrankaKitchenEnv).",
        build=_build_mujoco,
        available=_mujoco_available,
        hint=_MUJOCO_HINT,
    ),
)
registry.register(
    "franka_kitchen",
    Backend(
        "franka_kitchen",
        "Alias for the 'mujoco' FrankaKitchenEnv backend.",
        build=_build_mujoco,
        available=_mujoco_available,
        hint=_MUJOCO_HINT,
    ),
)
registry.register(
    "droid",
    Backend(
        "droid",
        "DROID env (DroidEnv).",
        build=_build_droid,
    ),
)
registry.register(
    "manipulator",
    Backend(
        "manipulator",
        "SAPIEN manipulator env (ManipulatorEnv).",
        build=_build_manipulator,
    ),
)

__all__ = ["registry", "resolve_env_spec"]
