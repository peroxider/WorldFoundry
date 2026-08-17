"""Tests for the *real* RLinf runner factory (M1-T4 P1 path).

The factory lives in :mod:`worldfoundry_rl.rlinf_factory` and is
wired through the orchestrator's ``--rl-backend rlinf`` seam (M1-T3)
via the ``--rlinf-runner-factory=worldfoundry_rl.rlinf_factory:build_runner``
flag. Unlike :mod:`worldfoundry_rl.rlinf_mock_factory`, this factory
imports the *real* ``rlinf`` package, inspects
:class:`rlinf.runners.embodied_runner.EmbodiedRunner`, and attempts to
construct it with a minimal ``DictConfig`` + duck-typed stub workers.

Why these tests matter
- They prove the integration plumbing is wired through the real rlinf
  package, not a stand-in. Specifically they assert:

  * ``rlinf`` is actually importable (and resolved to a real version
    string, not ``"unknown"`` or mocked out).
  * The factory's dotted-path import resolves (the CLI's
    ``--rlinf-runner-factory`` flag depends on this).
  * :meth:`RealRLinfRunner.run` returns the canonical summary dict
    shape the orchestrator's RLinf adapter merges into
    ``policy_bootstrap.rlinf_result``.
  * The :class:`EmbodiedRunner` constructor is actually exercised
    (``init_ok=True`` on a box with Ray, or ``init_ok=False`` with an
    informative ``init_error`` on one without). Either path is fine;
    what matters is that the factory tried to construct the real
    class.

Skipped when ``rlinf`` is not importable (e.g. on a host box that
hasn't run ``uv pip install rlinf`` against the rlinf venv yet).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

try:
    import rlinf  # noqa: F401  -- presence check for skipif
    _RLINF_AVAILABLE = True
except ImportError:
    _RLINF_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _RLINF_AVAILABLE,
    reason="rlinf not importable; install via 'uv pip install rlinf' (the M1-T4 P1 "
           "real-factory path requires the rlinf package on PYTHONPATH)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _Cfg:
    """Stand-in for an RLinf ``DictConfig`` — passed through unchanged."""


def _make_pool_and_policy(
    num_envs: int = 2,
    hidden_dim: int = 32,
    seed: int = 0,
) -> tuple[Any, Any]:
    """Construct a tiny env pool + matching :class:`VisionPolicy`.

    Uses :class:`worldfoundry_rl.env_mock.MockEnv` (no dm_control
    dependency) so the test runs on a host box without MuJoCo.
    """
    from worldfoundry_rl.env_mock import MockEnv
    from worldfoundry_rl.env_pool import SyncEnvPool
    from worldfoundry_rl.vision_policy import VisionPolicy

    pool = SyncEnvPool(
        [
            lambda i=i: MockEnv(obs_dim=8, action_dim=4, seed=seed + i)
            for i in range(num_envs)
        ]
    )
    obs = pool.reset(seed=seed)
    sample = obs[0]
    obs_dim = int(np.asarray(sample).ravel().size)
    action_dim = int(pool.single_action_space.shape[0])
    policy = VisionPolicy.from_state_dim(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
    )
    return pool, policy


# Session-scoped so the (heavy) EmbodiedRunner constructor — which spins
# up a local Ray cluster on a host box without one — runs only once per
# pytest invocation. Subsequent tests share the same ``RealRLinfRunner``
# instance.
@pytest.fixture(scope="session")
def real_runner() -> Any:
    pool, policy = _make_pool_and_policy()
    try:
        from worldfoundry_rl.rlinf_factory import build_runner
        runner = build_runner({"env_pool": pool, "policy": policy, "cfg": _Cfg()})
    finally:
        pool.close()
    return runner


# ---------------------------------------------------------------------------
# Import surface — dotted path resolves, build_runner is callable
# ---------------------------------------------------------------------------


def test_dotted_path_resolves_to_real_factory() -> None:
    """``--rlinf-runner-factory worldfoundry_rl.rlinf_factory:build_runner`` resolves."""
    module = importlib.import_module("worldfoundry_rl.rlinf_factory")
    attr = getattr(module, "build_runner")
    assert callable(attr), type(attr)


def test_module_exposes_real_rlinf_class() -> None:
    """The module imports the real :class:`EmbodiedRunner`, not a stub."""
    from worldfoundry_rl import rlinf_factory as mod
    from rlinf.runners.embodied_runner import EmbodiedRunner

    # The module should re-export ``EmbodiedRunner`` indirectly via its
    # ``embodied_runner_class`` attribute path; here we just check the
    # class itself was imported from the real rlinf module.
    assert mod.EmbodiedRunner is EmbodiedRunner
    # And the class file should be inside the rlinf package on disk.
    assert EmbodiedRunner.__module__.startswith("rlinf."), EmbodiedRunner.__module__


# ---------------------------------------------------------------------------
# rlinf_version + class fingerprint (proves rlinf is installed)
# ---------------------------------------------------------------------------


def test_real_rlinf_version_resolves_to_nonzero_string(real_runner: Any) -> None:
    """``rlinf_version`` is a non-empty, non-``"unknown"`` string.

    This proves the package is actually installed (not stubbed) and
    that :func:`importlib.metadata.version` can resolve it. We don't
    pin a specific version because 0.3.x is what the host venv has and
    0.4.x might land later; we only require "real, not unknown".
    """
    v = real_runner.rlinf_version
    assert isinstance(v, str) and v, v
    assert v != "unknown", v


def test_real_rlinf_file_points_at_installed_package(real_runner: Any) -> None:
    """``rlinf_file`` is a real on-disk path under a site-packages dir."""
    path = real_runner.rlinf_file
    assert isinstance(path, str) and path, path
    assert "site-packages" in path or "dist-packages" in path, path


def test_embodied_runner_class_is_canonical_dotted_path(real_runner: Any) -> None:
    """The reported class path is the canonical rlinf dotted name."""
    assert (
        real_runner.embodied_runner_class
        == "rlinf.runners.embodied_runner.EmbodiedRunner"
    )


# ---------------------------------------------------------------------------
# EmbodiedRunner constructor signature + methods (proves we inspected the real class)
# ---------------------------------------------------------------------------


def test_embodied_runner_init_signature_has_required_kwargs(real_runner: Any) -> None:
    """The captured signature lists cfg + actor + rollout + env + reward."""
    sig = real_runner.embodied_runner_init_signature
    for kw in ("cfg:", "actor:", "rollout:", "env:", "reward:"):
        assert kw in sig, f"expected {kw!r} in signature, got {sig!r}"


def test_embodied_runner_methods_list_is_nonempty(real_runner: Any) -> None:
    """The methods list contains the training-loop entry points we expect."""
    methods = set(real_runner.embodied_runner_methods)
    # ``set_max_steps`` is the zero-side-effect method we exercise in
    # :meth:`RealRLinfRunner.run` to prove the constructed runner is
    # alive; the others are documented public methods on the class.
    expected = {"run", "epoch", "evaluate", "init_workers", "set_max_steps"}
    missing = expected - methods
    assert not missing, f"missing methods: {missing} (got {sorted(methods)[:20]}...)"


# ---------------------------------------------------------------------------
# Duck-typed stub worker groups (actor/rollout/env/reward)
# ---------------------------------------------------------------------------


def test_stub_worker_groups_has_all_four(real_runner: Any) -> None:
    """Four groups: actor, rollout, env, reward."""
    names = set(real_runner.groups.keys())
    assert names == {"actor", "rollout", "env", "reward"}, names


def test_stub_worker_groups_have_dedicated_names(real_runner: Any) -> None:
    """Every stub carries its own ``name`` attribute matching the group key."""
    from worldfoundry_rl.rlinf_factory import _StubWorker

    for key, group in real_runner.groups.items():
        assert isinstance(group, _StubWorker), (key, type(group))
        assert group.name == key, (key, group.name)
        assert group.kind == "stub", (key, group.kind)


# ---------------------------------------------------------------------------
# run() summary dict — canonical contract for the adapter
# ---------------------------------------------------------------------------


def test_run_returns_canonical_summary_shape(real_runner: Any) -> None:
    """``.run()`` returns ``{framework, status, kind}`` matching the adapter contract."""
    out = real_runner.run()
    assert out["framework"] == "rlinf", out["framework"]
    assert out["status"] == "completed", out["status"]
    assert out["kind"] == "real_rlinf", out["kind"]


def test_run_summary_carries_init_outcome(real_runner: Any) -> None:
    """``init_ok`` is a bool and ``init_error`` is str | None.

    On a host box with Ray installed, ``init_ok`` is True (the real
    constructor succeeds after auto-starting a local Ray cluster);
    on one without Ray, ``init_ok`` is False and ``init_error``
    carries the actionable failure message. Both are acceptable;
    what matters is that the factory tried to construct the real class.
    """
    out = real_runner.run()
    assert isinstance(out["init_ok"], bool), out["init_ok"]
    if out["init_ok"]:
        assert out["init_error"] is None, out["init_error"]
    else:
        assert isinstance(out["init_error"], str) and out["init_error"], out["init_error"]


def test_run_summary_groups_are_serializable(real_runner: Any) -> None:
    """``groups`` is a JSON-safe dict (no dataclass leaking through)."""
    import json

    out = real_runner.run()
    # Must round-trip through json.dumps — catches stray dataclass / Ray-actor leakage.
    json.dumps(out["groups"])


def test_run_summary_wall_seconds_is_nonnegative(real_runner: Any) -> None:
    """``wall_seconds`` is a non-negative float."""
    out = real_runner.run()
    assert isinstance(out["wall_seconds"], (int, float)), out["wall_seconds"]
    assert out["wall_seconds"] >= 0.0, out["wall_seconds"]


def test_run_exercises_constructed_runner_when_init_ok(real_runner: Any) -> None:
    """When the constructor succeeded, ``.run()`` calls ``set_max_steps()``.

    This proves the factory wires the constructed runner through the
    summary dict instead of just reporting static fingerprint info.
    On a host without Ray, ``init_ok=False`` and this test is a no-op.
    """
    out = real_runner.run()
    if out["init_ok"]:
        assert out["set_max_steps_ok"] is True, out
        assert isinstance(out.get("max_steps"), int), out
        assert out["max_steps"] >= 1, out["max_steps"]


# ---------------------------------------------------------------------------
# Build-time validation — kwarg surface
# ---------------------------------------------------------------------------


def test_build_runner_requires_env_pool() -> None:
    """Missing ``env_pool`` raises :class:`ValueError` with a clear message."""
    from worldfoundry_rl.rlinf_factory import build_runner

    with pytest.raises(ValueError, match="env_pool"):
        build_runner({"policy": object(), "cfg": _Cfg()})


def test_build_runner_requires_policy() -> None:
    """Missing ``policy`` raises :class:`ValueError` with a clear message."""
    from worldfoundry_rl.rlinf_factory import build_runner

    pool, _ = _make_pool_and_policy()
    try:
        with pytest.raises(ValueError, match="policy"):
            build_runner({"env_pool": pool, "cfg": _Cfg()})
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Integration with the orchestrator adapter (M1-T3 contract)
# ---------------------------------------------------------------------------


def test_orchestrator_rlinf_runner_dispatches_to_real_factory() -> None:
    """The M1-T3 adapter wires the real factory through :class:`RLinfRunner`.

    Mirrors :func:`test_orchestrator_rlinf_runner_dispatches_to_mock_factory`
    in :mod:`test_rlinf_mock_factory` but goes through
    :mod:`worldfoundry_rl.rlinf_factory` instead.
    """
    from worldfoundry_common.adapters.rlinf import RLinfRunner
    from worldfoundry_rl.rlinf_factory import build_runner

    pool, policy = _make_pool_and_policy()
    try:
        runner = RLinfRunner(runner_factory=build_runner)
        result = runner.run(
            {"env_pool": pool, "policy": policy, "cfg": _Cfg()}
        )
    finally:
        pool.close()
    assert result["framework"] == "rlinf"
    assert result["status"] == "completed"
    # The adapter copies the inner factory's ``kind`` field if present.
    assert result["kind"] == "real_rlinf"
    # Inner result must carry the rlinf_version fingerprint.
    assert result["result"]["rlinf_version"], result["result"]["rlinf_version"]