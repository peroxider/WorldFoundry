"""Regression guard for ``scripts/verify_goal_evidence.py``.

The script is the canonical "is the goal achieved RIGHT NOW?" probe.
Running it as a subprocess and asserting exit code 0 + the structured
output guarantees the script remains useful as a CI gate.

The verifier exercises the orchestrator under
``--rl-backend=rlinf --inference-backend=vllm_omni`` and asserts the
three clauses of the persistent verbatim goal. See
``scripts/verify_goal_evidence.py`` docstring for the verbatim goal
text + clause definitions.

Per-clause check functions (``_check_training`` / ``_check_inference``
/ ``_check_diffusion``) are imported directly and exercised with
synthetic dicts to lock down the failure-path semantics — if any
clause's predicate regresses (e.g. a typo in the field name) the
in-process test fails immediately without a 10s subprocess.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_goal_evidence.py"


def _load_verify_module():
    """Import ``scripts/verify_goal_evidence.py`` as a module (not via subprocess)."""
    spec = importlib.util.spec_from_file_location("verify_goal_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None, SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _good_training_dict() -> dict:
    """Shape of a successful ``policy_bootstrap`` fingerprint.

    The M1-T4 P1 verifier requires ``rlinf_result.kind == "real_rlinf"``,
    proving the orchestrator reached the policy phase through the *real*
    :class:`rlinf.runners.embodied_runner.EmbodiedRunner`, not the mock
    factory (``kind == "mock"``) or the in-process self_ppo path.
    """
    return {
        "policy_bootstrap": {
            "rlinf_framework": "rlinf",
            "rlinf_status": "completed",
            "rlinf_result": {"kind": "real_rlinf", "n_updates": 1},
        }
    }


def _good_inference_dict() -> dict:
    """Shape of a successful ``inference_engine_dispatched`` fingerprint.

    The M2-T6 P1 verifier requires ``engine_type == "RealVLLMOmniPipeline"``,
    proving the inference axis goes through the real vLLM package and the
    RealVLLMOmniPipeline class, not the ORT stand-in ``VLLMOmniPipeline``
    (which only shares the registry name). The ``kind="real_vllm_omni"``
    parallel field carries the same information in the canonical
    :meth:`RealVLLMOmniPipeline.provider_summary` format for downstream
    dashboards.
    """
    return {
        "inference_backend": "vllm_omni",
        "inference_engine_dispatched": {
            "engine": "vllm_omni",
            "available": True,
            "engine_type": "RealVLLMOmniPipeline",
            "kind": "real_vllm_omni",
            "dispatch_error": None,
        },
    }


def _good_diffusion_dict() -> dict:
    return {
        "inference_engine_dispatched": {
            "inference_engine": "onnxruntime",
            "diffusion_engine": "onnxruntime",
            "providers_resolved": ["CPUExecutionProvider"],
        },
    }


# ---------------------------------------------------------------------------
# Per-clause check functions — failure-path coverage
# ---------------------------------------------------------------------------


def test_check_training_passes_when_rlinf_completed() -> None:
    mod = _load_verify_module()
    summary = _good_training_dict()
    ok, detail = mod._check_training(summary)
    assert ok is True, detail
    assert "rlinf_framework='rlinf'" in detail


def test_check_training_fails_when_framework_is_self_ppo() -> None:
    mod = _load_verify_module()
    summary = _good_training_dict()
    summary["policy_bootstrap"]["rlinf_framework"] = "self_ppo"
    ok, _detail = mod._check_training(summary)
    assert ok is False


def test_check_training_fails_when_status_not_completed() -> None:
    mod = _load_verify_module()
    summary = _good_training_dict()
    summary["policy_bootstrap"]["rlinf_status"] = "error"
    ok, _detail = mod._check_training(summary)
    assert ok is False


def test_check_inference_passes_when_vllm_omni_dispatched() -> None:
    mod = _load_verify_module()
    summary = _good_inference_dict()
    ok, detail = mod._check_inference(summary)
    assert ok is True, detail
    assert "dispatch_error=None" in detail


def test_check_inference_fails_when_backend_is_torch() -> None:
    mod = _load_verify_module()
    summary = _good_inference_dict()
    summary["inference_backend"] = "torch"
    ok, _detail = mod._check_inference(summary)
    assert ok is False


def test_check_inference_fails_when_available_false() -> None:
    mod = _load_verify_module()
    summary = _good_inference_dict()
    summary["inference_engine_dispatched"]["available"] = False
    summary["inference_engine_dispatched"]["dispatch_error"] = (
        "ImportError: vllm-ascend not installed"
    )
    ok, _detail = mod._check_inference(summary)
    assert ok is False


def test_check_inference_fails_when_engine_type_is_ort_standin() -> None:
    """M2-T6 P1 regression guard: ORT stand-in class must NOT pass the gate.

    Mirrors the M1-T4 P1 ``test_check_training_fails_when_framework_is_self_ppo``
    pattern: a stand-in that just shares the CLI flag name must NOT satisfy
    the inference clause. The ORT stand-in class is ``VLLMOmniPipeline``;
    the real path class is ``RealVLLMOmniPipeline``. If the orchestrator ever
    silently falls back to the stand-in (e.g. ``vllm`` import regresses), the
    clause fails immediately.
    """
    mod = _load_verify_module()
    summary = _good_inference_dict()
    summary["inference_engine_dispatched"]["engine_type"] = "VLLMOmniPipeline"
    summary["inference_engine_dispatched"]["kind"] = "vllm_omni"
    ok, _detail = mod._check_inference(summary)
    assert ok is False


def test_check_inference_fails_when_engine_type_missing() -> None:
    """M2-T6 P1 regression guard: orchestrator must record ``engine_type``.

    If the orchestrator's vllm_omni branch ever drops the
    ``engine_type`` capture (regression to pre-M2-T6-P1 fingerprint
    shape), the clause fails immediately. Without this guard the
    verifier would silently pass on a fingerprint that didn't carry
    the real-vs-stand-in distinction.
    """
    mod = _load_verify_module()
    summary = _good_inference_dict()
    del summary["inference_engine_dispatched"]["engine_type"]
    ok, _detail = mod._check_inference(summary)
    assert ok is False


def test_check_diffusion_passes_when_ort_engines_resolved() -> None:
    mod = _load_verify_module()
    summary = _good_diffusion_dict()
    ok, detail = mod._check_diffusion(summary)
    assert ok is True, detail
    assert "providers_resolved=['CPUExecutionProvider']" in detail


def test_check_diffusion_fails_when_pytorch_eager() -> None:
    """If diffusion ever regressed to PyTorch eager, the check must fail."""
    mod = _load_verify_module()
    summary = _good_diffusion_dict()
    summary["inference_engine_dispatched"]["diffusion_engine"] = "pytorch"
    ok, _detail = mod._check_diffusion(summary)
    assert ok is False


def test_check_diffusion_fails_when_providers_empty() -> None:
    mod = _load_verify_module()
    summary = _good_diffusion_dict()
    summary["inference_engine_dispatched"]["providers_resolved"] = []
    ok, _detail = mod._check_diffusion(summary)
    assert ok is False


# ---------------------------------------------------------------------------
# Full happy-path subprocess run (regression guard for the CLI gate)
# ---------------------------------------------------------------------------


def test_verify_goal_evidence_script_exits_zero_with_three_passes(tmp_path: Path) -> None:
    """The canonical goal-verification script reports 3/3 PASSES + exits 0.

    Subprocesses ``scripts/verify_goal_evidence.py`` and asserts:
    - returncode == 0
    - stdout contains three "[PASS]" lines (one per goal clause)
    - stdout contains the "VERIFIED: 3/3 clauses satisfied" summary
    - the structured ``goal_verdict.json`` file is written + has the expected
      shape (used by downstream dashboards / post-mortem tools)

    If this test fails after a refactor, either:
    - the orchestrator's summary schema changed (update _check_* in the script)
    - the orchestrator's dispatch path regressed (fix the underlying wiring)
    - the goal clauses themselves moved (re-derive from the verbatim goal)
    - the verdict file schema changed (update the assertions below)
    """
    if not SCRIPT.is_file():
        pytest.skip(f"verify_goal_evidence.py not present at {SCRIPT}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out-dir", str(tmp_path / "goal_ev")],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"verify_goal_evidence.py exited {result.returncode}.\n"
        f"stdout:\n{result.stdout[:2000]}\n"
        f"stderr:\n{result.stderr[:2000]}"
    )
    # All three clauses pass.
    pass_count = result.stdout.count("[PASS]")
    fail_count = result.stdout.count("[FAIL]")
    assert pass_count == 3, (
        f"Expected 3 [PASS] lines, got {pass_count}.\nstdout:\n{result.stdout}"
    )
    assert fail_count == 0, (
        f"Expected 0 [FAIL] lines, got {fail_count}.\nstdout:\n{result.stdout}"
    )
    # Summary line is present.
    assert "VERIFIED: 3/3 clauses satisfied" in result.stdout, (
        f"Missing VERIFIED summary line.\nstdout:\n{result.stdout}"
    )
    # Verbatim goal text is printed.
    assert "持续执行开发直到世界模型训练基于RLinf执行" in result.stdout, (
        f"Verbatim goal text missing from output.\nstdout:\n{result.stdout}"
    )
    # Structured verdict file written for downstream consumers / dashboards.
    verdict_path = tmp_path / "goal_ev" / "goal_verdict.json"
    assert verdict_path.is_file(), f"Missing {verdict_path}"
    import json as _json

    verdict = _json.loads(verdict_path.read_text())
    assert verdict["ok"] is True, verdict
    assert verdict["n_pass"] == 3, verdict
    assert verdict["n_total"] == 3, verdict
    assert verdict["exit_code"] == 0, verdict
    for clause_name in ("training", "inference", "diffusion"):
        clause = verdict["clauses"][clause_name]
        assert clause["ok"] is True, (clause_name, clause)
        assert isinstance(clause["detail"], str) and clause["detail"], (clause_name, clause)
    assert "持续执行开发直到世界模型训练基于RLinf执行" in verdict["goal"], verdict["goal"]