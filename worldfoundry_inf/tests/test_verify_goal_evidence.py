"""Regression guard for ``scripts/verify_goal_evidence.py``.

The script is the canonical "is the goal achieved RIGHT NOW?" probe.
Running it as a subprocess and asserting exit code 0 + the structured
output guarantees the script remains useful as a CI gate.

The verifier exercises the orchestrator under
``--rl-backend=rlinf --inference-backend=vllm_omni`` and asserts the
three clauses of the persistent verbatim goal. See
``scripts/verify_goal_evidence.py`` docstring for the verbatim goal
text + clause definitions.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_goal_evidence.py"


def test_verify_goal_evidence_script_exits_zero_with_three_passes(tmp_path: Path) -> None:
    """The canonical goal-verification script reports 3/3 PASSES + exits 0.

    Subprocesses ``scripts/verify_goal_evidence.py`` and asserts:
    - returncode == 0
    - stdout contains three "[PASS]" lines (one per goal clause)
    - stdout contains the "VERIFIED: 3/3 clauses satisfied" summary

    If this test fails after a refactor, either:
    - the orchestrator's summary schema changed (update _check_* in the script)
    - the orchestrator's dispatch path regressed (fix the underlying wiring)
    - the goal clauses themselves moved (re-derive from the verbatim goal)
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