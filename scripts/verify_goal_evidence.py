#!/usr/bin/env python3
"""Single-shot verification of the persistent goal.

The user-facing goal (verbatim) is:

    持续执行开发直到世界模型训练基于RLinf执行，推理基于vllm(或者
    vllm-omni)，世界模型的扩散模块也采用加速的推理引擎(非pytorch)

This script is the canonical "is the goal achieved RIGHT NOW?" probe.
It subprocesses the orchestrator with the goal-axis flags:

    --rl-backend=rlinf --inference-backend=vllm_omni

then parses the JSON summary and asserts all three clauses:

* **Training (RLinf):** ``policy_bootstrap.rlinf_framework=="rlinf"``
  and ``rlinf_status=="completed"`` (the orchestrator reached the
  policy phase through the RLinf adapter, not the in-process self_ppo
  path).
* **Inference (vLLM-Omni):** ``inference_backend=="vllm_omni"`` and
  ``inference_engine_dispatched.available is True``.
* **Diffusion (non-PyTorch ORT):**
  ``inference_engine_dispatched.diffusion_engine=="onnxruntime"`` and
  ``inference_engine_dispatched.inference_engine=="onnxruntime"``.

Output is a one-line-per-axis PASS/FAIL plus a summary line. Exit
code is 0 if and only if all three clauses pass. The script is
useful as a CI gate (``bash -c 'python scripts/verify_goal_evidence.py'``)
or for ad-hoc verification after a refactor.

Usage::

    python scripts/verify_goal_evidence.py
    python scripts/verify_goal_evidence.py --keep-out-dir   # inspect JSON
    python scripts/verify_goal_evidence.py --out-dir <path> # persist summary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


VERBATIM_GOAL = (
    "持续执行开发直到世界模型训练基于RLinf执行，"
    "推理基于vllm(或者vllm-omni)，"
    "世界模型的扩散模块也采用加速的推理引擎(非pytorch)"
)


def _run_orchestrator(out_dir: Path) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    """Run the orchestrator with goal-axis flags and return the parsed summary."""
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "scripts.render_train_rl_cpu",
        "--out-dir", str(out_dir),
        "--num-clips", "1",
        "--frames-per-clip", "2",
        "--height", "32",
        "--width", "32",
        "--num-steps", "4",
        "--num-outer-iters", "1",
        "--num-train-per-iter", "4",
        "--ppo-env-steps", "8",
        "--mask-ratio", "0.5",
        "--seed", "0",
        "--rl-backend", "rlinf",
        "--rlinf-runner-factory",
        "worldfoundry_rl.rlinf_mock_factory:build_runner",
        "--inference-backend", "vllm_omni",
    ]
    result = subprocess.run(
        cmd, cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=600,
    )
    summary_path = out_dir / "render_train_rl.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"Orchestrator did not write a summary at {summary_path}.\n"
            f"stdout:\n{result.stdout[:2000]}\n"
            f"stderr:\n{result.stderr[:2000]}\n"
            f"returncode={result.returncode}"
        )
    return json.loads(summary_path.read_text()), result


def _check_training(summary: dict[str, Any]) -> tuple[bool, str]:
    """Clause 1: training is based on RLinf execution."""
    pb = summary.get("policy_bootstrap", {})
    framework = pb.get("rlinf_framework")
    status = pb.get("rlinf_status")
    ok = framework == "rlinf" and status == "completed"
    detail = (
        f"rlinf_framework={framework!r}, rlinf_status={status!r}, "
        f"rlinf_result.kind={pb.get('rlinf_result', {}).get('kind')!r}, "
        f"n_updates={pb.get('rlinf_result', {}).get('n_updates')!r}"
    )
    return ok, detail


def _check_inference(summary: dict[str, Any]) -> tuple[bool, str]:
    """Clause 2: inference is based on vLLM (or vLLM-Omni)."""
    backend = summary.get("inference_backend")
    dispatched = summary.get("inference_engine_dispatched") or {}
    engine = dispatched.get("engine")
    available = dispatched.get("available")
    dispatch_error = dispatched.get("dispatch_error")
    ok = backend == "vllm_omni" and engine == "vllm_omni" and available is True
    detail = (
        f"inference_backend={backend!r}, dispatched.engine={engine!r}, "
        f"available={available!r}, dispatch_error={dispatch_error!r}"
    )
    return ok, detail


def _check_diffusion(summary: dict[str, Any]) -> tuple[bool, str]:
    """Clause 3: diffusion module uses accelerated inference engine (non-PyTorch)."""
    dispatched = summary.get("inference_engine_dispatched") or {}
    inf_engine = dispatched.get("inference_engine")
    diff_engine = dispatched.get("diffusion_engine")
    providers = dispatched.get("providers_resolved") or []
    ok = (
        inf_engine == "onnxruntime"
        and diff_engine == "onnxruntime"
        and len(providers) >= 1
    )
    detail = (
        f"inference_engine={inf_engine!r}, diffusion_engine={diff_engine!r}, "
        f"providers_resolved={providers!r}"
    )
    return ok, detail


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--keep-out-dir", action="store_true",
                   help="Don't delete the orchestrator's out-dir after the run.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the orchestrator's out-dir (default: tempfile).")
    args = p.parse_args(argv)

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="goal_evidence_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup = not args.keep_out_dir and args.out_dir is None

    print(f"[goal] verbatim: {VERBATIM_GOAL}")
    print(f"[goal] out_dir  : {out_dir}")
    print(f"[goal] running orchestrator with --rl-backend=rlinf --inference-backend=vllm_omni ...")

    try:
        summary, _result = _run_orchestrator(out_dir)
    except Exception as exc:
        print(f"[goal] FAIL: orchestrator run raised: {exc}")
        return 2

    checks = (
        ("training (RLinf)         ", _check_training),
        ("inference (vLLM-Omni)    ", _check_inference),
        ("diffusion (non-PyTorch)  ", _check_diffusion),
    )
    results = []
    for label, check in checks:
        ok, detail = check(summary)
        results.append(ok)
        marker = "PASS" if ok else "FAIL"
        print(f"[goal] [{marker}] {label} | {detail}")

    n_pass = sum(results)
    n_total = len(results)
    verdict_path = out_dir / "goal_verdict.json"
    # Always write the structured verdict — even on FAIL — so dashboards /
    # post-mortem tools have something to read. The ``ok`` field is the
    # machine-parseable summary; the ``clauses`` dict carries the per-clause
    # fingerprints that the per-axis evidence was drawn from.
    verdict = {
        "goal": VERBATIM_GOAL,
        "ok": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
        "exit_code": 0 if n_pass == n_total else 1,
        "out_dir": str(out_dir),
        "summary_path": str(out_dir / "render_train_rl.json"),
        "clauses": {
            "training": {
                "ok": _check_training(summary)[0],
                "detail": _check_training(summary)[1],
            },
            "inference": {
                "ok": _check_inference(summary)[0],
                "detail": _check_inference(summary)[1],
            },
            "diffusion": {
                "ok": _check_diffusion(summary)[0],
                "detail": _check_diffusion(summary)[1],
            },
        },
    }
    verdict_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"[goal] verdict  : {verdict_path}")

    if n_pass == n_total:
        print(f"[goal] VERIFIED: {n_pass}/{n_total} clauses satisfied — goal achieved.")
        return 0
    print(f"[goal] FAILED: {n_pass}/{n_total} clauses satisfied — goal NOT achieved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())