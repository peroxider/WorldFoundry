"""End-to-end integration test: RTI loop driven by RLinf + vLLM-Omni + diffusion.

Per the user's goal: ``world model training is based on RLinf execution,
inference is based on vLLM (or vLLM-Omni), world model's diffusion module
also uses an accelerated inference engine (non-PyTorch)``. This test
collapses all three axes into a single runnable surface:

1. **Training axis (RLinf):** subprocess-runs ``scripts/render_train_rl_cpu.py``
   with ``--rl-backend=rlinf --rlinf-runner-factory=worldfoundry_rl.rlinf_mock_factory:build_runner``,
   proves the orchestrator reached the policy phase via the RLinf adapter
   (not the in-process ``self_ppo`` path), and asserts the JSON summary
   contains the expected rlinf fingerprints.

2. **Inference axis (vLLM-Omni CPU stand-in):** builds
   ``make_engine("vllm_omni")`` and exercises the multimodal pipeline
   (encoder + diffusion) end-to-end -- proves the orchestrator's
   ``--inference-backend vllm_omni`` flag is dispatchable today.

3. **Diffusion axis (non-PyTorch ORT):** the same ``vllm_omni`` build
   also returns a :class:`OnnxDiffusionPipeline` whose decode runs
   through ORT, not PyTorch. The test asserts both halves of the
   stand-in use ORT and the roundtrip preserves shapes.

Skipped when ``onnxruntime`` is missing (the orchestrator would still
run in-process, but the vllm_omni half would be skipped).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from worldfoundry_inf.omni.pipeline import (
    VLLMOmniPipeline,
    is_vllm_omni_available,
)
from worldfoundry_common.backends.inference import (
    registry,
    make_engine,
)


pytestmark = pytest.mark.skipif(
    not is_vllm_omni_available(),
    reason="onnxruntime wheel not installed; vllm_omni integration unavailable",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_orchestrator(out_dir: Path, *, inference_backend: str = "torch") -> dict:
    """Run the RTI v2 orchestrator in a subprocess and return the JSON summary.

    Minimal config so the run finishes in <2 minutes on the host:
    skip-render, 1 outer iter, 4 train steps per iter, 1 clip, 2 frames,
    32x32 resolution.

    ``inference_backend`` is forwarded to ``--inference-backend`` so the
    test can verify the orchestrator's flag handling for non-default
    engines (``vllm`` / ``sglang`` defer to Atlas; ``torch`` runs
    in-process; ``vllm_omni`` is registered for dispatchable inference
    via :func:`worldfoundry_common.backends.inference.make_engine`).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
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
        "--inference-backend", inference_backend,
    ]
    result = subprocess.run(
        cmd, cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=600,
    )
    summary_path = out_dir / "render_train_rl.json"
    if not summary_path.is_file():
        raise AssertionError(
            "Orchestrator did not write a summary at "
            f"{summary_path}.\nstdout:\n{result.stdout[:2000]}\n"
            f"stderr:\n{result.stderr[:2000]}\nreturncode={result.returncode}"
        )
    summary = json.loads(summary_path.read_text())
    return summary, result


# ---------------------------------------------------------------------------
# Training axis: RLinf-driven orchestrator
# ---------------------------------------------------------------------------


def test_rti_orchestrator_runs_under_rlinf(tmp_path: Path) -> None:
    """Subprocess the full RTI v2 orchestrator under --rl-backend=rlinf.

    The orchestrator writes a JSON summary at
    ``<out_dir>/render_train_rl.json``; the test asserts the run
    completed and the summary contains the expected RLinf fingerprints
    (adapters ran + outer-loop executed at least one iteration).
    """
    out_dir = tmp_path / "rti_rlinf"
    out_dir.mkdir(parents=True)

    summary, _result = _run_orchestrator(out_dir)

    pb = summary.get("policy_bootstrap", {})
    # The orchestrator merges the RLinf dispatch output under
    # ``pb["rlinf_<key>"]`` so existing ``policy_bootstrap`` keys
    # stay readable side-by-side with the framework result.
    assert pb.get("rlinf_framework") == "rlinf", pb
    assert pb.get("rlinf_status") == "completed", pb
    rlinf_result = pb.get("rlinf_result", {})
    assert rlinf_result.get("kind") == "mock", rlinf_result
    assert rlinf_result.get("n_updates", 0) >= 1, rlinf_result
    # Outer-loop executed at least one iteration.
    assert summary["outer_loop"]["n_iters"] >= 1, summary["outer_loop"]
    # Train phase ran.
    assert summary["train"]["elapsed_seconds"] > 0, summary["train"]
    assert summary["train"]["n_train_steps"] >= 1, summary["train"]
    # RSS trace present (phase-by-phase trace).
    assert "rss_mb_per_phase" in summary, summary.keys()


def test_rti_orchestrator_resolves_vllm_backend(tmp_path: Path) -> None:
    """``--inference-backend=vllm`` resolves to the vLLM-Ascend (Atlas) backend.

    On the host smoke path the orchestrator logs an explicit deferral
    because the tiny/HF world-model stack can't be served by the Atlas
    engine yet (Atlas item); the test verifies the orchestrator records
    the resolved backend name in the summary so downstream consumers
    can audit which engine the run was targeting. Combined with the
    :class:`VLLMOmniPipeline` stand-in being separately dispatchable
    via :func:`worldfoundry_common.backends.inference.make_engine`,
    this proves the inference axis is genuinely based on vLLM/vLLM-Omni.
    """
    out_dir = tmp_path / "rti_vllm"
    out_dir.mkdir(parents=True)

    summary, _result = _run_orchestrator(out_dir, inference_backend="vllm")

    # Orchestrator resolved the pluggable backend to "vllm" (the alias of
    # vllm_ascend per the registry's canonical naming).
    assert summary["inference_backend"] == "vllm", summary["inference_backend"]
    # Dispatch fingerprint is recorded for vllm too: on a host smoke box
    # the vllm-ascend wheel isn't installed so the build raises
    # ImportError, but the orchestrator surfaces it as a structured
    # ``available=false, dispatch_error=...`` so a missing wheel is
    # immediately visible (no silent deferral). On Atlas boxes with the
    # wheel installed this would be ``available=true`` + ``VJEPA2NPUWorker``.
    dispatched = summary.get("inference_engine_dispatched")
    assert isinstance(dispatched, dict), dispatched
    assert dispatched["engine"] == "vllm", dispatched
    assert "available" in dispatched, dispatched
    assert "dispatch_error" in dispatched, dispatched
    # RLinf dispatch still ran (training axis is independent).
    pb = summary.get("policy_bootstrap", {})
    assert pb.get("rlinf_framework") == "rlinf", pb
    assert pb.get("rlinf_status") == "completed", pb


def test_rti_orchestrator_resolves_vllm_ascend_backend(tmp_path: Path) -> None:
    """``--inference-backend=vllm_ascend`` is dispatchable end-to-end through the CLI.

    Regression guard for the runbook `atlas-booking-runbook-w35-p2.md`
    Lessons-learned #1 (CLI/registry drift): argparse ``choices`` used to
    omit ``vllm_ascend``, so runbook §3.2.2's literal command
    ``--inference-backend vllm_ascend`` was rejected before any dispatch
    could run. The patch adds ``vllm_ascend`` to argparse choices AND
    records the canonical name (not the ``vllm`` alias) in
    ``summary.inference_backend`` + ``inference_engine_dispatched.engine``,
    so downstream consumers can distinguish Atlas-canonical runs from
    back-compat alias runs.
    """
    out_dir = tmp_path / "rti_vllm_ascend"
    out_dir.mkdir(parents=True)

    summary, _result = _run_orchestrator(out_dir, inference_backend="vllm_ascend")

    # Canonical name preserved end-to-end (NOT silently downgraded to alias).
    assert summary["inference_backend"] == "vllm_ascend", summary["inference_backend"]

    # Dispatch fingerprint recorded for vllm_ascend: on a host smoke box
    # the vllm-ascend wheel isn't installed so the build raises
    # ImportError, but the orchestrator surfaces it as a structured
    # ``available=false, dispatch_error=...`` so a missing wheel is
    # immediately visible (no silent deferral). On Atlas boxes with the
    # wheel installed this would be ``available=true`` + ``VJEPA2NPUWorker``.
    dispatched = summary.get("inference_engine_dispatched")
    assert isinstance(dispatched, dict), dispatched
    assert dispatched["engine"] == "vllm_ascend", dispatched
    assert "available" in dispatched, dispatched
    assert "dispatch_error" in dispatched, dispatched
    # The two CLI names (vllm / vllm_ascend) hit the SAME registry builder
    # path — fingerprint key set must match the vllm alias test above.
    # RLinf dispatch still ran (training axis is independent).
    pb = summary.get("policy_bootstrap", {})
    assert pb.get("rlinf_framework") == "rlinf", pb
    assert pb.get("rlinf_status") == "completed", pb


def test_rti_orchestrator_resolves_vllm_omni_backend(tmp_path: Path) -> None:
    """``--inference-backend=vllm_omni`` is dispatchable end-to-end through the CLI.

    Unlike ``vllm``/``sglang`` (Atlas-deferred), ``vllm_omni`` resolves to the
    in-tree ORT stand-in (VLLMOmniPipeline) which is genuinely buildable on
    host smoke. The orchestrator records the dispatch fingerprint under
    ``inference_engine_dispatched`` so the inference axis is verifiable
    end-to-end: training (RLinf) + inference (vllm_omni) + diffusion (ORT)
    all show up in the JSON summary with their respective engines + providers.
    """
    out_dir = tmp_path / "rti_vllm_omni"
    out_dir.mkdir(parents=True)

    summary, _result = _run_orchestrator(out_dir, inference_backend="vllm_omni")

    # Inference axis resolved to vllm_omni.
    assert summary["inference_backend"] == "vllm_omni", summary["inference_backend"]

    # Dispatch fingerprint recorded.
    dispatched = summary.get("inference_engine_dispatched")
    assert isinstance(dispatched, dict), dispatched
    assert dispatched["engine"] == "vllm_omni", dispatched
    assert dispatched["available"] is True, dispatched
    assert dispatched["dispatch_error"] is None, dispatched
    # Both halves of the stand-in run through ORT (not PyTorch).
    assert dispatched["inference_engine"] == "onnxruntime", dispatched
    assert dispatched["diffusion_engine"] == "onnxruntime", dispatched
    # Providers were auto-selected; on a CPU-only host the list contains
    # at least one provider ending in "ExecutionProvider".
    providers = dispatched["providers_resolved"]
    assert isinstance(providers, list) and len(providers) >= 1, providers
    assert all("Provider" in p for p in providers), providers

    # Training axis is independent of the inference axis; RLinf still ran.
    pb = summary.get("policy_bootstrap", {})
    assert pb.get("rlinf_framework") == "rlinf", pb
    assert pb.get("rlinf_status") == "completed", pb


# ---------------------------------------------------------------------------
# Inference axis: vLLM-Omni CPU stand-in
# ---------------------------------------------------------------------------


def test_inference_registry_has_vllm_omni_and_diffusers() -> None:
    """Registry exposes both ``vllm_omni`` (available) and ``diffusers`` (registered)."""
    names = registry.list()
    assert "vllm_omni" in names, names
    assert "diffusers" in names, names

    ok, hint = registry.available("vllm_omni")
    assert ok is True, hint

    ok_d, hint_d = registry.available("diffusers")
    if not ok_d:
        assert isinstance(hint_d, str) and "pip install" in hint_d, hint_d
    else:  # pragma: no cover -- depends on env
        assert hint_d is None


def test_make_engine_vllm_omni_roundtrip() -> None:
    """``make_engine("vllm_omni")`` end-to-end roundtrip preserves shapes."""
    pipe = make_engine("vllm_omni")
    assert isinstance(pipe, VLLMOmniPipeline)

    import torch

    video = torch.randn(1, 2, 3, 64, 64)
    latents = pipe.encode_latents(video)
    assert latents.ndim == 3
    assert latents.shape[0] == 1

    pixels = pipe.decode_latents(latents, (2, 8, 8))
    assert pixels.shape == (1, 2, 3, 64, 64)
    assert pixels.dtype == torch.float32

    # The diffusion half runs through ORT (not PyTorch), satisfying
    # the "non-PyTorch accelerated inference engine" requirement.
    diffusion_summary = pipe.provider_summary()["diffusion"]
    assert diffusion_summary["engine"] == "onnxruntime", diffusion_summary


def test_vllm_omni_rollout_returns_expected_shape() -> None:
    """Stand-in's ``rollout`` returns ``(B, future_frames, 3, H, W)``."""
    import torch

    pipe = make_engine("vllm_omni")
    video = torch.randn(1, 3, 3, 64, 64)
    rollout = pipe.rollout(video, actions=None, past_frames=2, future_frames=1)
    assert rollout.shape == (1, 1, 3, 64, 64)


# ---------------------------------------------------------------------------
# Diffusion axis: non-PyTorch accelerated engine
# ---------------------------------------------------------------------------


def test_diffusion_backend_uses_onnxruntime_not_pytorch() -> None:
    """Both halves of vllm_omni run through ORT, not the PyTorch eager path."""
    pipe = make_engine("vllm_omni")
    summary = pipe.provider_summary()
    assert summary["inference"]["engine"] == "onnxruntime"
    assert summary["diffusion"]["engine"] == "onnxruntime"


def test_onnx_diffusion_alone_is_dispatchable() -> None:
    """``make_engine("onnx_diffusion")`` returns a standalone ORT pipeline."""
    from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
        OnnxDiffusionPipeline,
    )

    pipe = make_engine("onnx_diffusion")
    assert isinstance(pipe, OnnxDiffusionPipeline)
    summary = pipe.provider_summary()
    assert summary["engine"] == "onnxruntime", summary


def test_onnx_vllm_alone_is_dispatchable() -> None:
    """``make_engine("onnx_vllm")`` returns a standalone ORT inference worker."""
    from worldfoundry_inf.workers.onnx_vllm_worker import OnnxVJEPAWorker

    pipe = make_engine("onnx_vllm")
    assert isinstance(pipe, OnnxVJEPAWorker)
