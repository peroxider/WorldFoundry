"""Surface-parity tests for VJEPA2NPUWorker vs VJEPA2Worker (W27 CPU prep + M2-T1..T3 closure).

The Atlas NPU worker :class:`VJEPA2NPUWorker` and the CPU worker
:class:`VJEPA2Worker` share the public ``predict_*`` contract: a request
routed to either worker must accept the same parameter names + ordering
+ return type, so the API endpoint layer can dispatch uniformly.

Post-W27 + M2-T1/T2/T3 (M2: vLLM inference closed-loop, see
``docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md`` §2): all
five ``predict_*`` methods now have **CPU fallback paths** that mirror
:mod:`worldfoundry_inf.workers.vjepa2_worker` byte-for-byte, so the API
endpoint dispatches the same way on both workers and the CPU ↔ NPU
parity contract can be exercised without Atlas hardware. This module
asserts that:

1. The NPU worker's ``predict_*`` methods have signatures **identical**
   to the CPU worker's (``inspect.signature`` cross-check). A future
   Atlas implementer who renames a parameter, reorders an argument, or
   drops the return annotation breaks this contract test loudly.
2. Calling the temporal / action methods **without** the corresponding
   ckpt (``predictor_ckpt`` or ``action_predictor_ckpt``) raises a
   :class:`RuntimeError` whose message names the missing ckpt — so a
   caller can distinguish "not configured for this mode" from "actual
   bug" without poking at private state. ``predict_masked_batch`` has
   no ckpt requirement and must run on the initialized model.
3. ``predict_action_conditioned_batch`` is present on the NPU worker
   (was missing before this commit; parity closure for W33/W25 batcher
   surface).

No NPU / vllm_ascend / msModelSlim required.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

import pytest
import torch
import transformers

from worldfoundry_inf.workers.vjepa2_worker import VJEPA2Worker
from worldfoundry_inf.workers.vllm_ascend_worker import VJEPA2NPUWorker


# ---------------------------------------------------------------------------
# Methods under test. ``predict`` / ``encode`` / ``plan`` are shared by both
# workers and have non-stub NPU implementations; only the five stubs plus
# the formerly-missing batched action-conditioned method are parity-gated.
# ---------------------------------------------------------------------------

_PARITY_METHODS = (
    "predict_masked_batch",
    "predict_temporal_batch",
    "predict_temporal",
    "predict_action_conditioned",
    "predict_action_conditioned_batch",
)


# ---------------------------------------------------------------------------
# 1. Signature parity
# ---------------------------------------------------------------------------


def _signature_dict(method):
    """Return a normalized signature view for comparison.

    Parameter ``kind`` is included so positional-only vs keyword-only
    mismatches surface (a future NPU implementer who flips a positional
    arg to keyword-only will fail this check).
    """
    sig = inspect.signature(method)
    params = {
        name: (p.kind, p.default, p.annotation)
        for name, p in sig.parameters.items()
    }
    return {
        "params": params,
        "return": sig.return_annotation,
    }


@pytest.mark.parametrize("method_name", _PARITY_METHODS)
def test_npu_predict_method_signature_matches_cpu(method_name):
    """Method exists on both workers and has identical parameter + return shape."""
    npu_method = getattr(VJEPA2NPUWorker, method_name)
    cpu_method = getattr(VJEPA2Worker, method_name)
    npu_sig = _signature_dict(npu_method)
    cpu_sig = _signature_dict(cpu_method)
    assert npu_sig == cpu_sig, (
        f"NPU worker signature for {method_name!r} drifted from CPU worker: "
        f"\n  NPU: {npu_sig}\n  CPU: {cpu_sig}"
    )


@pytest.mark.parametrize("method_name", _PARITY_METHODS)
def test_npu_predict_method_param_order_matches_cpu(method_name):
    """Parameter positional order must match (dispatch uses positional args)."""
    npu_params = list(inspect.signature(getattr(VJEPA2NPUWorker, method_name)).parameters)
    cpu_params = list(inspect.signature(getattr(VJEPA2Worker, method_name)).parameters)
    assert npu_params == cpu_params, (
        f"NPU {method_name!r} param order {npu_params} != CPU {cpu_params}"
    )


def test_npu_worker_exposes_all_parity_methods():
    """Sanity: every method in the parity table actually exists on the NPU class."""
    missing = [m for m in _PARITY_METHODS if not hasattr(VJEPA2NPUWorker, m)]
    assert not missing, f"NPU worker missing methods: {missing}"


# ---------------------------------------------------------------------------
# 2. Without the matching ckpt, the temporal/action paths raise a clear
# RuntimeError so callers can tell "not configured" from "actual bug".
# predict_masked_batch has no ckpt requirement; we exercise it on the
# initialized model.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name",
    [
        "predict_temporal_batch",
        "predict_temporal",
        "predict_action_conditioned",
        "predict_action_conditioned_batch",
    ],
)
def test_npu_predict_raises_runtime_error_when_ckpt_missing(method_name):
    """Calling a temporal/action method without predictor_ckpt /
    action_predictor_ckpt must raise with a message naming the missing ckpt.

    Construct a stub instance via ``__new__`` to bypass ``__init__`` (which
    requires a real VJEPA2 model path). The fallback paths check the
    ``predictor_ckpt`` / ``action_predictor_ckpt`` attribute on ``self``
    before any forward work, so an uninitialized instance is fine.
    """
    stub = VJEPA2NPUWorker.__new__(VJEPA2NPUWorker)
    # Match CPU-worker defaults so the configured-grid check passes and we
    # reach the ckpt-missing branch.
    stub.past_frames = 1
    stub.future_frames = 1
    stub.action_dim = 2
    # predictor_ckpt / action_predictor_ckpt stay None.
    stub.predictor_ckpt = None
    stub.action_predictor_ckpt = None
    # Lazy-bundle slots normally set by __init__.
    stub.bundle = None
    stub.action_bundle = None
    stub._past_grid = None
    stub._future_grid = None
    method = getattr(stub, method_name)
    args = _stub_args(method_name)
    with pytest.raises(RuntimeError, match="predictor_ckpt|action_predictor_ckpt"):
        method(*args)


def _stub_args(method_name: str) -> tuple:
    """Return minimal valid positional args to invoke each method (bypassing
    Python's required-arg check before the fallback path's ckpt check)."""
    tiny_video = torch.zeros(1, 1, 3, 8, 8)
    tiny_actions = torch.zeros(1, 1, 2)
    table = {
        "predict_masked_batch": ([tiny_video], [0.5]),
        "predict_temporal_batch": ([tiny_video], [1], [1]),
        "predict_temporal": (tiny_video, 1, 1),
        "predict_action_conditioned": (tiny_video, tiny_actions, 1, 1),
        "predict_action_conditioned_batch": (
            [tiny_video], [tiny_actions], [1], [1],
        ),
    }
    return table[method_name]


def test_npu_predict_masked_batch_runs_after_load_model(tmp_path):
    """predict_masked_batch has no ckpt dependency — after load_model the
    CPU fallback should produce per-sample outputs without raising."""
    cfg = transformers.VJEPA2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=196, patch_size=14, tubelet_size=1, in_chans=3,
        frames_per_clip=1, drop_path_rate=0.0,
    )
    model = transformers.VJEPA2Model(cfg)
    model_dir = Path(tmp_path) / "tiny"
    model.save_pretrained(model_dir)

    w = VJEPA2NPUWorker(model_dir, dtype=torch.float32)
    w.load_model()

    # ``videos[i].shape`` must be ``(T, C, H, W)``; the batcher stacks on
    # dim=0 to build ``(B, T, C, H, W)``. Use T=1, C=3 (in_chans=3), and
    # the configured H=W=196 image size.
    video = torch.zeros(1, 3, 196, 196)
    out = w.predict_masked_batch([video], [0.5])
    assert isinstance(out, list)
    assert len(out) == 1
    # HF VJEPA2 returns ``(B*N_tokens, hidden_size)`` when bool_masked_pos
    # is supplied; with the tiny fixture B=1, N=14*14=196, hidden=64.
    assert out[0].shape == (196, 64)


# ---------------------------------------------------------------------------
# 3. predict_action_conditioned_batch — was missing entirely pre-W27 CPU prep
# ---------------------------------------------------------------------------


def test_npu_predict_action_conditioned_batch_signature_exact():
    """Pre-W27 CPU prep this method didn't exist on NPU worker. This test
    pins the method down to the exact CPU signature so a future rename /
    re-order is caught at PR time."""
    npu_sig = _signature_dict(VJEPA2NPUWorker.predict_action_conditioned_batch)
    cpu_sig = _signature_dict(VJEPA2Worker.predict_action_conditioned_batch)
    assert npu_sig == cpu_sig


def test_cpu_predict_action_conditioned_batch_signature_introspectable():
    """Smoke: CPU worker's predict_action_conditioned_batch signature is
    introspectable and matches the parity test target. We don't run the
    real body (it requires an action_bundle + multi-frame video tensor
    that exceeds the tiny VJEPA2 fixture used elsewhere); we only confirm
    the signature is what we expect to mirror."""
    sig = inspect.signature(VJEPA2Worker.predict_action_conditioned_batch)
    params = list(sig.parameters)
    assert params == [
        "self", "videos", "actions_per_sample",
        "past_frames_list", "future_frames_list",
    ]
    try:
        hints = get_type_hints(VJEPA2Worker.predict_action_conditioned_batch)
    except Exception:
        hints = {}
    assert "return" in hints or sig.return_annotation is not inspect.Signature.empty