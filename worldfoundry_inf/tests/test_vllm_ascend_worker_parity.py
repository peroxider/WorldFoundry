"""Surface-parity tests for VJEPA2NPUWorker vs VJEPA2Worker (W27 CPU prep).

The Atlas NPU worker :class:`VJEPA2NPUWorker` and the CPU worker
:class:`VJEPA2Worker` share the public ``predict_*`` contract: a request
routed to either worker must accept the same parameter names + ordering
+ return type, so the API endpoint layer can dispatch uniformly.

Today (W27 v1) the NPU path raises :class:`NotImplementedError` for the
five batched / temporal / action-conditioned methods because the Atlas
wiring is gated on the hardware window (roadmap G11). This module
asserts that:

1. The NPU worker's ``predict_*`` methods have signatures **identical**
   to the CPU worker's (``inspect.signature`` cross-check). A future
   Atlas implementer who renames a parameter, reorders an argument, or
   drops the return annotation breaks this contract test loudly.
2. Calling any of the five ``NotImplementedError`` stubs raises with a
   message that mentions "Atlas pilot" — so a caller can distinguish
   "intentional deferral" from "actual bug".
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

from worldfoundry_inf.vjepa2_worker import VJEPA2Worker
from worldfoundry_inf.vllm_ascend_worker import VJEPA2NPUWorker


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
# 2. Stub raises with "Atlas pilot" message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", _PARITY_METHODS)
def test_npu_stub_raises_not_implemented_with_atlas_pilot_message(method_name):
    """All five stubs must raise NotImplementedError mentioning the Atlas deferral.

    Construct a stub instance via ``__new__`` to bypass ``__init__`` (which
    requires a real VJEPA2 model path). The stub bodies don't read ``self``,
    only raise, so the uninitialized instance is fine.
    """
    stub = VJEPA2NPUWorker.__new__(VJEPA2NPUWorker)
    method = getattr(stub, method_name)
    args = _stub_args(method_name)
    with pytest.raises(NotImplementedError, match="Atlas pilot"):
        method(*args)


def _stub_args(method_name: str) -> tuple:
    """Return minimal valid positional args to invoke each stub (bypassing
    Python's required-arg check before the stub body raises)."""
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


def test_npu_stub_does_not_silently_succeed_after_load_model(tmp_path):
    """Even after a real load_model, the five stubs must raise — guard
    against a future refactor that wires them through the eager CPU path
    and accidentally turns them into silently-working fallbacks (which
    would defeat the Atlas-pilot gate)."""
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

    video = torch.zeros(1, 1, 3, 196, 196)
    actions = torch.zeros(1, 1, 2)

    with pytest.raises(NotImplementedError, match="Atlas pilot"):
        w.predict_masked_batch([video], [0.5])
    with pytest.raises(NotImplementedError, match="Atlas pilot"):
        w.predict_temporal_batch([video], [1], [1])
    with pytest.raises(NotImplementedError, match="Atlas pilot"):
        w.predict_temporal(video, 1, 1)
    with pytest.raises(NotImplementedError, match="Atlas pilot"):
        w.predict_action_conditioned(video, actions, 1, 1)
    with pytest.raises(NotImplementedError, match="Atlas pilot"):
        w.predict_action_conditioned_batch([video], [actions], [1], [1])


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