"""Tests for the ONNX Runtime-backed vLLM-style inference worker (M2 followup).

The worker is the host-smoke non-PyTorch accelerated inference engine
for the world model. It builds a tiny ViT in pure PyTorch, exports it
to ONNX once, and serves :class:`onnxruntime.InferenceSession` for the
``encode`` / ``predict`` / ``predict_masked_batch`` / ``plan`` surface.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_inf.onnx_vllm_worker import (
    OnnxVJEPAWorker,
    TinyViTConfig,
    is_onnx_runtime_available,
)


pytestmark = pytest.mark.skipif(
    not is_onnx_runtime_available(),
    reason="onnxruntime wheel not installed; onnx_vllm backend unavailable",
)


def _tiny_cfg() -> TinyViTConfig:
    """Build a tiny ViT config that runs in <1 s on a stock CPU."""
    return TinyViTConfig(
        img_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=2,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_onnx_worker_builds_default_config() -> None:
    """Default construction picks :class:`TinyViTConfig` defaults."""
    worker = OnnxVJEPAWorker()
    summary = worker.provider_summary()
    assert summary["engine"] == "onnxruntime"
    assert summary["img_size"] == 224
    assert summary["onnx_bytes"] > 0


def test_onnx_worker_builds_custom_config() -> None:
    """Custom encoder config is reflected in ``provider_summary``."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    summary = worker.provider_summary()
    assert summary["img_size"] == 32
    assert summary["patch_size"] == 8
    assert summary["embed_dim"] == 32
    assert summary["depth"] == 1
    assert summary["num_patches"] == 16


def test_onnx_worker_resolves_providers() -> None:
    """``session.get_providers()`` returns at least the requested CPU EP."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg(), providers=["CPUExecutionProvider"])
    providers = worker.session.get_providers()
    assert "CPUExecutionProvider" in providers
    assert worker.provider_summary()["providers_resolved"] == providers


def test_onnx_worker_available_classmethod() -> None:
    """``available()`` returns True iff onnxruntime is importable."""
    assert OnnxVJEPAWorker.available() is True


# ---------------------------------------------------------------------------
# Inference surface
# ---------------------------------------------------------------------------


def test_encode_video_returns_expected_shape() -> None:
    """``encode(B, T, C, H, W)`` returns ``(B, T*N, D)``."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    video = torch.randn(2, 1, 3, 32, 32)
    feats = worker.encode(video)
    assert feats.shape == (2, 1 * 16, 32)
    assert feats.dtype == torch.float32


def test_encode_handles_different_batch_sizes() -> None:
    """``encode`` serves any batch size (dynamic batch axis in ONNX)."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    for b in (1, 2, 3, 4):
        video = torch.randn(b, 1, 3, 32, 32)
        feats = worker.encode(video)
        assert feats.shape[0] == b


def test_encode_handles_multiple_time_steps() -> None:
    """``encode`` serves ``T > 1`` (numpy reshape, not ONNX reshape)."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    video = torch.randn(2, 4, 3, 32, 32)
    feats = worker.encode(video)
    # 4 frames * 16 patches = 64 tokens per sample
    assert feats.shape == (2, 4 * 16, 32)


def test_encode_rejects_non_5d_input() -> None:
    """Non-5-D video raises :class:`ValueError`."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    with pytest.raises(ValueError, match="B, T, C, H, W"):
        worker.encode(torch.randn(2, 3, 32, 32))


def test_predict_ignores_mask_and_matches_encode() -> None:
    """``predict`` ignores the mask and returns the same as ``encode``."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    video = torch.randn(2, 1, 3, 32, 32)
    mask = torch.zeros(2, 1, dtype=torch.bool)
    feats_pred = worker.predict(video, mask)
    feats_enc = worker.encode(video)
    assert torch.allclose(feats_pred, feats_enc)


def test_predict_masked_batch_uniform_shape() -> None:
    """``predict_masked_batch`` returns per-sample ``(T*N, D)`` tensors."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    videos = [torch.randn(2, 3, 32, 32) for _ in range(3)]
    out = worker.predict_masked_batch(videos, [0.5, 0.5, 0.5])
    assert len(out) == 3
    for t in out:
        assert t.shape == (2 * 16, 32)


def test_predict_masked_batch_rejects_length_mismatch() -> None:
    """Length mismatch between ``videos`` and ``mask_ratios`` raises."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    videos = [torch.randn(2, 3, 32, 32) for _ in range(3)]
    with pytest.raises(ValueError, match="length mismatch"):
        worker.predict_masked_batch(videos, [0.5, 0.5])


def test_predict_masked_batch_rejects_non_uniform_shape() -> None:
    """Non-uniform video shapes raise :class:`ValueError`."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    videos = [torch.randn(2, 3, 32, 32), torch.randn(3, 3, 32, 32)]
    with pytest.raises(ValueError, match="uniform shape"):
        worker.predict_masked_batch(videos, [0.5, 0.5])


def test_predict_masked_batch_empty() -> None:
    """Empty input returns an empty list."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    assert worker.predict_masked_batch([], []) == []


def test_plan_returns_features_and_actions() -> None:
    """``plan`` returns ``(features, actions)`` matching the input."""
    worker = OnnxVJEPAWorker(cfg=_tiny_cfg())
    video = torch.randn(1, 1, 3, 32, 32)
    actions = [3, 1, 4, 1, 5]
    feats, returned_actions = worker.plan(video, actions)
    assert feats.shape == (1, 16, 32)
    assert returned_actions == [3, 1, 4, 1, 5]


# ---------------------------------------------------------------------------
# Backend registry integration
# ---------------------------------------------------------------------------


def test_inference_backend_registry_has_onnx_vllm() -> None:
    """``registry.list()`` exposes the ``onnx_vllm`` backend."""
    from worldfoundry_common.backends.inference import registry

    names = registry.list()
    assert "onnx_vllm" in names


def test_inference_backend_builds_onnx_vllm() -> None:
    """``make_engine("onnx_vllm")`` returns a :class:`OnnxVJEPAWorker`."""
    from worldfoundry_common.backends.inference import make_engine

    worker = make_engine("onnx_vllm")
    assert isinstance(worker, OnnxVJEPAWorker)
    assert worker.session is not None


# ---------------------------------------------------------------------------
# Determinism / caching
# ---------------------------------------------------------------------------


def test_onnx_export_is_deterministic() -> None:
    """Two workers built from the same seed produce identical outputs.

    The tiny ViT is initialized with ``torch.manual_seed(seed)`` in the
    export, so two workers constructed with the same seed produce the
    same ONNX bytes and same outputs for the same input.
    """
    torch.manual_seed(0)
    w1 = OnnxVJEPAWorker(cfg=_tiny_cfg(), providers=["CPUExecutionProvider"])
    torch.manual_seed(0)
    w2 = OnnxVJEPAWorker(cfg=_tiny_cfg(), providers=["CPUExecutionProvider"])
    v = torch.randn(1, 1, 3, 32, 32)
    f1 = w1.encode(v)
    f2 = w2.encode(v)
    # Same seed -> same model graph -> same output.
    assert torch.allclose(f1, f2)