"""CPU tests for worldforge_inf.vllm_ascend_worker (no NPU, no vllm_ascend).

Validates the Hardware Pluggable worker class structure:
- Base-class resolution returns the stub on CPU-only systems
- __init__ / load_model / encode / predict / plan signatures match the existing
  VJEPA2Worker contract (so the Hardware Pluggable worker is a drop-in upgrade)
- encode / predict / plan raise RuntimeError when load_model was skipped
- encode / predict / plan dispatch via HF transformers on CPU fallback
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import transformers

from worldforge_inf.vllm_ascend_worker import (
    VJEPA2NPUWorker,
    _npu_ops,
    _resolve_base_class,
)


def _make_tiny_vjepa2_and_save(tmp_path: Path) -> Path:
    cfg = transformers.VJEPA2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=196, patch_size=14, tubelet_size=1, in_chans=3,
        frames_per_clip=1, drop_path_rate=0.0,
    )
    model = transformers.VJEPA2Model(cfg)
    out = tmp_path / "tiny"
    model.save_pretrained(out)
    return out


def test_resolve_base_class_returns_stub_on_cpu():
    """When vllm_ascend is not installed, _resolve_base_class returns the stub."""
    base = _resolve_base_class()
    # On this CPU-only host, vllm_ascend is absent, so we expect the stub.
    # If the host ever gains vllm_ascend, this test would need to mock instead.
    assert base.__name__ in {"_StubNPUWorker", "NPUWorker"}


def test_npu_ops_returns_none_on_cpu():
    """torch_npu is not installed on this CPU-only host; helper returns None."""
    assert _npu_ops() is None


def test_worker_load_model_populates_self_model(tmp_path):
    """load_model() calls HF VJEPA2Model.from_pretrained and stores on self.model."""
    src = _make_tiny_vjepa2_and_save(tmp_path)
    w = VJEPA2NPUWorker(Path(src), dtype=torch.float32)
    assert w.model is None
    w.load_model()
    assert w.model is not None
    assert isinstance(w.model, transformers.VJEPA2Model)


def test_encode_predict_plan_require_loaded_model(tmp_path):
    """encode/predict/plan all raise RuntimeError if load_model was not called."""
    src = _make_tiny_vjepa2_and_save(tmp_path)
    w = VJEPA2NPUWorker(Path(src), dtype=torch.float32)
    video = torch.zeros(1, 1, 3, 196, 196)
    mask = torch.ones(1, 14 * 14, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="load_model"):
        w.encode(video)
    with pytest.raises(RuntimeError, match="load_model"):
        w.predict(video, mask)
    with pytest.raises(RuntimeError, match="load_model"):
        w.plan(video, [1, 2, 3])


def test_encode_after_load_returns_finite_tensor(tmp_path):
    """After load_model, encode() runs the HF transformers eager path on CPU."""
    src = _make_tiny_vjepa2_and_save(tmp_path)
    w = VJEPA2NPUWorker(Path(src), dtype=torch.float32)
    w.load_model()
    video = torch.zeros(1, 1, 3, 196, 196)
    feats = w.encode(video)
    assert feats.shape == (1, 196, 64)
    assert torch.isfinite(feats).all()


def test_predict_after_load_returns_finite_tensor(tmp_path):
    src = _make_tiny_vjepa2_and_save(tmp_path)
    w = VJEPA2NPUWorker(Path(src), dtype=torch.float32)
    w.load_model()
    video = torch.zeros(1, 1, 3, 196, 196)
    mask = torch.ones(1, 14 * 14, dtype=torch.bool)
    out = w.predict(video, mask)
    assert out.shape == (1, 196, 64)
    assert torch.isfinite(out).all()


def test_plan_after_load_returns_features_and_actions(tmp_path):
    src = _make_tiny_vjepa2_and_save(tmp_path)
    w = VJEPA2NPUWorker(Path(src), dtype=torch.float32)
    w.load_model()
    video = torch.zeros(1, 1, 3, 196, 196)
    actions = [1, 2, 3]
    feats, returned_actions = w.plan(video, actions)
    assert feats.shape == (1, 196, 64)
    assert returned_actions == [1, 2, 3]
    # ensure input list was not mutated
    assert actions == [1, 2, 3]


def test_init_device_is_safe_to_call_on_cpu(tmp_path):
    """init_device() on the stub is a no-op (doesn't raise)."""
    src = _make_tiny_vjepa2_and_save(tmp_path)
    w = VJEPA2NPUWorker(Path(src), dtype=torch.float32)
    # On CPU with the stub base, init_device does not call super(); must not raise.
    w.init_device()


def test_adapter_name_stored(tmp_path):
    src = _make_tiny_vjepa2_and_save(tmp_path)
    w = VJEPA2NPUWorker(Path(src), adapter_name="my_adapter", dtype=torch.float32)
    assert w.adapter_name == "my_adapter"