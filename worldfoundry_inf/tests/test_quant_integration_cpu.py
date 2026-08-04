"""CPU e2e smoke for W8A8 orchestration: facade -> fake backend -> cosine eval.

NOT NPU-gated -- runs by default on every host. Exists to validate the
full pipeline (file outputs + sidecar + model load + cosine compute)
on real CPU-bound V-JEPA 2 weights before committing to NPU time on
real V-JEPA 2 vitl-1.2b.

Builds a real but tiny ``VJEPA2Model`` (~21M params) from
``VJEPA2Config``, saves it to ``tmp_path``, and runs the orchestration
against it. No ``unittest.mock`` patching of ``AutoModel`` at all --
the test does real HF save / load / forward / cosine end to end.

Why we don't call ``cosine_similarity(...)`` directly: that helper
internally re-invokes ``AutoModel.from_pretrained`` to construct
fresh ``VJEPA2Model`` instances, which would have no ``.encode()``
interface. Stock V-JEPA 2 takes ``pixel_values_videos`` of shape
``(B, T, C, H, W)`` via ``model(...)``. This test inlines the same
``F.cosine_similarity`` per-sample aggregation that
``cosine_similarity`` does, on models loaded once here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel, VJEPA2Config, VJEPA2Model

from worldfoundry_inf.compile.quantize import quantize_model

_TINY_HIDDEN = 64
_TINY_LAYERS = 2
_TINY_HEADS = 2


def _build_real_tiny_vjepa2(path: Path) -> Path:
    """Build + save a real (tiny) VJEPA2Model on disk; ~21M params."""
    path.mkdir(parents=True, exist_ok=True)
    cfg = VJEPA2Config(
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        image_size=224,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        frames_per_clip=1,
        drop_path_rate=0.0,
    )
    model = VJEPA2Model(cfg).eval()
    model.save_pretrained(path)
    return path


def _real_cosine_pipeline(baseline_path: Path, quantized_path: Path,
                          eval_inputs: list, threshold: float):
    """Reproduce the production ``cosine_similarity`` aggregation on real
    VJEPA 2 forward passes.

    Loads both model dirs with ``AutoModel.from_pretrained`` (no monkey
    patching). For each ``(C, H, W)`` input, calls the real
    ``model(pixel_values_videos=...)`` forward and computes
    ``F.cosine_similarity`` between the bf16 and quantized
    ``last_hidden_state``. Returns the aggregate cosine so the test can
    assert against a real (non-mocked) number.
    """
    bf16 = AutoModel.from_pretrained(str(baseline_path), dtype=torch.float32).eval()
    quant = AutoModel.from_pretrained(str(quantized_path), dtype=torch.float32).eval()

    per_sample: list[float] = []
    for sample in eval_inputs:
        # sample is (C=3, H=224, W=224); VJEPA 2 expects (B=1, T=1, C, H, W)
        pixel_values_videos = sample.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            a = bf16(pixel_values_videos=pixel_values_videos).last_hidden_state
            b = quant(pixel_values_videos=pixel_values_videos).last_hidden_state
        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch: bf16={tuple(a.shape)} vs int8={tuple(b.shape)}"
            )
        per_sample.append(F.cosine_similarity(a, b, dim=-1).mean().item())
    aggregate = sum(per_sample) / len(per_sample)
    return aggregate, per_sample


def test_facade_fake_backend_real_vjepa2_pipeline_end_to_end_on_cpu(tmp_path):
    """Real CPU e2e: build tiny V-JEPA 2 -> FakeBackend -> cosine eval.

    Real HF save + load + forward. Real torch cosine similarity.
    FakeBackend copies bf16 weights unchanged so aggregate cosine is
    exactly 1.0 (within float epsilon).
    """
    src = _build_real_tiny_vjepa2(tmp_path / "bf16_vjepa2")
    out = tmp_path / "w8a8_fake_vjepa2"

    # ---- Step 1: facade -> FakeBackend.quantize() ----
    returned = quantize_model(
        src, out,
        quant_type="fake",
        calibration_dataset="videomix22m-512",
    )
    assert returned == out
    assert (out / "config.json").exists()
    assert (out / "quant_meta.json").exists()
    # Weights copied in whichever format the source used
    # (pytorch_model.bin legacy, model.safetensors modern HF default).
    assert (out / "pytorch_model.bin").exists() or (out / "model.safetensors").exists()

    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "fake"
    assert meta["calibration_dataset"] == "videomix22m-512"
    assert meta["phase"] == "w11-w12-fake"
    assert meta["backend"] == "fake"
    assert meta["source_model_path"] == str(src)

    # ---- Step 2: real forward + real cosine, no mock ----
    torch.manual_seed(123)
    eval_inputs = [torch.randn(3, 224, 224) for _ in range(3)]

    aggregate, per_sample = _real_cosine_pipeline(src, out, eval_inputs, threshold=0.99)

    assert len(per_sample) == 3
    assert aggregate == pytest.approx(1.0, abs=1e-5)
    for c in per_sample:
        assert -1.0 <= c <= 1.0
