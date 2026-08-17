"""M3-T5: SDXL/VAE W8A8 quant smoke path tests.

Locks down:
- FakeSDXLBackend self-registers as "fake_sdxl"
- Round-tripped artifact's cosine vs source ≥ 0.95 (M3-T5 acceptance gate)
- cosine_similarity_sdxl returns SDXLEvalReport with the right shape
- Non-2-D tensors (e.g. 4-D conv, int64 token embeddings) are skipped unchanged
- quant_meta.json sidecar has the right schema

These tests run on any host with torch + safetensors installed. No
msModelSlim / Atlas hardware required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry_inf.compile.quantize_backends import SUPPORTED_QUANT_TYPES, get_backend
from worldfoundry_inf.compile.quantize_backends.fake import FakeSDXLBackend
from worldfoundry_inf.eval.quant_eval import (
    SDXLEvalReport,
    cosine_similarity_sdxl,
)


@pytest.fixture
def fake_sdxl_source(tmp_path: Path) -> Path:
    """Build a tiny SDXL-shaped source directory with 2-D + 4-D tensors."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"_class_name": "SDXL", "model_type": "sdxl"})
    )
    weights = {
        # 2-D float → round-tripped (linear.weight + proj_in.weight)
        "linear.weight": torch.randn(64, 32, dtype=torch.float32),
        "proj_in.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        # 1-D float → SKIPPED (pack_w8a8_weight requires 2-D)
        "linear.bias": torch.randn(64, dtype=torch.float32),
        # 4-D float → SKIPPED (conv-aware quant is out of scope for host smoke)
        "conv.weight": torch.randn(8, 3, 3, 3, dtype=torch.float32),
        # int64 → SKIPPED (non-floating-point)
        "token_embeds": torch.randint(0, 1000, (100, 64), dtype=torch.int64),
    }
    save_file(weights, str(src / "diffusion_pytorch_model.safetensors"))
    return src


def test_fake_sdxl_in_supported_quant_types() -> None:
    """M3-T5 fake backend must be in SUPPORTED_QUANT_TYPES for facade dispatch."""
    assert "fake_sdxl" in SUPPORTED_QUANT_TYPES


def test_fake_sdxl_self_registers_via_get_backend() -> None:
    backend = get_backend("fake_sdxl")
    assert isinstance(backend, FakeSDXLBackend)


def test_fake_sdxl_round_trip_produces_artifact_with_quant_meta(
    fake_sdxl_source: Path, tmp_path: Path
) -> None:
    out = tmp_path / "quant"
    backend = FakeSDXLBackend()
    returned = backend.quantize(fake_sdxl_source, out, calibration_dataset=None)
    assert returned == out
    assert (out / "config.json").is_file()
    assert (out / "diffusion_pytorch_model.safetensors").is_file()
    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "fake_sdxl"
    assert meta["backend"] == "fake_sdxl"
    assert meta["phase"] == "m3-t5-fake"
    assert meta["round_trip_method"] == "pack_w8a8_weight"
    assert meta["mse_threshold"] == pytest.approx(1e-3)
    assert meta["weight_filename"] == "diffusion_pytorch_model.safetensors"
    # 5 tensors in source: 2 × 2-D float (round-tripped) + 3 skipped (1-D + 4-D + int64)
    assert meta["n_weights_total"] == 5
    assert meta["n_weights_rounded"] == 2
    assert meta["n_weights_skipped"] == 3


def test_fake_sdxl_missing_source_weights_raises(
    tmp_path: Path,
) -> None:
    """Missing ``diffusion_pytorch_model.safetensors`` raises actionable error."""
    src = tmp_path / "no_weights"
    src.mkdir()
    (src / "config.json").write_text("{}")
    backend = FakeSDXLBackend()
    with pytest.raises(FileNotFoundError, match="diffusion_pytorch_model.safetensors"):
        backend.quantize(src, tmp_path / "out", calibration_dataset=None)


def test_cosine_similarity_sdxl_passes_after_round_trip(
    fake_sdxl_source: Path, tmp_path: Path
) -> None:
    """Round-tripped artifact passes the 0.95 cosine gate (M3-T5 acceptance)."""
    out = tmp_path / "quant"
    FakeSDXLBackend().quantize(fake_sdxl_source, out, calibration_dataset=None)
    report = cosine_similarity_sdxl(fake_sdxl_source, out)
    assert isinstance(report, SDXLEvalReport)
    # All round-tripped 2-D tensors should pass with cosine ≥ 0.99 by the
    # packer's MSE < 1e-3 contract; aggregate is mean over ALL tensors
    # (including skipped ones which have cosine = 1.0 since unchanged).
    assert report.passed is True, report
    assert report.aggregate_cosine >= 0.99
    assert report.threshold == pytest.approx(0.95)
    assert report.n_layers == 5
    # Per-tensor cosine populated for every key in the safetensors file
    assert set(report.per_tensor_cosine.keys()) == {
        "linear.weight", "linear.bias", "proj_in.weight",
        "conv.weight", "token_embeds",
    }
    # Skipped tensors are unchanged → cosine exactly 1.0
    assert report.per_tensor_cosine["conv.weight"] == pytest.approx(1.0)
    assert report.per_tensor_cosine["token_embeds"] == pytest.approx(1.0)


def test_cosine_similarity_sdxl_fails_when_artifact_is_pure_noise(
    fake_sdxl_source: Path, tmp_path: Path
) -> None:
    """Quantized artifact with random noise fails the gate (regression guard)."""
    out = tmp_path / "quant"
    out.mkdir()
    (out / "config.json").write_text((fake_sdxl_source / "config.json").read_text())
    # Write totally random weights (NOT round-tripped) to break cosine.
    torch.manual_seed(0)
    save_file(
        {"linear.weight": torch.randn(64, 32), "linear.bias": torch.randn(64),
         "proj_in.weight": torch.randn(128, 128), "conv.weight": torch.randn(8, 3, 3, 3),
         "token_embeds": torch.randint(0, 1000, (100, 64))},
        str(out / "diffusion_pytorch_model.safetensors"),
    )
    report = cosine_similarity_sdxl(fake_sdxl_source, out)
    assert report.passed is False
    assert report.aggregate_cosine < 0.95


def test_cosine_similarity_sdxl_raises_on_missing_baseline_weights(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="baseline weights"):
        cosine_similarity_sdxl(tmp_path, tmp_path)


def test_cosine_similarity_sdxl_raises_on_key_mismatch(
    fake_sdxl_source: Path, tmp_path: Path
) -> None:
    out = tmp_path / "quant"
    out.mkdir()
    (out / "config.json").write_text((fake_sdxl_source / "config.json").read_text())
    save_file(
        {"different_key.weight": torch.randn(8, 8)},
        str(out / "diffusion_pytorch_model.safetensors"),
    )
    with pytest.raises(ValueError, match="Key mismatch"):
        cosine_similarity_sdxl(fake_sdxl_source, out)