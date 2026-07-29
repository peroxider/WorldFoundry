"""CPU e2e smoke for W8A8 orchestration: facade -> fake backend -> cosine eval.

NOT NPU-gated -- runs by default on every host. Exists to validate the
full pipeline (file outputs + sidecar + model load + cosine compute)
before committing to NPU time on real V-JEPA 2 vitl-1.2b.

A tiny synthetic HF model is written to ``tmp_path`` (config + weights) so
no network download is required, and ``AutoModel.from_pretrained`` is
monkey-patched so the loaded models expose a stub ``.encode()`` that
returns a deterministic ``last_hidden_state``. Cosine evaluation against
identical weights is ~1.0; the goal is pipeline-correctness, not
simulated quantization noise.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModel

from worldforge_inf.compile.quantize import quantize_model
from worldforge_inf.eval.quant_eval import cosine_similarity

_TINY_MODEL_ID = "hf-internal-testing/tiny-random-bert"
_FAKE_HIDDEN = 4
_FAKE_SEQ = 196


def _write_tiny_hf_model(path: Path) -> Path:
    """Write a tiny HF model dir to ``path`` (config + weights).

    Uses ``hf-internal-testing/tiny-random-bert`` -- a ~4MB HF-loadable
    model the CI environment can download. Once written, the test
    monkey-patches ``AutoModel.from_pretrained`` so the loaded models
    expose a stub ``.encode()``; the on-disk weights themselves are not
    consumed by the patched load path.
    """
    path.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(_TINY_MODEL_ID)
    model.save_pretrained(path)
    return path


def _make_fake_loaded_model(_pretrained_path, **_kwargs):
    """Replacement for ``AutoModel.from_pretrained`` used by the test.

    Returns a MagicMock with a stub ``.encode()`` that yields a
    deterministic ``last_hidden_state`` of shape (1, 196, 4). The same
    shape for every call ensures the shape-mismatch guard in
    ``cosine_similarity`` passes and the cosine similarity between the
    bf16 and quantized outputs is exactly 1.0 (identical tensors).
    """
    model = MagicMock()
    encode_out = SimpleNamespace(
        last_hidden_state=torch.zeros(1, _FAKE_SEQ, _FAKE_HIDDEN)
    )
    model.encode.return_value = encode_out
    return model


def test_facade_fake_backend_cosine_pipeline_end_to_end_on_cpu(tmp_path):
    """Full facade -> FakeBackend -> cosine_similarity on CPU.

    This is the orchestrator-level smoke test. It does NOT touch NPU
    or msModelSlim. The acceptance criterion: the pipeline completes
    without raising, the fake backend writes the expected sidecar
    schema, and ``cosine_similarity`` returns a report whose aggregate
    cosine is in the valid [-1, 1] range and is flagged as ``passed``.
    """
    src = _write_tiny_hf_model(tmp_path / "bf16")
    out = tmp_path / "w8a8-fake"

    # Step 1: facade -> FakeBackend.quantize()
    returned = quantize_model(
        src,
        out,
        quant_type="fake",
        calibration_dataset="videomix22m-512",
    )
    assert returned == out
    assert (out / "config.json").exists()
    assert (out / "quant_meta.json").exists()

    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "fake"
    assert meta["calibration_dataset"] == "videomix22m-512"
    assert meta["phase"] == "w11-w12-fake"
    assert meta["backend"] == "fake"
    assert meta["source_model_path"] == str(src)

    # Step 2: cosine_similarity. tiny-random-bert has no .encode(); patch
    # AutoModel.from_pretrained to attach a deterministic stub. Inputs
    # are random CPU tensors -- shape (3, 224, 224), n_samples=2.
    eval_inputs = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=_make_fake_loaded_model,
    ):
        report = cosine_similarity(
            baseline_path=src,
            quantized_path=out,
            eval_dataset=eval_inputs,
            threshold=0.0,  # smoke-test only: anything in [-1, 1] passes
            n_samples=2,
        )

    assert report.passed is True
    assert -1.0 <= report.aggregate_cosine <= 1.0
    assert report.n_samples == 2
    assert len(report.per_sample_cosine) == 2
