"""End-to-end W8A8 quantization + cosine eval (NPU-gated, default skip).

Run with: WORLDFORGE_RUN_QUANT_E2E=1 pytest worldfoundry_inf/tests/test_quant_integration_npu.py

Requires:
- torch_npu installed + NPU available
- msmodelslim installed (worldfoundry[quant])
- V-JEPA 2 vitl-1.2b weights downloaded to models/vjepa2-vitl-1.2b/

Uses a smaller surrogate ViT (facebook/dino-vits8, ~5M params) for CI;
production W11 acceptance runs against real V-JEPA 2 vitl-1.2b on 8xNPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch_npu_available = pytest.importorskip("torch_npu").npu.is_available()


@pytest.mark.skipif(
    not torch_npu_available or not os.environ.get("WORLDFORGE_RUN_QUANT_E2E"),
    reason="Requires NPU + WORLDFORGE_RUN_QUANT_E2E=1",
)
def test_w8a8_quantization_end_to_end(tmp_path):
    """End-to-end: load bf16 -> W8A8 quant -> cosine >= 0.95."""
    from worldfoundry_inf.compile.quantize import quantize_model
    from worldfoundry_inf.eval.quant_eval import cosine_similarity

    src = Path(os.environ.get("WORLDFORGE_QUANT_TEST_MODEL",
                              "models/dino-vits8"))  # tiny ViT for CI
    out = tmp_path / "w8a8"

    quantize_model(src, out, quant_type="w8a8",
                   calibration_dataset="videomix22m-512")

    assert (out / "quant_meta.json").exists()
    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "w8a8"

    report = cosine_similarity(src, out, threshold=0.95, n_samples=4)
    assert report.passed, f"cosine={report.aggregate_cosine} below 0.95"
