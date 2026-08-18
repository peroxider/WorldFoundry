"""Unit tests for the M3-T5 W8A8 memory-reduction estimate.

Covers :func:`compute_w8a8_memory_reduction` against the real W26
packer: byte-size accounting, the ≥30% reduction gate, honest skipped
tensor counting, and CLI wiring (``--sdxl --memory-threshold``).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry_inf.eval.quant_eval import (
    W8A8MemoryReport,
    compute_w8a8_memory_reduction,
)


@pytest.fixture()
def sdxl_source(tmp_path: Path) -> Path:
    """SDXL/VAE-shaped source: one 2-D linear + one 4-D conv weight."""
    src = tmp_path / "src"
    src.mkdir()
    w = torch.randn(64, 128).to(torch.bfloat16)
    conv = torch.randn(8, 8, 3, 3).to(torch.bfloat16)
    save_file(
        {"linear.weight": w, "conv.weight": conv},
        str(src / "diffusion_pytorch_model.safetensors"),
    )
    (src / "config.json").write_text("{}")
    return src


def test_memory_reduction_shape_and_gate(sdxl_source: Path) -> None:
    """2-D weights shrink via W8A8 (int8 + row scales); gate passes ≥30%."""
    mem = compute_w8a8_memory_reduction(sdxl_source)
    assert isinstance(mem, W8A8MemoryReport)
    assert mem.source_bytes > 0
    assert mem.w8a8_bytes < mem.source_bytes
    assert mem.reduction_pct >= 0.30
    assert mem.passed is True
    # One 2-D linear rounded, one 4-D conv skipped.
    assert mem.n_weights_rounded == 1
    assert mem.n_weights_skipped == 1
    assert mem.n_tensors == 2


def test_memory_reduction_skipped_conv_counts_at_source(sdxl_source: Path) -> None:
    """Skipped 4-D tensors are counted at source size (honest accounting)."""
    mem = compute_w8a8_memory_reduction(sdxl_source)
    # conv.weight is 8*8*3*3 bf16 = 1152 bytes, unchanged in the estimate.
    assert mem.w8a8_bytes >= 1152


def test_memory_reduction_all_linear_is_higher() -> None:
    """All-2-D sources shrink ~50% (int8 halves, scales add a little back)."""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    w = torch.randn(256, 256).to(torch.bfloat16)
    save_file(
        {"a.weight": w, "b.weight": torch.randn(128, 64).to(torch.bfloat16)},
        str(tmp / "diffusion_pytorch_model.safetensors"),
    )
    (tmp / "config.json").write_text("{}")
    mem = compute_w8a8_memory_reduction(tmp)
    assert mem.n_weights_skipped == 0
    assert mem.n_weights_rounded == 2
    # int8 + per-row fp32 scale vs bf16 source -> ~50% for large rows.
    assert 0.40 <= mem.reduction_pct <= 0.55


def test_memory_reduction_fails_gate_when_threshold_strict() -> None:
    """A 45%-ish reduction fails an artificially strict 90% gate."""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    w = torch.randn(64, 128).to(torch.bfloat16)
    save_file(
        {"linear.weight": w},
        str(tmp / "diffusion_pytorch_model.safetensors"),
    )
    (tmp / "config.json").write_text("{}")
    mem = compute_w8a8_memory_reduction(tmp, threshold=0.90)
    assert mem.passed is False
    assert mem.reduction_pct < 0.90


def test_memory_reduction_missing_weights_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing baseline weights"):
        compute_w8a8_memory_reduction(empty)


def test_cli_sdxl_reports_memory(tmp_path: Path, monkeypatch) -> None:
    """The --sdxl CLI prints both cosine and memory lines; rc=0 when both pass."""
    from worldfoundry_inf.compile.quantize_backends import get_backend
    from worldfoundry_inf.eval import quant_eval

    src = tmp_path / "src"
    src.mkdir()
    w = torch.randn(64, 128).to(torch.bfloat16)
    save_file(
        {"linear.weight": w},
        str(src / "diffusion_pytorch_model.safetensors"),
    )
    (src / "config.json").write_text("{}")
    out = tmp_path / "out"
    get_backend("fake_sdxl").quantize(src, out, calibration_dataset=None)

    import io
    import sys

    captured = io.StringIO()
    # __main__ strips "--sdxl" from argv before calling _cli_sdxl; mirror it.
    monkeypatch.setattr(sys, "argv", [
        "quant_eval.py", "--sdxl",
        "--baseline", str(src), "--quantized", str(out),
    ])
    sys.argv.remove("--sdxl")
    monkeypatch.setattr(sys, "stdout", captured)
    rc = quant_eval._cli_sdxl()
    text = captured.getvalue()
    assert "SDXL cosine:" in text
    assert "W8A8 memory:" in text
    assert "-" in text  # reduction % printed
    assert rc == 0
