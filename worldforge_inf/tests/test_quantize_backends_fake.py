"""FakeBackend self-registers + reproduces quantize outputs without msModelSlim."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldforge_inf.compile.quantize_backends import get_backend, backend_available


def test_fake_backend_self_registers_on_import():
    from worldforge_inf.compile.quantize_backends import fake  # noqa: F401
    available, _ = backend_available("fake")
    assert available is True


def test_fake_backend_quantize_copies_weights_and_meta(tmp_path):
    """FakeBackend copies weights + writes quant_meta.json sidecar."""
    src = tmp_path / "bf16"
    src.mkdir()
    (src / "config.json").write_text('{"hidden_size": 4, "model_type": "tiny"}')
    (src / "pytorch_model.bin").write_bytes(b"fake-bf16-weights")

    out = tmp_path / "w8a8"
    backend = get_backend("fake")
    returned = backend.quantize(src, out, calibration_dataset=None)

    assert returned == out
    assert (out / "config.json").exists()
    assert (out / "pytorch_model.bin").exists()
    assert (out / "pytorch_model.bin").read_bytes() == b"fake-bf16-weights"


def test_fake_backend_writes_quant_meta_with_required_fields(tmp_path):
    """quant_meta.json writes the 6 fields FakeBackend owns (msmodelslim_version
    intentionally omitted — no msModelSlim involvement in the CPU smoke path).
    Field set is a strict subset of W8A8Backend's schema."""
    from worldforge_inf.compile.quantize_backends.fake import FakeBackend

    src = tmp_path / "bf16"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_bytes(b"x")

    out = tmp_path / "w8a8"
    FakeBackend().quantize(src, out, calibration_dataset="videomix22m-512")

    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "fake"  # the registered name (deferred-name contract)
    assert meta["calibration_dataset"] == "videomix22m-512"
    assert meta["phase"] == "w11-w12-fake"
    assert meta["backend"] == "fake"
    assert meta["source_model_path"] == str(src)
    # msmodelslim_version intentionally omitted (no msModelSlim involvement)
    assert "note" in meta


def test_fake_backend_missing_config_raises(tmp_path):
    from worldforge_inf.compile.quantize_backends.fake import FakeBackend

    src = tmp_path / "bad"
    src.mkdir()
    # no config.json
    with pytest.raises(FileNotFoundError, match="config.json"):
        FakeBackend().quantize(src, tmp_path / "out", calibration_dataset=None)
