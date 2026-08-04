"""CPU tests for worldfoundry_train.quantize_bridge (no NPU, no model load)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def bridge_module():
    """Re-import the bridge to pick up any monkeypatching done elsewhere."""
    if "worldfoundry_train.quantize_bridge" in sys.modules:
        del sys.modules["worldfoundry_train.quantize_bridge"]
    return importlib.import_module("worldfoundry_train.quantize_bridge")


def _make_source_dir(tmp_path: Path) -> Path:
    """Create a minimal HF-saved model dir (config + pytorch_model.bin)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_bytes(b"\x00" * 100)
    return src


def test_no_quant_block_is_noop(bridge_module):
    cfg = {"model": {"hf_repo": "facebook/vjepa2-vitl-1.2b"}}
    out = bridge_module.maybe_quantize_first(cfg)
    assert out is cfg
    assert cfg["model"]["hf_repo"] == "facebook/vjepa2-vitl-1.2b"


def test_empty_quant_block_is_noop(bridge_module):
    cfg = {"model": {"hf_repo": "x"}, "quant": None}
    out = bridge_module.maybe_quantize_first(cfg)
    assert out is cfg
    assert cfg["model"]["hf_repo"] == "x"


def test_valid_block_calls_quantize_and_mutates_hf_repo(bridge_module, tmp_path, monkeypatch):
    src = _make_source_dir(tmp_path)
    out_dir = tmp_path / "q"
    cfg = {
        "model": {"hf_repo": str(src)},
        "quant": {
            "type": "fake",
            "source_dir": str(src),
            "output_dir": str(out_dir),
            "calibration_dataset": "videomix22m-512",
        },
    }
    seen = {}

    def fake_quantize_model(model_path, output_path, quant_type, calibration_dataset):
        seen["mp"] = model_path
        seen["op"] = output_path
        seen["qt"] = quant_type
        seen["cd"] = calibration_dataset
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    monkeypatch.setattr(
        "worldfoundry_inf.compile.quantize.quantize_model", fake_quantize_model,
    )
    out = bridge_module.maybe_quantize_first(cfg)
    assert out is cfg
    assert seen["mp"] == src
    assert seen["op"] == out_dir
    assert seen["qt"] == "fake"
    assert seen["cd"] == "videomix22m-512"
    assert cfg["model"]["hf_repo"] == str(out_dir)


def test_missing_required_key_raises(bridge_module):
    cfg = {
        "model": {"hf_repo": "x"},
        "quant": {"type": "fake", "source_dir": "/tmp"},  # missing output_dir
    }
    with pytest.raises(ValueError, match="output_dir"):
        bridge_module.maybe_quantize_first(cfg)


def test_missing_source_dir_raises(bridge_module, tmp_path):
    cfg = {
        "model": {"hf_repo": "x"},
        "quant": {
            "type": "fake",
            "source_dir": str(tmp_path / "does_not_exist"),
            "output_dir": str(tmp_path / "out"),
        },
    }
    with pytest.raises(ValueError, match="not an existing directory"):
        bridge_module.maybe_quantize_first(cfg)


def test_output_dir_eq_source_dir_raises(bridge_module, tmp_path):
    same = tmp_path / "same"
    same.mkdir()
    cfg = {
        "model": {"hf_repo": "x"},
        "quant": {"type": "fake", "source_dir": str(same), "output_dir": str(same)},
    }
    with pytest.raises(ValueError, match="equals source_dir"):
        bridge_module.maybe_quantize_first(cfg)


def test_calibration_dataset_optional(bridge_module, tmp_path, monkeypatch):
    src = _make_source_dir(tmp_path)
    cfg = {
        "model": {"hf_repo": "x"},
        "quant": {
            "type": "fake",
            "source_dir": str(src),
            "output_dir": str(tmp_path / "o"),
        },
    }
    seen = {}

    def fake_q(model_path, output_path, quant_type, calibration_dataset):
        seen["cd"] = calibration_dataset
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    monkeypatch.setattr(
        "worldfoundry_inf.compile.quantize.quantize_model", fake_q,
    )
    bridge_module.maybe_quantize_first(cfg)
    assert seen["cd"] is None