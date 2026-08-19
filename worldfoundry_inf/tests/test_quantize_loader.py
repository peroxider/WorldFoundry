"""CPU tests for worldfoundry_inf.compile.quantize_primitives + _vllm_ascend_plugin loader hook.

Two halves:
1. Round-trip + edge-case tests for the W8A8 pack/unpack utilities in
   :mod:`worldfoundry_inf.compile.quantize_primitives` (W26 CPU prep). The acceptance gate
   is MSE <= ``ACCEPTANCE_MSE_THRESHOLD`` (= 1e-3) on a random bf16
   tensor — tight enough that a broken unpacker fails loudly but
   generous enough that float32 round-off is invisible.
2. Tests for :func:`worldfoundry_inf._vllm_ascend_plugin._patch_quantized_loader`,
   using a fake ``ModelRegistry`` injected via ``monkeypatch`` (the same
   pattern as ``test_vllm_ascend_plugin.py``).

The loader closure yields ``(name, tensor)`` pairs; we exercise the
quantized-vs-passthrough path by feeding it a small iterable of mock
tensors + sidecar state. No safetensors / msModelSlim / vLLM required.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock  # noqa: F401  (kept for symmetry w/ sibling tests)

import pytest
import torch

from worldfoundry_inf import _vllm_ascend_plugin as plugin
from worldfoundry_inf.compile import quantize_primitives as quantize_mod
from worldfoundry_inf.compile.quantize_primitives import (
    ACCEPTANCE_MSE_THRESHOLD,
    DEFAULT_QUANT_TYPE,
    is_w8a8_quantized,
    load_quant_meta,
    pack_linear,
    pack_w8a8_activation,
    pack_w8a8_weight,
    unpack_linear,
    unpack_w8a8_activation,
    unpack_w8a8_weight,
)


# ---------------------------------------------------------------------------
# Round-trip + validation: pack_w8a8_weight / unpack_w8a8_weight
# ---------------------------------------------------------------------------


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.to(torch.float32) - b.to(torch.float32)) ** 2).mean().item())


def test_pack_w8a8_weight_returns_int8_weight_and_per_channel_scale():
    torch.manual_seed(0)
    weight = torch.randn(8, 16)
    packed = pack_w8a8_weight(weight)
    assert packed["weight"].dtype == torch.int8
    assert packed["weight"].shape == weight.shape
    assert packed["scale"].dtype == torch.float32
    assert packed["scale"].shape == (8,)  # one scale per out-channel


def test_pack_w8a8_weight_round_trip_mse_under_threshold():
    """The W26 acceptance gate: MSE of pack→unpack vs the bf16 source."""
    torch.manual_seed(1)
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    packed = pack_w8a8_weight(weight)
    restored = unpack_w8a8_weight(packed)
    assert restored.dtype == torch.bfloat16
    assert restored.shape == weight.shape
    mse = _mse(restored, weight)
    assert mse < ACCEPTANCE_MSE_THRESHOLD, (
        f"round-trip MSE {mse:.3e} exceeds {ACCEPTANCE_MSE_THRESHOLD:.3e}"
    )


def test_pack_w8a8_weight_handles_zero_row_gracefully():
    """An all-zero output channel must not produce a NaN scale."""
    weight = torch.zeros(4, 8)
    weight[2] = torch.randn(8)  # only one non-zero row
    packed = pack_w8a8_weight(weight)
    assert torch.isfinite(packed["scale"]).all()
    assert torch.equal(unpack_w8a8_weight(packed)[0], torch.zeros(8))


def test_pack_w8a8_weight_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        pack_w8a8_weight(torch.randn(8))  # 1-D


def test_unpack_w8a8_weight_rejects_non_int8_weight():
    bogus = {"weight": torch.randn(4, 4), "scale": torch.ones(4)}
    with pytest.raises(ValueError, match="int8"):
        unpack_w8a8_weight(bogus)


def test_unpack_w8a8_weight_rejects_mismatched_scale_shape():
    bogus = {"weight": torch.zeros(4, 4, dtype=torch.int8), "scale": torch.ones(3)}
    with pytest.raises(ValueError, match="scale"):
        unpack_w8a8_weight(bogus)


def test_pack_linear_is_alias_for_pack_w8a8_weight():
    torch.manual_seed(2)
    w = torch.randn(3, 5)
    assert pack_linear(w)["weight"].shape == (3, 5)
    unpacked = unpack_linear(pack_linear(w))
    assert unpacked.shape == (3, 5)


# ---------------------------------------------------------------------------
# Round-trip: pack_w8a8_activation / unpack_w8a8_activation
# ---------------------------------------------------------------------------


def test_pack_w8a8_activation_returns_int8_and_scalar_scale():
    torch.manual_seed(3)
    x = torch.randn(2, 4, 8)
    packed = pack_w8a8_activation(x)
    assert packed["activation"].dtype == torch.int8
    assert packed["activation"].shape == x.shape
    assert packed["scale"].dtype == torch.float32
    assert packed["scale"].ndim == 0  # per-tensor


def test_pack_w8a8_activation_round_trip_mse_under_threshold():
    torch.manual_seed(4)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    packed = pack_w8a8_activation(x)
    restored = unpack_w8a8_activation(packed)
    assert restored.dtype == torch.bfloat16
    mse = _mse(restored, x)
    assert mse < ACCEPTANCE_MSE_THRESHOLD, f"activation MSE {mse:.3e} too high"


def test_pack_w8a8_activation_handles_zero_input():
    packed = pack_w8a8_activation(torch.zeros(3, 5))
    assert packed["scale"].item() == 0.0
    assert torch.equal(unpack_w8a8_activation(packed), torch.zeros(3, 5))


def test_unpack_w8a8_activation_rejects_non_int8():
    bogus = {"activation": torch.zeros(4), "scale": torch.tensor(1.0)}
    with pytest.raises(ValueError, match="int8"):
        unpack_w8a8_activation(bogus)


def test_unpack_w8a8_activation_rejects_vector_scale():
    bogus = {"activation": torch.zeros(4, dtype=torch.int8),
             "scale": torch.ones(4)}
    with pytest.raises(ValueError, match="0-D"):
        unpack_w8a8_activation(bogus)


# ---------------------------------------------------------------------------
# Sidecar: load_quant_meta / is_w8a8_quantized
# ---------------------------------------------------------------------------


def test_load_quant_meta_returns_none_when_absent(tmp_path):
    assert load_quant_meta(tmp_path) is None


def test_load_quant_meta_returns_dict_when_present(tmp_path):
    meta = {"quant_type": "w8a8", "calibration_dataset": "videomix22m-512"}
    (tmp_path / "quant_meta.json").write_text(json.dumps(meta))
    loaded = load_quant_meta(tmp_path)
    assert loaded == meta


def test_load_quant_meta_raises_on_malformed_json(tmp_path):
    (tmp_path / "quant_meta.json").write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_quant_meta(tmp_path)


def test_is_w8a8_quantized_true_when_sidecar_matches(tmp_path):
    (tmp_path / "quant_meta.json").write_text(
        json.dumps({"quant_type": DEFAULT_QUANT_TYPE})
    )
    assert is_w8a8_quantized(tmp_path) is True


def test_is_w8a8_quantized_false_when_absent(tmp_path):
    assert is_w8a8_quantized(tmp_path) is False


def test_is_w8a8_quantized_false_when_other_quant_type(tmp_path):
    (tmp_path / "quant_meta.json").write_text(json.dumps({"quant_type": "w4a8"}))
    assert is_w8a8_quantized(tmp_path) is False


# ---------------------------------------------------------------------------
# Loader hook: _patch_quantized_loader
# ---------------------------------------------------------------------------


def _make_fake_registry(calls: dict):
    class _FakeRegistry:
        @staticmethod
        def register_model_loader(name, loader_fn):  # type: ignore[no-untyped-def]
            calls["name"] = name
            calls["loader"] = loader_fn
    return _FakeRegistry


def test_patch_quantized_loader_no_op_when_vllm_missing(caplog):
    """On a CPU-only host (no vLLM), the hook must log + return, never raise."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(plugin, "_vllm_model_registry", lambda: None)
        with caplog.at_level(
            logging.INFO, logger="worldfoundry_inf._vllm_ascend_plugin"
        ):
            plugin._patch_quantized_loader()  # must not raise
        assert "skipping W8A8 quantized loader hook" in caplog.text
    finally:
        monkeypatch.undo()


def test_patch_quantized_loader_registers_loader_when_vllm_present(monkeypatch):
    """When vLLM's ModelRegistry is available, the hook registers a loader for VJEPA2Model."""
    calls: dict = {}
    monkeypatch.setattr(plugin, "_vllm_model_registry", lambda: _make_fake_registry(calls))

    plugin._patch_quantized_loader()

    assert calls["name"] == "VJEPA2Model"
    assert callable(calls["loader"])


def test_patch_quantized_loader_falls_back_when_registry_lacks_method(monkeypatch, caplog):
    """Older vLLM without register_model_loader: log + skip, do not raise."""
    class _LegacyRegistry:
        pass

    monkeypatch.setattr(plugin, "_vllm_model_registry", lambda: _LegacyRegistry)
    with caplog.at_level(
        logging.INFO, logger="worldfoundry_inf._vllm_ascend_plugin"
    ):
        plugin._patch_quantized_loader()  # must not raise
    assert "lacks register_model_loader" in caplog.text


def test_patch_quantized_loader_swallows_unexpected_exceptions(monkeypatch, caplog):
    """If registration raises something other than AttributeError, log + return."""

    class _BoomRegistry:
        @staticmethod
        def register_model_loader(name, loader_fn):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated registry failure")

    monkeypatch.setattr(plugin, "_vllm_model_registry", lambda: _BoomRegistry)
    with caplog.at_level(
        logging.WARNING, logger="worldfoundry_inf._vllm_ascend_plugin"
    ):
        plugin._patch_quantized_loader()
    assert "W8A8 loader hook failed" in caplog.text
    assert "simulated registry failure" in caplog.text


# ---------------------------------------------------------------------------
# Loader closure behavior (the actual unpack wiring)
# ---------------------------------------------------------------------------


def _trigger_loader_registration(monkeypatch) -> "callable":
    """Register the W8A8 loader hook and return the registered closure."""
    calls: dict = {}
    monkeypatch.setattr(
        plugin, "_vllm_model_registry", lambda: _make_fake_registry(calls)
    )
    plugin._patch_quantized_loader()
    return calls["loader"]


def test_loader_passes_through_when_not_quantized(tmp_path, monkeypatch):
    """Non-W8A8 model dir: weights iterator yields original tensors unchanged."""
    loader = _trigger_loader_registration(monkeypatch)
    weight = torch.zeros(4, 4, dtype=torch.bfloat16)
    seen = list(loader(iter([("layer.weight", weight)]), tmp_path))
    assert len(seen) == 1
    assert seen[0][0] == "layer.weight"
    assert torch.equal(seen[0][1], weight)


def test_loader_unpacks_int8_weight_when_quantized(tmp_path, monkeypatch):
    """W8A8 model dir + int8 weight + adjacent scale tensor: yields unpacked bf16.

    We monkeypatch ``_load_w8a8_scales`` to inject the scale directly; the
    on-disk ``safetensors`` reader is exercised separately (see
    ``test_loader_handles_missing_safetensors_lib_for_scales``).
    """
    loader = _trigger_loader_registration(monkeypatch)

    (tmp_path / "quant_meta.json").write_text(
        json.dumps({"quant_type": DEFAULT_QUANT_TYPE})
    )

    src = torch.randn(4, 8, dtype=torch.bfloat16)
    packed = pack_w8a8_weight(src)
    q = packed["weight"]
    scale = packed["scale"]

    monkeypatch.setattr(
        plugin, "_load_w8a8_scales",
        lambda d: {"layer.weight_scale": scale},
    )

    weights_iter = iter([
        ("layer.weight", q),
        ("layer.weight_scale", scale),
    ])

    seen = list(loader(weights_iter, tmp_path))
    seen_dict = dict(seen)
    assert seen_dict["layer.weight"].dtype == torch.bfloat16
    assert seen_dict["layer.weight"].shape == src.shape
    mse = _mse(seen_dict["layer.weight"], src)
    assert mse < ACCEPTANCE_MSE_THRESHOLD, f"loader MSE {mse:.3e} too high"
    assert torch.equal(seen_dict["layer.weight_scale"], scale)


def test_loader_passes_int8_through_when_no_adjacent_scale(tmp_path, monkeypatch):
    """W8A8 sidecar present but no scale tensor found: int8 weight passes
    through unmodified (fail-fast: the inference graph will reject it,
    surfacing the missing-scale condition rather than silently NaN-ing)."""
    loader = _trigger_loader_registration(monkeypatch)
    (tmp_path / "quant_meta.json").write_text(
        json.dumps({"quant_type": DEFAULT_QUANT_TYPE})
    )

    q = torch.zeros(4, 4, dtype=torch.int8)
    seen = list(loader(iter([("orphan.weight", q)]), tmp_path))
    assert len(seen) == 1
    assert seen[0][1].dtype == torch.int8
    assert torch.equal(seen[0][1], q)


def test_loader_falls_back_to_passthrough_on_malformed_sidecar(tmp_path, monkeypatch, caplog):
    """A malformed ``quant_meta.json`` must NOT raise — log + fall back to bf16 pass-through."""
    loader = _trigger_loader_registration(monkeypatch)
    (tmp_path / "quant_meta.json").write_text("{not valid json")

    weight = torch.zeros(2, 2, dtype=torch.bfloat16)
    with caplog.at_level(
        logging.WARNING, logger="worldfoundry_inf._vllm_ascend_plugin"
    ):
        seen = list(loader(iter([("w", weight)]), tmp_path))
    assert len(seen) == 1
    assert torch.equal(seen[0][1], weight)
    assert "Failed to read quant_meta.json" in caplog.text


def test_loader_handles_missing_safetensors_lib_for_scales(tmp_path, monkeypatch):
    """If the safetensors lib is absent, _load_w8a8_scales returns {} and
    the loader still yields (passthrough or unpacked) without crashing."""
    import builtins
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "safetensors" or name.startswith("safetensors."):
            raise ImportError("simulated: safetensors not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    loader = _trigger_loader_registration(monkeypatch)
    (tmp_path / "quant_meta.json").write_text(
        json.dumps({"quant_type": DEFAULT_QUANT_TYPE})
    )

    src = torch.randn(2, 4, dtype=torch.bfloat16)
    packed = pack_w8a8_weight(src)
    # With safetensors hidden, scales dict is empty → int8 passes through.
    seen = list(loader(
        iter([("layer.weight", packed["weight"])]),
        tmp_path,
    ))
    assert seen[0][1].dtype == torch.int8


def test_loader_module_constants_match_spec():
    """ACCEPTANCE_MSE_THRESHOLD is intentionally tight (1e-3)."""
    assert ACCEPTANCE_MSE_THRESHOLD == 1e-3
    assert DEFAULT_QUANT_TYPE == "w8a8"
    expected = {
        "pack_w8a8_weight", "unpack_w8a8_weight",
        "pack_w8a8_activation", "unpack_w8a8_activation",
        "pack_linear", "unpack_linear",
        "load_quant_meta", "is_w8a8_quantized",
        "DEFAULT_QUANT_TYPE", "ACCEPTANCE_MSE_THRESHOLD",
    }
    assert set(quantize_mod.__all__) == expected