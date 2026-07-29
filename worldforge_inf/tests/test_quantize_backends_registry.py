"""Registry + factory tests for quantize_backends subpackage."""
from __future__ import annotations

import pytest

from worldforge_inf.compile.quantize_backends import (
    QuantBackend,
    backend_available,
    get_backend,
    register,
)


def test_register_and_get_backend_roundtrip():
    """A class decorated with @register('foo') is retrievable via get_backend('foo')."""
    @register("foo")
    class FakeBackend:
        def quantize(self, model_path, output_path, calibration_dataset):
            return output_path
    backend = get_backend("foo")
    assert isinstance(backend, FakeBackend)


def test_get_backend_unknown_raises_value_error():
    """get_backend('nope') raises ValueError with available types listed."""
    with pytest.raises(ValueError, match="No backend registered for quant_type='nope'"):
        get_backend("nope")


def test_backend_available_true_when_registered():
    """backend_available returns (True, None) for a registered type."""
    @register("bar")
    class BarBackend:
        def quantize(self, model_path, output_path, calibration_dataset):
            return output_path
    available, hint = backend_available("bar")
    assert available is True
    assert hint is None


def test_protocol_quantize_signature():
    """QuantBackend Protocol declares the expected quantize signature."""
    # Static check via runtime introspection
    import inspect
    sig = inspect.signature(QuantBackend.quantize)
    params = list(sig.parameters.keys())
    assert params == ["self", "model_path", "output_path", "calibration_dataset"]