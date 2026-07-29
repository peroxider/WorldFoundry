"""Verify quantize_model facade dispatches to the registered backend (G3 + G6)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worldforge_inf.compile.quantize import quantize_model


def test_facade_dispatches_w8a8_to_backend(tmp_path):
    """quantize_model('w8a8') calls W8A8Backend.quantize with the same args."""
    src = tmp_path / "bf16"
    src.mkdir()
    out = tmp_path / "w8a8"

    fake_backend = MagicMock()
    fake_backend.quantize.return_value = out

    with patch(
        "worldforge_inf.compile.quantize.get_backend", return_value=fake_backend
    ) as mock_get:
        with patch(
            "worldforge_inf.compile.quantize.backend_available",
            return_value=(True, None),
        ):
            result = quantize_model(src, out, quant_type="w8a8",
                                    calibration_dataset="videomix22m-512")

    assert result == out
    mock_get.assert_called_once_with("w8a8")
    fake_backend.quantize.assert_called_once_with(
        src, out, "videomix22m-512"
    )


def test_facade_unknown_quant_type_raises():
    """quantize_model(quant_type='w3a3') raises ValueError (Phase-1 preserved)."""
    with pytest.raises(ValueError, match="Unsupported quant_type"):
        quantize_model(
            Path("/fake"), Path("/fake"),
            quant_type="w3a3",
        )


def test_facade_backend_unavailable_raises_import_error():
    """When backend_available returns (False, hint), facade raises ImportError."""
    src = Path("/fake/bf16")
    out = Path("/fake/w8a8")
    with patch(
        "worldforge_inf.compile.quantize.backend_available",
        return_value=(False, "pip install worldforge[quant]"),
    ):
        with pytest.raises(ImportError, match="pip install worldforge\\[quant\\]"):
            quantize_model(src, out, quant_type="w8a8")
