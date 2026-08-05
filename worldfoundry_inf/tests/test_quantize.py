from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worldfoundry_inf.compile.quantize import quantize_model


def test_quantize_model_creates_output_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_bytes(b"fake-weights")

    out_dir = tmp_path / "quantized"

    # W11 facade dispatches to the registered backend; mock it so this
    # contract test doesn't depend on msModelSlim or real model artifacts
    # (G3 exception per task-7-brief.md Step 6).
    fake_backend = MagicMock()
    fake_backend.quantize.return_value = out_dir
    with patch(
        "worldfoundry_inf.compile.quantize.get_backend",
        return_value=fake_backend,
    ):
        with patch(
            "worldfoundry_inf.compile.quantize.backend_available",
            return_value=(True, None),
        ):
            result = quantize_model(src, out_dir, quant_type="w8a8")

    assert result == out_dir
    fake_backend.quantize.assert_called_once()


def test_quantize_model_rejects_unknown_type(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsupported quant_type"):
        quantize_model(src, out, quant_type="w3a3")
