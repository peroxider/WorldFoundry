from pathlib import Path

from worldforge_inf.compile.quantize import quantize_model


def test_quantize_model_creates_output_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_bytes(b"fake-weights")

    out_dir = tmp_path / "quantized"
    result = quantize_model(src, out_dir, quant_type="w8a8")

    assert result.exists()
    assert result.is_dir()
    assert (result / "quant_meta.json").exists()


def test_quantize_model_rejects_unknown_type(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    import pytest
    with pytest.raises(ValueError, match="Unsupported quant_type"):
        quantize_model(src, out, quant_type="w3a3")
