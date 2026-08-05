
import pytest

from worldfoundry_common.exceptions import OMCompileError
from worldfoundry_inf.compile.mindie_sd import compile_to_om


def test_compile_to_om_writes_om_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    out = tmp_path / "model.om"
    result = compile_to_om(
        src, out, target_hardware="Atlas 800 A2 / 910B", require_quantized=False
    )
    assert result.exists()
    assert result.suffix == ".om"


def test_compile_to_om_raises_on_missing_quant_meta(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    out = tmp_path / "model.om"
    with pytest.raises(OMCompileError):
        compile_to_om(src, out, require_quantized=True)
