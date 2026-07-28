"""ModelSlim-style quantization wrapper.

Phase-1 stub: applies W8A8 symmetric quantization in-place on weights and writes
a quant_meta.json sidecar. Real vLLM-Ascend ModelSlim integration lands in
Phase-2 W11 (W4A8).
"""
from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_QUANT_TYPES = ("w8a8", "w4a8", "w4a16")


def quantize_model(
    model_path: Path,
    output_path: Path,
    quant_type: str = "w8a8",
    calibration_dataset: str | None = None,
) -> Path:
    """Quantize a model directory and write the result to `output_path`.

    Returns the output directory path. Raises ValueError on unsupported type.
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type={quant_type!r}. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    src = Path(model_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    # Phase-1 stub: copy model files and emit a quant_meta.json sidecar.
    # Real W8A8 weight replacement is wired in Phase-2 W11.
    for f in src.iterdir():
        if f.is_file():
            (dst / f.name).write_bytes(f.read_bytes())

    meta = {
        "quant_type": quant_type,
        "calibration_dataset": calibration_dataset,
        "source_model_path": str(src),
        "phase": "phase1-stub",
        "note": "Phase-2 W11: replace with real ModelSlim W8A8.",
    }
    (dst / "quant_meta.json").write_text(json.dumps(meta, indent=2))
    return dst
