"""Optional quantization step inserted before training.

Reads ``cfg["quant"]`` (if present). When populated, runs the W11-W12
``quantize_model`` facade on ``cfg["quant"]["source_dir"]``, writes to
``cfg["quant"]["output_dir"]``, and mutates ``cfg["model"]["hf_repo"]``
to the output dir so the existing trainer loads the quantized weights
via its single ``AutoModel.from_pretrained`` call.

Absent ``quant:`` block -> no-op (existing W14-W16 behavior preserved).

This module is the seam between the W11-W12 quantization stack and the
W14-W16 training pipeline. It does NOT modify the facade or the trainer.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["maybe_quantize_first"]


def _validate_quant_block(block: dict) -> None:
    """Raise ValueError on missing/wrong-type keys. Never silent."""
    if not isinstance(block, dict):
        raise ValueError(f"cfg['quant'] must be a dict, got {type(block).__name__}")
    for key in ("type", "source_dir", "output_dir"):
        if key not in block:
            raise ValueError(f"cfg['quant'] missing required key: {key!r}")
    if not isinstance(block["type"], str):
        raise ValueError(
            f"cfg['quant']['type'] must be str, got {type(block['type']).__name__}"
        )
    src = Path(block["source_dir"])
    if not src.is_dir():
        raise ValueError(
            f"cfg['quant']['source_dir']={src} is not an existing directory. "
            "Quantize requires a local saved-HF model dir (use huggingface-cli "
            "download or save_pretrained() first)."
        )
    out = Path(block["output_dir"])
    if out == src:
        raise ValueError(
            f"cfg['quant']['output_dir']={out} equals source_dir; cannot "
            "overwrite the baseline with quantized weights."
        )


def maybe_quantize_first(cfg: dict) -> dict:
    """If cfg has a 'quant' block, run quantize_model and mutate hf_repo.

    Returns the (possibly-mutated) cfg. The mutation is in-place AND
    returned for convenience so callers can write
    ``cfg = maybe_quantize_first(cfg)``.
    """
    block = cfg.get("quant")
    if not block:
        return cfg
    _validate_quant_block(block)

    from worldforge_inf.compile.quantize import quantize_model

    out_path = quantize_model(
        model_path=Path(block["source_dir"]),
        output_path=Path(block["output_dir"]),
        quant_type=block["type"],
        calibration_dataset=block.get("calibration_dataset"),
    )
    cfg["model"]["hf_repo"] = str(out_path)
    return cfg


if __name__ == "__main__":
    raise SystemExit(0)  # library module; CLI lives in worldforge_train.train