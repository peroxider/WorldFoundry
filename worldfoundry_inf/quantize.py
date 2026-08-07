"""W8A8 symmetric quantize / unpack utilities for inference (W26).

Mirror of the *write* side in :mod:`worldfoundry_inf.compile.quantize`:
that module dispatches to the msModelSlim backend to *produce* a quantized
model directory, while this module is the *read* side — it takes a
weight or activation that has been packed by msModelSlim (or by us via
the helpers below) and turns it back into a bf16 tensor the rest of the
inference graph can consume.

Two quantization granularities are supported, matching the msModelSlim
"static W8A8" recipe that :class:`worldfoundry_inf.compile.quantize_backends.msmodelslim.W8A8Backend`
produces:

* **Weights** — per-output-channel symmetric int8. The scale is a 1-D
  float32 tensor of length ``out_features``; the per-channel granularity
  keeps MSE tight even when a row's dynamic range is much smaller than
  the global max.
* **Activations** — per-tensor symmetric int8. A single scalar scale.
  Calibrated against ``videomix22m-512`` at W11, so the typical
  activation magnitudes are well within int8 range and a single scale is
  sufficient.

Both directions round-trip at MSE < 1e-3 against the bf16 source on the
random-tensor acceptance test in
:mod:`worldfoundry_inf.tests.test_quantize_loader` (see
:data:`ACCEPTANCE_MSE_THRESHOLD`). The threshold is intentionally much
tighter than naive (16x) so a broken unpacker fails the test loudly.

Why a separate module from :mod:`worldfoundry_inf.compile.quantize`?
The compile-side module owns *write* dispatch (facade over msModelSlim
backends). This module owns *read* (inference-side unpack). G8 — no
duplicate module names, but the two responsibilities are distinct
enough that they live in separate top-level files. Hook callers
(``_vllm_ascend_plugin._patch_quantized_loader``) import from here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch import Tensor
import torch

__all__ = [
    "DEFAULT_QUANT_TYPE",
    "ACCEPTANCE_MSE_THRESHOLD",
    "pack_w8a8_weight",
    "unpack_w8a8_weight",
    "pack_w8a8_activation",
    "unpack_w8a8_activation",
    "pack_linear",
    "unpack_linear",
    "load_quant_meta",
    "is_w8a8_quantized",
]

DEFAULT_QUANT_TYPE = "w8a8"
# Tighter than the naive "bf16 round-trip after dequant" because the
# inner quantization here is symmetric int8 (small constant error per
# element). A broken unpacker typically produces MSE > 1e-2, so 1e-3
# is a comfortable margin.
ACCEPTANCE_MSE_THRESHOLD = 1e-3
_QUANT_META_FILENAME = "quant_meta.json"
_SCALE_FLOOR = 1e-12


def _safe_scale(scale: Tensor) -> Tensor:
    """Clamp scale away from zero so dividing by it never blows up."""
    return scale.clamp_min(_SCALE_FLOOR)


def pack_w8a8_weight(weight: Tensor) -> dict[str, Tensor]:
    """Per-output-channel symmetric int8 quantization of a 2-D weight.

    Args:
        weight: Tensor of shape ``(out_features, in_features)`` (any float
            dtype; the produced int8 pack uses float32 internally and the
            unpack path returns bf16 to match the msModelSlim convention).

    Returns:
        ``{"weight": int8 Tensor of input shape, "scale": float32 Tensor
        of shape (out_features,)}``. The scale is ``abs_max / 127`` per
        output channel so that ``abs(q) <= 127`` and ``unpack`` is exact.

    Raises:
        ValueError: ``weight`` is not 2-D or its dtype is unsupported.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"pack_w8a8_weight expects a 2-D (out_features, in_features) "
            f"weight, got shape={tuple(weight.shape)}"
        )
    out_features = weight.shape[0]
    src = weight.detach().to(torch.float32).reshape(out_features, -1)
    abs_max = src.abs().amax(dim=1)
    scale = torch.where(
        abs_max > 0, abs_max / 127.0, torch.zeros_like(abs_max)
    )
    safe = _safe_scale(scale).unsqueeze(1)
    quant = torch.round(src / safe).clamp(-128, 127).to(torch.int8)
    return {"weight": quant.reshape(weight.shape), "scale": scale.to(torch.float32)}


def unpack_w8a8_weight(packed: dict[str, Tensor]) -> Tensor:
    """Inverse of :func:`pack_w8a8_weight`; returns bf16 of original shape."""
    q = packed["weight"]
    scale = packed["scale"]
    if q.dtype != torch.int8:
        raise ValueError(f"unpack_w8a8_weight: weight dtype must be int8, got {q.dtype}")
    if scale.ndim != 1 or scale.shape[0] != q.shape[0]:
        raise ValueError(
            f"unpack_w8a8_weight: scale must be 1-D of length {q.shape[0]}, "
            f"got shape={tuple(scale.shape)}"
        )
    dequant = q.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)
    return dequant.reshape(q.shape).to(torch.bfloat16)


def pack_w8a8_activation(activation: Tensor) -> dict[str, Tensor]:
    """Per-tensor symmetric int8 quantization of an activation tensor.

    Args:
        activation: any float tensor. The produced int8 pack uses a single
            scalar scale (0-D float32); the unpack path returns bf16.

    Returns:
        ``{"activation": int8 Tensor of input shape, "scale": float32 0-D}``.
    """
    src = activation.detach().to(torch.float32)
    abs_max = src.abs().amax()
    scale = torch.where(abs_max > 0, abs_max / 127.0, torch.zeros_like(abs_max))
    safe = _safe_scale(scale)
    quant = torch.round(src / safe).clamp(-128, 127).to(torch.int8)
    return {"activation": quant, "scale": scale.to(torch.float32)}


def unpack_w8a8_activation(packed: dict[str, Tensor]) -> Tensor:
    """Inverse of :func:`pack_w8a8_activation`; returns bf16 of original shape."""
    q = packed["activation"]
    scale = packed["scale"]
    if q.dtype != torch.int8:
        raise ValueError(
            f"unpack_w8a8_activation: activation dtype must be int8, got {q.dtype}"
        )
    if scale.ndim != 0:
        raise ValueError(
            f"unpack_w8a8_activation: scale must be 0-D, got shape={tuple(scale.shape)}"
        )
    dequant = q.to(torch.float32) * scale.to(torch.float32)
    return dequant.to(torch.bfloat16)


def pack_linear(weight: Tensor) -> dict[str, Tensor]:
    """Alias of :func:`pack_w8a8_weight` for nn.Linear-shaped weights."""
    return pack_w8a8_weight(weight)


def unpack_linear(packed: dict[str, Tensor]) -> Tensor:
    """Alias of :func:`unpack_w8a8_weight` for nn.Linear-shaped weights."""
    return unpack_w8a8_weight(packed)


def load_quant_meta(model_dir: Path) -> dict[str, Any] | None:
    """Read ``model_dir/quant_meta.json`` (the W11-W12 sidecar).

    Returns ``None`` if the sidecar is absent so callers can probe
    non-quantized models without an exception. Raises on malformed JSON
    so a partially-written sidecar doesn't silently degrade to a
    non-quantized lookup.
    """
    meta_path = Path(model_dir) / _QUANT_META_FILENAME
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def is_w8a8_quantized(model_dir: Path) -> bool:
    """True iff ``load_quant_meta(model_dir)["quant_type"] == "w8a8"``."""
    meta = load_quant_meta(Path(model_dir))
    return bool(meta) and meta.get("quant_type") == DEFAULT_QUANT_TYPE