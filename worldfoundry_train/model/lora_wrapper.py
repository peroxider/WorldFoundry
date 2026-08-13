"""LoRA wrapping for V-JEPA 2 with mandatory sanity check (Global Constraint G4).

Per spec §3.2 + risk row "LoRA silent zero training":
- Before training, MUST verify `model.print_trainable_parameters()` > 0
- If matches=0, raise ValueError; never silently proceed with no trainable params

peft 0.18.1 compatibility:
- peft's torchao dispatcher auto-detects `AffineQuantizedTensor` weights
  and wraps them with `TorchaoLoraLinear`. That class requires a
  `get_apply_tensor_subclass` kwarg which peft reads via
  ``operator.attrgetter("hf_quantizer.quantization_config.
  get_apply_tensor_subclass")(self.model)``.
- When a model is hand-quantized via ``torchao.quantize_(...)`` (no
  ``from_pretrained`` path), ``hf_quantizer`` is absent and the
  ``AttributeError`` is silently swallowed, leaving the kwarg missing
  → ``TypeError`` in ``TorchaoLoraLinear.__init__``.
- We attach a minimal shim only when torchao-tensorized weights are
  present. The callable is only invoked by ``merge()``/``unmerge()``,
  so the shim is dormant during forward/backward training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from worldfoundry_train.model.module_names import discover_lora_targets

# Default candidates: q/k/v/o projections + MLP linears.
# HF VJEPA2's attention layers name the projections `query`/`key`/`value`/`proj`
# (not `q_proj`/`k_proj`/...), so we list both conventions — candidates that
# don't match the actual model are silently filtered by discover_lora_targets.
DEFAULT_LORA_CANDIDATES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query", "key", "value", "proj",
    "mlp.fc1", "mlp.fc2",
]


@dataclass
class LoraBuildResult:
    peft_model: object
    trainable_params: int
    total_params: int
    target_modules: list[str]


def _has_torchao_quantized_weight(model: nn.Module) -> bool:
    """Return True iff any nn.Linear weight is a torchao AffineQuantizedTensor.

    Detects hand-quantized models. The HF-from_pretrained path goes
    through ``hf_quantizer`` which is already set on the model and
    doesn't need our shim.
    """
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:  # torchao not installed → no torchao quant path
        return False
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, AffineQuantizedTensor):
            return True
    return False


def _attach_torchao_quantizer_shim(model: nn.Module) -> bool:
    """Attach a minimal hf_quantizer chain so peft 0.18.1's torchao
    dispatcher can construct ``TorchaoLoraLinear``.

    Returns True if a shim was attached (i.e., the model has
    torchao-tensorized weights and was missing the chain).
    Idempotent: if the chain is already present, does nothing.
    """
    if not _has_torchao_quantized_weight(model):
        return False
    if hasattr(model, "hf_quantizer") and hasattr(
        model.hf_quantizer, "quantization_config"
    ) and hasattr(
        model.hf_quantizer.quantization_config, "get_apply_tensor_subclass"
    ):
        return False  # chain already present (HF from_pretrained path)

    class _ShimQuantConfig:
        """Minimal quantization_config surrogate.

        ``get_apply_tensor_subclass`` is only invoked by
        ``TorchaoLoraLinear.merge``/``unmerge``; for forward/backward
        LoRA training it just needs to exist to satisfy __init__'s
        keyword-only signature. Returning a fresh ``Int8WeightOnlyConfig``
        makes merge() work correctly if a caller ever invokes it.
        """

        @staticmethod
        def get_apply_tensor_subclass():
            from torchao.quantization import Int8WeightOnlyConfig
            return Int8WeightOnlyConfig()

    class _ShimHfQuantizer:
        quantization_config = _ShimQuantConfig

    model.hf_quantizer = _ShimHfQuantizer()
    return True


def build_lora_model(
    base_model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    candidate_targets: list[str] | None = None,
) -> LoraBuildResult:
    """Wrap a V-JEPA 2 base model with PEFT LoRA.

    Raises ValueError if LoRA matches zero modules (per G4).
    """
    from peft import LoraConfig, get_peft_model  # imported lazily to keep tests light

    _attach_torchao_quantizer_shim(base_model)

    candidates = candidate_targets or DEFAULT_LORA_CANDIDATES
    targets = discover_lora_targets(base_model, candidates)
    if not targets:
        raise ValueError(
            f"LoRA targets matched 0 modules. Candidates={candidates}. "
            f"Inspect the model's named_modules() and update candidates. "
            f"Per spec §3.2 + Global Constraint G4."
        )

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias=bias,
    )
    peft_model = get_peft_model(base_model, peft_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    if trainable_params == 0:
        raise ValueError(
            f"LoRA config produced 0 trainable params (trainable_params={trainable_params}). "
            f"Aborting per Global Constraint G4. targets={targets}"
        )

    return LoraBuildResult(
        peft_model=peft_model,
        trainable_params=trainable_params,
        total_params=total_params,
        target_modules=targets,
    )