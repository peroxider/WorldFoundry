"""LoRA wrapping for V-JEPA 2 with mandatory sanity check (Global Constraint G4).

Per spec §3.2 + risk row "LoRA silent zero training":
- Before training, MUST verify `model.print_trainable_parameters()` > 0
- If matches=0, raise ValueError; never silently proceed with no trainable params
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from worldfoundry_train.model.module_names import discover_lora_targets

# Default candidates: q/k/v/o projections + MLP linears.
# These will be filtered by discover_lora_targets against the actual model.
DEFAULT_LORA_CANDIDATES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "mlp.fc1", "mlp.fc2",
]


@dataclass
class LoraBuildResult:
    peft_model: object
    trainable_params: int
    total_params: int
    target_modules: list[str]


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