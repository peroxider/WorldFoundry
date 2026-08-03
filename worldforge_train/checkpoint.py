"""Checkpoint save/load for WorldForge training.

Persists trainable model state + optimizer state + torch/numpy RNG + step
counter + LoRA signature into a single .pt blob. Atomic via tmp+rename.

On multi-rank FSDP, callers must gate save/load themselves (see W16 plan
section 'Multi-rank save is deferred to Phase-2 W9+'). This module raises
FileNotFoundError / IOError / KeyError / RuntimeError / ValueError per G4.

G3 boundary: only files in worldforge_train/ are modified.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class LoraSignature:
    """Architectural + count signature of a LoRA-wrapped PEFT model.

    Used by validate_lora_signature() on resume. r/alpha/dropout/target_modules
    must match exactly; trainable_params may drift (peft internal counts can
    float across peft versions) and triggers a >1% drift warning.
    """
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    trainable_params: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "trainable_params": self.trainable_params,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoraSignature":
        return cls(
            r=int(d["r"]),
            alpha=int(d["alpha"]),
            dropout=float(d["dropout"]),
            target_modules=tuple(d["target_modules"]),
            trainable_params=int(d["trainable_params"]),
        )


_REQUIRED_KEYS: tuple[str, ...] = (
    "step",
    "model_state",
    "optimizer_state",
    "rng_state_torch",
    "rng_state_numpy",
    "lora_signature",
)


def _is_fsdp_module(model: nn.Module) -> bool:
    """Return True if model is wrapped by FSDP."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:  # pragma: no cover -- fsdp unavailable on very old torch
        return False
    return isinstance(model, FSDP)


def _trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Filter state_dict() to params with requires_grad=True.

    For LoRA: keeps only LoRA adapter params (~10MB for ViT-L).
    For full fine-tune: keeps everything (~2.4GB for ViT-L bf16).

    When ``model`` is FSDP-wrapped, this uses FULL_STATE_DICT so the resulting
    checkpoint is loadable on both single-rank and multi-rank deployments.
    """
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

    if _is_fsdp_module(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
        return {
            k: v.detach().cpu().clone()
            for k, v in state_dict.items()
            if k in trainable_names
        }

    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k in trainable_names
    }


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """torch.save is not atomic for partial writes. Save to .tmp then os.replace."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    *,
    lora_signature: LoraSignature | None = None,
) -> Path:
    """Save trainable model state + optimizer state + RNG + step to .pt.

    On multi-rank FSDP only rank 0 performs the disk write; other ranks return
    the resolved path without writing. Parent directories are created.

    Returns the resolved path.
    """
    path = Path(path)

    import torch.distributed as dist
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if _is_fsdp_module(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        optim_state = optimizer.state_dict()

    payload = {
        "step": int(step),
        "model_state": _trainable_state_dict(model),
        "optimizer_state": optim_state,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "lora_signature": lora_signature.to_dict() if lora_signature else None,
    }

    if world_size > 1 and _is_fsdp_module(model) and rank != 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load .pt, restore trainable params + optimizer + RNG. Returns {step, lora_signature}.

    Raises:
        FileNotFoundError: path doesn't exist.
        IOError: torch.load fails (corrupted file / wrong format).
        KeyError: payload missing required keys.
        RuntimeError: model param not found (LoRA config changed) or shape mismatch.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise IOError(f"failed to load checkpoint {path}: {exc}") from exc

    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise KeyError(f"checkpoint {path} missing keys: {missing}")

    model_state: dict[str, torch.Tensor] = payload["model_state"]

    if _is_fsdp_module(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import StateDictType

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            raise RuntimeError(
                f"FSDP checkpoint missing params: {missing}; "
                f"likely LoRA config changed since save."
            )

        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, payload["optimizer_state"]
        )
        optimizer.load_state_dict(optim_state)
    else:
        named_params = dict(model.named_parameters())
        for k, v in model_state.items():
            if k not in named_params:
                raise RuntimeError(
                    f"checkpoint expects param {k!r} not in current model; "
                    f"likely LoRA config changed since save."
                )
            with torch.no_grad():
                param = named_params[k]
                param.copy_(v.to(param.device))

        optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["rng_state_torch"])
    np.random.set_state(payload["rng_state_numpy"])

    return {
        "step": int(payload["step"]),
        "lora_signature": payload.get("lora_signature"),
    }


def validate_lora_signature(
    ckpt_signature: dict[str, Any] | None,
    current: LoraSignature,
) -> None:
    """Raise ValueError if architectural fields differ; warn on trainable_params drift > 1%."""
    if ckpt_signature is None:
        return
    expected = current.to_dict()
    for key, current_val in expected.items():
        ckpt_val = ckpt_signature.get(key)
        if key == "trainable_params":
            if ckpt_val is None:
                continue
            drift = abs(int(ckpt_val) - current_val) / max(current_val, 1)
            if drift > 0.01:
                warnings.warn(
                    f"checkpoint trainable_params={ckpt_val} differs >1% "
                    f"from current={current_val} (drift {drift:.1%}); "
                    f"may indicate peft version drift.",
                    UserWarning,
                )
            continue
        if ckpt_val != current_val:
            raise ValueError(
                f"checkpoint lora_signature.{key}={ckpt_val!r} does not match "
                f"current cfg={current_val!r}. Use the same LoRA config to resume."
            )
