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


@dataclass(frozen=True)
class JEPASignature:
    """Architectural signature of a JEPA objective (W22, design §7.4).

    All fields are architecture-identifying: a mismatch means the checkpoint's
    predictor/EMA state cannot be loaded into the current objective, so
    validate_jepa_signature() raises ValueError (same semantics as
    validate_lora_signature). Non-architectural hyperparams (EMA momentum
    endpoints, loss beta) are deliberately NOT part of the signature.
    """
    pred_dim: int
    pred_depth: int
    pred_heads: int
    encoder_dim: int
    ema_schedule: str
    loss_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pred_dim": self.pred_dim,
            "pred_depth": self.pred_depth,
            "pred_heads": self.pred_heads,
            "encoder_dim": self.encoder_dim,
            "ema_schedule": self.ema_schedule,
            "loss_type": self.loss_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JEPASignature":
        return cls(
            pred_dim=int(d["pred_dim"]),
            pred_depth=int(d["pred_depth"]),
            pred_heads=int(d["pred_heads"]),
            encoder_dim=int(d["encoder_dim"]),
            ema_schedule=str(d["ema_schedule"]),
            loss_type=str(d["loss_type"]),
        )


def validate_jepa_signature(
    ckpt_signature: dict[str, Any] | None,
    current: JEPASignature,
) -> None:
    """Raise ValueError if any architectural JEPA field differs (§7.4)."""
    if ckpt_signature is None:
        return
    for key, current_val in current.to_dict().items():
        ckpt_val = ckpt_signature.get(key)
        if ckpt_val != current_val:
            raise ValueError(
                f"checkpoint jepa_signature.{key}={ckpt_val!r} does not match "
                f"current cfg={current_val!r}. Use the same JEPA architecture "
                f"to resume."
            )


_REQUIRED_KEYS: tuple[str, ...] = (
    "step",
    "model_state",
    "optimizer_state",
    "rng_state_torch",
    "rng_state_numpy",
    "lora_signature",
)

# W22 (design §7.4): all JEPA keys are optional; old W16 checkpoints without
# them stay loadable when jepa.enabled=false.
_OPTIONAL_JEPA_KEYS: tuple[str, ...] = (
    "predictor_state",
    "ema_state",
    "ema_step",
    "jepa_signature",
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


def _cpu_clone_tree(obj: Any) -> Any:
    """Recursively detach/clone tensors onto CPU (checkpoint portability).

    Used for the optional W22 JEPA payload: the predictor state_dict holds
    references to live params and the fp32 EMA master may live on an
    accelerator; both must be snapshotted before torch.save.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu_clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cpu_clone_tree(v) for v in obj]
    return obj


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    *,
    lora_signature: LoraSignature | None = None,
    jepa_objective: Any | None = None,
) -> Path:
    """Save trainable model state + optimizer state + RNG + step to .pt.

    On multi-rank FSDP only rank 0 performs the disk write; other ranks return
    the resolved path without writing. Parent directories are created.

    ``jepa_objective`` (W22, optional): when given, the JEPA payload keys
    (``predictor_state`` / ``ema_state`` / ``ema_step`` / ``jepa_signature``,
    see ``_OPTIONAL_JEPA_KEYS``) are added. The predictor never participates
    in FSDP (design §8.1), so its state_dict needs no FULL_STATE_DICT path.

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

    if jepa_objective is not None:
        jepa_state = jepa_objective.state_dict()
        payload["predictor_state"] = _cpu_clone_tree(jepa_state["predictor_state"])
        payload["ema_state"] = _cpu_clone_tree(jepa_state["ema_state"])
        payload["ema_step"] = int(jepa_state["ema_step"])
        payload["jepa_signature"] = dict(jepa_state["jepa_signature"])

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
    jepa_objective: Any | None = None,
) -> dict[str, Any]:
    """Load .pt, restore trainable params + optimizer + RNG. Returns {step, lora_signature}.

    ``jepa_objective`` (W22, optional): when given, the checkpoint MUST carry
    the JEPA payload keys — a pre-W22 checkpoint raises KeyError telling the
    caller to disable jepa.enabled or train from scratch (design §7.4). The
    stored ``jepa_signature`` is validated against the live objective
    (architectural mismatch -> ValueError) before any state is restored.

    Raises:
        FileNotFoundError: path doesn't exist.
        IOError: torch.load fails (corrupted file / wrong format).
        KeyError: payload missing required keys (or JEPA keys when a
            jepa_objective is supplied).
        RuntimeError: model param not found (LoRA config changed) or shape mismatch.
        ValueError: jepa_signature architectural mismatch.
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

    if jepa_objective is not None:
        jepa_missing = [k for k in _OPTIONAL_JEPA_KEYS if k not in payload]
        if jepa_missing:
            raise KeyError(
                f"checkpoint {path} does not contain JEPA state "
                f"(missing {jepa_missing}); this checkpoint predates W22 — "
                f"disable jepa.enabled or train from scratch."
            )
        validate_jepa_signature(
            payload["jepa_signature"],
            JEPASignature.from_dict(jepa_objective.signature()),
        )

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

    if jepa_objective is not None:
        jepa_objective.load_state_dict({
            "predictor_state": payload["predictor_state"],
            "ema_state": payload["ema_state"],
        })

    result = {
        "step": int(payload["step"]),
        "lora_signature": payload.get("lora_signature"),
    }
    if "ema_step" in payload:
        result["ema_step"] = int(payload["ema_step"])
    return result


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
