"""Tiny-EMA + tiny-checkpoint helpers for the CPU render-train-inference demo.

The full :mod:`worldfoundry_train.ema.EMAUpdater` ships the
cosine-scheduled, FSDP-aware fp32-master EMA that the W22 design
requires — that class is overkill for a 450k-param ViT-Tiny running
on a single CPU thread. This module provides a minimal stand-in:

- :class:`TinyEMAUpdater` — constant-momentum EMA with a single
  ``update(target, online)`` method that lerps ``target`` toward
  ``online``. No fp32 master copies, no schedule, no FSDP.
- :func:`save_tiny_checkpoint` / :func:`load_tiny_checkpoint` — a
  checkpoint format mirroring
  :func:`worldfoundry_train.checkpoint.save_checkpoint`'s shape
  (atomic tmp+rename via :func:`torch.save` + :func:`os.replace`,
  identical .pt container) but with the tiny encoder/predictor/decoder
  payloads instead of the W22 JEPA bundle.

The orchestrator (RTI-3) consumes this surface; no other code does.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


class TinyEMAUpdater:
    """Constant-momentum EMA: ``target <- m * target + (1 - m) * online``.

    No fp32 master copies (single dtype, single device, single rank).
    Call :meth:`update` once per optimizer step with the freshly-updated
    online module and the EMA target module.
    """

    def __init__(self, *, momentum: float = 0.99) -> None:
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"momentum must be in [0, 1]; got {momentum}")
        self.momentum = momentum

    @torch.no_grad()
    def update(self, target: nn.Module, online: nn.Module) -> None:
        for t_p, o_p in zip(target.parameters(), online.parameters()):
            t_p.mul_(self.momentum).add_(o_p.detach(), alpha=1.0 - self.momentum)
        # Buffer copy (none on the tiny encoder, but keep the contract).
        for t_b, o_b in zip(target.buffers(), online.buffers()):
            t_b.copy_(o_b.detach())


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """torch.save is not atomic for partial writes. Save to .tmp then os.replace."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_tiny_checkpoint(
    path: str | Path,
    *,
    encoder: nn.Module,
    predictor: nn.Module,
    decoder: nn.Module,
    optimizer: Optimizer,
    step: int,
    target: nn.Module | None = None,
) -> Path:
    """Save encoder + predictor + decoder + optimizer + step + config signature.

    Mirrors :func:`worldfoundry_train.checkpoint.save_checkpoint`'s
    atomic write path but with a tiny-specific payload (no JEPA bundle
    since the tiny stack does not need the W22 signature/hyperparams
    validation).

    If ``target`` is provided, its state is also persisted under the
    ``target_state`` key. This is the EMA target encoder used by the
    JEPA loss; without it, callers that resume training would have to
    rebuild the target from scratch (losing the EMA's slow-moving
    bias). The orchestrator passes ``target`` so the on-disk
    checkpoint is a complete (encoder, target) state dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "encoder_state": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
        "predictor_state": {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()},
        "decoder_state": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
        "optimizer_state": {k: v for k, v in optimizer.state_dict().items()},
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
    }
    if target is not None:
        payload["target_state"] = {
            k: v.detach().cpu().clone() for k, v in target.state_dict().items()
        }
    # Encoder config signature is the architectural fingerprint used to
    # rebuild the module at load time. The orchestrator consumes this via
    # `bundle["encoder_config"]` to instantiate TinyVJEPA2Encoder with
    # the right shape, then load_state_dict() into it.
    if hasattr(encoder, "config"):
        payload["encoder_config"] = encoder.config.signature()  # type: ignore[attr-defined]
    if hasattr(predictor, "encoder_config"):
        payload["predictor_config"] = predictor.encoder_config.signature()  # type: ignore[attr-defined]
    _atomic_save(payload, path)
    return path


def load_tiny_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load the tiny checkpoint and return a bundle with rebuilt modules.

    Returns ``{"encoder", "predictor", "decoder", "step", "encoder_config"}``.
    The caller is responsible for calling ``.eval()`` and moving modules
    to the desired device (we move them on the way out so the caller
    doesn't have to track which kwargs they passed).
    """
    from worldfoundry_train.model.tiny_jepa import (
        TinyActionConditionedPredictor,
        TinyEncoderConfig,
        TinyJEPAPredictor,
        TinyPixelDecoder,
        TinyVJEPA2Encoder,
    )

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"tiny checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise IOError(f"failed to load tiny checkpoint {path}: {exc}") from exc

    for key in ("encoder_state", "predictor_state", "decoder_state", "encoder_config"):
        if key not in payload:
            raise KeyError(f"tiny checkpoint {path} missing key: {key}")

    cfg_dict = dict(payload["encoder_config"])
    # W2.18+: the HF encoder path writes a ``encoder_kind == "hf_vjepa2"``
    # marker into ``encoder_config.signature()``. Branch here so the
    # LoRA-wrapped HF state can be rebuilt via the shim factory. Tiny
    # checkpoints (and any pre-W2.18 checkpoint that omits the marker)
    # keep the v2.x tiny path.
    if cfg_dict.get("encoder_kind") == "hf_vjepa2":
        from worldfoundry_train.model.hf_vjepa_shim import (
            HfJEPAPredictorShim,
            HfPixelDecoderShim,
            HfVJEPA2EncoderShim,
        )
        # Reconstruct the random-init factory kwargs from the saved
        # signature. Defaults match the train-phase defaults
        # (``scripts/render_train_rl_cpu.py:_build_stack`` HF branch).
        hf_kwargs = {
            "image_size": int(cfg_dict["image_size"]),
            "patch_size": int(cfg_dict["patch_size"]),
            "in_channels": int(cfg_dict["in_channels"]),
            "hidden_size": int(cfg_dict["hidden_size"]),
            "num_hidden_layers": int(cfg_dict["depth"]),
            "num_attention_heads": int(cfg_dict["num_attention_heads"]),
            "tubelet_size": int(cfg_dict["tubelet_size"]),
            "frames_per_clip": int(cfg_dict["frames_per_clip"]),
            "lora_r": int(cfg_dict.get("lora_r", 4)),
            "lora_alpha": int(cfg_dict.get("lora_alpha", 8)),
            "lora_dropout": float(cfg_dict.get("lora_dropout", 0.0)),
        }
        online = HfVJEPA2EncoderShim.from_random_init(**hf_kwargs)
        # The LoRA-wrapped VJEPA2Model state_dict includes the HF
        # model's internal predictor submodule (12 transformer layers
        # with their own LoRA adapters); these keys don't match a fresh
        # instance, so the encoder load uses ``strict=False`` and the
        # unmatched keys are dropped. Predictor + decoder are local
        # modules and load strictly.
        missing_unexpected = online.load_state_dict(
            payload["encoder_state"], strict=False,
        )
        # Sanity-check the G4 invariant: we should have loaded at least
        # the LoRA adapters from the saved state. "missing" must be
        # empty (LoRA adapters were initialised in the factory and are
        # therefore present in both saved + fresh instances).
        # The "unexpected" check is intentionally relaxed: VJEPA2Model
        # with ``skip_predictor=True`` may or may not expose its
        # internal predictor submodule's parameters in
        # ``state_dict()`` (depends on HF's lazy-init policy), so we
        # can't guarantee any specific "unexpected" keys.
        if missing_unexpected.missing_keys:
            raise RuntimeError(
                f"HF encoder state load had unexpected missing keys: "
                f"{missing_unexpected.missing_keys[:5]}..."
            )
        # Predictor: re-instantiate via the shim with the same grid
        # geometry the train phase used. ``grid_size`` is derived from
        # ``image_size // patch_size`` (square assumption matches the
        # orchestrator's HF branch).
        Hg = int(cfg_dict["image_size"]) // int(cfg_dict["patch_size"])
        Wg = Hg
        N = int(cfg_dict["frames_per_clip"]) * Hg * Wg
        predictor = HfJEPAPredictorShim(
            num_patches=N,
            grid_size=(int(cfg_dict["frames_per_clip"]), Hg, Wg),
            encoder_dim=int(cfg_dict["embed_dim"]),
            predictor_dim=64, depth=2, num_heads=2,
        )
        predictor.load_state_dict(payload["predictor_state"])
        decoder = HfPixelDecoderShim(online.config)
        decoder.load_state_dict(payload["decoder_state"])

        online = online.to(device)
        predictor = predictor.to(device)
        decoder = decoder.to(device)

        bundle = {
            "encoder": online,
            "predictor": predictor,
            "decoder": decoder,
            "step": int(payload.get("step", 0)),
            "encoder_config": cfg_dict,
        }

        if "target_state" in payload:
            target = HfVJEPA2EncoderShim.from_random_init(**hf_kwargs)
            target.load_state_dict(payload["target_state"], strict=False)
            target = target.to(device)
            target.eval()
            for p in target.parameters():
                p.requires_grad_(False)
            bundle["target"] = target

        return bundle

    cfg = TinyEncoderConfig(**{k: cfg_dict[k] for k in (
        "image_size", "patch_size", "in_channels", "embed_dim",
        "depth", "num_heads", "mlp_ratio",
    ) if k in cfg_dict})
    encoder = TinyVJEPA2Encoder(cfg)
    # Action-conditioned variant stores `_action_proj.{weight,bias}`.
    # Infer action_dim from that weight's input dim so the loader
    # doesn't need an external schema flag.
    is_action_conditioned = "_action_proj.weight" in payload["predictor_state"]
    if is_action_conditioned:
        action_dim = int(payload["predictor_state"]["_action_proj.weight"].shape[1])
        predictor = TinyActionConditionedPredictor(cfg, action_dim=action_dim)
    else:
        predictor = TinyJEPAPredictor(cfg)
    decoder = TinyPixelDecoder(cfg)
    encoder.load_state_dict(payload["encoder_state"])
    predictor.load_state_dict(payload["predictor_state"])
    decoder.load_state_dict(payload["decoder_state"])

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    decoder = decoder.to(device)

    bundle = {
        "encoder": encoder,
        "predictor": predictor,
        "decoder": decoder,
        "step": int(payload.get("step", 0)),
        "encoder_config": cfg_dict,
    }

    # If the checkpoint carries the EMA target_state (v2.9+), rebuild
    # the target encoder and load it. This is the EMA's slow-moving
    # bias accumulated across training steps; without restoring it,
    # the JEPA loss would use a target that drifts toward the
    # just-initialized online encoder.
    if "target_state" in payload:
        target = TinyVJEPA2Encoder(cfg)
        target.load_state_dict(payload["target_state"])
        target = target.to(device)
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        bundle["target"] = target

    return bundle


__all__ = [
    "TinyEMAUpdater",
    "save_tiny_checkpoint",
    "load_tiny_checkpoint",
]