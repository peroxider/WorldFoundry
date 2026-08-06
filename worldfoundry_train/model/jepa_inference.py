"""Inference-side loader for W22 JEPA checkpoints (W24).

Loads online encoder + target encoder (from EMA) + JEPAPredictor +
TemporalPredictor from a W22 training ``.pt`` file. The result is a
frozen :class:`JEPInferenceBundle` that the inference worker can use
directly to answer ``mode=temporal`` requests on ``/v1/world/predict``.

Scope (W24 v1)
--------------
- Full-parameter training checkpoints (``lora_signature is None``).
- Caller-supplied ``online_encoder`` whose architecture matches the
  training-time encoder; the loader applies ``payload["model_state"]``
  in-place and then deepcopy's the loaded model to seed the target
  encoder.

Out of scope (deferred)
-----------------------
- LoRA-trained checkpoints (``lora_signature`` set): inferring the base
  HF model requires a config-side reference; deferred to a follow-up
  that takes ``hf_repo + lora_signature`` and rebuilds PEFT.
- HF ``from_pretrained`` discovery: the loader does not sniff
  ``cfg["model"]["hf_repo"]`` from the ckpt (the training side never
  wrote it). Callers pass a pre-constructed encoder.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.predictor import JEPAPredictor
from worldfoundry_train.model.temporal_predictor import TemporalPredictor
from worldfoundry_train.model.action_predictor import ActionConditionedPredictor

__all__ = ["JEPInferenceBundle", "load_jepa_inference_model"]


@dataclass(frozen=True)
class JEPInferenceBundle:
    """Inference-ready bundle loaded from a W22 (or W27) JEPA ckpt.

    All ``nn.Module`` fields are ``eval()`` mode with
    ``requires_grad=False`` on every parameter and buffer. ``target``
    weights reflect the EMA snapshot at ``ema_step`` (not the latest
    online params), so ``predict_temporal`` / ``predict_action_conditioned``
    confidence comparisons against ``target`` carry the JEPA semantics.

    The ``action_predictor`` / ``action_signature`` fields are populated
    only by the W27 v2 loader (``load_action_conditioned_inference_model``);
    W22-only ckpts leave them as ``None`` so the W24 surface is byte-compatible.
    """

    online: nn.Module
    target: nn.Module
    jepa_predictor: JEPAPredictor
    temporal_predictor: TemporalPredictor
    ema_step: int
    jepa_signature: dict
    jepa_hyperparams: dict
    action_predictor: ActionConditionedPredictor | None = None
    action_signature: dict | None = None


# W22 fingerprints: a ckpt is a W22 JEPA ckpt iff all four keys are present.
_REQUIRED_JEPA_KEYS = (
    "predictor_state",
    "ema_state",
    "jepa_signature",
    "jepa_hyperparams",
)


def _validate_w22_payload(payload: dict) -> None:
    """Raise ValueError if ``payload`` is not a W22 JEPA checkpoint."""
    missing = [k for k in _REQUIRED_JEPA_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"checkpoint is not a W22 JEPA checkpoint; missing keys: {missing}. "
            f"(The presence of predictor_state + ema_state + jepa_signature "
            f"+ jepa_hyperparams is the W22 fingerprint.)"
        )
    # LoRA-trained ckpts: defer (need base model + peft build path).
    if payload.get("lora_signature") is not None:
        raise NotImplementedError(
            "W24 v1 does not yet load LoRA-trained W22 JEPA checkpoints "
            "(the base HF model is not recorded in the .pt). Pass a "
            "full-parameter ckpt or extend jepa_inference to accept "
            "hf_repo + lora_signature and use peft."
        )


def _freeze(model: nn.Module) -> None:
    """Set eval() and freeze all params + buffers."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for b in model.buffers():
        b.requires_grad = False


def load_jepa_inference_model(
    ckpt_path: str | Path,
    online_encoder: nn.Module,
    *,
    past_grid: tuple[int, int, int],
    future_grid: tuple[int, int, int],
    map_location: str | torch.device = "cpu",
) -> JEPInferenceBundle:
    """Load a W22 JEPA ckpt into online/target encoders + predictor(s).

    Args:
        ckpt_path: path to the ``.pt`` produced by
            ``worldfoundry_train.checkpoint.save_checkpoint`` with
            ``jepa_objective`` supplied.
        online_encoder: a freshly-constructed ``nn.Module`` with the same
            architecture as the trained encoder. Weights are populated
            from ``payload["model_state"]`` (``strict=False`` to allow
            unrelated buffers; param shapes must still match).
        past_grid: ``(K, Hg, Wg)`` of the past frame grid; must match
            the training-time grid that produced this ckpt (the
            ``JEPAPredictor`` ``pos_embed`` is keyed on this grid).
        future_grid: ``(H, Hg', Wg')`` of the future frame grid.
        map_location: ``torch.device`` or str forwarded to ``torch.load``.

    Returns:
        A frozen :class:`JEPInferenceBundle`. All modules are ``eval()``
        mode with frozen params.

    Raises:
        FileNotFoundError: ``ckpt_path`` doesn't exist.
        IOError: ``torch.load`` fails (corrupted file / wrong format).
        ValueError: payload missing JEPA fingerprint keys (not a W22
            JEPA checkpoint).
        NotImplementedError: LoRA-trained ckpt (``lora_signature`` set).
        RuntimeError: ``model_state`` / ``predictor_state`` have shapes
            incompatible with the supplied modules.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    try:
        payload = torch.load(
            ckpt_path, map_location=map_location, weights_only=False
        )
    except Exception as exc:
        raise IOError(
            f"failed to load checkpoint {ckpt_path}: {exc}"
        ) from exc

    _validate_w22_payload(payload)

    # 1) online encoder: caller-provided; populate model_state in place.
    online = online_encoder
    # strict=False: _trainable_state_dict skips non-trainable buffers (e.g.
    # BN running stats), so the encoder may have extra buffers that the
    # ckpt doesn't carry. Param shapes must match (otherwise RuntimeError).
    online.load_state_dict(payload["model_state"], strict=False)

    # 2) target encoder: deepcopy + EMAUpdater.load_state_dict(ema_state).
    # EMAUpdater.load_state_dict copies ema_state["master"] into the
    # fp32 _master buffers, then re-syncs target_params from _master;
    # so after the call target has the EMA snapshot, online has the
    # loaded online weights (they will differ whenever the EMA hasn't
    # saturated at momentum_end == 1.0).
    target = copy.deepcopy(online)
    # Constructor defaults are placeholders; ema.load_state_dict below
    # overwrites every EMA-config field from the saved state.
    ema = EMAUpdater(
        online_params=list(online.parameters()),
        target_params=list(target.parameters()),
        momentum_start=0.996,
        momentum_end=1.0,
        schedule="cosine",
        schedule_steps=1,
        update_every=1,
    )
    ema.load_state_dict(payload["ema_state"])

    # 3) JEPAPredictor: construct from signature, load weights.
    sig = payload["jepa_signature"]
    n_past = past_grid[0] * past_grid[1] * past_grid[2]
    jepa = JEPAPredictor(
        num_patches=n_past,
        grid_size=past_grid,
        encoder_dim=int(sig["encoder_dim"]),
        predictor_dim=int(sig["pred_dim"]),
        depth=int(sig["pred_depth"]),
        num_heads=int(sig["pred_heads"]),
    )
    jepa.load_state_dict(payload["predictor_state"])

    # 4) TemporalPredictor: warm-start from the loaded JEPAPredictor.
    n_future = future_grid[0] * future_grid[1] * future_grid[2]
    temporal = TemporalPredictor.from_w22_jepa_predictor(
        jepa,
        num_past_positions=n_past,
        num_future_positions=n_future,
        past_grid=past_grid,
        future_grid=future_grid,
    )

    # Inference mode everywhere.
    _freeze(online)
    _freeze(target)
    _freeze(jepa)
    _freeze(temporal)

    return JEPInferenceBundle(
        online=online,
        target=target,
        jepa_predictor=jepa,
        temporal_predictor=temporal,
        ema_step=int(payload.get("ema_step", 0)),
        jepa_signature=dict(sig),
        jepa_hyperparams=dict(payload["jepa_hyperparams"]),
    )