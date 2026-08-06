"""Inference-side loader for W27 V-JEPA 2-AC action-conditioned ckpts (W27 v2).

Loads online encoder + target encoder (from EMA) + ActionConditionedPredictor
from a W27 training ``.pt`` file. The result is a frozen
:class:`JEPInferenceBundle` (with ``action_predictor`` populated) that the
inference worker can use directly to answer ``mode=action`` requests on
``/v1/world/predict``.

Scope (W27 v2)
--------------
- Full-parameter training checkpoints (``lora_signature is None``).
- Caller-supplied ``online_encoder`` whose architecture matches the
  training-time encoder; the loader applies ``payload["model_state"]``
  in-place and then deepcopy's the loaded model to seed the target
  encoder.
- The ``temporal_predictor`` field of the returned bundle is warm-started
  from the action predictor's trunk (``proj_in / blocks / proj_out``) so
  that the W22 ``mode=temporal`` path still works after the trunk has been
  replaced by an action-trained checkpoint.

Out of scope (deferred)
-----------------------
- LoRA-trained checkpoints (``lora_signature`` set): inferring the base
  HF model requires a config-side reference; deferred to a follow-up
  that takes ``hf_repo + lora_signature`` and rebuilds PEFT.
- A dedicated W27 training objective (mirroring
  :class:`worldfoundry_train.temporal_objective.TemporalObjective`): the
  ckpt format described here is forward-compatible with such an objective
  landing later; for now the loader accepts any ``.pt`` that carries the
  W27 fingerprint (action_predictor_state + action_signature +
  action_hyperparams + model_state + ema_state).
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from worldfoundry_train.model.action_predictor import ActionConditionedPredictor
from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.jepa_inference import (
    JEPInferenceBundle,
    _freeze,
)
from worldfoundry_train.model.temporal_predictor import TemporalPredictor

__all__ = ["load_action_conditioned_inference_model"]


# W27 v2 fingerprints: a ckpt is a W27 v2 ckpt iff all five keys are present.
# The W22 fingerprint keys (``jepa_signature / jepa_hyperparams /
# predictor_state``) are NOT required — a W27 ckpt is action-focused and
# does not need to carry a W22 JEPA predictor alongside.
_REQUIRED_W27_KEYS = (
    "action_predictor_state",
    "action_signature",
    "action_hyperparams",
    "model_state",
    "ema_state",
)


def _validate_w27_payload(payload: dict) -> None:
    """Raise ValueError if ``payload`` is not a W27 v2 action-conditioned ckpt."""
    missing = [k for k in _REQUIRED_W27_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"checkpoint is not a W27 v2 action-conditioned ckpt; missing "
            f"keys: {missing}. (The presence of action_predictor_state + "
            f"action_signature + action_hyperparams + model_state + "
            f"ema_state is the W27 v2 fingerprint.)"
        )
    # LoRA-trained ckpts: defer (need base model + peft build path).
    if payload.get("lora_signature") is not None:
        raise NotImplementedError(
            "W27 v2 does not yet load LoRA-trained W27 action-conditioned "
            "checkpoints (the base HF model is not recorded in the .pt). "
            "Pass a full-parameter ckpt or extend action_jepa_inference to "
            "accept hf_repo + lora_signature and use peft."
        )


def load_action_conditioned_inference_model(
    ckpt_path: str | Path,
    online_encoder: nn.Module,
    *,
    past_grid: tuple[int, int, int],
    future_grid: tuple[int, int, int],
    map_location: str | torch.device = "cpu",
    jepa_predictor: nn.Module | None = None,
) -> JEPInferenceBundle:
    """Load a W27 v2 action-conditioned ckpt into online/target encoders + ACP.

    Args:
        ckpt_path: path to the ``.pt`` produced by the W27 training entry
            (forward-compatible with a future
            ``ActionConditionedObjective.state_dict`` — the only contract
            is the W27 fingerprint keys).
        online_encoder: a freshly-constructed ``nn.Module`` with the same
            architecture as the trained encoder. Weights are populated
            from ``payload["model_state"]`` (``strict=False`` to allow
            unrelated buffers; param shapes must still match).
        past_grid: ``(K, Hg, Wg)`` of the past frame grid; must match
            the training-time grid that produced this ckpt.
        future_grid: ``(H, Hg', Wg')`` of the future frame grid.
        map_location: ``torch.device`` or str forwarded to ``torch.load``.
        jepa_predictor: optional W22 :class:`JEPAPredictor`. If provided,
            the returned bundle's ``jepa_predictor`` is this module and its
            trunk is used to warm-start the :class:`TemporalPredictor`
            (preferred when a W22-trained JEPAPredictor is available —
            it preserves the W22 JEPA signal). If ``None``, the
            :class:`TemporalPredictor` is warm-started from the action
            predictor's trunk (still valid; the action predictor shares
            the trunk shape with W22).

    Returns:
        A frozen :class:`JEPInferenceBundle` with ``action_predictor`` and
        ``action_signature`` populated. All modules are ``eval()`` mode
        with frozen params.

    Raises:
        FileNotFoundError: ``ckpt_path`` doesn't exist.
        IOError: ``torch.load`` fails (corrupted file / wrong format).
        ValueError: payload missing W27 fingerprint keys (not a W27 ckpt).
        NotImplementedError: LoRA-trained ckpt (``lora_signature`` set).
        RuntimeError: ``model_state`` / ``action_predictor_state`` have
            shapes incompatible with the supplied modules.
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

    _validate_w27_payload(payload)

    # 1) online encoder: caller-provided; populate model_state in place.
    online = online_encoder
    online.load_state_dict(payload["model_state"], strict=False)

    # 2) target encoder: deepcopy + EMAUpdater.load_state_dict(ema_state).
    target = copy.deepcopy(online)
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

    # 3) ActionConditionedPredictor: construct from action_signature, load weights.
    sig = payload["action_signature"]
    n_past = past_grid[0] * past_grid[1] * past_grid[2]
    n_future = future_grid[0] * future_grid[1] * future_grid[2]
    action_dim = int(sig["action_dim"])
    ac = ActionConditionedPredictor(
        past_frames=past_grid[0],
        action_dim=action_dim,
        num_past_positions=n_past,
        num_future_positions=n_future,
        past_grid=past_grid,
        future_grid=future_grid,
        encoder_dim=int(sig["encoder_dim"]),
        predictor_dim=int(sig["pred_dim"]),
        depth=int(sig["pred_depth"]),
        num_heads=int(sig["pred_heads"]),
    )
    ac.load_state_dict(payload["action_predictor_state"])

    # 4) TemporalPredictor: warm-start from the JEPAPredictor (preferred) or
    # from the action predictor's trunk (fallback). The latter preserves
    # trunk-level information when only a W27 ckpt is available.
    if jepa_predictor is not None:
        # The trunk dimensions must match the action predictor's; the
        # JEPAPredictor and ActionConditionedPredictor share the trunk
        # layout, so this is a structural check rather than a stretch.
        if (
            jepa_predictor.encoder_dim != ac.encoder_dim
            or jepa_predictor.predictor_dim != ac.predictor_dim
            or len(jepa_predictor.blocks.layers) != len(ac.blocks.layers)
        ):
            raise RuntimeError(
                f"jepa_predictor trunk shape "
                f"(enc={jepa_predictor.encoder_dim}, "
                f"pred={jepa_predictor.predictor_dim}, "
                f"depth={len(jepa_predictor.blocks.layers)}) does not "
                f"match action_predictor trunk shape "
                f"(enc={ac.encoder_dim}, pred={ac.predictor_dim}, "
                f"depth={len(ac.blocks.layers)}); warm-start rejected"
            )
        temporal = TemporalPredictor.from_w22_jepa_predictor(
            jepa_predictor,
            num_past_positions=n_past,
            num_future_positions=n_future,
            past_grid=past_grid,
            future_grid=future_grid,
        )
        # Carry the W22 JEPA signature fields when known (so downstream
        # consumers can introspect the bundle without re-reading the
        # W22 ckpt).
        jepa_signature = dict(payload.get("jepa_signature", {}))
        jepa_hyperparams = dict(payload.get("jepa_hyperparams", {}))
    else:
        temporal = TemporalPredictor.from_action_conditioned_predictor(
            ac,
            num_past_positions=n_past,
            num_future_positions=n_future,
            past_grid=past_grid,
            future_grid=future_grid,
        )
        jepa_signature = {}
        jepa_hyperparams = {}

    # Inference mode everywhere.
    _freeze(online)
    _freeze(target)
    _freeze(ac)
    _freeze(temporal)

    return JEPInferenceBundle(
        online=online,
        target=target,
        # If a JEPAPredictor was passed, use it; otherwise build a fresh
        # one matching the action-predictor shape so the bundle stays
        # structurally consistent (W22 consumers ignore action_predictor).
        jepa_predictor=jepa_predictor if jepa_predictor is not None else _placeholder_jepa(ac),
        temporal_predictor=temporal,
        ema_step=int(payload.get("ema_step", 0)),
        jepa_signature=jepa_signature,
        jepa_hyperparams=jepa_hyperparams,
        action_predictor=ac,
        action_signature=dict(sig),
    )


def _placeholder_jepa(ac: ActionConditionedPredictor) -> nn.Module:
    """Build a fresh :class:`JEPAPredictor` mirroring the action predictor's trunk.

    Used when the caller does not supply a W22-trained JEPAPredictor. The
    placeholder is structurally identical to the action predictor's trunk
    (encoder/predictor dim, depth, heads) but uses the W22 grid shape — it
    is NOT warm-started from any ckpt.
    """
    from worldfoundry_train.model.predictor import JEPAPredictor

    n_past = ac.num_past_positions
    jepa = JEPAPredictor(
        num_patches=n_past,
        grid_size=ac.past_grid,
        encoder_dim=ac.encoder_dim,
        predictor_dim=ac.predictor_dim,
        depth=len(ac.blocks.layers),
        num_heads=ac.num_heads,
    )
    return jepa
