"""Tests for the W27 v2 action-conditioned inference loader.

Covers: ckpt-not-found, corrupt-file, non-W27-payload rejection, LoRA-ckpt
rejection, happy path assembly (with and without jepa_predictor), target!=online
after EMA load, all-modules-eval/freeze invariants, action_signature
introspection, action_dim consistency check.
"""
from __future__ import annotations

import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

from worldfoundry_train.model.action_jepa_inference import (
    load_action_conditioned_inference_model,
)
from worldfoundry_train.model.action_predictor import ActionConditionedPredictor
from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.jepa_inference import JEPInferenceBundle


# Tiny config: same shape as test_jepa_inference.py for symmetry.
ENC_DIM = 64
PRED_DIM = 32
DEPTH = 2
NUM_HEADS = 2
PAST_GRID = (1, 4, 4)          # N_past = 16
FUTURE_GRID = (1, 4, 4)        # N_future = 16
PAST_FRAMES = 1
FUTURE_FRAMES = 1
ACTION_DIM = 7


def _encoder() -> VJEPA2Model:
    cfg = VJEPA2Config(
        hidden_size=ENC_DIM,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=224,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        frames_per_clip=1,
        drop_path_rate=0.0,
    )
    return VJEPA2Model(cfg)


def _action_predictor() -> ActionConditionedPredictor:
    n_past = PAST_GRID[0] * PAST_GRID[1] * PAST_GRID[2]
    n_future = FUTURE_GRID[0] * FUTURE_GRID[1] * FUTURE_GRID[2]
    return ActionConditionedPredictor(
        past_frames=PAST_FRAMES,
        action_dim=ACTION_DIM,
        num_past_positions=n_past,
        num_future_positions=n_future,
        past_grid=PAST_GRID,
        future_grid=FUTURE_GRID,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )


def _fake_w27_payload(
    encoder: VJEPA2Model,
    *,
    ema_offset: float = 1.0,
) -> dict:
    """Build a synthetic W27 v2 ckpt payload in-memory.

    ``ema_offset`` mirrors the W22 fixture: 1.0 means master == online,
    0.5 means master = 0.5*online (used to verify the EMA load actually
    replaces target weights).
    """
    ac = _action_predictor()

    # VJEPA2Model has no trainable params by default (frozen). Flip every
    # param to requires_grad=True so model_state captures the full state_dict.
    for p in encoder.parameters():
        p.requires_grad = True
    trainable = list(encoder.parameters())

    target = type(encoder)(encoder.config).eval()
    target.load_state_dict(encoder.state_dict())
    for p in target.parameters():
        p.requires_grad = True

    ema = EMAUpdater(
        online_params=trainable,
        target_params=list(target.parameters()),
        momentum_start=0.9,
        momentum_end=0.99,
        schedule="cosine",
        schedule_steps=100,
        update_every=1,
    )
    with torch.no_grad():
        for buf in ema._master:
            buf.mul_(ema_offset)
    ema.step = 42
    ema_state = ema.state_dict()

    return {
        "step": 7,
        "model_state": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
        "ema_state": ema_state,
        "ema_step": 42,
        "action_signature": {
            "pred_dim": PRED_DIM,
            "pred_depth": DEPTH,
            "pred_heads": NUM_HEADS,
            "encoder_dim": ENC_DIM,
            "ema_schedule": "cosine",
            "loss_type": "smooth_l1",
            "action_dim": ACTION_DIM,
            "past_frames": PAST_FRAMES,
            "future_frames": FUTURE_FRAMES,
            "past_grid": list(PAST_GRID),
            "future_grid": list(FUTURE_GRID),
        },
        "action_hyperparams": {
            "momentum_end": 0.99,
            "loss_beta": 1.0,
        },
        "action_predictor_state": ac.state_dict(),
        "lora_signature": None,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": __import__("numpy").random.get_state(),
        "optimizer_state": {},
    }


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_ckpt_raises_file_not_found(tmp_path):
    encoder = _encoder()
    with pytest.raises(FileNotFoundError, match="not found"):
        load_action_conditioned_inference_model(
            tmp_path / "nope.pt", encoder,
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


def test_corrupt_ckpt_raises_io_error(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises(IOError, match="failed to load"):
        load_action_conditioned_inference_model(
            bad, _encoder(),
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


def test_non_w27_payload_missing_action_predictor_state_raises_value_error(tmp_path):
    bad = tmp_path / "no_action.pt"
    torch.save({"model_state": {}, "step": 0}, bad)
    with pytest.raises(ValueError, match="not a W27 v2 action-conditioned ckpt"):
        load_action_conditioned_inference_model(
            bad, _encoder(),
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


def test_non_w27_payload_missing_ema_state_raises_value_error(tmp_path):
    bad = tmp_path / "no_ema.pt"
    torch.save(
        {
            "model_state": {},
            "action_predictor_state": {},
            "action_signature": {},
            "action_hyperparams": {},
            "step": 0,
        },
        bad,
    )
    with pytest.raises(ValueError, match="not a W27 v2 action-conditioned ckpt"):
        load_action_conditioned_inference_model(
            bad, _encoder(),
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


def test_lora_signature_present_raises_not_implemented(tmp_path):
    """LoRA-trained W27 ckpts are deferred (peft build path)."""
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    payload["lora_signature"] = {"r": 4, "alpha": 8}
    bad = tmp_path / "lora.pt"
    torch.save(payload, bad)
    with pytest.raises(NotImplementedError, match="LoRA"):
        load_action_conditioned_inference_model(
            bad, _encoder(),
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


def test_encoder_shape_mismatch_raises_runtime_error(tmp_path):
    """encoder with wrong hidden_size -> load_state_dict RuntimeError."""
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    p = tmp_path / "mismatch.pt"
    torch.save(payload, p)

    cfg_wrong = VJEPA2Config(
        hidden_size=ENC_DIM * 2,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=224,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        frames_per_clip=1,
        drop_path_rate=0.0,
    )
    wrong_encoder = VJEPA2Model(cfg_wrong)
    with pytest.raises(RuntimeError):
        load_action_conditioned_inference_model(
            p, wrong_encoder,
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


def test_action_predictor_signature_mismatch_raises_runtime_error(tmp_path):
    """action_signature claims a depth the predictor_state doesn't have."""
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    payload["action_signature"] = dict(payload["action_signature"])
    payload["action_signature"]["pred_depth"] = 99
    p = tmp_path / "shape.pt"
    torch.save(payload, p)
    with pytest.raises((RuntimeError, ValueError)):
        load_action_conditioned_inference_model(
            p, encoder,
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_w27_full_param_ckpt_returns_bundle_with_action_predictor(tmp_path):
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    p = tmp_path / "ok.pt"
    torch.save(payload, p)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    assert isinstance(bundle, JEPInferenceBundle)
    assert bundle.online is encoder  # in-place load (caller-owned)
    assert bundle.action_predictor is not None
    assert isinstance(bundle.action_predictor, ActionConditionedPredictor)
    assert bundle.action_signature["action_dim"] == ACTION_DIM
    assert bundle.action_signature["encoder_dim"] == ENC_DIM
    assert bundle.ema_step == 42


def test_target_encoder_differs_from_online_when_ema_offset_is_not_one(tmp_path):
    encoder = _encoder()
    payload = _fake_w27_payload(encoder, ema_offset=0.5)
    p = tmp_path / "delta.pt"
    torch.save(payload, p)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )

    online_params = list(bundle.online.parameters())
    target_params = list(bundle.target.parameters())
    assert len(online_params) == len(target_params)
    any_diff = False
    for op, tp in zip(online_params, target_params):
        if not torch.equal(op, tp):
            any_diff = True
            break
    assert any_diff, "target encoder must differ from online after non-trivial EMA load"


def test_target_encoder_equals_online_when_ema_offset_is_one(tmp_path):
    encoder = _encoder()
    payload = _fake_w27_payload(encoder, ema_offset=1.0)
    p = tmp_path / "identity.pt"
    torch.save(payload, p)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    for op, tp in zip(bundle.online.parameters(), bundle.target.parameters()):
        assert torch.equal(op, tp)


def test_action_predictor_weights_match_ckpt_payload(tmp_path):
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    saved_action = {k: v.clone() for k, v in payload["action_predictor_state"].items()}
    p = tmp_path / "act.pt"
    torch.save(payload, p)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    got = bundle.action_predictor.state_dict()
    for k, v in saved_action.items():
        assert k in got, f"missing key in action predictor: {k}"
        assert torch.equal(got[k], v), f"action_predictor.{k} differs from ckpt"


def test_bundle_all_modules_in_eval_mode(tmp_path):
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    p = tmp_path / "eval.pt"
    torch.save(payload, p)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    for name, mod in [
        ("online", bundle.online),
        ("target", bundle.target),
        ("temporal_predictor", bundle.temporal_predictor),
        ("action_predictor", bundle.action_predictor),
    ]:
        assert not mod.training, f"{name} should be eval()"


def test_bundle_all_params_frozen(tmp_path):
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    p = tmp_path / "frozen.pt"
    torch.save(payload, p)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    for name, mod in [
        ("online", bundle.online),
        ("target", bundle.target),
        ("temporal_predictor", bundle.temporal_predictor),
        ("action_predictor", bundle.action_predictor),
    ]:
        for pn, p in mod.named_parameters():
            assert not p.requires_grad, f"{name}.{pn} should be frozen"
        for bn, b in mod.named_buffers():
            assert not b.requires_grad, f"{name}.{bn} buffer should be frozen"


def test_temporal_predictor_warm_starts_from_action_predictor(tmp_path):
    """Without a JEPAPredictor, temporal_predictor is warm-started from the action predictor's trunk."""
    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    p = tmp_path / "warm.pt"
    torch.save(payload, p)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    assert torch.equal(
        bundle.temporal_predictor.proj_in.weight,
        bundle.action_predictor.proj_in.weight,
    )
    assert torch.equal(
        bundle.temporal_predictor.blocks.state_dict()["layers.0.linear1.weight"],
        bundle.action_predictor.blocks.state_dict()["layers.0.linear1.weight"],
    )
    assert torch.equal(
        bundle.temporal_predictor.proj_out.weight,
        bundle.action_predictor.proj_out.weight,
    )


def test_jepa_predictor_passed_uses_w22_warm_start(tmp_path):
    """When a JEPAPredictor is passed, temporal_predictor warm-starts from its trunk."""
    from worldfoundry_train.model.predictor import JEPAPredictor

    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    p = tmp_path / "w22.pt"
    torch.save(payload, p)

    n_past = PAST_GRID[0] * PAST_GRID[1] * PAST_GRID[2]
    jepa = JEPAPredictor(
        num_patches=n_past,
        grid_size=PAST_GRID,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )
    # Make JEPAPredictor weights distinct from the action predictor so the
    # warm-start-from-W22 path is observable.
    with torch.no_grad():
        jepa.proj_in.weight.fill_(0.42)
        jepa.proj_out.weight.fill_(0.42)

    bundle = load_action_conditioned_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        jepa_predictor=jepa,
    )
    assert bundle.jepa_predictor is jepa
    assert torch.equal(
        bundle.temporal_predictor.proj_in.weight,
        jepa.proj_in.weight,
    )
    # And NOT the action predictor's weights (since W22 was passed).
    assert not torch.equal(
        bundle.temporal_predictor.proj_in.weight,
        bundle.action_predictor.proj_in.weight,
    )


def test_jepa_predictor_trunk_mismatch_raises(tmp_path):
    """JEPAPredictor with different trunk dims -> RuntimeError."""
    from worldfoundry_train.model.predictor import JEPAPredictor

    encoder = _encoder()
    payload = _fake_w27_payload(encoder)
    p = tmp_path / "mismatch.pt"
    torch.save(payload, p)

    n_past = PAST_GRID[0] * PAST_GRID[1] * PAST_GRID[2]
    # Mismatched encoder_dim: 128 (jepa) vs 64 (ACP) — triggers the trunk
    # shape mismatch check.
    jepa = JEPAPredictor(
        num_patches=n_past,
        grid_size=PAST_GRID,
        encoder_dim=ENC_DIM * 2,
        predictor_dim=PRED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )
    with pytest.raises(RuntimeError, match="trunk shape"):
        load_action_conditioned_inference_model(
            p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
            jepa_predictor=jepa,
        )
