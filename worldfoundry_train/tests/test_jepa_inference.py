"""Tests for the W24 jepa_inference loader.

Covers: ckpt-not-found, corrupt-file, non-W22-payload rejection,
LoRA-ckpt rejection, happy path assembly, target!=online after EMA load,
all-modules-eval/freeze invariants, encoder/predictor shape mismatch.
"""
from __future__ import annotations

import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.jepa_inference import (
    JEPInferenceBundle,
    load_jepa_inference_model,
)
from worldfoundry_train.model.predictor import JEPAPredictor


# Tiny config: encoder_dim=64, predictor_dim=32, depth=2, heads=2.
ENC_DIM = 64
PRED_DIM = 32
DEPTH = 2
NUM_HEADS = 2
PAST_GRID = (1, 4, 4)          # N_past = 16
FUTURE_GRID = (1, 4, 4)        # N_future = 16


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


def _jepa_predictor() -> JEPAPredictor:
    n_past = PAST_GRID[0] * PAST_GRID[1] * PAST_GRID[2]
    return JEPAPredictor(
        num_patches=n_past,
        grid_size=PAST_GRID,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )


def _fake_w22_payload(
    encoder: VJEPA2Model,
    *,
    ema_offset: float = 1.0,
) -> dict:
    """Build a synthetic W22 JEPA ckpt payload in-memory.

    ``ema_offset`` controls the EMA master delta vs the online params
    (1.0 means master == online; 0.5 means master = 0.5*online — used to
    verify EMA load actually replaces target weights).
    """
    predictor = _jepa_predictor()

    # VJEPA2Model has no trainable params by default (frozen). Flip every
    # param to requires_grad=True so model_state captures the full
    # state_dict (mirrors full-FT ckpts).
    for p in encoder.parameters():
        p.requires_grad = True
    trainable = list(encoder.parameters())

    target = type(encoder)(encoder.config).eval()
    # Mirror production training: target starts as a deepcopy of online, so
    # EMAUpdater._master (initialized from target) equals online at t=0.
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
    # Manually set the master to a delta copy so the EMA load actually
    # changes target weights (verifiable in tests).
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
        "jepa_signature": {
            "pred_dim": PRED_DIM,
            "pred_depth": DEPTH,
            "pred_heads": NUM_HEADS,
            "encoder_dim": ENC_DIM,
            "ema_schedule": "cosine",
            "loss_type": "smooth_l1",
        },
        "jepa_hyperparams": {
            "momentum_end": 0.99,
            "loss_beta": 1.0,
        },
        "predictor_state": predictor.state_dict(),
        "lora_signature": None,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": __import__("numpy").random.get_state(),
        "optimizer_state": {},
    }


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_ckpt_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_jepa_inference_model(
            tmp_path / "nope.pt",
            _encoder(),
            past_grid=PAST_GRID,
            future_grid=FUTURE_GRID,
        )


def test_corrupt_ckpt_raises_io_error(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises(IOError, match="failed to load"):
        load_jepa_inference_model(
            bad,
            _encoder(),
            past_grid=PAST_GRID,
            future_grid=FUTURE_GRID,
        )


def test_non_w22_payload_missing_predictor_state_raises_value_error(tmp_path):
    bad = tmp_path / "no_predictor.pt"
    torch.save({"model_state": {}, "step": 0}, bad)
    with pytest.raises(ValueError, match="not a W22 JEPA checkpoint"):
        load_jepa_inference_model(
            bad,
            _encoder(),
            past_grid=PAST_GRID,
            future_grid=FUTURE_GRID,
        )


def test_non_w22_payload_missing_ema_state_raises_value_error(tmp_path):
    bad = tmp_path / "no_ema.pt"
    torch.save(
        {
            "model_state": {},
            "predictor_state": {},
            "jepa_signature": {},
            "jepa_hyperparams": {},
            "step": 0,
        },
        bad,
    )
    with pytest.raises(ValueError, match="not a W22 JEPA checkpoint"):
        load_jepa_inference_model(
            bad,
            _encoder(),
            past_grid=PAST_GRID,
            future_grid=FUTURE_GRID,
        )


def test_lora_signature_present_raises_not_implemented(tmp_path):
    bad = tmp_path / "lora.pt"
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    payload["lora_signature"] = {"r": 4, "alpha": 8}  # simulate LoRA training
    torch.save(payload, bad)
    with pytest.raises(NotImplementedError, match="LoRA"):
        load_jepa_inference_model(
            bad,
            _encoder(),
            past_grid=PAST_GRID,
            future_grid=FUTURE_GRID,
        )


def test_encoder_shape_mismatch_raises_runtime_error(tmp_path):
    """encoder with wrong hidden_size -> load_state_dict RuntimeError."""
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    p = tmp_path / "mismatch.pt"
    torch.save(payload, p)

    # Build a mismatched encoder (different hidden_size).
    cfg_wrong = VJEPA2Config(
        hidden_size=ENC_DIM * 2,  # 128 instead of 64
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
        load_jepa_inference_model(
            p, wrong_encoder,
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_w22_full_param_ckpt_returns_bundle_with_all_components(tmp_path):
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    p = tmp_path / "ok.pt"
    torch.save(payload, p)

    bundle = load_jepa_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    assert isinstance(bundle, JEPInferenceBundle)
    assert bundle.online is encoder  # in-place load (caller-owned)
    assert bundle.jepa_predictor is not None
    assert bundle.temporal_predictor is not None
    assert bundle.ema_step == 42
    assert bundle.jepa_signature["encoder_dim"] == ENC_DIM
    assert bundle.jepa_hyperparams["momentum_end"] == 0.99


def test_target_encoder_differs_from_online_when_ema_offset_is_not_one(tmp_path):
    """EMA-load replaces target weights; with offset != 1 they differ from online."""
    encoder = _encoder()
    payload = _fake_w22_payload(encoder, ema_offset=0.5)  # EMA = 0.5 * online
    p = tmp_path / "delta.pt"
    torch.save(payload, p)

    bundle = load_jepa_inference_model(
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
    """EMA-load is a no-op when master == online; target should equal online."""
    encoder = _encoder()
    payload = _fake_w22_payload(encoder, ema_offset=1.0)
    p = tmp_path / "identity.pt"
    torch.save(payload, p)

    bundle = load_jepa_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    online_params = list(bundle.online.parameters())
    target_params = list(bundle.target.parameters())
    for op, tp in zip(online_params, target_params):
        assert torch.equal(op, tp)


def test_predictor_weights_match_ckpt_payload(tmp_path):
    """Bundle's JEPAPredictor carries the same weights as the saved ckpt."""
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    saved_predictor = {k: v.clone() for k, v in payload["predictor_state"].items()}
    p = tmp_path / "pred.pt"
    torch.save(payload, p)

    bundle = load_jepa_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    got = bundle.jepa_predictor.state_dict()
    for k, v in saved_predictor.items():
        assert k in got, f"missing key in predictor: {k}"
        assert torch.equal(got[k], v), f"predictor.{k} differs from ckpt"


def test_bundle_all_modules_in_eval_mode(tmp_path):
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    p = tmp_path / "eval.pt"
    torch.save(payload, p)

    bundle = load_jepa_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    for name, mod in [
        ("online", bundle.online),
        ("target", bundle.target),
        ("jepa_predictor", bundle.jepa_predictor),
        ("temporal_predictor", bundle.temporal_predictor),
    ]:
        assert not mod.training, f"{name} should be eval()"


def test_bundle_all_params_frozen(tmp_path):
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    p = tmp_path / "frozen.pt"
    torch.save(payload, p)

    bundle = load_jepa_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    for name, mod in [
        ("online", bundle.online),
        ("target", bundle.target),
        ("jepa_predictor", bundle.jepa_predictor),
        ("temporal_predictor", bundle.temporal_predictor),
    ]:
        for pn, p in mod.named_parameters():
            assert not p.requires_grad, f"{name}.{pn} should be frozen"
        for bn, b in mod.named_buffers():
            assert not b.requires_grad, f"{name}.{bn} buffer should be frozen"


def test_temporal_predictor_warm_starts_from_loaded_jepa(tmp_path):
    """temporal_predictor's trunk must equal jepa_predictor's trunk (warm-start)."""
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    p = tmp_path / "warm.pt"
    torch.save(payload, p)

    bundle = load_jepa_inference_model(
        p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    assert torch.equal(
        bundle.temporal_predictor.proj_in.weight,
        bundle.jepa_predictor.proj_in.weight,
    )
    assert torch.equal(
        bundle.temporal_predictor.blocks.state_dict()["layers.0.linear1.weight"],
        bundle.jepa_predictor.blocks.state_dict()["layers.0.linear1.weight"],
    )
    assert torch.equal(
        bundle.temporal_predictor.proj_out.weight,
        bundle.jepa_predictor.proj_out.weight,
    )


def test_predictor_state_shape_mismatch_raises(tmp_path):
    """jepa_signature claims a dim the predictor_state doesn't carry."""
    encoder = _encoder()
    payload = _fake_w22_payload(encoder)
    payload["jepa_signature"] = dict(payload["jepa_signature"])
    payload["jepa_signature"]["pred_depth"] = 99  # claim 99 layers
    p = tmp_path / "shape.pt"
    torch.save(payload, p)

    with pytest.raises((RuntimeError, ValueError)):
        load_jepa_inference_model(
            p, encoder, past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )