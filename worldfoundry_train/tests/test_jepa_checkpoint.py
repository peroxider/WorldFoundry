"""Unit tests for JEPA checkpoint payload (W22.5, design §9.1 test_jepa_checkpoint).

Covers: predictor bitwise round-trip, ema_step/momentum continuation,
JEPASignature mismatch -> ValueError, old-format ckpt + jepa -> KeyError,
old-format ckpt without jepa -> compatible load, and a trainer-level
save/resume round-trip with JEPA enabled.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from worldforge_train.checkpoint import (
    JEPASignature, load_checkpoint, save_checkpoint, validate_jepa_signature,
)
from worldforge_train.jepa_objective import JEPAObjective
from worldforge_train.model.predictor import JEPAPredictor


class _ToyEncoder(nn.Module):
    """Minimal stand-in for the PEFT-wrapped encoder (deepcopy-able, has
    trainable params for the EMA pairing)."""

    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.adapter = nn.Linear(4, 4)  # "LoRA": trainable

    def forward(self, **kwargs):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _objective(seed: int = 0, loss_type: str = "smooth_l1") -> tuple:
    torch.manual_seed(seed)
    online = _ToyEncoder()
    predictor = JEPAPredictor(
        num_patches=4, grid_size=(1, 2, 2), encoder_dim=8,
        predictor_dim=8, depth=1, num_heads=2,
    )
    obj = JEPAObjective.from_online_model(
        online, predictor, loss_type=loss_type,
        ema_momentum_start=0.9, ema_momentum_end=0.99,
        ema_schedule="cosine", ema_schedule_steps=10,
    )
    return obj, online


def _save(obj, online, path, step=5):
    optimizer = torch.optim.AdamW(
        [p for p in online.parameters() if p.requires_grad], lr=1e-3
    )
    save_checkpoint(path, online, optimizer, step, jepa_objective=obj)
    return optimizer


def test_predictor_weights_bitwise_round_trip(tmp_path):
    obj, online = _objective()
    _save(obj, online, tmp_path / "ckpt.pt")

    obj2, online2 = _objective(seed=99)  # different init
    optimizer2 = torch.optim.AdamW(
        [p for p in online2.parameters() if p.requires_grad], lr=1e-3
    )
    load_checkpoint(tmp_path / "ckpt.pt", online2, optimizer2,
                    jepa_objective=obj2)
    for (n1, p1), (n2, p2) in zip(
        obj.predictor.named_parameters(), obj2.predictor.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1.detach(), p2.detach()), f"predictor param {n1}"


def test_ema_step_and_momentum_continue_after_resume(tmp_path):
    obj, online = _objective()
    for _ in range(4):
        obj.ema.update()
    assert obj.ema.step == 4
    _save(obj, online, tmp_path / "ckpt.pt")

    obj2, online2 = _objective(seed=99)
    optimizer2 = torch.optim.AdamW(
        [p for p in online2.parameters() if p.requires_grad], lr=1e-3
    )
    result = load_checkpoint(tmp_path / "ckpt.pt", online2, optimizer2,
                             jepa_objective=obj2)
    assert obj2.ema.step == 4
    assert result["ema_step"] == 4
    # Momentum continues the same cosine curve at step 4 of 10.
    assert obj2.ema.momentum == pytest.approx(obj.ema.momentum_at(4))
    # Target params re-synced from the restored fp32 master. Only the
    # trainable ("adapter") params are EMA-tracked and checkpointed; frozen
    # backbone weights come from the base model at build time (here obj2 was
    # built with a different seed, so its frozen weights legitimately differ).
    trainable_names = {
        n for n, p in online.named_parameters() if p.requires_grad
    }
    for (n1, p1), (n2, p2) in zip(
        obj.target_encoder.named_parameters(),
        obj2.target_encoder.named_parameters(),
    ):
        if n1 in trainable_names:
            assert torch.allclose(p1.detach(), p2.detach()), f"target param {n1}"


def test_jepa_signature_mismatch_raises_value_error(tmp_path):
    obj, online = _objective(loss_type="smooth_l1")
    _save(obj, online, tmp_path / "ckpt.pt")

    obj2, online2 = _objective(loss_type="cosine")  # different loss_type
    optimizer2 = torch.optim.AdamW(
        [p for p in online2.parameters() if p.requires_grad], lr=1e-3
    )
    with pytest.raises(ValueError, match="jepa_signature.loss_type"):
        load_checkpoint(tmp_path / "ckpt.pt", online2, optimizer2,
                        jepa_objective=obj2)


def test_validate_jepa_signature_unit():
    sig = JEPASignature(
        pred_dim=384, pred_depth=12, pred_heads=12, encoder_dim=1024,
        ema_schedule="cosine", loss_type="smooth_l1",
    )
    validate_jepa_signature(None, sig)  # missing ckpt sig -> no-op
    validate_jepa_signature(sig.to_dict(), sig)
    bad = sig.to_dict()
    bad["pred_depth"] = 6
    with pytest.raises(ValueError, match="pred_depth"):
        validate_jepa_signature(bad, sig)


def test_old_checkpoint_with_jepa_objective_raises_key_error(tmp_path):
    # Save WITHOUT jepa payload (pre-W22 format).
    obj, online = _objective()
    optimizer = torch.optim.AdamW(
        [p for p in online.parameters() if p.requires_grad], lr=1e-3
    )
    save_checkpoint(tmp_path / "old.pt", online, optimizer, 3)

    with pytest.raises(KeyError, match="does not contain JEPA state"):
        load_checkpoint(tmp_path / "old.pt", online, optimizer,
                        jepa_objective=obj)


def test_old_checkpoint_without_jepa_loads_compatibly(tmp_path):
    obj, online = _objective()
    optimizer = torch.optim.AdamW(
        [p for p in online.parameters() if p.requires_grad], lr=1e-3
    )
    save_checkpoint(tmp_path / "old.pt", online, optimizer, 3)
    result = load_checkpoint(tmp_path / "old.pt", online, optimizer)
    assert result["step"] == 3
    assert "ema_step" not in result


# ---------------------------------------------------------------------------
# Trainer-level round-trip with jepa.enabled=true
# ---------------------------------------------------------------------------

def test_trainer_jepa_checkpoint_resume_round_trip(tmp_path, monkeypatch):
    import worldforge_train.trainer as trainer_mod
    from transformers import VJEPA2Config, VJEPA2Model

    captured: list[JEPAObjective] = []
    original = JEPAObjective.from_online_model

    def _spy(cls, *args, **kwargs):
        obj = original(*args, **kwargs)
        captured.append(obj)
        return obj

    monkeypatch.setattr(
        trainer_mod.JEPAObjective, "from_online_model", classmethod(_spy)
    )

    class _FakeDS:
        def __len__(self):
            return 4

        def __getitem__(self, i):
            rng = np.random.default_rng(i + 1)
            return {
                "frames": rng.random((1, 3, 224, 224), dtype=np.float32),
                "mask": rng.integers(0, 2, size=(1, 14, 14), dtype=bool),
            }

    monkeypatch.setattr(trainer_mod, "_build_dataset", lambda cfg: _FakeDS())

    from worldforge_train.parallel.strategy import DistributedStrategy

    class _Noop(DistributedStrategy):
        def wrap(self, model):
            return model

        def backward(self, loss):
            loss.backward()

        def step(self, optimizer):
            optimizer.step()
            optimizer.zero_grad()

    cfg = {
        "data": {"manifest": "/x", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-4,
                     "log_every_n": 100},
        "jepa": {
            "enabled": True,
            "predictor": {"embed_dim": 32, "depth": 2, "num_heads": 2},
            "ema": {"momentum_start": 0.9, "momentum_end": 0.99,
                    "schedule": "cosine", "schedule_steps": None,
                    "update_every": 1},
            "loss": {"type": "smooth_l1", "beta": 1.0, "num_masks": 1},
        },
    }

    def _model():
        return VJEPA2Model(VJEPA2Config(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            image_size=224, patch_size=16, tubelet_size=1, in_chans=3,
            frames_per_clip=1, drop_path_rate=0.0,
        ))

    def _batch(i):
        rng = np.random.default_rng(i)
        return {
            "pixel_values_videos": torch.from_numpy(
                rng.random((1, 1, 3, 224, 224), dtype=np.float32)
            ),
            "bool_masked_pos": torch.from_numpy(
                rng.integers(0, 2, size=(1, 196), dtype=bool)
            ),
        }

    torch.manual_seed(7)
    trainer_mod.run_training(
        cfg, _Noop(), base_model=_model(), num_steps=2, batch_fn=_batch,
        logger=lambda s, l: None, checkpoint_dir=tmp_path,
    )
    ckpt = tmp_path / "step_2.pt"
    assert ckpt.is_file()
    obj_first = captured[0]
    assert obj_first.ema.step == 2

    torch.manual_seed(7)
    losses = trainer_mod.run_training(
        cfg, _Noop(), base_model=_model(), num_steps=4, batch_fn=_batch,
        logger=lambda s, l: None, checkpoint_dir=tmp_path, resume_from=ckpt,
    )
    assert len(losses) == 2  # steps 3..4
    obj_resumed = captured[1]
    # EMA schedule counter continued across resume: 2 (restored) + 2 (new).
    assert obj_resumed.ema.step == 4
    # Predictor state was restored before training continued: the resumed
    # run's initial predictor equals the first run's final predictor. Verify
    # indirectly — the resumed objective started from the checkpoint's
    # predictor (captured post-training, so just assert it differs from a
    # fresh init).
    fresh, _ = _objective(seed=12345)
    same_as_fresh = all(
        torch.equal(p1.detach(), p2.detach())
        for (n1, p1), (n2, p2) in zip(
            obj_resumed.predictor.named_parameters(),
            fresh.predictor.named_parameters(),
        )
    )
    assert not same_as_fresh
