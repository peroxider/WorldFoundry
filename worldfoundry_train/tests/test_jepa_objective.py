"""Unit tests for JEPAObjective (W22.3, design §9.1 test_jepa_loss +
test_training_loop_jepa smoke) and the bool_masked_pos empirical resolution
(design §11 risk #1).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldforge_train.jepa_objective import JEPAObjective, jepa_loss
from worldforge_train.model.ema import EMAUpdater
from worldforge_train.model.predictor import JEPAPredictor

ENC_DIM = 32
GRID = (1, 4, 4)
N = 16


def _predictor() -> JEPAPredictor:
    return JEPAPredictor(
        num_patches=N, grid_size=GRID, encoder_dim=ENC_DIM,
        predictor_dim=16, depth=2, num_heads=2,
    )


# ---------------------------------------------------------------------------
# jepa_loss numeric branches (design §9.1 test_jepa_loss ①②)
# ---------------------------------------------------------------------------

def test_jepa_loss_smooth_l1_numeric():
    pred = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    target = torch.zeros(2, 2)
    # beta=1: |d|=0 -> 0 ; |d|=2 -> linear: 2 - 0.5 = 1.5. sum=1.5, M=2.
    loss = jepa_loss(pred, target, "smooth_l1", beta=1.0)
    assert loss.item() == pytest.approx(1.5 / 2)


def test_jepa_loss_l1_numeric_and_m_normalization():
    pred = torch.ones(4, 3)  # M=4 tokens, dim=3, every |diff| = 1
    target = torch.zeros(4, 3)
    loss = jepa_loss(pred, target, "l1")
    # sum = 12, normalized by M=4 -> 3.0 (NOT mean over elements = 1.0).
    assert loss.item() == pytest.approx(3.0)


def test_jepa_loss_cosine_numeric():
    pred = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = jepa_loss(pred, target, "cosine")
    # cos = 1 and 0 -> mean(1 - cos) = 0.5
    assert loss.item() == pytest.approx(0.5)


def test_jepa_loss_rejects_empty_and_unknown_type():
    with pytest.raises(ValueError, match="M=0"):
        jepa_loss(torch.zeros(0, 4), torch.zeros(0, 4), "l1")
    with pytest.raises(ValueError, match="unknown loss_type"):
        jepa_loss(torch.ones(1, 2), torch.ones(1, 2), "mse")


# ---------------------------------------------------------------------------
# Objective wiring with mocks (design §9.1 test_jepa_loss ③④)
# ---------------------------------------------------------------------------

class _RecordingEncoder(nn.Module):
    """Mock encoder: records call kwargs, returns fixed features."""

    def __init__(self, feats: torch.Tensor):
        super().__init__()
        self._feats = feats
        self.calls: list[dict] = []
        self.dummy = nn.Parameter(torch.zeros(1))  # keeps it a real Module

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=self._feats)


def _objective_with_mocks(loss_type="smooth_l1"):
    torch.manual_seed(0)
    online_feats = torch.randn(1, N, ENC_DIM)
    target_feats = torch.randn(1, N, ENC_DIM)
    online = _RecordingEncoder(online_feats)
    target = _RecordingEncoder(target_feats)
    p_on = nn.Parameter(torch.randn(2, 2))
    p_tg = nn.Parameter(torch.randn(2, 2))
    ema = EMAUpdater([p_on], [p_tg], momentum_start=0.5, momentum_end=0.5,
                     schedule="constant")
    obj = JEPAObjective(_predictor(), target, ema, loss_type=loss_type)
    return obj, online, target


def test_loss_fn_call_contract_online_and_target():
    obj, online, target = _objective_with_mocks()
    mask = torch.rand(1, N) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, N) < 0.5
    batch = {
        "pixel_values_videos": torch.randn(1, 1, 3, 32, 32),
        "bool_masked_pos": mask,
    }
    loss = obj.loss_fn(online, batch)
    assert torch.isfinite(loss)

    assert len(online.calls) == 1 and len(target.calls) == 1
    for calls in (online.calls, target.calls):
        # bool_masked_pos is NEVER forwarded to the HF encoder (it is
        # silently ignored there — see empirical test below); masking lives
        # inside JEPAPredictor. skip_predictor avoids dead compute in HF's
        # built-in frozen predictor.
        assert "bool_masked_pos" not in calls[0]
        assert calls[0].get("skip_predictor") is True
        assert "pixel_values_videos" in calls[0]


def test_target_features_are_stop_gradient():
    obj, online, target = _objective_with_mocks()
    assert not target._feats.requires_grad
    mask = torch.rand(1, N) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, N) < 0.5
    h_ctx = online._feats.clone().requires_grad_(True)
    h_tgt = target._feats
    loss = obj._loss_for_masks(h_ctx, h_tgt, [mask])
    assert not h_tgt.requires_grad
    loss.backward()
    assert h_ctx.grad is not None


def test_loss_fn_rejects_grad_carrying_target():
    obj, _, _ = _objective_with_mocks()
    h_ctx = torch.randn(1, N, ENC_DIM, requires_grad=True)
    h_tgt = torch.randn(1, N, ENC_DIM, requires_grad=True)
    mask = torch.ones(1, N, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="stop-gradient"):
        obj._loss_for_masks(h_ctx, h_tgt, [mask])


def test_multi_mask_path_averages_over_k():
    obj, _, _ = _objective_with_mocks()
    torch.manual_seed(1)
    h_ctx = torch.randn(1, N, ENC_DIM, requires_grad=True)
    h_tgt = torch.randn(1, N, ENC_DIM)
    m1 = torch.rand(1, N) < 0.5
    m2 = torch.rand(1, N) < 0.3
    while int(m1.sum()) == 0 or int(m2.sum()) == 0:
        m1 = torch.rand(1, N) < 0.5
        m2 = torch.rand(1, N) < 0.3
    loss_k2 = obj._loss_for_masks(h_ctx, h_tgt, [m1, m2])
    l1 = obj._loss_for_masks(h_ctx, h_tgt, [m1])
    l2 = obj._loss_for_masks(h_ctx, h_tgt, [m2])
    assert loss_k2.item() == pytest.approx((l1.item() + l2.item()) / 2, rel=1e-5)


def test_all_false_mask_yields_graph_attached_zero_loss():
    obj, online, target = _objective_with_mocks()
    batch = {
        "pixel_values_videos": torch.randn(1, 1, 3, 32, 32),
        "bool_masked_pos": torch.zeros(1, N, dtype=torch.bool),
    }
    loss = obj.loss_fn(online, batch)
    assert loss.item() == 0.0
    assert loss.requires_grad  # backward must stay well-defined
    loss.backward()


# ---------------------------------------------------------------------------
# bool_masked_pos empirical resolution (design §11 risk #1)
# ---------------------------------------------------------------------------

def test_hf_encoder_ignores_bool_masked_pos():
    """Empirical: transformers 5.1.0 VJEPA2Model swallows bool_masked_pos.

    ``VJEPA2Model.forward`` has no such parameter; the kwarg falls into
    ``**kwargs`` and is discarded. Encoder output is bit-identical with no
    mask, and with two different masks. Masking therefore happens exclusively
    inside JEPAPredictor (late masking); the question "True = masked vs True
    = predict" is moot at the HF level and NO mask flip is needed in the
    predictor input construction (design §11 open question #1: RESOLVED).
    """
    from transformers import VJEPA2Config, VJEPA2Model

    cfg = VJEPA2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=224, patch_size=16, tubelet_size=1, in_chans=3,
        frames_per_clip=1, drop_path_rate=0.0,
    )
    model = VJEPA2Model(cfg).eval()
    assert "bool_masked_pos" not in inspect.signature(model.forward).parameters

    x = torch.randn(1, 1, 3, 224, 224)
    mask_a = torch.rand(1, 196) < 0.5
    mask_b = torch.rand(1, 196) < 0.5
    with torch.no_grad():
        base = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        out_a = model(
            pixel_values_videos=x, bool_masked_pos=mask_a, skip_predictor=True
        ).last_hidden_state
        out_b = model(
            pixel_values_videos=x, bool_masked_pos=mask_b, skip_predictor=True
        ).last_hidden_state
    assert torch.equal(base, out_a)
    assert torch.equal(out_a, out_b)


# ---------------------------------------------------------------------------
# End-to-end CPU smoke: real tiny V-JEPA 2 + LoRA + 3 optimizer steps
# (design §9.1 test_training_loop_jepa ②)
# ---------------------------------------------------------------------------

def test_jepa_objective_three_step_cpu_smoke():
    from transformers import VJEPA2Config, VJEPA2Model

    from worldforge_train.model.lora_wrapper import build_lora_model

    torch.manual_seed(42)
    cfg = VJEPA2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=224, patch_size=16, tubelet_size=1, in_chans=3,
        frames_per_clip=1, drop_path_rate=0.0,
    )
    base = VJEPA2Model(cfg)
    lora = build_lora_model(base, r=4, alpha=8, dropout=0.0)
    online = lora.peft_model
    online.train()

    # Grid (1, 14, 14) -> N=196, encoder dim 64.
    predictor = JEPAPredictor(
        num_patches=196, grid_size=(1, 14, 14), encoder_dim=64,
        predictor_dim=32, depth=2, num_heads=2,
    )
    obj = JEPAObjective.from_online_model(
        online, predictor,
        ema_momentum_start=0.9, ema_momentum_end=0.99,
        ema_schedule="cosine", ema_schedule_steps=3,
    )
    # Target encoder is fully frozen and in eval mode (design §5.5).
    assert all(not p.requires_grad for p in obj.target_encoder.parameters())
    assert not obj.target_encoder.training

    trainable = [p for p in online.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable + list(predictor.parameters()), lr=1e-3
    )

    ema_master0 = [b.clone() for b in obj.ema.state_dict()["master"]]
    pred0 = {n: p.detach().clone() for n, p in predictor.named_parameters()}
    tgt_lora0 = {
        n: p.detach().clone()
        for n, p in obj.target_encoder.named_parameters()
    }

    losses = []
    for _ in range(3):
        mask = torch.rand(1, 196) < 0.5
        batch = {
            "pixel_values_videos": torch.randn(1, 1, 3, 224, 224),
            "bool_masked_pos": mask,
        }
        loss = obj.loss_fn(online, batch)
        assert loss.requires_grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        obj.after_step()  # once per optimizer step, never per micro-batch
        losses.append(float(loss.detach()))

    assert len(losses) == 3 and all(v == v and v >= 0.0 for v in losses)
    assert obj.ema.step == 3

    # EMA master moved; target LoRA weights moved with it.
    ema_master1 = obj.ema.state_dict()["master"]
    assert any(not torch.equal(a, b) for a, b in zip(ema_master0, ema_master1))
    tgt_changed = any(
        not torch.equal(tgt_lora0[n], p.detach())
        for n, p in obj.target_encoder.named_parameters()
    )
    assert tgt_changed
    # Predictor received gradients and updated.
    pred_changed = any(
        not torch.equal(pred0[n], p.detach())
        for n, p in predictor.named_parameters()
    )
    assert pred_changed
    # Target remains frozen: requires_grad still all False.
    assert all(not p.requires_grad for p in obj.target_encoder.parameters())
