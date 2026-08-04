"""Unit tests for W22.7 probing metrics (worldforge_train/eval/jepa_probing.py)
and the default-off throughput instrumentation (design §12-② hooks).

All CPU, synthetic data: each metric is validated on constructions with a
known expected value.
"""
from __future__ import annotations

import json

import pytest
import torch

from worldforge_train.eval.jepa_probing import (
    append_jsonl, collapse_metrics, effective_rank, knn_hit_rate,
    linear_probe_accuracy, masked_reconstruction_cosine, read_jsonl,
    token_variance,
)


def test_recon_cosine_identical_orthogonal():
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert masked_reconstruction_cosine(v, v.clone()) == pytest.approx(1.0)
    orth = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert masked_reconstruction_cosine(v, orth) == pytest.approx(0.0, abs=1e-6)


def test_recon_cosine_rejects_shape_mismatch_and_empty():
    with pytest.raises(ValueError):
        masked_reconstruction_cosine(torch.zeros(2, 3), torch.zeros(3, 3))
    with pytest.raises(ValueError, match="M=0"):
        masked_reconstruction_cosine(torch.zeros(0, 3), torch.zeros(0, 3))


def test_token_variance_constant_vs_random():
    const = torch.ones(2, 8, 4)
    assert token_variance(const) == pytest.approx(0.0)
    torch.manual_seed(0)
    rand = torch.randn(2, 8, 4)
    assert token_variance(rand) > 0.5


def test_effective_rank_extremes():
    # Rank-1 matrix (all tokens colinear) -> erank ~= 1 (collapsed).
    base = torch.randn(1, 16)
    collapsed = base.repeat(64, 1) * torch.linspace(0.5, 1.5, 64).unsqueeze(1)
    assert effective_rank(collapsed) < 1.5
    # Well-spread random matrix -> erank close to min(M, D).
    torch.manual_seed(0)
    spread = torch.randn(64, 8)
    assert effective_rank(spread) > 5.0
    # Fully collapsed (all zeros) -> defined as 1.0, no NaN.
    assert effective_rank(torch.zeros(8, 4)) == 1.0


def test_collapse_metrics_bundle():
    torch.manual_seed(0)
    out = collapse_metrics(torch.randn(2, 16, 8))
    assert set(out) == {"token_variance", "effective_rank"}
    assert out["token_variance"] > 0 and out["effective_rank"] > 1.0


def test_linear_probe_separable_clusters():
    g = torch.Generator().manual_seed(0)
    c0 = torch.tensor([2.0, 2.0])
    c1 = torch.tensor([-2.0, -2.0])
    x0 = c0 + 0.3 * torch.randn(40, 2, generator=g)
    x1 = c1 + 0.3 * torch.randn(40, 2, generator=g)
    feats = torch.cat([x0, x1])
    labels = torch.cat([torch.zeros(40), torch.ones(40)]).long()
    acc = linear_probe_accuracy(
        feats[:60], labels[:60], feats[60:], labels[60:], epochs=100
    )
    assert acc >= 0.9


def test_knn_hit_rate_perfect_clusters():
    g = torch.Generator().manual_seed(1)
    proto = torch.randn(3, 8, generator=g)
    q = proto + 0.01 * torch.randn(3, 8, generator=g)  # queries near protos
    k = proto + 0.01 * torch.randn(3, 8, generator=g)  # keys near protos
    labels = torch.tensor([0, 1, 2])
    assert knn_hit_rate(q, k, labels, labels, k=1) == pytest.approx(1.0)


def test_knn_rejects_bad_k():
    with pytest.raises(ValueError, match="out of range"):
        knn_hit_rate(torch.randn(2, 4), torch.randn(2, 4),
                     torch.tensor([0, 1]), torch.tensor([0, 1]), k=3)


def test_jsonl_roundtrip_sorted_by_step(tmp_path):
    p = tmp_path / "sub" / "metrics.jsonl"
    append_jsonl(p, {"step": 8, "recon_cosine": 0.5})
    append_jsonl(p, {"step": 4, "recon_cosine": 0.3})
    records = read_jsonl(p)
    assert [r["step"] for r in records] == [4, 8]
    lines = p.read_text().strip().splitlines()
    assert all(json.loads(line) for line in lines)
    assert read_jsonl(tmp_path / "missing.jsonl") == []


# ---------------------------------------------------------------------------
# throughput instrumentation: default OFF, env-gated ON (design §12-②)
# ---------------------------------------------------------------------------

def _objective():
    from types import SimpleNamespace

    import torch.nn as nn

    from worldforge_train.jepa_objective import JEPAObjective
    from worldforge_train.model.predictor import JEPAPredictor

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 8)

        def forward(self, *, pixel_values_videos, skip_predictor=False):
            b = pixel_values_videos.shape[0]
            x = pixel_values_videos.reshape(b, 3, 4).permute(0, 2, 1)
            return SimpleNamespace(last_hidden_state=self.lin(x))

    online = _Enc()
    predictor = JEPAPredictor(
        num_patches=4, grid_size=(1, 2, 2), encoder_dim=8,
        predictor_dim=8, depth=1, num_heads=2,
    )
    return JEPAObjective.from_online_model(
        online, predictor, ema_momentum_start=0.5, ema_momentum_end=0.5,
        ema_schedule="constant",
    ), _Enc


def _batch():
    return {
        "pixel_values_videos": torch.randn(1, 3, 4),
        "bool_masked_pos": torch.tensor([[True, False, True, False]]),
    }


def test_timing_default_off(monkeypatch):
    monkeypatch.delenv("WORLDFORGE_JEPA_TIMING", raising=False)
    obj, enc_cls = _objective()
    assert obj.timing is None
    assert obj.ema.timing is None
    loss = obj.loss_fn(enc_cls(), _batch())  # noqa: F841 — must not error
    obj.after_step()
    assert obj.timing is None


def test_timing_env_gated_on(monkeypatch):
    monkeypatch.setenv("WORLDFORGE_JEPA_TIMING", "1")
    obj, enc_cls = _objective()
    assert obj.timing == {}
    assert obj.ema.timing is obj.timing  # shared dict
    obj.loss_fn(enc_cls(), _batch())
    obj.after_step()
    t = obj.timing
    assert t["loss_fn_s"] >= 0.0
    assert t["context_forward_s"] >= 0.0
    assert t["target_forward_s"] >= 0.0
    assert t["ema_update_s"] >= 0.0
    assert t["ema_updates"] == 1
    # target forward is INSIDE loss_fn wall time.
    assert t["target_forward_s"] <= t["loss_fn_s"] + 1e-9
