"""W22.8 multi-mask data pipeline tests (design §5.1).

Covers:
- dataset: K=1 backward compat (byte-identical to W22), K>1 produces masks,
  determinism, validation (mask_count < 1).
- collator: K=1 path emits no ``bool_masked_pos_k``, K>1 emits (B,K,N);
  K consistency and mixed-batch validation.
- objective: loss_fn dispatches on bool_masked_pos_k, predictor called K
  times, encoder called once; mean-of-K equivalence; all-empty K keeps
  backward well-defined; K=1 contract preserved.
- trainer: default mask_count=1 when cfg lacks jepa section;
  ``jepa.loss.num_masks`` propagates to Vjepa2Dataset; num_masks<1 raises.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from worldfoundry_train.data.collator import collate_vjepa2_batch
from worldfoundry_train.data.dataset import Vjepa2Dataset
from worldfoundry_train.jepa_objective import JEPAObjective
from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.predictor import JEPAPredictor


ENC_DIM = 32
N = 16
GRID = (1, 4, 4)


def _predictor() -> JEPAPredictor:
    return JEPAPredictor(
        num_patches=N, grid_size=GRID, encoder_dim=ENC_DIM,
        predictor_dim=16, depth=2, num_heads=2,
    )


def _write_manifest(path: Path, video: Path) -> None:
    path.write_text(
        f"video_path,num_frames,height,width\n{video},20,64,64\n"
    )


@pytest.fixture
def manifest(tmp_path, sample_mp4):
    csv_path = tmp_path / "manifest.csv"
    _write_manifest(csv_path, sample_mp4)
    return csv_path


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def test_dataset_mask_count_one_matches_legacy_outer_rng(manifest):
    """K=1 is byte-identical to the W22 behaviour. Pre-W22.8 used
    ``np.random.default_rng(idx)`` for the only mask; ``idx * 1 + 0 == idx``
    so the seed, and therefore the mask grid, must match exactly.
    """
    captured: list[int] = []

    def capture(shape, rng):
        captured.append(int(rng.integers(0, 2**31 - 1)))
        return np.zeros(shape, dtype=bool)

    Vjepa2Dataset(
        manifest_path=manifest, mask_fn=capture, mask_count=1,
    )[0]
    fresh = np.random.default_rng(0)
    expected_draws = [int(fresh.integers(0, 2**31 - 1)) for _ in range(len(captured))]
    assert captured == expected_draws

    # And the sample dict shape is exactly the W22 contract.
    item = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
        mask_count=1,
    )[0]
    assert "masks" not in item
    assert "mask" in item
    assert item["mask"].dtype == bool


def test_dataset_mask_count_three_emits_masks_list(manifest):
    """K=3 -> sample has 'masks' of length 3 and 'mask' is masks[0]."""
    ds = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
        mask_count=3,
    )
    item = ds[0]
    assert "mask" in item
    assert "masks" in item
    assert len(item["masks"]) == 3
    assert np.array_equal(item["mask"], item["masks"][0])


def test_dataset_multi_mask_seeds_differ_across_k(manifest):
    """The K masks of one sample must not all be identical (a regression
    that would silently degenerate the multi-mask regulariser).
    """
    def make(shape, rng):
        return rng.random(shape) < 0.5

    ds = Vjepa2Dataset(
        manifest_path=manifest, mask_fn=make, mask_count=4,
    )
    item = ds[0]
    ms = item["masks"]
    any_diff = False
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            if not np.array_equal(ms[i], ms[j]):
                any_diff = True
                break
        if any_diff:
            break
    assert any_diff


def test_dataset_multi_mask_is_deterministic(manifest):
    """Same idx twice in a row -> identical K masks."""
    def make(shape, rng):
        return rng.random(shape) < 0.5

    ds = Vjepa2Dataset(
        manifest_path=manifest, mask_fn=make, mask_count=3,
    )
    a = ds[0]
    b = ds[0]
    for i in range(3):
        assert np.array_equal(a["masks"][i], b["masks"][i])


def test_dataset_rejects_mask_count_below_one(manifest):
    with pytest.raises(ValueError, match="mask_count"):
        Vjepa2Dataset(
            manifest_path=manifest,
            mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
            mask_count=0,
        )
    with pytest.raises(ValueError, match="mask_count"):
        Vjepa2Dataset(
            manifest_path=manifest,
            mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
            mask_count=-1,
        )


# ---------------------------------------------------------------------------
# collator
# ---------------------------------------------------------------------------

def _np_sample(T=4, H=28, W=28, Hg=2, Wg=2, seed=0, *, K=1):
    rng = np.random.default_rng(seed)
    s = {
        "frames": rng.random((T, 3, H, W), dtype=np.float32),
        "mask": rng.integers(0, 2, size=(T, Hg, Wg), dtype=bool),
    }
    if K > 1:
        s["masks"] = [
            rng.integers(0, 2, size=(T, Hg, Wg), dtype=bool)
            for _ in range(K)
        ]
        s["mask"] = s["masks"][0]
    return s


def test_collate_k1_does_not_emit_bool_masked_pos_k():
    """K=1 path: no bool_masked_pos_k key, bool_masked_pos unchanged."""
    samples = [_np_sample(seed=i) for i in range(2)]
    out = collate_vjepa2_batch(samples)
    assert "bool_masked_pos_k" not in out
    assert out["bool_masked_pos"].shape == (2, 4 * 2 * 2)


def test_collate_k2_emits_bool_masked_pos_k_and_first_mask_matches():
    """K=2 path: bool_masked_pos_k.shape == (B, K, N); [:, 0] == bool_masked_pos."""
    samples = [_np_sample(seed=i, K=2) for i in range(2)]
    out = collate_vjepa2_batch(samples)
    assert "bool_masked_pos_k" in out
    assert out["bool_masked_pos_k"].shape == (2, 2, 4 * 2 * 2)
    assert out["bool_masked_pos_k"].dtype == torch.bool
    assert torch.equal(out["bool_masked_pos_k"][:, 0], out["bool_masked_pos"])


def test_collate_raises_on_inconsistent_k_across_samples():
    """Different K within a batch -> ValueError naming the bad sample."""
    s0 = _np_sample(seed=0, K=2)
    s1 = _np_sample(seed=1, K=3)
    with pytest.raises(ValueError, match="K=3"):
        collate_vjepa2_batch([s0, s1])


def test_collate_raises_on_mixed_single_and_multi_mask_batch():
    """A batch where some samples carry 'masks' and some do not -> ValueError.

    Silently dropping to K=1 would make a num_masks>1 config a no-op.
    """
    s0 = _np_sample(seed=0, K=2)
    s1 = _np_sample(seed=1, K=1)
    with pytest.raises(ValueError, match="multi-mask"):
        collate_vjepa2_batch([s0, s1])


# ---------------------------------------------------------------------------
# objective / loss_fn
# ---------------------------------------------------------------------------

class _RecordingEncoder(nn.Module):
    """Mock encoder. Records call kwargs; returns fixed features."""

    def __init__(self, feats: torch.Tensor):
        super().__init__()
        self._feats = feats
        self.calls: list[dict] = []
        self.dummy = nn.Parameter(torch.zeros(1))

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


def test_loss_fn_with_bool_masked_pos_k_calls_predictor_k_times(monkeypatch):
    """Encoder stays at 1 forward each; predictor is invoked K times."""
    obj, online, target = _objective_with_mocks()
    pred_calls = {"n": 0}
    original = obj.predictor.forward

    def spy(*args, **kwargs):
        pred_calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(obj.predictor, "forward", spy)

    torch.manual_seed(7)
    masks = []
    while len(masks) < 3:
        m = torch.rand(1, N) < 0.5
        if int(m.sum()) > 0:
            masks.append(m)
    batch = {
        "pixel_values_videos": torch.randn(1, 1, 3, 32, 32),
        "bool_masked_pos": masks[0],
        "bool_masked_pos_k": torch.stack(masks, dim=1),  # (1, K=3, N)
    }
    loss = obj.loss_fn(online, batch)
    assert torch.isfinite(loss)
    assert pred_calls["n"] == 3
    assert len(online.calls) == 1
    assert len(target.calls) == 1


def test_loss_fn_mean_over_k_equals_arithmetic_mean():
    """K>1 with all-equal masks -> mean (not sum) over K."""
    obj, online, _ = _objective_with_mocks()
    torch.manual_seed(2)
    mask = torch.rand(1, N) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, N) < 0.5
    batch_one = {
        "pixel_values_videos": torch.randn(1, 1, 3, 32, 32),
        "bool_masked_pos": mask,
    }
    batch_k = {
        "pixel_values_videos": batch_one["pixel_values_videos"],
        "bool_masked_pos": mask,
        "bool_masked_pos_k": torch.stack([mask, mask], dim=1),
    }
    l1 = obj.loss_fn(online, batch_one)
    lk = obj.loss_fn(online, batch_k)
    assert l1.item() == pytest.approx(lk.item(), rel=1e-5, abs=1e-7)


def test_loss_fn_all_empty_masks_k_returns_graph_attached_zero():
    """K>1 with every mask all-False -> loss is 0 with grad attached."""
    obj, online, _ = _objective_with_mocks()
    empty = torch.zeros(1, N, dtype=torch.bool)
    batch = {
        "pixel_values_videos": torch.randn(1, 1, 3, 32, 32),
        "bool_masked_pos": empty,
        "bool_masked_pos_k": torch.stack([empty, empty, empty], dim=1),
    }
    loss = obj.loss_fn(online, batch)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()  # must not raise


def test_loss_fn_without_bool_masked_pos_k_uses_k1_path():
    """W22 contract: a batch with no bool_masked_pos_k still works (K=1)."""
    obj, online, _ = _objective_with_mocks()
    mask = torch.rand(1, N) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, N) < 0.5
    batch = {
        "pixel_values_videos": torch.randn(1, 1, 3, 32, 32),
        "bool_masked_pos": mask,
    }
    loss = obj.loss_fn(online, batch)
    assert torch.isfinite(loss)
    assert len(online.calls) == 1


# ---------------------------------------------------------------------------
# trainer passthrough + default
# ---------------------------------------------------------------------------

def _patch_video_source(monkeypatch):
    """Stub build_video_source: returns frames of all-zeros uint8."""
    from worldfoundry_train.data import dataset as ds_mod

    class FakeSource:
        def read_all(self):
            return np.zeros((20, 64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(
        ds_mod, "build_video_source", lambda *a, **kw: FakeSource(),
    )


def test_trainer_build_dataset_defaults_mask_count_to_one(tmp_path, monkeypatch):
    """A cfg without a jepa section still builds a dataset with mask_count=1."""
    from worldfoundry_train import trainer as trainer_mod

    _patch_video_source(monkeypatch)
    csv_path = tmp_path / "manifest.csv"
    _write_manifest(csv_path, tmp_path / "dummy.mp4")
    cfg = {
        "data": {
            "manifest": str(csv_path),
            "mask_tubelet_size": [1, 14, 14],
            "mask_ratio": 0.5,
        },
    }
    ds = trainer_mod._build_dataset(cfg)
    assert ds.mask_count == 1


def test_trainer_build_dataset_reads_jepa_loss_num_masks(tmp_path, monkeypatch):
    """``jepa.loss.num_masks`` propagates to Vjepa2Dataset as mask_count."""
    from worldfoundry_train import trainer as trainer_mod

    _patch_video_source(monkeypatch)
    csv_path = tmp_path / "manifest.csv"
    _write_manifest(csv_path, tmp_path / "dummy.mp4")
    cfg = {
        "data": {
            "manifest": str(csv_path),
            "mask_tubelet_size": [1, 14, 14],
            "mask_ratio": 0.5,
        },
        "jepa": {"loss": {"num_masks": 2}},
    }
    ds = trainer_mod._build_dataset(cfg)
    assert ds.mask_count == 2
