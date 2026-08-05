import pytest

from worldfoundry_inf.vjepa2_worker import VJEPA2Worker


class _FakeHFModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        class Out:
            last_hidden_state = __import__("torch").zeros(1, 196, 1024)
        return Out()


def test_worker_init_loads_model(monkeypatch):
    monkeypatch.setattr("worldfoundry_inf.vjepa2_worker._load_hf_model", lambda path, dtype: _FakeHFModel())
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    assert worker.model is not None


def test_worker_encode_returns_tensor(monkeypatch):
    import torch
    monkeypatch.setattr("worldfoundry_inf.vjepa2_worker._load_hf_model", lambda path, dtype: _FakeHFModel())
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    fake_video = torch.zeros(1, 16, 3, 224, 224)
    out = worker.encode(fake_video)
    assert out.shape == (1, 196, 1024)


# ---------------------------------------------------------------------------
# W24: predictor_ckpt / predict_temporal wiring
# ---------------------------------------------------------------------------


def test_worker_without_predictor_ckpt_has_none_bundle(monkeypatch):
    monkeypatch.setattr(
        "worldfoundry_inf.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    assert worker.bundle is None
    with pytest.raises(RuntimeError, match="temporal"):
        worker.predict_temporal(__import__("torch").zeros(1, 4, 3, 64, 64), 2, 2)


def test_worker_with_predictor_ckpt_loads_bundle(monkeypatch):
    """predictor_ckpt != None -> _build_bundle() is called -> bundle is set."""
    import types

    fake_bundle = types.SimpleNamespace(
        online=_FakeHFModel(),
        target=_FakeHFModel(),
        temporal_predictor=lambda feats, future_grid=None: __import__("torch").zeros(1, 196, 1024),
        jepa_predictor=None,
        ema_step=0,
        jepa_signature={},
        jepa_hyperparams={},
    )

    def fake_load(ckpt, encoder, *, past_grid, future_grid, map_location="cpu"):
        # Record the grids for assertion.
        fake_bundle._grids = (past_grid, future_grid)
        return fake_bundle

    monkeypatch.setattr(
        "worldfoundry_inf.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.vjepa2_worker.load_jepa_inference_model", fake_load,
    )

    worker = VJEPA2Worker(
        "/fake/model",
        adapter_name=None,
        predictor_ckpt="/fake/ckpt.pt",
        past_frames=2,
        future_frames=3,
    )
    assert worker.bundle is fake_bundle
    assert worker.bundle._grids == ((2, 14, 14), (3, 14, 14))
