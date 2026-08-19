import pytest

from worldfoundry_inf.workers.vjepa2_worker import VJEPA2Worker


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
    monkeypatch.setattr("worldfoundry_inf.workers.vjepa2_worker._load_hf_model", lambda path, dtype: _FakeHFModel())
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    assert worker.model is not None


def test_worker_encode_returns_tensor(monkeypatch):
    import torch
    monkeypatch.setattr("worldfoundry_inf.workers.vjepa2_worker._load_hf_model", lambda path, dtype: _FakeHFModel())
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    fake_video = torch.zeros(1, 16, 3, 224, 224)
    out = worker.encode(fake_video)
    assert out.shape == (1, 196, 1024)


# ---------------------------------------------------------------------------
# W24: predictor_ckpt / predict_temporal wiring
# ---------------------------------------------------------------------------


def test_worker_without_predictor_ckpt_has_none_bundle(monkeypatch):
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
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
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.load_jepa_inference_model", fake_load,
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


# ---------------------------------------------------------------------------
# W27: predict_action_conditioned API stub
# ---------------------------------------------------------------------------


def test_worker_predict_action_conditioned_raises(monkeypatch):
    """Without action_predictor_ckpt, the worker raises RuntimeError (W27 v2 surface)."""
    import torch
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    video = torch.zeros(1, 4, 3, 64, 64)
    actions = torch.zeros(1, 2, 7)
    with pytest.raises(RuntimeError, match="action_predictor_ckpt"):
        worker.predict_action_conditioned(video, actions, past_frames=2, future_frames=2)


# ---------------------------------------------------------------------------
# W27 v2: action_predictor_ckpt / predict_action_conditioned wiring
# ---------------------------------------------------------------------------


def test_worker_without_action_predictor_ckpt_has_none_action_bundle(monkeypatch):
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    assert worker.action_bundle is None
    assert worker.action_dim == 7  # default


def test_worker_with_action_predictor_ckpt_loads_action_bundle(monkeypatch):
    """action_predictor_ckpt != None -> _build_action_bundle() called -> action_bundle is set."""
    fake_bundle = _make_fake_action_bundle(action_dim=7)

    def fake_load(ckpt, encoder, *, past_grid, future_grid, map_location="cpu",
                  jepa_predictor=None):
        fake_bundle._grids = (past_grid, future_grid)
        return fake_bundle

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.load_action_conditioned_inference_model",
        fake_load,
    )

    worker = VJEPA2Worker(
        "/fake/model",
        adapter_name=None,
        action_predictor_ckpt="/fake/ckpt.pt",
        past_frames=2,
        future_frames=3,
        action_dim=7,
    )
    assert worker.action_bundle is fake_bundle
    assert worker.action_bundle._grids == ((2, 14, 14), (3, 14, 14))


def test_worker_with_action_predictor_ckpt_action_dim_mismatch_raises(monkeypatch):
    """action_dim mismatch between worker config and ckpt signature -> ValueError."""
    fake_bundle = _make_fake_action_bundle(action_dim=8)  # ckpt says 8

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.load_action_conditioned_inference_model",
        lambda ckpt, encoder, *, past_grid, future_grid, map_location="cpu",
             jepa_predictor=None: fake_bundle,
    )

    with pytest.raises(ValueError, match="action_dim"):
        VJEPA2Worker(
            "/fake/model",
            adapter_name=None,
            action_predictor_ckpt="/fake/ckpt.pt",
            action_dim=7,  # worker says 7
        )


def test_worker_predict_action_conditioned_runs_through_action_bundle(monkeypatch):
    """Happy path: action_bundle is set -> predict_action_conditioned returns (features, confidence)."""
    import torch
    fake_bundle = _make_fake_action_bundle(action_dim=7)

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.load_action_conditioned_inference_model",
        lambda ckpt, encoder, *, past_grid, future_grid, map_location="cpu",
             jepa_predictor=None: fake_bundle,
    )

    worker = VJEPA2Worker(
        "/fake/model",
        adapter_name=None,
        action_predictor_ckpt="/fake/ckpt.pt",
        past_frames=2,
        future_frames=2,
        action_dim=7,
    )
    video = torch.zeros(1, 4, 3, 64, 64)  # 4 frames = past 2 + future 2
    actions = torch.zeros(1, 2, 7)
    predicted, confidence = worker.predict_action_conditioned(
        video, actions, past_frames=2, future_frames=2,
    )
    assert predicted.shape == (1, 196, 1024)
    assert isinstance(confidence, float)
    assert -1.0 <= confidence <= 1.0


def test_worker_predict_action_conditioned_rejects_wrong_past_future(monkeypatch):
    """past_frames/future_frames must match worker config."""
    import torch
    fake_bundle = _make_fake_action_bundle(action_dim=7)

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.load_action_conditioned_inference_model",
        lambda ckpt, encoder, *, past_grid, future_grid, map_location="cpu",
             jepa_predictor=None: fake_bundle,
    )

    worker = VJEPA2Worker(
        "/fake/model",
        adapter_name=None,
        action_predictor_ckpt="/fake/ckpt.pt",
        past_frames=2,
        future_frames=2,
    )
    video = torch.zeros(1, 4, 3, 64, 64)
    actions = torch.zeros(1, 3, 7)  # wrong frame count
    with pytest.raises(RuntimeError, match="past_frames"):
        worker.predict_action_conditioned(
            video, actions, past_frames=3, future_frames=2,
        )


def test_worker_predict_action_conditioned_rejects_video_too_short(monkeypatch):
    """video length must be >= past_frames + future_frames."""
    import torch
    fake_bundle = _make_fake_action_bundle(action_dim=7)

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.load_action_conditioned_inference_model",
        lambda ckpt, encoder, *, past_grid, future_grid, map_location="cpu",
             jepa_predictor=None: fake_bundle,
    )

    worker = VJEPA2Worker(
        "/fake/model",
        adapter_name=None,
        action_predictor_ckpt="/fake/ckpt.pt",
        past_frames=2,
        future_frames=2,
    )
    video = torch.zeros(1, 3, 3, 64, 64)  # only 3 frames, need 4
    actions = torch.zeros(1, 2, 7)
    with pytest.raises(RuntimeError, match="past_frames\\+future_frames"):
        worker.predict_action_conditioned(
            video, actions, past_frames=2, future_frames=2,
        )


def test_worker_predict_action_conditioned_rejects_wrong_actions_shape(monkeypatch):
    """actions must be (B, past_frames, action_dim)."""
    import torch
    fake_bundle = _make_fake_action_bundle(action_dim=7)

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker._load_hf_model",
        lambda path, dtype: _FakeHFModel(),
    )
    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.load_action_conditioned_inference_model",
        lambda ckpt, encoder, *, past_grid, future_grid, map_location="cpu",
             jepa_predictor=None: fake_bundle,
    )

    worker = VJEPA2Worker(
        "/fake/model",
        adapter_name=None,
        action_predictor_ckpt="/fake/ckpt.pt",
        past_frames=2,
        future_frames=2,
        action_dim=7,
    )
    video = torch.zeros(1, 4, 3, 64, 64)
    # Wrong: action_dim=8 instead of 7
    actions = torch.zeros(1, 2, 8)
    with pytest.raises(RuntimeError, match="action_dim"):
        worker.predict_action_conditioned(
            video, actions, past_frames=2, future_frames=2,
        )


def _make_fake_action_bundle(*, action_dim: int = 7):
    """Build a `SimpleNamespace` mimicking the W27 v2 action bundle."""
    import types
    return types.SimpleNamespace(
        online=_FakeHFModel(),
        target=_FakeHFModel(),
        action_predictor=lambda feats, actions, future_grid=None: __import__("torch").zeros(1, 196, 1024),
        temporal_predictor=None,
        jepa_predictor=None,
        ema_step=0,
        jepa_signature={},
        jepa_hyperparams={},
        action_signature={"action_dim": action_dim},
    )
