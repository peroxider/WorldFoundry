import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from worldfoundry_inf.server import create_app


@pytest.fixture
def fake_video_bytes():
    # 1 second of 64x64 zeros
    return base64.b64encode(np.zeros((10, 64, 64, 3), dtype=np.uint8).tobytes()).decode()


@pytest.fixture
def client(monkeypatch):
    """Build a FastAPI test client with a stubbed VJEPA2Worker."""
    class _StubWorker:
        def encode(self, x):
            import torch
            return torch.zeros(1, 196, 1024)

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.VJEPA2Worker",
        lambda *a, **kw: _StubWorker(),
    )

    app = create_app(model_path="/fake/model", secret="test-secret")
    return TestClient(app)


def test_encode_requires_auth(client):
    resp = client.post("/v1/world/encode", json={"video": "abc"})
    assert resp.status_code == 401


def test_encode_with_valid_auth_returns_200(client, fake_video_bytes):
    import hmac
    import hashlib
    import time

    method = "POST"
    path = "/v1/world/encode"
    body = f'{{"video":"{fake_video_bytes}"}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()

    resp = client.post(
        path,
        content=body,
        headers={
            "Authorization": f"HMAC {sig}",
            "X-WorldFoundry-Timestamp": ts,
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert "features" in resp.json()


def test_encode_rejects_stale_timestamp(client, fake_video_bytes):
    import hmac
    import hashlib
    method = "POST"
    path = "/v1/world/encode"
    body = f'{{"video":"{fake_video_bytes}"}}'.encode()
    ts = "1000000000"  # year 2001
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()
    resp = client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldFoundry-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 401


@pytest.fixture
def predict_client(monkeypatch):
    """Test client whose stub worker implements both encode() and predict()."""
    class _StubWorker:
        def encode(self, x):
            import torch
            return torch.zeros(1, 196, 1024)

        def predict(self, x, mask):
            import torch
            return torch.zeros(1, 196, 1024)

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.VJEPA2Worker",
        lambda *a, **kw: _StubWorker(),
    )
    return TestClient(create_app(model_path="/fake/model", secret="test-secret"))


def test_predict_with_valid_auth_returns_200(predict_client, fake_video_bytes):
    import hmac
    import hashlib
    import time

    method = "POST"
    path = "/v1/world/predict"
    body = f'{{"video":"{fake_video_bytes}","mask_ratio":0.7}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()

    resp = predict_client.post(
        path,
        content=body,
        headers={
            "Authorization": f"HMAC {sig}",
            "X-WorldFoundry-Timestamp": ts,
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "predicted_features" in payload
    assert "loss_score" in payload


@pytest.fixture
def plan_client(monkeypatch):
    """Test client whose stub worker implements encode/predict/plan."""
    class _StubWorker:
        def encode(self, x):
            import torch
            return torch.zeros(1, 196, 1024)

        def predict(self, x, mask):
            import torch
            return torch.zeros(1, 196, 1024)

        def plan(self, x, actions):
            import torch
            return torch.zeros(1, 196, 1024), [int(a) for a in actions]

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.VJEPA2Worker",
        lambda *a, **kw: _StubWorker(),
    )
    return TestClient(create_app(model_path="/fake/model", secret="test-secret"))


def test_plan_with_valid_auth_returns_200(plan_client, fake_video_bytes):
    import hmac
    import hashlib
    import time

    method = "POST"
    path = "/v1/world/plan"
    body = f'{{"video":"{fake_video_bytes}","actions":[1,2,3]}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()

    resp = plan_client.post(
        path,
        content=body,
        headers={
            "Authorization": f"HMAC {sig}",
            "X-WorldFoundry-Timestamp": ts,
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "next_features" in payload
    assert payload["planned_actions"] == [1, 2, 3]  # stub returns ints


def test_plan_rejects_null_actions(plan_client, fake_video_bytes):
    import hmac
    import hashlib
    import time

    method = "POST"
    path = "/v1/world/plan"
    body = f'{{"video":"{fake_video_bytes}","actions":null}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()
    resp = plan_client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldFoundry-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_plan_rejects_boolean_actions(plan_client, fake_video_bytes):
    import hmac
    import hashlib
    import time

    method = "POST"
    path = "/v1/world/plan"
    body = f'{{"video":"{fake_video_bytes}","actions":[true,false]}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()
    resp = plan_client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldFoundry-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_predict_rejects_malformed_video(predict_client):
    import hmac
    import hashlib
    import time

    method = "POST"
    path = "/v1/world/predict"
    # Invalid base64 + insufficient length → should be 400, not 200 (G8).
    body = b'{"video":"%%%","mask_ratio":0.5}'
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()
    resp = predict_client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldFoundry-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_plan_rejects_malformed_video(plan_client):
    import hmac
    import hashlib
    import time

    method = "POST"
    path = "/v1/world/plan"
    # Invalid base64 + insufficient length → should be 400, not 500.
    body = b'{"video":"%%%","actions":[1,2,3]}'
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()
    resp = plan_client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldFoundry-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# W24: temporal-mode predict (mode="temporal")
# ---------------------------------------------------------------------------


@pytest.fixture
def temporal_client(monkeypatch):
    """Test client whose stub worker exposes ``predict_temporal``."""
    class _TemporalStubWorker:
        def __init__(self) -> None:
            # W33: surface action_dim so /predict?mode=action validation
            # reaches the 503 path instead of failing earlier on the
            # action_dim check.
            self.action_dim = 7

        def encode(self, x):
            import torch
            return torch.zeros(1, 196, 1024)

        def predict(self, x, mask):
            import torch
            return torch.zeros(1, 196, 1024)

        def predict_temporal(self, x, past_frames, future_frames):
            import torch
            # (B, N_future, hidden_dim), confidence in [-1, 1]
            return torch.zeros(1, future_frames * 196, 1024), 0.5

    monkeypatch.setattr("worldfoundry_inf.workers.vjepa2_worker.VJEPA2Worker", lambda *a, **kw: _TemporalStubWorker())
    return TestClient(create_app(model_path="/fake/model", secret="test-secret"))


def _sign_request(secret: str, method: str, path: str, body: bytes) -> dict:
    import hashlib
    import hmac
    import time
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(secret.encode(), canonical, hashlib.sha256).hexdigest()
    return {
        "Authorization": f"HMAC {sig}",
        "X-WorldFoundry-Timestamp": ts,
        "Content-Type": "application/json",
    }


def test_predict_temporal_with_valid_auth_returns_future_features_and_confidence(
    temporal_client, fake_video_bytes,
):
    method = "POST"
    path = "/v1/world/predict"
    # fake_video_bytes is 10 frames; past_frames=2 + future_frames=2 = 4 <= 10.
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"temporal",'
        f'"past_frames":2,"future_frames":2}}'
    ).encode()
    resp = temporal_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["mode"] == "temporal"
    assert "future_features" in payload
    assert isinstance(payload["confidence"], float)
    assert -1.0 <= payload["confidence"] <= 1.0


def test_predict_temporal_rejects_past_frames_below_one(temporal_client, fake_video_bytes):
    method = "POST"
    path = "/v1/world/predict"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"temporal",'
        f'"past_frames":0,"future_frames":2}}'
    ).encode()
    resp = temporal_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "past_frames" in resp.json()["detail"]


def test_predict_temporal_rejects_future_frames_below_one(temporal_client, fake_video_bytes):
    method = "POST"
    path = "/v1/world/predict"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"temporal",'
        f'"past_frames":2,"future_frames":0}}'
    ).encode()
    resp = temporal_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "future_frames" in resp.json()["detail"]


def test_predict_temporal_rejects_past_plus_future_exceeding_video_length(
    temporal_client, fake_video_bytes,
):
    method = "POST"
    path = "/v1/world/predict"
    # fake_video_bytes has 10 frames; past=8 + future=8 = 16 > 10.
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"temporal",'
        f'"past_frames":8,"future_frames":8}}'
    ).encode()
    resp = temporal_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "exceeds video length" in resp.json()["detail"]


def test_predict_temporal_returns_503_when_worker_has_no_predict_temporal_method(
    predict_client, fake_video_bytes,
):
    """predict_client's stub has only encode() + predict(), no predict_temporal()."""
    method = "POST"
    path = "/v1/world/predict"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"temporal",'
        f'"past_frames":2,"future_frames":2}}'
    ).encode()
    resp = predict_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 503
    assert "temporal mode" in resp.json()["detail"]


def test_predict_temporal_requires_auth(temporal_client, fake_video_bytes):
    """No HMAC headers -> 401, even in temporal mode."""
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"temporal",'
        f'"past_frames":2,"future_frames":2}}'
    ).encode()
    resp = temporal_client.post(
        "/v1/world/predict", content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 401


def test_predict_unknown_mode_returns_400(temporal_client, fake_video_bytes):
    method = "POST"
    path = "/v1/world/predict"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"bogus"}}'
    ).encode()
    resp = temporal_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "mode must be" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# W33: action-mode (mode="action") and W33 plan endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def action_client(monkeypatch):
    """Test client whose stub worker exposes ``predict_action_conditioned``
    and an ``action_dim`` of 7."""
    class _ActionStubWorker:
        def __init__(self) -> None:
            self.action_dim = 7
            self.past_frames = 2
            self.future_frames = 2

        def encode(self, x):
            import torch
            return torch.zeros(1, 196, 1024)

        def predict(self, x, mask):
            import torch
            return torch.zeros(1, 196, 1024)

        def predict_action_conditioned(
            self, x, actions, past_frames, future_frames,
        ):
            import torch
            return torch.zeros(1, future_frames * 196, 1024), 0.6

    monkeypatch.setattr(
        "worldfoundry_inf.workers.vjepa2_worker.VJEPA2Worker",
        lambda *a, **kw: _ActionStubWorker(),
    )
    return TestClient(create_app(model_path="/fake/model", secret="test-secret"))


def test_predict_mode_action_returns_200(action_client, fake_video_bytes):
    method = "POST"
    path = "/v1/world/predict"
    actions_json = "[[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.0,0.1,0.2,0.3,0.4,0.5,0.6]]"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"action",'
        f'"past_frames":2,"future_frames":2,"actions":{actions_json}}}'
    ).encode()
    resp = action_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["mode"] == "action"
    assert "future_features" in payload
    assert isinstance(payload["confidence"], float)
    assert -1.0 <= payload["confidence"] <= 1.0


def test_predict_mode_action_rejects_missing_actions(action_client, fake_video_bytes):
    method = "POST"
    path = "/v1/world/predict"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"action",'
        f'"past_frames":2,"future_frames":2}}'
    ).encode()
    resp = action_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "actions" in resp.json()["detail"]


def test_predict_mode_action_rejects_action_dim_mismatch(action_client, fake_video_bytes):
    """Worker action_dim=7; client sends 5-vector past_frames=2 actions -> 400."""
    method = "POST"
    path = "/v1/world/predict"
    # Each action vector has 5 floats, but worker.action_dim=7.
    actions_json = "[[0.1,0.2,0.3,0.4,0.5],[0.0,0.1,0.2,0.3,0.4]]"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"action",'
        f'"past_frames":2,"future_frames":2,"actions":{actions_json}}}'
    ).encode()
    resp = action_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "action_dim" in resp.json()["detail"]


def test_predict_mode_action_rejects_wrong_actions_count(action_client, fake_video_bytes):
    """past_frames=2 but only 1 action vector -> 400."""
    method = "POST"
    path = "/v1/world/predict"
    actions_json = "[[0.1,0.2,0.3,0.4,0.5,0.6,0.7]]"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"action",'
        f'"past_frames":2,"future_frames":2,"actions":{actions_json}}}'
    ).encode()
    resp = action_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "past_frames" in resp.json()["detail"]


def test_predict_mode_action_returns_503_when_worker_lacks_method(
    temporal_client, fake_video_bytes,
):
    """temporal_client's stub has predict_temporal but not predict_action_conditioned."""
    method = "POST"
    path = "/v1/world/predict"
    actions_json = "[[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.0,0.1,0.2,0.3,0.4,0.5,0.6]]"
    body = (
        f'{{"video":"{fake_video_bytes}","mode":"action",'
        f'"past_frames":2,"future_frames":2,"actions":{actions_json}}}'
    ).encode()
    resp = temporal_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 503
    assert "action mode" in resp.json()["detail"]


def test_plan_action_mode_with_valid_auth_returns_200(action_client, fake_video_bytes):
    """W33: /v1/world/plan with 2-D action list -> 200 action-conditioned."""
    method = "POST"
    path = "/v1/world/plan"
    actions_json = "[[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.0,0.1,0.2,0.3,0.4,0.5,0.6]]"
    body = (
        f'{{"video":"{fake_video_bytes}","actions":{actions_json},'
        f'"past_frames":2,"future_frames":2}}'
    ).encode()
    resp = action_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["mode"] == "action"
    assert "next_features" in payload
    assert "planned_actions" in payload
    assert payload["planned_actions"] == [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ]
    assert isinstance(payload["confidence"], float)


def test_plan_action_mode_rejects_wrong_action_dim(action_client, fake_video_bytes):
    method = "POST"
    path = "/v1/world/plan"
    actions_json = "[[0.1,0.2,0.3,0.4,0.5],[0.0,0.1,0.2,0.3,0.4]]"  # 5-vector vs worker 7
    body = (
        f'{{"video":"{fake_video_bytes}","actions":{actions_json},'
        f'"past_frames":2,"future_frames":2}}'
    ).encode()
    resp = action_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 400
    assert "action_dim" in resp.json()["detail"]


def test_plan_legacy_passthrough_returns_200(plan_client, fake_video_bytes):
    """Pre-W33 worker without predict_action_conditioned: flat integer list
    still hits the legacy plan() passthrough (backward compat)."""
    method = "POST"
    path = "/v1/world/plan"
    body = f'{{"video":"{fake_video_bytes}","actions":[1,2,3]}}'.encode()
    resp = plan_client.post(
        path, content=body, headers=_sign_request("test-secret", method, path, body),
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["planned_actions"] == [1, 2, 3]
