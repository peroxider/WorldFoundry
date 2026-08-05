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
        "worldfoundry_inf.vjepa2_worker.VJEPA2Worker",
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
        "worldfoundry_inf.vjepa2_worker.VJEPA2Worker",
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
        "worldfoundry_inf.vjepa2_worker.VJEPA2Worker",
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

    monkeypatch.setattr("worldfoundry_inf.vjepa2_worker.VJEPA2Worker", lambda *a, **kw: _TemporalStubWorker())
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
