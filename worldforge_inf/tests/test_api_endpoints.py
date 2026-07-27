import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from worldforge_inf.server import create_app


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

    monkeypatch.setattr("worldforge_inf.vjepa2_worker.VJEPA2Worker", lambda p: _StubWorker())

    app = create_app(model_path="/fake/model", secret="test-secret")
    return TestClient(app)


def test_encode_requires_auth(client):
    resp = client.post("/v1/world/encode", json={"video": "abc"})
    assert resp.status_code == 401


def test_encode_with_valid_auth_returns_200(client, fake_video_bytes):
    import hmac, hashlib, time

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
            "X-WorldForge-Timestamp": ts,
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert "features" in resp.json()


def test_encode_rejects_stale_timestamp(client, fake_video_bytes):
    import hmac, hashlib
    method = "POST"
    path = "/v1/world/encode"
    body = f'{{"video":"{fake_video_bytes}"}}'.encode()
    ts = "1000000000"  # year 2001
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()
    resp = client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldForge-Timestamp": ts,
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

    monkeypatch.setattr("worldforge_inf.vjepa2_worker.VJEPA2Worker", lambda p: _StubWorker())
    return TestClient(create_app(model_path="/fake/model", secret="test-secret"))


def test_predict_with_valid_auth_returns_200(predict_client, fake_video_bytes):
    import hmac, hashlib, time

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
            "X-WorldForge-Timestamp": ts,
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

    monkeypatch.setattr("worldforge_inf.vjepa2_worker.VJEPA2Worker", lambda p: _StubWorker())
    return TestClient(create_app(model_path="/fake/model", secret="test-secret"))


def test_plan_with_valid_auth_returns_200(plan_client, fake_video_bytes):
    import hmac, hashlib, time

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
            "X-WorldForge-Timestamp": ts,
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "next_features" in payload
    assert payload["planned_actions"] == [1, 2, 3]  # stub returns ints
