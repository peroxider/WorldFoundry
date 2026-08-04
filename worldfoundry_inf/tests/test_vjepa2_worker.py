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
