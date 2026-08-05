import pytest

from worldfoundry_rl.actor import VJEPAActor


def test_actor_init_loads_model(monkeypatch):
    class _FakeEncoder:
        def eval(self): return self
        def to(self, d): return self
        def __call__(self, **kw):
            class Out: last_hidden_state = __import__("torch").zeros(1, 196, 1024)
            return Out()

    monkeypatch.setattr(
        "worldfoundry_rl.actor._load_hf_model",
        lambda *args, **kwargs: _FakeEncoder(),
    )
    actor = VJEPAActor.from_pretrained("/fake/model")
    assert actor.encoder is not None


def test_predict_latent_action_returns_tensor(monkeypatch):
    import torch
    class _FakeEncoder:
        def eval(self): return self
        def to(self, d): return self
        def __call__(self, **kw):
            class Out: last_hidden_state = torch.zeros(1, 196, 1024)
            return Out()

    monkeypatch.setattr(
        "worldfoundry_rl.actor._load_hf_model",
        lambda *args, **kwargs: _FakeEncoder(),
    )
    actor = VJEPAActor.from_pretrained("/fake/model")
    state = torch.zeros(1, 16, 3, 224, 224)
    actions = [1, 2, 3]
    out = actor.predict_latent_action(state, actions)
    assert out.shape == (1, 196, 1024)