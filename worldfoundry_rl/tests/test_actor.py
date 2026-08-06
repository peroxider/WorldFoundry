import pytest
import torch

from worldfoundry_rl.actor import VJEPAActor


class _FakeCfg:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size


class _FakeEncoder:
    """Drop-in encoder for VJEPAActor tests (exposes `config.hidden_size`)."""

    def __init__(self, hidden_size: int = 1024, patch_tokens: int = 196) -> None:
        self.config = _FakeCfg(hidden_size=hidden_size)
        self._patch_tokens = patch_tokens

    def eval(self):
        return self

    def to(self, d):  # HF VJEPA2Model has a .to() but we ignore dtype here
        return self

    def __call__(self, **kw):
        # The actor only consumes pixel_values_videos, so the batch dim
        # is the leading dim of the state tensor it was given.
        state = kw.get("pixel_values_videos")
        batch = state.shape[0] if state is not None else 1
        out = torch.zeros(batch, self._patch_tokens, self.config.hidden_size)
        return type("Out", (), {"last_hidden_state": out})()


@pytest.fixture
def fake_encoder():
    return _FakeEncoder()


def test_actor_init_loads_model(monkeypatch):
    monkeypatch.setattr(
        "worldfoundry_rl.actor._load_hf_model",
        lambda *args, **kwargs: _FakeEncoder(),
    )
    actor = VJEPAActor.from_pretrained("/fake/model")
    assert actor.encoder is not None
    assert actor.head is not None
    assert actor.action_dim == 7  # default


def test_predict_latent_action_shape_and_dtype(fake_encoder):
    actor = VJEPAActor(fake_encoder, action_dim=7)
    state = torch.zeros(1, 16, 3, 224, 224)
    current_action = torch.zeros(1, 7)
    out = actor.predict_latent_action(state, current_action)
    # W32 contract: (B, action_dim), not (B, N, encoder_dim) anymore.
    assert out.shape == (1, 7)


def test_predict_latent_action_zero_at_init(fake_encoder):
    """W32: head is zero-initialized, so the actor is a zero function at init.

    Matches the pre-W32 passthrough baseline (no behavior change at day-0).
    """
    actor = VJEPAActor(fake_encoder, action_dim=7)
    state = torch.randn(2, 16, 3, 224, 224)
    current_action = torch.randn(2, 7)
    out = actor.predict_latent_action(state, current_action)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_predict_latent_action_changes_after_head_perturbation(fake_encoder):
    """W32: not a passthrough — perturbing the head's last linear's bias
    (which is zero at init) makes the output non-zero, breaking the
    passthrough equivalence. We perturb the bias because the last
    linear's weight is also zero-init, so perturbing the weight only
    would still produce a near-zero output (the head is still
    well-behaved under W22-style warmup stability)."""
    actor = VJEPAActor(fake_encoder, action_dim=7)
    state = torch.randn(1, 16, 3, 224, 224)
    current_action = torch.randn(1, 7)
    out_before = actor.predict_latent_action(state, current_action)
    with torch.no_grad():
        actor.head.net[-1].bias.add_(0.5)
    out_after = actor.predict_latent_action(state, current_action)
    assert not torch.allclose(out_before, out_after)
    assert torch.allclose(out_before, torch.zeros_like(out_before), atol=1e-6)
    assert not torch.allclose(out_after, torch.zeros_like(out_after), atol=1e-6)


def test_actor_custom_action_dim(fake_encoder):
    actor = VJEPAActor(fake_encoder, action_dim=4)
    state = torch.zeros(1, 16, 3, 224, 224)
    current_action = torch.zeros(1, 4)
    out = actor.predict_latent_action(state, current_action)
    assert out.shape == (1, 4)
    assert actor.head.action_dim == 4


def test_actor_init_rejects_encoder_without_config():
    class _NoCfg:
        def eval(self): return self
        def __call__(self, **kw):
            raise AssertionError("should not be called")

    with pytest.raises(ValueError, match="config.hidden_size"):
        VJEPAActor(_NoCfg(), action_dim=7)
