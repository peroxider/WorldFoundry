"""Unit tests for the M3-T1 VAE building blocks.

Covers the real architecture (not the random-projection stub): shape
contracts, encode/decode round trips, determinism, training step, and
the ``from_pretrained`` weight-source hook.
"""
from __future__ import annotations

import io

import pytest
import torch

from worldfoundry_inf.diffusion.vae import (
    VAEConfig,
    VAEDecoder,
    VAEEncoder,
    VAEModel,
)


@pytest.fixture()
def cfg() -> VAEConfig:
    # Small CI-sized config: 3 downsample steps -> 8x compression,
    # latent_dim=4, base 32 channels.
    return VAEConfig(
        in_channels=3,
        latent_dim=4,
        base_channels=32,
        downsample_steps=3,
        groups=8,
    )


def test_vae_config_downscale_factor(cfg: VAEConfig) -> None:
    """3 downsample steps = 8x spatial compression."""
    assert cfg.downscale_factor == 8


def test_vae_encoder_shape(cfg: VAEConfig) -> None:
    """Encoder maps (B, 3, H, W) -> (mean, logvar) each (B, latent_dim, H/8, W/8)."""
    encoder = VAEEncoder(cfg)
    x = torch.randn(2, 3, 32, 32)
    mean, logvar = encoder(x)
    assert mean.shape == (2, 4, 4, 4)
    assert logvar.shape == (2, 4, 4, 4)


def test_vae_decoder_shape(cfg: VAEConfig) -> None:
    """Decoder maps (B, latent_dim, H/8, W/8) -> (B, 3, H, W)."""
    decoder = VAEDecoder(cfg)
    z = torch.randn(2, 4, 4, 4)
    out = decoder(z)
    assert out.shape == (2, 3, 32, 32)


def test_vae_model_encode_decode_roundtrip(cfg: VAEConfig) -> None:
    """encode -> decode round trips to the same spatial size."""
    model = VAEModel(cfg)
    x = torch.randn(2, 3, 32, 32)
    z = model.encode(x)
    assert z.shape == (2, 4, 4, 4)
    recon = model.decode(z)
    assert recon.shape == x.shape


def test_vae_reconstruct_matches_shape(cfg: VAEConfig) -> None:
    model = VAEModel(cfg)
    x = torch.randn(1, 3, 64, 64)
    assert model.reconstruct(x).shape == x.shape


def test_vae_model_determinism(cfg: VAEConfig) -> None:
    """Same seed -> same weights -> identical outputs."""
    torch.manual_seed(0)
    m1 = VAEModel(cfg)
    torch.manual_seed(0)
    m2 = VAEModel(cfg)
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(m1.encode(x), m2.encode(x))


def test_vae_training_step_returns_loss_and_metrics(cfg: VAEConfig) -> None:
    """training_step optimizes a KL-weighted reconstruction objective."""
    model = VAEModel(cfg)
    x = torch.randn(2, 3, 32, 32)
    loss, metrics = model.training_step(x)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(metrics) == {"recon", "kl"}
    assert metrics["recon"].ndim == 0
    assert metrics["kl"].ndim == 0


def test_vae_training_step_losses_are_seed_reproducible(cfg: VAEConfig) -> None:
    """Same weights + same sampling seed -> identical training-step loss.

    training_step uses reparameterization (``z = mean + eps*std`` with
    ``torch.randn_like``), so the sampling RNG must be fixed in addition
    to the weight init seed for the loss to be reproducible.
    """
    torch.manual_seed(1234)
    m1 = VAEModel(cfg)
    x = torch.randn(2, 3, 32, 32)
    torch.manual_seed(99)  # fix the reparam sampling for m1
    l1, _ = m1.training_step(x)
    torch.manual_seed(1234)
    m2 = VAEModel(cfg)
    torch.manual_seed(99)  # same sampling seed for m2
    l2, _ = m2.training_step(x)
    assert torch.allclose(l1, l2, atol=1e-6)


def test_vae_from_pretrained_no_checkpoint_seeded(cfg: VAEConfig) -> None:
    """from_pretrained(None) builds deterministic random weights."""
    m1 = VAEModel.from_pretrained(None, seed=7, cfg=cfg)
    m2 = VAEModel.from_pretrained(None, seed=7, cfg=cfg)
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(m1.encode(x), m2.encode(x))


def test_vae_from_pretrained_checkpoint_loads(cfg: VAEConfig, tmp_path) -> None:
    """A saved state_dict loads back through from_pretrained."""
    src = VAEModel(cfg)
    ckpt = tmp_path / "vae.pt"
    torch.save({"state_dict": src.state_dict()}, ckpt)
    loaded = VAEModel.from_pretrained(str(ckpt), seed=0, cfg=cfg)
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(src.encode(x), loaded.encode(x))


def test_vae_from_pretrained_bad_checkpoint_raises(cfg: VAEConfig, tmp_path) -> None:
    """A mismatched state_dict surfaces a ValueError with the keys."""
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {"not.a.key": torch.zeros(1)}}, bad)
    with pytest.raises(ValueError, match="missing|unexpected"):
        VAEModel.from_pretrained(str(bad), seed=0, cfg=cfg)


def test_vae_state_dict_is_portable(cfg: VAEConfig) -> None:
    """The full model state_dict survives a save/load round trip (memory)."""
    model = VAEModel(cfg)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, map_location="cpu")
    model2 = VAEModel(cfg)
    model2.load_state_dict(state)
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(model.encode(x), model2.encode(x))
