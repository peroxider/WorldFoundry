"""Unit tests for the M3-T1 DiT-class transformer backbone.

Covers shape contracts, time conditioning, optional action
conditioning, determinism, and the ``from_pretrained`` weight-source
hook.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_inf.diffusion.transformer import (
    DiTConfig,
    LatentDiffusionTransformer,
)


@pytest.fixture()
def cfg() -> DiTConfig:
    # Small CI-sized config.
    return DiTConfig(
        latent_dim=64,
        hidden_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cond_dim=0,
        embed_dim=64,
    )


@pytest.fixture()
def cond_cfg() -> DiTConfig:
    # Same but with action conditioning.
    return DiTConfig(
        latent_dim=64,
        hidden_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cond_dim=8,
        embed_dim=64,
    )


def test_transformer_shape_unconditioned(cfg: DiTConfig) -> None:
    """(B, N, latent_dim) + (B,) t -> (B, N, latent_dim)."""
    model = LatentDiffusionTransformer(cfg)
    x = torch.randn(2, 16, 64)
    t = torch.tensor([0.5, 0.9])
    out = model(x, t)
    assert out.shape == x.shape


def test_transformer_output_is_finite(cfg: DiTConfig) -> None:
    model = LatentDiffusionTransformer(cfg)
    x = torch.randn(1, 8, 64)
    t = torch.tensor([0.3])
    out = model(x, t)
    assert torch.isfinite(out).all()


def test_transformer_bad_input_ndim_raises(cfg: DiTConfig) -> None:
    model = LatentDiffusionTransformer(cfg)
    with pytest.raises(ValueError, match=r"\(B, N, latent_dim\)"):
        model(torch.randn(2, 16), torch.tensor([0.5, 0.5]))


def test_transformer_bad_latent_dim_raises(cfg: DiTConfig) -> None:
    model = LatentDiffusionTransformer(cfg)
    with pytest.raises(ValueError, match="latent_dim"):
        model(torch.randn(2, 16, 32), torch.tensor([0.5, 0.5]))


def test_transformer_bad_t_shape_raises(cfg: DiTConfig) -> None:
    model = LatentDiffusionTransformer(cfg)
    with pytest.raises(ValueError, match="t must be"):
        model(torch.randn(2, 16, 64), torch.tensor([0.5]))


def test_transformer_time_conditioning_changes_output(cfg: DiTConfig) -> None:
    """Different timesteps -> different denoised outputs for same input."""
    model = LatentDiffusionTransformer(cfg)
    x = torch.randn(1, 8, 64)
    out_a = model(x, torch.tensor([0.1]))
    out_b = model(x, torch.tensor([0.9]))
    assert not torch.allclose(out_a, out_b)


def test_transformer_determinism(cfg: DiTConfig) -> None:
    """Same seed -> same weights -> identical outputs."""
    torch.manual_seed(0)
    m1 = LatentDiffusionTransformer(cfg)
    torch.manual_seed(0)
    m2 = LatentDiffusionTransformer(cfg)
    x = torch.randn(2, 16, 64)
    t = torch.tensor([0.5, 0.7])
    assert torch.equal(m1(x, t), m2(x, t))


def test_transformer_cond_missing_raises(cond_cfg: DiTConfig) -> None:
    """cond_dim>0 without cond raises an actionable error."""
    model = LatentDiffusionTransformer(cond_cfg)
    with pytest.raises(ValueError, match="cond=None"):
        model(torch.randn(2, 16, 64), torch.tensor([0.5, 0.5]))


def test_transformer_cond_shape(cond_cfg: DiTConfig) -> None:
    """(B, cond_dim) conditioning is accepted and shape-preserving."""
    model = LatentDiffusionTransformer(cond_cfg)
    x = torch.randn(2, 16, 64)
    t = torch.tensor([0.5, 0.7])
    cond = torch.randn(2, 8)
    out = model(x, t, cond)
    assert out.shape == x.shape


def test_transformer_cond_changes_output(cond_cfg: DiTConfig) -> None:
    """Different conditioning vectors -> different outputs."""
    model = LatentDiffusionTransformer(cond_cfg)
    x = torch.randn(1, 8, 64)
    t = torch.tensor([0.5])
    out_a = model(x, t, torch.randn(1, 8))
    out_b = model(x, t, torch.randn(1, 8))
    assert not torch.allclose(out_a, out_b)


def test_transformer_state_dict_portable(cfg: DiTConfig, tmp_path) -> None:
    """A saved state_dict loads back through from_pretrained."""
    src = LatentDiffusionTransformer(cfg)
    ckpt = tmp_path / "dit.pt"
    torch.save({"state_dict": src.state_dict()}, ckpt)
    loaded = LatentDiffusionTransformer.from_pretrained(str(ckpt), seed=0, cfg=cfg)
    x = torch.randn(1, 8, 64)
    t = torch.tensor([0.5])
    assert torch.equal(src(x, t), loaded(x, t))


def test_transformer_from_pretrained_seeded(cfg: DiTConfig) -> None:
    """from_pretrained(None) builds deterministic random weights."""
    m1 = LatentDiffusionTransformer.from_pretrained(None, seed=7, cfg=cfg)
    m2 = LatentDiffusionTransformer.from_pretrained(None, seed=7, cfg=cfg)
    x = torch.randn(1, 8, 64)
    t = torch.tensor([0.5])
    assert torch.equal(m1(x, t), m2(x, t))


def test_transformer_backward_passes(cfg: DiTConfig) -> None:
    """The model is trainable end-to-end (gradient flows)."""
    model = LatentDiffusionTransformer(cfg)
    x = torch.randn(2, 16, 64)
    t = torch.tensor([0.5, 0.7])
    target = torch.zeros_like(x)
    loss = torch.nn.functional.mse_loss(model(x, t), target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one parameter with a gradient"
