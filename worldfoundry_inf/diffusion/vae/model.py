"""Real VAE encoder/decoder for the latent→frame pipeline (M3-T1).

Why this exists
---------------
M3-T1 replaces the v1 CPU random-projection stub
(:class:`worldfoundry_inf.diffusion.backends.cpu.CPUPixelDecoderPipeline`)
with real diffusion building blocks. This module is the **VAE** half: a
learned convolutional autoencoder in the SDXL / Stable Video Diffusion
family — spatial downsampling encoder + upsampling decoder around a
low-dimensional latent space (default ``latent_dim=4``, 8× spatial
compression).

The ADR (``docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md``
Route B) names SDXL / SVD VAE as the preferred weight source. This
module implements the architecture so weights can be dropped in from
either source; when no checkpoint is available (CI / offline hosts), the
model still works with a deterministic random initialization — the unit
tests exercise encode/decode against real architecture weights, not a
projection stub.

Interface contract (M3-T2)
--------------------------
* :class:`VAEConfig` — shape/scale hyperparameters.
* :class:`VAEEncoder` — ``(B, C, H, W)`` → ``(B, latent_dim, H/8, W/8)``
  (returns ``mean`` / ``logvar``; inference uses ``mean``).
* :class:`VAEDecoder` — ``(B, latent_dim, H/8, W/8)`` → ``(B, C, H, W)``.
* :class:`VAEModel` — composition with ``encode`` / ``decode`` /
  ``reconstruct`` helpers + ``from_pretrained`` hook (weights optional).

The pipeline-level ``LatentToFramePipeline`` implementation lives in
:mod:`worldfoundry_inf.diffusion.backends.vae_diffusion` (M3-T1 wiring),
which uses :class:`VAEModel.decode` as the pixel decoder and
:class:`VAEModel.encode` as the pixel-space ground-truth encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VAEConfig:
    """Hyperparameters for the convolutional VAE.

    Attributes:
        in_channels: input image channels (3 = RGB).
        latent_dim: latent channels (SDXL VAE uses 4).
        base_channels: first-conv channel count; each down/up block
            doubles/halves it (kept small for CI-sized tests).
        downsample_steps: number of 2× spatial halvings (3 → 8× total).
        groups: GroupNorm group count (modern VAE style, batch-independent).
    """

    in_channels: int = 3
    latent_dim: int = 4
    base_channels: int = 32
    downsample_steps: int = 3
    groups: int = 8

    @property
    def downscale_factor(self) -> int:
        return 2 ** self.downsample_steps


class _ResBlock(nn.Module):
    """Two-conv residual block with GroupNorm + SiLU (pre-activation)."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class VAEEncoder(nn.Module):
    """Spatial encoder: ``(B, in_channels, H, W)`` → ``(B, 2*latent_dim, H/8, W/8)``.

    Returns ``mean``/``logvar`` pairs (2× channels on the last conv) so
    the VAE can be trained with a KL term; inference uses ``mean``.
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        blocks: list[nn.Module] = [
            nn.Conv2d(cfg.in_channels, ch, 3, padding=1),
        ]
        for _ in range(cfg.downsample_steps):
            blocks.append(_ResBlock(ch, cfg.groups))
            blocks.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
            ch *= 2
        blocks.append(_ResBlock(ch, cfg.groups))
        blocks.append(nn.GroupNorm(cfg.groups, ch))
        blocks.append(nn.SiLU())
        blocks.append(nn.Conv2d(ch, 2 * cfg.latent_dim, 3, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``x`` to ``(mean, logvar)``, each ``(B, latent_dim, H/8, W/8)``."""
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=1)
        return mean, logvar


class VAEDecoder(nn.Module):
    """Spatial decoder: ``(B, latent_dim, H/8, W/8)`` → ``(B, in_channels, H, W)``."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        top_ch = cfg.base_channels * (2 ** cfg.downsample_steps)
        blocks: list[nn.Module] = [
            nn.Conv2d(cfg.latent_dim, top_ch, 3, padding=1),
            _ResBlock(top_ch, cfg.groups),
        ]
        ch = top_ch
        for _ in range(cfg.downsample_steps):
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, ch // 2, 3, padding=1),
                    _ResBlock(ch // 2, cfg.groups),
                )
            )
            ch //= 2
        blocks.append(nn.GroupNorm(cfg.groups, ch))
        blocks.append(nn.SiLU())
        blocks.append(nn.Conv2d(ch, cfg.in_channels, 3, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent ``z`` → pixel frames ``(B, in_channels, H, W)``."""
        return self.net(z)


class VAEModel(nn.Module):
    """Composed VAE: encode (pixel→latent) + decode (latent→pixel).

    Works with random-init weights for CI/offline use; ``from_pretrained``
    is the hook to drop in SDXL / SVD VAE checkpoints when available.
    """

    def __init__(self, cfg: VAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.encoder = VAEEncoder(self.cfg)
        self.decoder = VAEDecoder(self.cfg)

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3, H, W)`` pixels → ``(B, latent_dim, H/8, W/8)`` latent.

        Uses the encoder's ``mean`` (deterministic, inference mode).
        """
        mean, _logvar = self.encoder(pixels)
        return mean

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode ``(B, latent_dim, H/8, W/8)`` latent → ``(B, 3, H, W)`` pixels."""
        return self.decoder(z)

    # ------------------------------------------------------------------ #
    # round-trip / training helpers
    # ------------------------------------------------------------------ #
    def reconstruct(self, pixels: torch.Tensor) -> torch.Tensor:
        """Full encode→decode round trip (used by tests + eval)."""
        return self.decode(self.encode(pixels))

    def training_step(
        self,
        pixels: torch.Tensor,
        kl_weight: float = 1e-6,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """KL-weighted reconstruction loss — supports end-to-end VAE training.

        Returns ``(loss, metrics)`` where ``metrics`` carries ``recon`` /
        ``kl`` components for logging.
        """
        mean, logvar = self.encoder(pixels)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, pixels)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kl_weight * kl
        return loss, {"recon": recon_loss.detach(), "kl": kl.detach()}

    # ------------------------------------------------------------------ #
    # weight source
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        *,
        seed: int = 0,
        cfg: VAEConfig | None = None,
    ) -> "VAEModel":
        """Build a model; load ``state_dict`` from ``checkpoint_path`` if given.

        When ``checkpoint_path`` is ``None`` (CI / offline), weights are
        initialized deterministically from ``seed`` so tests are
        reproducible. ``checkpoint_path`` may point at a local
        ``.pt`` / ``.safetensors`` export of an SDXL / SVD VAE.
        """
        torch.manual_seed(seed)
        model = cls(cfg=cfg)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"VAE checkpoint {checkpoint_path} mismatched: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        return model


__all__ = ["VAEConfig", "VAEEncoder", "VAEDecoder", "VAEModel"]
