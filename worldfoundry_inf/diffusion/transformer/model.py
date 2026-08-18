"""DiT-class backbone for the latent→frame diffusion pipeline (M3-T1).

Why this exists
---------------
M3-T1 replaces the v1 CPU random-projection stub with real diffusion
building blocks. This module is the **transformer** half: a
DiT-class (Diffusion Transformer) backbone in the UViT / PixArt-α
family — sinusoidal time embedding + adaLN-Zero modulated transformer
blocks over the latent token sequence. It is the denoiser that maps
``(noisy_latent_tokens, timestep, optional_condition)`` →
``(predicted_clean_latent_tokens)`` inside the DDIM / Flow-Matching
sampling loop (see :mod:`worldfoundry_inf.diffusion.scheduler`).

Design notes
------------
* **adaLN-Zero modulation** (DiT, Peebles & Xie 2023): the time +
  condition embedding drives per-block ``shift/scale/gate`` for the
  pre-attention norm — the standard DiT conditioning mechanism.
* **Token-first layout**: input is ``(B, N, latent_dim)`` (patch
  tokens, ``N = T*Hg*Wg`` from the JEPA grid), so it plugs directly
  into :class:`LatentToFramePipeline.decode_latents` semantics.
* **Optional action conditioning**: ``cond`` ``(B, cond_dim)`` is
  concatenated into the adaLN embedding, matching the orchestrator's
  action-conditioned rollout seam (FiLM-style additive conditioning).
* Works with random-init weights for CI / offline; ``from_pretrained``
  is the hook for a UViT / PixArt-α checkpoint when available.

Interface contract (M3-T2)
--------------------------
* :class:`DiTConfig` — model hyperparameters.
* :class:`LatentDiffusionTransformer` — ``(B, N, latent_dim)`` +
  ``(B,)`` timestep (+ optional ``(B, cond_dim)`` cond) →
  ``(B, N, latent_dim)`` predicted clean latent.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DiTConfig:
    """Hyperparameters for the latent diffusion transformer.

    Attributes:
        latent_dim: input/output token dimension (JEPA latent dim).
        hidden_dim: transformer hidden width.
        depth: number of DiT blocks.
        num_heads: attention head count.
        mlp_ratio: MLP hidden = ``hidden_dim * mlp_ratio``.
        cond_dim: dimension of the optional conditioning vector
            (e.g. action features), or 0 for no conditioning.
        embed_dim: sinusoidal time-embedding dimension (must be
            divisible by 2).
    """

    latent_dim: int = 64
    hidden_dim: int = 128
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    cond_dim: int = 0
    embed_dim: int = 64


class _TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding → MLP projection."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        half = embed_dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half) / half))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` (B,) → ``(B, hidden_dim)``."""
        emb = t.float().unsqueeze(-1) * self.freqs  # (B, half)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class _Modulation(nn.Module):
    """adaLN-Zero per-block modulation generator.

    Maps the combined (time + cond) embedding to the six modulation
    parameters (shift/scale/gate for the pre-attention norm, then
    shift/scale/gate for the pre-MLP norm) laid out as
    ``(B, 6, hidden_dim)``:
        channel 0/1/2 = shift1/scale1/gate1 (attention norm)
        channel 3/4/5 = shift2/scale2/gate2 (MLP norm)
    """

    def __init__(self, cond_hidden: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_hidden, 6 * hidden_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 6, hidden_dim)`` modulation parameters."""
        return self.net(emb).view(-1, 6, self.hidden_dim)


class _DiTBlock(nn.Module):
    """Pre-norm attention + MLP with adaLN-Zero modulation."""

    def __init__(self, cfg: DiTConfig, cond_hidden: int) -> None:
        super().__init__()
        h = cfg.hidden_dim
        self.norm1 = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(h, cfg.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(h, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(h, int(h * cfg.mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(h * cfg.mlp_ratio), h),
        )
        self.modulation = _Modulation(cond_hidden, h)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.modulation(emb)  # (B, 6, hidden)
        shift1, scale1, gate1 = mod[:, 0:1], mod[:, 1:2], mod[:, 2:3]
        shift2, scale2, gate2 = mod[:, 3:4], mod[:, 4:5], mod[:, 5:6]
        # Pre-attention adaLN: shift1/scale1/gate1
        h = self.norm1(x) * (1.0 + scale1) + shift1
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate1 * h
        # Pre-MLP adaLN: shift2/scale2/gate2
        h = self.norm2(x) * (1.0 + scale2) + shift2
        h = self.mlp(h)
        x = x + gate2 * h
        return x


class LatentDiffusionTransformer(nn.Module):
    """DiT-class denoiser over latent tokens (M3-T1 transformer).

    Args:
        cfg: model hyperparameters.

    Forward:
        ``(x: (B, N, latent_dim), t: (B,), cond: (B, cond_dim) | None)``
        → ``(B, N, latent_dim)`` predicted clean latent tokens.
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        cond_hidden = cfg.hidden_dim
        if cfg.cond_dim > 0:
            self.cond_proj = nn.Linear(cfg.cond_dim, cfg.hidden_dim)
        else:
            self.cond_proj = None
        self.time_embed = _TimestepEmbedder(cfg.embed_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [_DiTBlock(cfg, cond_hidden) for _ in range(cfg.depth)]
        )
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def _embed(self, t: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        emb = self.time_embed(t)
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError(
                    "LatentDiffusionTransformer configured with cond_dim>0 "
                    "but forward() received cond=None."
                )
            emb = emb + self.cond_proj(cond)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoise ``x`` at timestep ``t`` → predicted clean latent tokens."""
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, latent_dim); got {tuple(x.shape)}")
        if x.shape[-1] != self.cfg.latent_dim:
            raise ValueError(
                f"x last dim must equal latent_dim={self.cfg.latent_dim}; "
                f"got {x.shape[-1]}"
            )
        if t.shape != (x.shape[0],):
            raise ValueError(f"t must be (B,); got {tuple(t.shape)}")
        h = self.in_proj(x)
        emb = self._embed(t, cond)
        for block in self.blocks:
            h = block(h, emb)
        h = self.final_norm(h)
        return self.out_proj(h)

    # ------------------------------------------------------------------ #
    # weight source
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        *,
        seed: int = 0,
        cfg: DiTConfig | None = None,
    ) -> "LatentDiffusionTransformer":
        """Build a model; load ``state_dict`` from ``checkpoint_path`` if given.

        ``None`` checkpoint (CI / offline) → deterministic random init
        from ``seed``. ``checkpoint_path`` may point at a local export of
        a UViT / PixArt-α checkpoint (strict=False tolerant load).
        """
        torch.manual_seed(seed)
        model = cls(cfg=cfg or DiTConfig())
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"DiT checkpoint {checkpoint_path} mismatched: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        return model


__all__ = ["DiTConfig", "LatentDiffusionTransformer"]
