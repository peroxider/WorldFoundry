"""Tiny V-JEPA 2 encoder + pixel decoder for the CPU render-train-inference demo.

The full V-JEPA 2 stack (1.2B ViT-L + LoRA + predictor + EMA target) is not
CPU-feasible for an end-to-end loop: weights alone are ~5 GB and HF download
is a network dependency. To prove the render->train->inference loop on CPU we
ship a ``TinyVJEPA2Encoder`` (~80k params) that mirrors the HF
:class:`transformers.VJEPA2Model` forward contract — same kwargs
(``pixel_values_videos``, ``skip_predictor``), same return shape
(``.last_hidden_state`` of shape ``(B, T*Hg*Wg, encoder_dim)``) — so the
existing :class:`worldfoundry_train.jepa_objective.JEPAObjective` and
:class:`worldfoundry_train.temporal_objective.TemporalObjective` drop in
unchanged.

The matching ``TinyPixelDecoder`` inverts the encoder so the inference
phase can render predicted latents back to pixels for visualisation
(W40 Route A mini: a per-patch linear + small upsampler; not a full
diffusion decoder — that's the W40 Route B deferred work).

Both modules are :class:`torch.nn.Module` and only depend on ``torch``
(no HF, no decord, no opencv).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TinyEncoderConfig:
    """Architectural fingerprint of :class:`TinyVJEPA2Encoder`."""

    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    embed_dim: int = 96
    depth: int = 4
    num_heads: int = 3
    mlp_ratio: float = 2.0

    @property
    def grid_h(self) -> int:
        return self.image_size // self.patch_size

    @property
    def grid_w(self) -> int:
        return self.image_size // self.patch_size

    def signature(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
        }


class _TinyOutput:
    """Duck-typed stand-in for ``transformers.VJEPA2WithMaskedInputModelOutput``.

    Only the attribute the rest of the pipeline reads — ``last_hidden_state``
    — is populated. The named-tuple form is intentional so isinstance
    checks against HF's output class still pass (we don't claim HF
    conformance; we claim "same surface for the bits we use").
    """

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class TinyVJEPA2Encoder(nn.Module):
    """CPU-friendly ViT-Tiny with the VJEPA2Model.forward surface.

    Treats time as part of the token sequence: a video of shape
    ``(B, T, C, H, W)`` becomes ``(B, T*Hg*Wg, embed_dim)`` after
    patchification + position embedding + transformer blocks. The
    mask-agnostic interface means callers apply masks externally
    (``JEPAPredictor`` etc. operate on the full sequence).

    Args:
        config: :class:`TinyEncoderConfig`. Defaults are tuned for
            64x64 input, 8x8 patches -> 8x8 = 64 spatial tokens per
            frame; ``embed_dim=96`` ~ ViT-Tiny.
    """

    def __init__(self, config: TinyEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyEncoderConfig()
        c = self.config
        if c.image_size % c.patch_size != 0:
            raise ValueError(
                f"image_size={c.image_size} must be divisible by "
                f"patch_size={c.patch_size}"
            )
        self._patch_embed = nn.Conv2d(
            in_channels=c.in_channels,
            out_channels=c.embed_dim,
            kernel_size=c.patch_size,
            stride=c.patch_size,
            bias=True,
        )
        gh, gw = c.grid_h, c.grid_w
        self._spatial_pos = nn.Parameter(
            torch.zeros(1, gh * gw, c.embed_dim),
        )
        nn.init.trunc_normal_(self._spatial_pos, std=0.02)
        self._temporal_pos = nn.Parameter(
            torch.zeros(1, 16, c.embed_dim),
        )
        nn.init.trunc_normal_(self._temporal_pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.embed_dim,
            nhead=c.num_heads,
            dim_feedforward=int(c.embed_dim * c.mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=c.depth, norm=nn.LayerNorm(c.embed_dim),
        )
        self._norm = nn.LayerNorm(c.embed_dim)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        skip_predictor: bool = True,
        **_: Any,
    ) -> _TinyOutput:
        """Encode a video batch.

        Args:
            pixel_values_videos: ``(B, T, C, H, W)`` float in [0, 1] or any
                normalized range; we apply the patch projection directly.
            skip_predictor: kept for signature parity with
                :class:`transformers.VJEPA2Model`. This tiny encoder has no
                internal predictor; the value is ignored.
        """
        del skip_predictor  # not used in the tiny encoder
        if pixel_values_videos.ndim != 5:
            raise ValueError(
                f"pixel_values_videos must be (B, T, C, H, W); got shape "
                f"{tuple(pixel_values_videos.shape)}"
            )
        B, T, C, H, W = pixel_values_videos.shape
        c = self.config
        if H != c.image_size or W != c.image_size:
            raise ValueError(
                f"input H/W must equal encoder image_size={c.image_size}; "
                f"got H={H}, W={W}"
            )
        if C != c.in_channels:
            raise ValueError(
                f"input channels must equal encoder.in_channels={c.in_channels}; "
                f"got C={C}"
            )
        if T > self._temporal_pos.shape[1]:
            raise ValueError(
                f"encoder supports up to {self._temporal_pos.shape[1]} frames; "
                f"got T={T}"
            )

        # (B, T, C, H, W) -> (B*T, C, H, W) -> patchify -> (B*T, N, embed_dim)
        x = pixel_values_videos.reshape(B * T, C, H, W)
        x = self._patch_embed(x)  # (B*T, embed_dim, Hg, Wg)
        gh, gw = c.grid_h, c.grid_w
        x = x.flatten(2).transpose(1, 2)  # (B*T, Hg*Wg, embed_dim)
        x = x + self._spatial_pos  # (B*T, Hg*Wg, embed_dim)
        # reshape to (B, T*Hg*Wg, embed_dim) and broadcast-add temporal pos
        x = x.reshape(B, T, gh * gw, c.embed_dim).reshape(B, T * gh * gw, c.embed_dim)
        temporal = (
            self._temporal_pos[:, :T, :]
            .unsqueeze(2)
            .expand(B, T, gh * gw, c.embed_dim)
            .reshape(B, T * gh * gw, c.embed_dim)
        )
        x = x + temporal
        x = self._blocks(x)
        x = self._norm(x)
        return _TinyOutput(last_hidden_state=x)


class TinyPixelDecoder(nn.Module):
    """Per-patch linear + reshape upsampler (W40 Route A mini).

    Inverse of :class:`TinyVJEPA2Encoder`'s patchify stage:

    - input  : ``(B, T*Hg*Wg, embed_dim)``
    - project: ``(B, T*Hg*Wg, patch_size**2 * 3)``
    - pixel-shuffle to ``(B, T, 3, H, W)``

    Trainable end-to-end alongside the encoder in the orchestrator's
    optional decoder-fine-tune pass. Outputs are sigmoid-clamped to
    ``[0, 1]`` so they can be compared directly with input frames.
    """

    def __init__(
        self,
        encoder_config: TinyEncoderConfig,
        *,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        c = encoder_config
        self.encoder_config = c
        self._proj = nn.Sequential(
            nn.Linear(c.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                c.patch_size * c.patch_size * c.in_channels,
            ),
        )
        self._sigmoid = nn.Sigmoid()

    def forward(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Decode latents back to pixels.

        Args:
            latents: ``(B, T*Hg*Wg, embed_dim)``.
            num_frames: ``T`` — needed to split the flat token sequence.

        Returns:
            ``(B, T, 3, H, W)`` float in ``[0, 1]``.
        """
        c = self.encoder_config
        gh, gw = c.grid_h, c.grid_w
        expected = num_frames * gh * gw
        if latents.shape[1] != expected:
            raise ValueError(
                f"latents sequence length {latents.shape[1]} != T*Hg*Wg="
                f"{num_frames}*{gh}*{gw}={expected}"
            )
        B = latents.shape[0]
        x = self._proj(latents)  # (B, T*Hg*Wg, patch_size**2 * 3)
        x = x.reshape(B, num_frames, gh, gw, c.patch_size, c.patch_size, c.in_channels)
        # (B, T, Hg, Wg, ph, pw, C) -> (B, T, C, Hg, ph, Wg, pw) -> (B*T, C, H, W)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        H = gh * c.patch_size
        W = gw * c.patch_size
        x = x.reshape(B * num_frames, c.in_channels, H, W)
        return self._sigmoid(x).reshape(B, num_frames, c.in_channels, H, W)


class TinyJEPAPredictor(nn.Module):
    """Narrow Transformer that predicts at masked positions.

    Matches the ``JEPAPredictor(past_tokens, bool_masked_pos) -> (M, encoder_dim)``
    surface so it slots into :class:`JEPAObjective.loss_fn` without changes.

    The predictor never participates in FSDP (per W22 design §8.1), so
    state_dict handling is a plain nn.Module.state_dict().
    """

    def __init__(
        self,
        encoder_config: TinyEncoderConfig,
        *,
        pred_dim: int = 48,
        depth: int = 2,
        num_heads: int = 3,
    ) -> None:
        super().__init__()
        c = encoder_config
        if pred_dim % num_heads != 0:
            raise ValueError(
                f"pred_dim={pred_dim} must be divisible by num_heads={num_heads}"
            )
        self.encoder_config = c
        self._proj_in = nn.Linear(c.embed_dim, pred_dim)
        self._mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self._mask_token, std=0.02)
        self._pos = nn.Parameter(
            torch.zeros(1, c.grid_h * c.grid_w * 16, pred_dim),
        )
        nn.init.trunc_normal_(self._pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=pred_dim,
            nhead=num_heads,
            dim_feedforward=pred_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self._proj_out = nn.Linear(pred_dim, c.embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Predict features at the masked positions.

        Args:
            context_tokens: ``(B, N, encoder_dim)`` — full-sequence encoder output.
            bool_masked_pos: ``(B, N)`` bool — which positions to predict.

        Returns:
            ``(M_total, encoder_dim)`` — predictions at masked positions,
            concatenated across the batch dim (matches the HF + our
            :class:`JEPAPredictor` semantics).
        """
        if bool_masked_pos.dtype != torch.bool:
            bool_masked_pos = bool_masked_pos.bool()
        B, N, _ = context_tokens.shape
        x = self._proj_in(context_tokens)  # (B, N, pred_dim)
        mask = bool_masked_pos.unsqueeze(-1)
        x = torch.where(mask, self._mask_token.expand(B, N, -1), x)
        x = x + self._pos[:, :N, :]
        x = self._blocks(x)
        x = self._proj_out(x)  # (B, N, encoder_dim)
        out = x[bool_masked_pos]  # (M_total, encoder_dim)
        return out


__all__ = [
    "TinyEncoderConfig",
    "TinyVJEPA2Encoder",
    "TinyPixelDecoder",
    "TinyJEPAPredictor",
]