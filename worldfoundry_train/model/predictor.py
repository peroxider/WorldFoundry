"""JEPA predictor Transformer (W22.1).

Design source: docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md §4.

A narrow-but-deep Transformer (default 12 x 384 x 12 heads) that consumes the
online encoder's full-sequence output, replaces masked positions with a
learnable mask token (late masking, done *inside* this module), adds a
learnable 3D spatiotemporal position embedding, and predicts the target
encoder's representation at the masked positions.

Mask semantics note (empirically verified for transformers 5.1.0, see
``tests/test_jepa_objective.py::test_hf_encoder_ignores_bool_masked_pos``):
HF ``VJEPA2Model.forward`` does NOT consume ``bool_masked_pos`` — the encoder
always processes the full video. Masking therefore happens exclusively here,
in the predictor input construction, which is exactly the dataflow of design
§4.3 step 2. This module is self-contained and does not depend on any HF
masking behavior.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["JEPAPredictor"]


def _init_trunc_normal(module: nn.Module, std: float = 0.02) -> None:
    for p in module.parameters():
        if p.ndim >= 2:
            nn.init.trunc_normal_(p, std=std)


class JEPAPredictor(nn.Module):
    """Predict target-encoder features at masked positions from context features.

    Args:
        num_patches: N = T*Hg*Wg of the training grid (must match the
            collator's flattened ``bool_masked_pos`` layout, C-order
            ``(T, Hg, Wg)`` row-major, cf. ``data/collator.py:36``).
        grid_size: ``(T, Hg, Wg)`` matching ``num_patches``. Used to lay out
            the learnable 3D position embedding and as the source grid for
            trilinear interpolation when a batch has a different grid.
        encoder_dim: online/target encoder hidden size (1024 for ViT-L).
        predictor_dim: bottleneck width (official vitl ``pred_embed_dim=384``).
        depth: number of Transformer layers (official ``pred_depth=12``).
        num_heads: attention heads (12 -> 32 dims/head).
        mlp_ratio: feedforward expansion (4.0, ViT convention).
    """

    def __init__(
        self,
        num_patches: int,
        grid_size: tuple[int, int, int],
        encoder_dim: int = 1024,
        predictor_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        t, hg, wg = grid_size
        if t * hg * wg != num_patches:
            raise ValueError(
                f"grid_size {grid_size} implies {t * hg * wg} patches, "
                f"but num_patches={num_patches}"
            )
        self.num_patches = num_patches
        self.grid_size = grid_size
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_heads = num_heads

        self.proj_in = nn.Linear(encoder_dim, predictor_dim)
        # Learnable mask token, trunc_normal(0.02) init (design §4.2).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Learnable 3D spatiotemporal position embedding on the training grid.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_out = nn.Linear(predictor_dim, encoder_dim)

        # trunc_normal(0.02) for all matmul-like weights (design §4.1);
        # zero-init the *last* block's MLP output projection so the predictor
        # starts near zero output and the EMA target dominates early gradient
        # scale (stable warmup).
        _init_trunc_normal(self.proj_in)
        _init_trunc_normal(self.blocks)
        _init_trunc_normal(self.proj_out)
        last: nn.TransformerEncoderLayer = self.blocks.layers[-1]
        nn.init.zeros_(last.linear2.weight)
        nn.init.zeros_(last.linear2.bias)

    def _pos_embed_for(self, grid_size: tuple[int, int, int]) -> torch.Tensor:
        """Position embedding for ``grid_size``, interpolating if it differs.

        Layout matches the collator: index ``i`` of the flattened axis maps to
        ``(t, h, w)`` in C-order over ``(T, Hg, Wg)``.
        """
        if tuple(grid_size) == tuple(self.grid_size):
            return self.pos_embed
        t0, h0, w0 = self.grid_size
        d = self.predictor_dim
        pe = self.pos_embed.reshape(1, t0, h0, w0, d).permute(0, 4, 1, 2, 3)
        pe = F.interpolate(
            pe, size=grid_size, mode="trilinear", align_corners=False
        )
        return pe.permute(0, 2, 3, 4, 1).reshape(1, -1, d)

    def forward(
        self,
        context_feats: torch.Tensor,
        bool_masked_pos: torch.Tensor,
        grid_size: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Predict target features at masked positions.

        Args:
            context_feats: ``(B, N, encoder_dim)`` online encoder full-sequence
                output (has grad).
            bool_masked_pos: ``(B, N)`` bool, True = position to predict.
            grid_size: actual ``(T, Hg, Wg)`` of this batch; ``None`` = the
                construction grid. A different grid triggers trilinear
                interpolation of the position embedding (design §4.2).

        Returns:
            ``(M_total, encoder_dim)`` predictions at masked positions, in the
            same order as ``context_feats[bool_masked_pos]``. An all-False
            mask yields an empty ``(0, encoder_dim)`` tensor (no error).
        """
        if context_feats.ndim != 3:
            raise ValueError(
                f"context_feats must be (B, N, D), got {tuple(context_feats.shape)}"
            )
        b, n, _ = context_feats.shape
        if bool_masked_pos.shape != (b, n):
            raise ValueError(
                f"bool_masked_pos shape {tuple(bool_masked_pos.shape)} does not "
                f"match (B, N)=({b}, {n})"
            )
        mask = bool_masked_pos.bool()

        x = self.proj_in(context_feats)  # (B, N, predictor_dim)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, n, -1), x)
        x = x + self._pos_embed_for(grid_size or self.grid_size)
        x = self.blocks(x)
        x = self.proj_out(x)  # (B, N, encoder_dim)
        return x[mask]
