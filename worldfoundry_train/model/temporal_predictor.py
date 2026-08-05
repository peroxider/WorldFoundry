"""TemporalPredictor: past K frame latents -> future H frame latents (W24).

Reuses :class:`JEPAPredictor`'s ``proj_in / blocks / proj_out`` trunk (the
~21M-param Transformer body) so a W22-pretrained trunk can be warm-started;
adds task-specific learnable future query tokens plus separate past/future
position embeddings for the temporal "predict next" task.

Unlike :class:`JEPAPredictor` (which predicts target-encoder features at
**masked positions within the same N-token input grid**), TemporalPredictor
predicts at **positions not in the input**: it generates H future frame
latents from K past frame latents, with N_future possibly different from
N_past. The output sequence lives in a different grid (H, Hg', Wg') from
the input grid (K, Hg, Wg').

Mechanism
---------
Concatenate ``proj_in(past_feats) + past_pos_embed`` (B, N_past, pred_dim)
with the learned future queries plus future position embedding
``future_query + future_pos_embed`` (B, N_future, pred_dim) along the sequence
axis, run the reused TransformerEncoder trunk over the combined
N_past + N_future tokens, slice off the trailing N_future positions, and
project back to encoder_dim via ``proj_out``. Self-attention across the
combined sequence is how past information flows into future predictions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry_train.model.predictor import JEPAPredictor, _init_trunc_normal

__all__ = ["TemporalPredictor"]


class TemporalPredictor(nn.Module):
    """Past K frame latents -> Future H frame latents (world-model roll-out).

    Args:
        num_past_positions: ``N_past = K * Hg * Wg`` of the past grid.
        num_future_positions: ``N_future = H * Hg' * Wg'`` of the future grid.
        past_grid: ``(K, Hg, Wg)`` past frame grid.
        future_grid: ``(H, Hg', Wg')`` future frame grid.
        encoder_dim: online/target encoder hidden size (1024 for ViT-L).
        predictor_dim: bottleneck width; should match the source
            :class:`JEPAPredictor` for warm-start to be byte-for-byte.
        depth: number of Transformer blocks; should match the source
            :class:`JEPAPredictor`.
        num_heads: attention heads; should match the source.
        mlp_ratio: feedforward expansion.
    """

    def __init__(
        self,
        num_past_positions: int,
        num_future_positions: int,
        past_grid: tuple[int, int, int],
        future_grid: tuple[int, int, int],
        encoder_dim: int = 1024,
        predictor_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if past_grid[0] * past_grid[1] * past_grid[2] != num_past_positions:
            raise ValueError(
                f"past_grid {past_grid} implies "
                f"{past_grid[0] * past_grid[1] * past_grid[2]} positions, "
                f"but num_past_positions={num_past_positions}"
            )
        if future_grid[0] * future_grid[1] * future_grid[2] != num_future_positions:
            raise ValueError(
                f"future_grid {future_grid} implies "
                f"{future_grid[0] * future_grid[1] * future_grid[2]} positions, "
                f"but num_future_positions={num_future_positions}"
            )
        self.num_past_positions = num_past_positions
        self.num_future_positions = num_future_positions
        self.past_grid = past_grid
        self.future_grid = future_grid
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_heads = num_heads

        # Reused from JEPAPredictor (warm-start target via
        # from_w22_jepa_predictor below): identical submodule types / shapes
        # so load_state_dict transfers weights byte-for-byte.
        self.proj_in = nn.Linear(encoder_dim, predictor_dim)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=predictor_dim,
                nhead=num_heads,
                dim_feedforward=int(predictor_dim * mlp_ratio),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.proj_out = nn.Linear(predictor_dim, encoder_dim)

        # Task-specific (NOT warm-started): past and future grids differ, so
        # JEPAPredictor's single pos_embed (indexing one grid) cannot be
        # reused without introducing cross-grid interpolation ambiguity.
        # A separate learnable embed per side is simpler and ~3 KB / side.
        self.past_pos_embed = nn.Parameter(
            torch.zeros(1, num_past_positions, predictor_dim)
        )
        self.future_pos_embed = nn.Parameter(
            torch.zeros(1, num_future_positions, predictor_dim)
        )
        # Learnable query tokens: initial content of each future slot,
        # replaced by the predictor's predicted representation after the
        # Transformer trunk consumes past + future.
        self.future_query = nn.Parameter(
            torch.zeros(1, num_future_positions, predictor_dim)
        )

        # trunc_normal(0.02) for all matmul-like weights; zero-init the last
        # block's MLP output projection so the predictor starts near zero
        # output (stable warmup; same trick as JEPAPredictor, see
        # worldfoundry_train/model/predictor.py:101-103).
        _init_trunc_normal(self.proj_in)
        _init_trunc_normal(self.blocks)
        _init_trunc_normal(self.proj_out)
        nn.init.trunc_normal_(self.past_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.future_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.future_query, std=0.02)
        last: nn.TransformerEncoderLayer = self.blocks.layers[-1]
        nn.init.zeros_(last.linear2.weight)
        nn.init.zeros_(last.linear2.bias)

    @classmethod
    def from_w22_jepa_predictor(
        cls,
        w22: JEPAPredictor,
        num_past_positions: int,
        num_future_positions: int,
        past_grid: tuple[int, int, int],
        future_grid: tuple[int, int, int],
    ) -> "TemporalPredictor":
        """Construct a TemporalPredictor warm-started from a W22 JEPAPredictor.

        Copies ``proj_in / blocks / proj_out`` weights from ``w22``;
        ``past_pos_embed / future_pos_embed / future_query`` stay at their
        fresh trunc_normal init (the W22 pos_embed indexes the source grid
        only, not the past-vs-future split, so it is not transferable).

        Caller is responsible for ensuring ``encoder_dim / predictor_dim /
        depth / num_heads`` are compatible — the constructor's load_state_dict
        calls will raise on shape mismatch.
        """
        temporal = cls(
            num_past_positions=num_past_positions,
            num_future_positions=num_future_positions,
            past_grid=past_grid,
            future_grid=future_grid,
            encoder_dim=w22.encoder_dim,
            predictor_dim=w22.predictor_dim,
            depth=len(w22.blocks.layers),
            num_heads=w22.num_heads,
            mlp_ratio=4.0,
        )
        temporal.proj_in.load_state_dict(w22.proj_in.state_dict())
        temporal.blocks.load_state_dict(w22.blocks.state_dict())
        temporal.proj_out.load_state_dict(w22.proj_out.state_dict())
        return temporal

    def forward(
        self,
        past_feats: torch.Tensor,
        future_grid: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Predict future H frame latents from past K frame latents.

        Args:
            past_feats: ``(B, N_past, encoder_dim)`` online encoder
                full-sequence output of the past K frames.
            future_grid: actual ``(H, Hg', Wg')`` of this batch; ``None`` =
                ``self.future_grid``. A different grid triggers trilinear
                interpolation of the future position embedding.

        Returns:
            ``(B, N_future, encoder_dim)`` — predicted features at future
            slots, in C-order over ``(H, Hg', Wg')``.
        """
        if past_feats.ndim != 3:
            raise ValueError(
                f"past_feats must be (B, N_past, D), got {tuple(past_feats.shape)}"
            )
        b, n_past, _ = past_feats.shape
        if n_past != self.num_past_positions:
            raise ValueError(
                f"past_feats has {n_past} positions but TemporalPredictor was "
                f"built for {self.num_past_positions} "
                f"(past_grid={self.past_grid})"
            )

        # Cast learned params to the encoder's dtype so the cat + add stays
        # in-band (same dtype-cache trick as JEPAPredictor.forward;
        # worldfoundry_train/model/predictor.py:154-164).
        target_dtype = past_feats.dtype
        target_grid = future_grid or self.future_grid
        past_pe = self.past_pos_embed.to(dtype=target_dtype)
        future_pe = self._interp_3d(self.future_pos_embed, target_grid).to(
            dtype=target_dtype
        )
        future_q = self._interp_3d(self.future_query, target_grid).to(
            dtype=target_dtype
        )

        past = self.proj_in(past_feats) + past_pe  # (B, N_past, pred_dim)
        future = future_q.expand(b, -1, -1) + future_pe  # (B, N_future, pred_dim)

        x = torch.cat([past, future], dim=1)  # (B, N_past + N_future, pred_dim)
        x = self.blocks(x)
        future_out = x[:, n_past:]  # (B, N_future, pred_dim)
        return self.proj_out(future_out)  # (B, N_future, encoder_dim)

    def _interp_3d(
        self, tensor: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Trilinear-interpolate a (1, N_construction, D) tensor to ``grid``.

        ``N_construction = self.future_grid[0] * [1] * [2]``. Returns the
        tensor unchanged when ``grid == self.future_grid`` (fast path).
        """
        if tuple(grid) == tuple(self.future_grid):
            return tensor
        t0, h0, w0 = self.future_grid
        d = tensor.shape[-1]
        pe = tensor.reshape(1, t0, h0, w0, d).permute(0, 4, 1, 2, 3)
        pe = F.interpolate(pe, size=grid, mode="trilinear", align_corners=False)
        return pe.permute(0, 2, 3, 4, 1).reshape(1, -1, d)