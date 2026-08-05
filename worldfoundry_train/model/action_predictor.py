"""ActionConditionedPredictor: past K frames + actions -> future H latents (W27).

V-JEPA 2-AC family (Meta 2025): same world-model roll-out as
:class:`TemporalPredictor`, but with robot actions (or any per-frame
control signal) injected into the past-token sequence. Reuses the W22
attention trunk (``proj_in / blocks / proj_out``) so a W22-pretrained
JEPAPredictor can warm-start the trunk; action conditioning is task-
specific and is fresh-init.

Design choices (W27 v1, kept minimal):

- **Action broadcast, not cross-attention.** Each past frame ``k`` has
  one action ``a_k``; we embed it once into ``pred_dim`` and broadcast
  across the ``Hg*Wg`` spatial positions of that frame. This keeps the
  sequence length identical to :class:`TemporalPredictor` and avoids
  adding an attention layer (one ``Linear`` action_proj + a tile vs a
  full cross-attention block).
- **Action token is added** to the embedded past tokens, not
  concatenated. Concatenation would change ``N_past`` and break the
  warm-start trunk's head; addition is a learned residual on top of
  position-and-trunk processing.
- **Future side is unchanged** vs :class:`TemporalPredictor` — we don't
  yet accept future actions / planned policy (W27 v2 if the user wants
  the robot-policy-conditioned variant).
- **No new loss.** W27 v1 is wiring parity with W24/W26: same
  ``jepa_loss(smooth_l1 | l1 | cosine)`` from W22 applies to the
  predictor's output; the action-prediction auxiliary of full
  V-JEPA 2-AC is deferred.

W27 v1 ships the module, its tests, and a worker API stub so the
architecture is in tree. Training it on real robot data is a separate
effort and not part of W27 v1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry_train.model.predictor import _init_trunc_normal
from worldfoundry_train.model.temporal_predictor import TemporalPredictor

__all__ = ["ActionConditionedPredictor"]


class ActionConditionedPredictor(nn.Module):
    """Past K frames + K actions -> Future H frame latents (W27).

    Args:
        past_frames: K (number of past frames per request).
        action_dim: per-frame action vector width (e.g. 7 for joint
            positions + gripper; 8 for joint + gripper_state).
        num_past_positions: ``N_past = K * Hg * Wg`` of the past grid.
        num_future_positions: ``N_future = H * Hg' * Wg'`` of the
            future grid.
        past_grid: ``(K, Hg, Wg)`` past frame grid.
        future_grid: ``(H, Hg', Wg')`` future frame grid.
        encoder_dim: online/target encoder hidden size (1024 for ViT-L).
        predictor_dim: bottleneck width; must match the warm-start
            source :class:`JEPAPredictor` for byte-for-byte transfer.
        depth: number of Transformer blocks; must match source.
        num_heads: attention heads; must match source.
        mlp_ratio: feedforward expansion.
    """

    def __init__(
        self,
        past_frames: int,
        action_dim: int,
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
        if past_grid[0] != past_frames:
            raise ValueError(
                f"past_grid frame dimension {past_grid[0]} != past_frames "
                f"{past_frames}"
            )
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
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        self.past_frames = past_frames
        self.action_dim = action_dim
        self.num_past_positions = num_past_positions
        self.num_future_positions = num_future_positions
        self.past_grid = past_grid
        self.future_grid = future_grid
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_heads = num_heads

        # Same trunk as TemporalPredictor: warm-start-friendly load_state_dict
        # from a W22 JEPAPredictor or a TemporalPredictor.
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

        # Action embedding: (B, K, action_dim) -> (B, K, pred_dim)
        # then broadcast across (Hg, Wg) spatial positions of that frame.
        self.action_proj = nn.Linear(action_dim, predictor_dim)

        # Position embeddings + learnable future queries, same scheme as
        # TemporalPredictor (constructor-side fresh init; warm-start skips these).
        self.past_pos_embed = nn.Parameter(
            torch.zeros(1, num_past_positions, predictor_dim)
        )
        self.future_pos_embed = nn.Parameter(
            torch.zeros(1, num_future_positions, predictor_dim)
        )
        self.future_query = nn.Parameter(
            torch.zeros(1, num_future_positions, predictor_dim)
        )

        _init_trunc_normal(self.proj_in)
        _init_trunc_normal(self.blocks)
        _init_trunc_normal(self.proj_out)
        _init_trunc_normal(self.action_proj)
        nn.init.trunc_normal_(self.past_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.future_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.future_query, std=0.02)
        last: nn.TransformerEncoderLayer = self.blocks.layers[-1]
        nn.init.zeros_(last.linear2.weight)
        nn.init.zeros_(last.linear2.bias)

    @classmethod
    def from_temporal_predictor(
        cls,
        temporal: TemporalPredictor,
        past_frames: int,
        action_dim: int,
    ) -> "ActionConditionedPredictor":
        """Warm-start from a :class:`TemporalPredictor` (W24 -> W27 transfer).

        Copies the trunk (``proj_in / blocks / proj_out``) and the past /
        future position embeddings / future queries (same shapes as in
        ``temporal``). The ``action_proj`` is fresh-initialized.

        Caller is responsible for ``past_frames`` (typically ``temporal.past_grid[0]``)
        and ``action_dim`` (a config-time choice; no default makes sense).
        """
        ac = cls(
            past_frames=past_frames,
            action_dim=action_dim,
            num_past_positions=temporal.num_past_positions,
            num_future_positions=temporal.num_future_positions,
            past_grid=temporal.past_grid,
            future_grid=temporal.future_grid,
            encoder_dim=temporal.encoder_dim,
            predictor_dim=temporal.predictor_dim,
            depth=len(temporal.blocks.layers),
            num_heads=temporal.num_heads,
        )
        ac.proj_in.load_state_dict(temporal.proj_in.state_dict())
        ac.blocks.load_state_dict(temporal.blocks.state_dict())
        ac.proj_out.load_state_dict(temporal.proj_out.state_dict())
        # Position embeddings and future queries: transfer byte-for-byte
        # so the warm-start trunk sees the exact same positional signal
        # it was trained on (trunk still has the action-broadcast residual
        # on top, but the no-action baseline is preserved at init).
        ac.past_pos_embed.data.copy_(temporal.past_pos_embed.data)
        ac.future_pos_embed.data.copy_(temporal.future_pos_embed.data)
        ac.future_query.data.copy_(temporal.future_query.data)
        # Re-zero last block's MLP output so the trunk starts as identity
        # (the temporal predictor's last linear2 may have grown during its
        # own training; this resets the warm-start stability guarantee).
        last = ac.blocks.layers[-1]
        nn.init.zeros_(last.linear2.weight)
        nn.init.zeros_(last.linear2.bias)
        return ac

    def forward(
        self,
        past_feats: torch.Tensor,
        actions: torch.Tensor,
        future_grid: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Predict future H frame latents from past K frames + K actions.

        Args:
            past_feats: ``(B, N_past, encoder_dim)`` full-sequence output
                of the online encoder on the past K frames.
            actions: ``(B, K, action_dim)`` per-frame action sequence.
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
        if actions.ndim != 3:
            raise ValueError(
                f"actions must be (B, K, action_dim), got {tuple(actions.shape)}"
            )
        b, n_past, _ = past_feats.shape
        if n_past != self.num_past_positions:
            raise ValueError(
                f"past_feats has {n_past} positions but "
                f"ActionConditionedPredictor was built for "
                f"{self.num_past_positions} (past_grid={self.past_grid})"
            )
        if actions.shape[0] != b:
            raise ValueError(
                f"actions batch size {actions.shape[0]} != past_feats "
                f"batch size {b}"
            )
        if actions.shape[1] != self.past_frames:
            raise ValueError(
                f"actions has {actions.shape[1]} frames but "
                f"ActionConditionedPredictor was built for "
                f"{self.past_frames} past frames"
            )
        if actions.shape[2] != self.action_dim:
            raise ValueError(
                f"actions last dim {actions.shape[2]} != action_dim "
                f"{self.action_dim}"
            )

        target_dtype = past_feats.dtype
        target_grid = future_grid or self.future_grid
        past_pe = self.past_pos_embed.to(dtype=target_dtype)
        future_pe = self._interp_3d(self.future_pos_embed, target_grid).to(
            dtype=target_dtype
        )
        future_q = self._interp_3d(self.future_query, target_grid).to(
            dtype=target_dtype
        )

        # Action embed: (B, K, action_dim) -> (B, K, pred_dim).
        action_emb = self.action_proj(actions.to(dtype=target_dtype))
        # Broadcast across (Hg, Wg) spatial positions of each frame so the
        # past side stays exactly N_past tokens (matches the trunk's warm-
        # start sequence length and the no-action TemporalPredictor's shape).
        k_frame, h_frame, w_frame = self.past_grid
        per_frame = h_frame * w_frame
        action_broadcast = (
            action_emb.unsqueeze(2)
            .expand(b, k_frame, per_frame, self.predictor_dim)
            .reshape(b, n_past, self.predictor_dim)
        )

        past = self.proj_in(past_feats) + past_pe + action_broadcast
        future = future_q.expand(b, -1, -1) + future_pe

        x = torch.cat([past, future], dim=1)
        x = self.blocks(x)
        future_out = x[:, n_past:]
        return self.proj_out(future_out)

    def _interp_3d(
        self, tensor: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Trilinear-interpolate a (1, N_construction, D) tensor to ``grid``.

        Returns the tensor unchanged when ``grid == self.future_grid``.
        """
        if tuple(grid) == tuple(self.future_grid):
            return tensor
        t0, h0, w0 = self.future_grid
        d = tensor.shape[-1]
        pe = tensor.reshape(1, t0, h0, w0, d).permute(0, 4, 1, 2, 3)
        pe = F.interpolate(pe, size=grid, mode="trilinear", align_corners=False)
        return pe.permute(0, 2, 3, 4, 1).reshape(1, -1, d)
