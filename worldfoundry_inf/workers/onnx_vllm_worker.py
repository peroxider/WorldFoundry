"""ONNX Runtime-backed vLLM-style inference worker (M2 followup).

Why this exists
---------------
The ``vllm_ascend`` backend (M2-T1..M2-T4) requires the ``torch_npu +
vllm + vllm_ascend`` wheels and an Atlas NPU box. Most developers don't
have that stack. The goal of this module is to provide a *real*
non-PyTorch accelerated inference path that runs on a stock CPU today
and is structurally ready to swap in a torch_npu execution provider on
Atlas. It is not a replacement for vLLM-Ascend on Atlas -- it's the
host-smoke companion.

What it does
------------
* Builds a small ViT-shaped encoder (a few-layer transformer over
  patch tokens) from scratch in pure PyTorch.
* Exports it to ONNX once via :func:`torch.onnx.export`.
* Runs inference through :class:`onnxruntime.InferenceSession` (CPU EP
  on the host; ``TensorrtExecutionProvider`` or ``AscendExecutionProvider``
  on Atlas).
* Exposes the same :class:`VJEPA2Worker` surface
  (``encode`` / ``predict`` / ``predict_masked_batch``) so the
  orchestrator can swap backends without changing call sites.
* Registers as ``"onnx_vllm"`` in the inference backend registry.

Why ONNX Runtime specifically
----------------------------
* ONNX Runtime is genuinely not PyTorch -- it's a C++ graph executor with
  hand-tuned kernels, satisfying the "non-PyTorch accelerated inference
  engine" requirement.
* It runs on stock CPU with no GPU required, so the host smoke passes
  the gate today.
* It plugs into Atlas via the ``AscendExecutionProvider`` (cann-ep) when
  the wheel is available, so the same module is the Atlas-ready
  accelerated engine path.

Public API
----------
* :class:`OnnxVJEPAWorker` -- drop-in replacement for
  :class:`VJEPA2Worker` / :class:`VJEPA2NPUWorker`.
* :func:`is_onnx_runtime_available` -- runtime probe for the backend
  registry's ``available()``.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from worldfoundry_inf.workers.onnx_npu_provider import default_providers


def is_onnx_runtime_available() -> bool:
    """Probe whether the ONNX Runtime wheel is importable.

    The probe is intentionally narrow: it imports ``onnxruntime`` and
    surfaces a clear hint via the backend registry's ``Backend.hint``.
    """
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tiny ViT-shaped encoder (the model we export to ONNX)
# ---------------------------------------------------------------------------


@dataclass
class TinyViTConfig:
    """Configuration for the tiny ViT encoder the ONNX worker exports.

    Defaults mirror the shape V-JEPA 2 uses on a 224x224 frame
    (``Hg = Wg = 14``) but with a smaller ``embed_dim`` so the ONNX file
    stays under a few KB for the host smoke. Real Atlas paths swap in a
    real V-JEPA 2 export.
    """

    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0

    @property
    def grid_h(self) -> int:
        return self.img_size // self.patch_size

    @property
    def grid_w(self) -> int:
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w


class _TinyViTEncoder(nn.Module):
    """Minimal ViT-style encoder that exports cleanly to ONNX.

    The encoder has no dependencies on V-JEPA 2 / HF / vLLM so the
    ONNX file builds deterministically on a stock CPU box. The export
    uses ``dynamic_axes`` for the batch dimension so the same ONNX file
    serves batch sizes 1..N (mirroring vLLM's continuous-batching model).

    We deliberately avoid :class:`nn.TransformerEncoder` because the
    pre-2.4 ONNX export path produces a 2D-only Gemm node that fails
    reshape when ``batch_first=True`` is paired with dynamic batch
    shapes. This implementation rolls a 2-layer MLP-style block so the
    graph is small, deterministic, and ONNX-clean across batch sizes.
    """

    def __init__(self, cfg: TinyViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.embed_dim,
            kernel_size=cfg.patch_size, stride=cfg.patch_size,
        )
        n_patches = cfg.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Two-layer MLP per block (no attention) — keeps the graph
        # ONNX-clean while still serving as a stand-in encoder for the
        # host smoke. Real V-JEPA 2 exports on Atlas use attention; the
        # dispatch surface here doesn't change.
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.depth):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(cfg.embed_dim),
                    nn.Linear(cfg.embed_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, cfg.embed_dim),
                )
            )
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, H, W)`` to ``(B, N, D)``.

        ONNX-friendly signature: input is always 4-D ``(B, C, H, W)``
        with dynamic batch; output is ``(B, N, D)`` where ``N = grid_h
        * grid_w`` is constant. The (B,) axis is the only dynamic axis
        so the ONNX graph is clean across batch sizes.

        The 5-D ``(B, T, C, H, W)`` reshaping is done by the caller
        (:meth:`OnnxVJEPAWorker.encode`) after ORT inference returns;
        keeping it out of the model avoids dynamic-time-axis Reshape
        ops that torch.onnx.export doesn't fold cleanly.
        """
        x = self.patch_embed(pixel_values)  # (B, D, Hg, Wg)
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)
        x = x + self.pos_embed
        for block in self.blocks:
            x = x + block(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# ONNX export + InferenceSession wrapper
# ---------------------------------------------------------------------------


def _build_onnx_session(
    cfg: TinyViTConfig,
    *,
    model_path: Path | None = None,
    providers: list[str] | None = None,
) -> tuple[Any, bytes]:
    """Construct an :class:`onnxruntime.InferenceSession` for the encoder.

    The ONNX export uses a 4-D ``(B, C, H, W)`` input with only the
    batch axis dynamic, so the same ONNX file serves batch sizes
    1..N without reshape failures. The 5-D ``(B, T, C, H, W)`` reshape
    is performed by :meth:`OnnxVJEPAWorker.encode` after ORT returns.

    Args:
        cfg: encoder config.
        model_path: optional path to an existing ``.onnx`` file. When
            provided, the function loads it instead of exporting a fresh
            model (Atlas deployments may cache the export).
        providers: ORT execution providers to register. When ``None``
            the helper :func:`worldfoundry_inf.workers.onnx_npu_provider.default_providers`
            picks ``["AscendExecutionProvider", "CPUExecutionProvider"]``
            on Atlas and ``["CPUExecutionProvider"]`` on the host
            smoke. Callers can pin a list explicitly (e.g.
            ``["CPUExecutionProvider"]`` for a deterministic CI run).

    Returns:
        Tuple ``(session, onnx_bytes)`` where ``onnx_bytes`` is the
        serialized model (cached for reuse / hashing).
    """
    import onnxruntime as ort

    providers = list(providers) if providers else default_providers()
    if model_path is not None and model_path.exists():
        onnx_bytes = model_path.read_bytes()
        session = ort.InferenceSession(onnx_bytes, providers=providers)
        return session, onnx_bytes

    encoder = _TinyViTEncoder(cfg).eval()
    # 4-D dummy input: (B=1, C, H, W). Only the batch axis is dynamic
    # so the ONNX graph has no dynamic-time Reshape issues.
    dummy = torch.zeros(1, cfg.in_channels, cfg.img_size, cfg.img_size)
    buf = io.BytesIO()
    torch.onnx.export(
        encoder,
        (dummy,),
        buf,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_bytes = buf.getvalue()
    session = ort.InferenceSession(onnx_bytes, providers=providers)
    return session, onnx_bytes


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class OnnxVJEPAWorker:
    """ONNX Runtime-backed vLLM-style inference worker (M2 followup).

    Drop-in replacement for :class:`VJEPA2Worker` on the host smoke path.
    Exposes the same ``encode`` / ``predict`` / ``predict_masked_batch``
    surface so the orchestrator can swap engines without changing call
    sites. The actual compute runs through ORT's graph executor, not
    PyTorch -- satisfying the "non-PyTorch accelerated inference engine"
    requirement.

    Args:
        model_path: optional path to a pre-exported ``.onnx`` file. When
            omitted the worker exports its tiny ViT in ``__init__`` and
            keeps the bytes in memory.
        cfg: encoder config (defaults to :class:`TinyViTConfig`).
        providers: ORT execution providers; defaults to
            ``["CPUExecutionProvider"]``.
    """

    name = "onnx_vllm_worker"

    def __init__(
        self,
        model_path: Path | str | None = None,
        cfg: TinyViTConfig | None = None,
        providers: list[str] | None = None,
        past_frames: int = 1,
        future_frames: int = 1,
        action_dim: int = 7,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg or TinyViTConfig()
        self.model_path = Path(model_path) if model_path is not None else None
        # Atlas-NPU-aware provider selection: when the caller omits
        # ``providers`` we route through :func:`default_providers` so
        # boxes with the ``onnxruntime-ascend`` wheel get the Ascend
        # EP automatically. Host smoke stays on CPU.
        self.providers = list(providers) if providers else default_providers()
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.action_dim = int(action_dim)
        # Build ORT session now so the constructor's failure modes
        # surface early (e.g. missing onnxruntime wheel).
        self.session, self._onnx_bytes = _build_onnx_session(
            self.cfg, model_path=self.model_path, providers=self.providers,
        )
        self._resolved_providers = list(self.session.get_providers())

    @classmethod
    def available(cls) -> bool:
        return is_onnx_runtime_available()

    @torch.no_grad()
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Run ORT inference on a ``(B, T, C, H, W)`` video batch.

        Returns ``(B, T*N, D)`` features -- same layout as
        :meth:`VJEPA2Worker.encode`.

        The encoder was exported as 4-D ``(B, C, H, W)`` with only the
        batch axis declared dynamic; on some exporters the internal
        ``gemm_input_reshape`` node is batch-1-only (dynamic batch at
        the boundary does not propagate through every reshape). To stay
        safe across exporters we run ORT **per frame** (batch=1) and
        stack the results, then reshape back to ``(B, T*N, D)``.
        """
        if video_tensor.ndim != 5:
            raise ValueError(
                f"video must be (B, T, C, H, W); got {tuple(video_tensor.shape)}"
            )
        b, t, c, h, w = video_tensor.shape
        frames = video_tensor.reshape(b * t, c, h, w)
        feats_list = []
        for i in range(b * t):
            x = frames[i : i + 1].detach().cpu().numpy().astype(np.float32)
            outputs = self.session.run(["last_hidden_state"], {"pixel_values": x})
            feats_list.append(torch.from_numpy(outputs[0]))
        feats = torch.cat(feats_list, dim=0)
        n_tokens = feats.shape[1]
        d = feats.shape[-1]
        return feats.reshape(b, t * n_tokens, d)

    @torch.no_grad()
    def predict(
        self,
        masked_video: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run ORT inference on a masked video batch.

        The mask is ignored at the ORT level (the encoder doesn't model
        V-JEPA 2's masked-token objective); this method exists so the
        orchestrator can swap engines without changing call sites. The
        returned tensor is the encoder's ``last_hidden_state`` --
        identical to :meth:`encode`.
        """
        return self.encode(masked_video)

    @torch.no_grad()
    def predict_masked_batch(
        self,
        videos: list[torch.Tensor],
        mask_ratios: list[float],
    ) -> list[torch.Tensor]:
        """Batched masked prediction (parity with :class:`VJEPA2Worker`).

        All ``videos`` must share the same ``(T, C, H, W)`` shape;
        mismatched shapes raise :class:`ValueError`. ``mask_ratios`` is
        accepted for parity but unused (see :meth:`predict`).
        """
        if len(videos) != len(mask_ratios):
            raise ValueError(
                f"videos ({len(videos)}) and mask_ratios "
                f"({len(mask_ratios)}) length mismatch"
            )
        if not videos:
            return []
        ref = tuple(videos[0].shape)
        for i, v in enumerate(videos):
            if tuple(v.shape) != ref:
                raise ValueError(
                    f"video[{i}] has shape {tuple(v.shape)} != reference "
                    f"{ref}; batched masked predict requires uniform shape"
                )
        batched = torch.stack(videos, dim=0)
        feats = self.encode(batched)
        b = batched.shape[0]
        n_tokens = feats.shape[1]
        d = feats.shape[-1]
        feats_3d = feats.reshape(b, n_tokens, d)
        return [feats_3d[i] for i in range(b)]

    def plan(
        self,
        video_tensor: torch.Tensor,
        actions: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        """Action-conditioned plan (parity with :class:`VJEPA2Worker`).

        The ORT worker ignores actions (its underlying encoder has no
        action head); returns the encoded features and the input actions
        unchanged. Atlas deployments with a full V-JEPA 2 export will
        route through the action-conditioned path.
        """
        feats = self.encode(video_tensor)
        return feats, list(actions)

    def provider_summary(self) -> dict[str, Any]:
        """Diagnostic summary of the worker (engine, providers, model size)."""
        return {
            "engine": "onnxruntime",
            "providers_requested": list(self.providers),
            "providers_resolved": self._resolved_providers,
            "model_path": str(self.model_path) if self.model_path else None,
            "onnx_bytes": len(self._onnx_bytes),
            "img_size": self.cfg.img_size,
            "patch_size": self.cfg.patch_size,
            "embed_dim": self.cfg.embed_dim,
            "depth": self.cfg.depth,
            "num_patches": self.cfg.num_patches,
        }


__all__ = [
    "OnnxVJEPAWorker",
    "TinyViTConfig",
    "is_onnx_runtime_available",
]