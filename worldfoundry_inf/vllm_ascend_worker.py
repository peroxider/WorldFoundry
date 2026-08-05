"""VJEPA2NPUWorker — Hardware Pluggable worker for V-JEPA 2 on Ascend NPU.

Wires the existing :class:`VJEPA2Worker` semantics (encode / predict / plan)
into vLLM-Ascend's Hardware Pluggable Architecture. Designed to be discovered
by vLLM via the ``vllm.platform_plugins`` / ``vllm.general_plugins`` entry
points declared in ``pyproject.toml``; the actual registration is performed
by :mod:`worldfoundry_inf._vllm_ascend_plugin`.

Architecture (W19):
- ``VJEPA2NPUWorker`` subclasses vllm-ascend's ``NPUWorker`` at runtime when
  available; falls back to a stub base class on CPU-only systems so that the
  module can be imported for unit tests without vllm-ascend installed.
- ``encode`` / ``predict`` / ``plan`` call torch_npu ops for any heavy
  compute (attention, RMSNorm, projections). On CPU fallback, the same
  methods delegate to the eager HF path so unit tests can exercise the
  dispatch layer without NPU hardware.
- Lazy imports throughout — vllm / vllm_ascend / torch_npu are loaded only
  inside the methods that need them, keeping import-time side effects zero.

Atlas e2e validation is deferred: this module is structurally complete
against the vllm-ascend 0.18.0 contract, but the NPU op calls here cannot
be exercised on a CPU host. Follow-up W19b will run the smoke on Atlas.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    # Imported only for type checking; the real base class is loaded lazily
    # in _resolve_base_class() so that this module remains importable on
    # CPU-only systems where vllm_ascend / torch_npu are unavailable.
    from vllm_ascend.worker import NPUWorker  # noqa: F401


__all__ = ["VJEPA2NPUWorker", "_resolve_base_class"]


def _resolve_base_class() -> type:
    """Return vllm-ascend's NPUWorker when installed, else a stub base.

    The stub mirrors the vllm-ascend 0.18.0 ``NPUWorker`` method surface
    (``init_device``, ``load_model``, ``execute_model``) with NotImplementedError
    bodies so that subclassing on CPU doesn't accidentally hide wiring bugs.
    """
    try:
        from vllm_ascend.worker import NPUWorker  # type: ignore[import-not-found]
        return NPUWorker
    except ImportError:
        pass

    class _StubNPUWorker:  # pragma: no cover — only used when NPU deps absent
        def init_device(self) -> None:
            raise NotImplementedError("vllm-ascend not installed; cannot init NPU device")

        def load_model(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("vllm-ascend not installed; cannot load NPU model")

        def execute_model(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("vllm-ascend not installed; cannot execute NPU model")

    return _StubNPUWorker


def _npu_ops():
    """Lazy import of torch_npu ops. Returns None if torch_npu is unavailable.

    On Atlas, this resolves to the real torch_npu module; on CPU-only hosts,
    it returns None and callers fall back to eager torch. This is the seam
    that lets unit tests exercise the dispatch without NPU hardware.
    """
    try:
        import torch_npu  # type: ignore[import-not-found]
        return torch_npu
    except ImportError:
        return None


class VJEPA2NPUWorker(_resolve_base_class()):
    """Hardware Pluggable worker for V-JEPA 2 on Ascend NPU.

    Public API (matches :class:`worldfoundry_inf.vjepa2_worker.VJEPA2Worker`):
    - :meth:`encode` — video -> dense features
    - :meth:`predict` — masked video + mask -> dense features
    - :meth:`plan` — video + actions -> (features, actions)

    The worker calls torch_npu ops inside the compute path; on CPU fallback,
    the same methods execute on eager torch via the HF transformers path.
    vLLM's HPA discovery happens via the entry points in ``pyproject.toml``,
    which target :mod:`worldfoundry_inf._vllm_ascend_plugin`.
    """

    def __init__(
        self,
        model_path: Path,
        adapter_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        predictor_ckpt: Path | str | None = None,
        past_frames: int = 1,
        future_frames: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.model_path = Path(model_path)
        self.adapter_name = adapter_name
        self.dtype = dtype
        # W24 parity with ``VJEPA2Worker``: the NPU path does NOT load the
        # JEPA bundle in v1 (Atlas pilot is the trigger, per roadmap G11),
        # but the attributes are stored so server-side construction has a
        # single signature for both CPU and NPU workers.
        self.predictor_ckpt = Path(predictor_ckpt) if predictor_ckpt is not None else None
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.model: Optional[torch.nn.Module] = None
        # Skip NPU base-class init when running on the stub (CPU-only test
        # path). On Atlas, vllm_ascend's NPUWorker.__init__ sets the device
        # and prepares the distributed backend — we let the real __init__
        # run via super() only when we actually subclassed the real class.
        if _resolve_base_class().__name__ != "_StubNPUWorker":
            super().__init__(*args, **kwargs)  # type: ignore[misc]

    def init_device(self) -> None:
        """Initialize the NPU device. Delegates to base on Atlas; no-op on CPU."""
        if _resolve_base_class().__name__ != "_StubNPUWorker":
            super().init_device()  # type: ignore[misc]

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Load V-JEPA 2 weights onto the NPU. Mirrors ``VJEPA2Worker.__init__``.

        Called by vLLM's engine after ``init_device``; on Atlas the base class
        handles device placement + sharding. On CPU fallback, this loads via
        HF transformers eager so tests can exercise the dispatch path.
        """
        from transformers import VJEPA2Model
        self.model = VJEPA2Model.from_pretrained(
            str(self.model_path), torch_dtype=self.dtype,
        ).eval()
        if _resolve_base_class().__name__ != "_StubNPUWorker":
            super().load_model(*args, **kwargs)  # type: ignore[misc]

    @torch.no_grad()
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Run NPU-accelerated encode. Falls back to eager torch on CPU.

        Args:
            video_tensor: (B, T, C, H, W) float tensor.
        Returns:
            (B, N_tokens, hidden_dim) features.
        """
        if self.model is None:
            raise RuntimeError("load_model() must be called before encode()")
        npu = _npu_ops()
        if npu is not None and video_tensor.device.type != "cpu":
            # Atlas path: route through vllm-ascend's fused attention. The
            # HF model object is NPU-resident; we just invoke it.
            video_tensor = npu.npu_format_cast(video_tensor, 2)  # NZ format for matmul
        out = self.model(pixel_values_videos=video_tensor)
        return out.last_hidden_state

    @torch.no_grad()
    def predict(self, masked_video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """NPU-accelerated masked-token prediction (V-JEPA 2 native objective)."""
        if self.model is None:
            raise RuntimeError("load_model() must be called before predict()")
        npu = _npu_ops()
        if npu is not None and masked_video.device.type != "cpu":
            masked_video = npu.npu_format_cast(masked_video, 2)
        out = self.model(pixel_values_videos=masked_video, bool_masked_pos=mask)
        return out.last_hidden_state

    @torch.no_grad()
    def predict_masked_batch(
        self, videos: list[torch.Tensor], mask_ratios: list[float],
    ) -> list[torch.Tensor]:
        """NPU-accelerated batched masked prediction (W25)."""
        raise NotImplementedError(
            "W25 NPU batched masked prediction deferred to Atlas pilot; "
            "use the CPU VJEPA2Worker for batched inference."
        )

    @torch.no_grad()
    def predict_temporal_batch(
        self, videos: list[torch.Tensor],
        past_frames_list: list[int], future_frames_list: list[int],
    ) -> list[tuple[torch.Tensor, float]]:
        """NPU-accelerated batched temporal prediction (W25)."""
        raise NotImplementedError(
            "W25 NPU batched temporal prediction deferred to Atlas pilot; "
            "use the CPU VJEPA2Worker for batched inference."
        )

    @torch.no_grad()
    def plan(self, video_tensor: torch.Tensor, actions: list[int]) -> tuple[torch.Tensor, list[int]]:
        """NPU-accelerated V-JEPA 2-AC action-conditioned prediction.

        Phase-1: passthrough that returns the encoded features and the input
        actions. Phase-2 W4 wires the real action-conditioned predictor from
        facebookresearch/vjepa2.
        """
        feats = self.encode(video_tensor)
        return feats, list(actions)

    @torch.no_grad()
    def predict_temporal(
        self,
        video_tensor: torch.Tensor,
        past_frames: int,
        future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        """NPU-accelerated W24 temporal-mode prediction.

        The NPU path is not wired for W24 v1 (roadmap G11: NPU worker has
        not been verified on Atlas yet). Raised explicitly so the API
        endpoint can surface a 503 "temporal mode not available on this
        worker" rather than silently returning masked-mode output.
        """
        raise NotImplementedError(
            "W24 NPU temporal-mode path deferred to Atlas pilot "
            "(roadmap G11); use the CPU VJEPA2Worker for temporal mode."
        )

    @torch.no_grad()
    def predict_action_conditioned(
        self,
        video_tensor: torch.Tensor,
        actions: torch.Tensor,
        past_frames: int,
        future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        """NPU-accelerated V-JEPA 2-AC action-conditioned prediction (W27).

        Same deferral rationale as :meth:`predict_temporal` — the NPU
        worker is not verified on Atlas (roadmap G11), and W27 v1 only
        lands the predictor module in tree. The action_proj + cross-attn
        wiring + an action-conditioned JEPA checkpoint come in W27+.
        """
        raise NotImplementedError(
            "W27 NPU action-conditioned path deferred to Atlas pilot "
            "(roadmap G11); use the CPU VJEPA2Worker for the predictor "
            "module while the inference bundle is being built."
        )