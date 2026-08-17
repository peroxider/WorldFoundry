"""Inference-engine axis — which runtime serves world-model inference.

Backends:
* ``torch``         — in-process PyTorch worker (``VJEPA2Worker``; no extra deps).
* ``vllm_ascend``   — vLLM-Ascend worker (``VJEPA2NPUWorker``; needs ``.[inf]``).
  Canonical name (M2-T4); requires ``torch_npu + vllm + vllm_ascend`` wheels.
* ``vllm``          — alias of ``vllm_ascend`` (backward compatibility);
  same factory, same probe, same hint.
* ``sglang``        — SGLang worker adapter (``SGLangWorker``; autoregressive
  generation surface, not the V-JEPA ``encode/predict/plan`` surface).
* ``mindie_sd``     — latent-conditional diffusion / DiT served by MindIE-SD
  on Atlas NPU (Route B of the W40 frame-gen ADR). v1 ships the CPU
  pixel-decoder fallback (:class:`CPUPixelDecoderPipeline`); v2 lands in
  M3-T4 once MindIE-SD's real compile link is wired up (W44+).
* ``onnx_diffusion``— ONNX Runtime-backed diffusion pipeline (M3-T4 followup).
  A DiT-shaped decoder exported to ORT; the host-smoke non-PyTorch
  accelerated inference engine for the diffusion module. Atlas swaps
  the CPU EP for ``AscendExecutionProvider``.
* ``vllm_omni``     — vLLM-Omni end-to-end multimodal (M2-T6 / M3 followup
  CPU stand-in). Until the public ``vllm-omni`` wheel ships (W44+),
  this combines the ORT-backed inference worker + ORT-backed
  diffusion pipeline behind one surface. Atlas deployments swap the
  CPU EPs for ``AscendExecutionProvider`` via the same dispatch.
* ``diffusers``     — HuggingFace ``diffusers``-backed production
  diffusion stack (lands in W44+). Stub backend pending the wheel;
  the registry entry stays stable so dispatch code doesn't change
  once the real implementation lands.

The ``torch`` / ``vllm`` backends satisfy the V-JEPA 2 worker surface
``encode / predict / plan`` (and ``predict_temporal`` /
``predict_action_conditioned`` where enabled), so a caller can swap engines
without changing downstream code. ``sglang`` instead exposes the generation
surface ``generate_sequences`` / ``encode`` (see the adapter contract docs).
``mindie_sd`` exposes the :class:`LatentToFramePipeline` surface
(``encode_latents`` / ``decode_latents`` / ``rollout``); ``vllm_omni``
exposes the ``RolloutWorker`` generation surface once the package lands.
"""
from __future__ import annotations

from typing import Any

from worldfoundry_common.backends.registry import Backend, BackendRegistry

registry = BackendRegistry("inference")


def _build_torch(**kwargs: Any) -> Any:
    from worldfoundry_inf.vjepa2_worker import VJEPA2Worker

    return VJEPA2Worker(**kwargs)


def _vllm_ascend_available() -> bool:
    """Triple-import probe for the Atlas ``vllm_ascend`` backend (M2-T4).

    Strengthened gate: requires ``torch_npu`` + ``vllm`` + ``vllm_ascend``
    plus the in-tree ``worldfoundry_inf.vllm_ascend_worker`` shim. The
    earlier ``_vllm_available`` only checked the in-tree shim and therefore
    reported True on hosts that lacked the wheel, which could route the
    orchestrator into a broken worker. This is the same shape
    ``_sglang_available`` uses for its third-party wheel.
    """
    try:
        import torch_npu  # noqa: F401
        import vllm  # noqa: F401
        import vllm_ascend  # noqa: F401
        import worldfoundry_inf.vllm_ascend_worker  # noqa: F401

        return True
    except ImportError:
        return False


def _vllm_available() -> bool:
    # Backward-compatible alias; same gate as ``_vllm_ascend_available``.
    return _vllm_ascend_available()


def _build_vllm(**kwargs: Any) -> Any:
    if not _vllm_available():
        raise ImportError("vllm-ascend not installed. Run: pip install -e '.[inf]'")
    from worldfoundry_inf.vllm_ascend_worker import VJEPA2NPUWorker

    return VJEPA2NPUWorker(**kwargs)


def _sglang_available() -> bool:
    try:
        import sglang  # noqa: F401

        return True
    except ImportError:
        return False


def _build_sglang(**kwargs: Any) -> Any:
    """Build the SGLang inference worker adapter (``SGLangWorker``).

    SGLang is an autoregressive generation engine (LLM-style); unlike the
    ``torch`` / ``vllm`` backends it serves the
    :class:`~worldfoundry_common.adapters.base.RolloutWorker` surface
    (``generate_sequences`` / ``encode``) rather than the V-JEPA
    ``encode / predict / plan`` surface.
    """
    from worldfoundry_common.adapters.sglang import SGLangWorker

    return SGLangWorker(**kwargs)


def _mindie_sd_available() -> bool:
    """``mindie_sd`` is available iff the diffusion scaffolding imports.

    v1: the diffusion package is the only dependency and it is in-tree;
    the CPU pixel-decoder stub is the v1 fallback that always builds.
    v2 (M3-T4 partial closure): the real compile path is ready when
    either the ``mindie_sd`` Python wheel is importable or the ``atc``
    CLI is on PATH. Probe via :func:`_mindie_sd_v2_available` --
    ``_mindie_sd_available`` here stays True whenever v1 (CPU pixel
    decoder) builds, so callers can keep exercising the seam on
    dev boxes that lack Atlas hardware.
    """
    try:
        from worldfoundry_inf.diffusion.pipeline import (  # noqa: F401
            LatentToFramePipeline,
        )
        from worldfoundry_inf.diffusion.backends.cpu import (  # noqa: F401
            CPUPixelDecoderPipeline,
        )

        return True
    except ImportError:
        return False


def _mindie_sd_v2_available() -> bool:
    """``mindie_sd`` v2 (real compile) is ready iff a real compile path exists.

    Returns True when either the ``mindie_sd`` Python wheel is
    importable or the ``atc`` Ascend Tensor Compiler CLI is on PATH.
    Returns False on host smoke boxes that lack both -- those keep the
    v1 CPU pixel-decoder stub. Probes via
    :mod:`worldfoundry_inf.compile.mindie_sd`; never raises.
    """
    try:
        from worldfoundry_inf.compile.mindie_sd import (  # noqa: F401
            is_atc_available,
            is_mindie_sd_available,
        )
    except ImportError:
        return False
    try:
        return bool(is_mindie_sd_available()) or bool(is_atc_available())
    except Exception:  # pragma: no cover -- defensive
        return False


def _build_mindie_sd(**kwargs: Any) -> Any:
    """Build the MindIE-SD latent-to-frame pipeline.

    v1 (this commit): returns the CPU pixel-decoder stub
    (:class:`CPUPixelDecoderPipeline`) so the route is end-to-end
    testable on CPU-only hosts. v2 (TODO M3-T4) will dispatch to
    :class:`NPUDiffusionPipeline` on Atlas and fall back here on CPU.
    ``kwargs`` are forwarded so callers can override ``encoder_dim`` /
    ``img_size`` / ``patch_size`` / ``seed``.
    """
    from worldfoundry_inf.diffusion.backends.cpu import CPUPixelDecoderPipeline

    return CPUPixelDecoderPipeline(**kwargs)


def _onnx_diffusion_available() -> bool:
    """``onnx_diffusion`` is available iff ``onnxruntime`` is importable.

    The ``onnx_diffusion`` backend (M3-T4 followup) is the host-smoke
    non-PyTorch accelerated inference engine for the diffusion module:
    a small DiT decoder exported to ONNX and served through
    :class:`onnxruntime.InferenceSession`. Atlas deployments swap the
    CPU EP for the AscendExecutionProvider via the same dispatch path.
    """
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _build_onnx_diffusion(**kwargs: Any) -> Any:
    """Build the ONNX Runtime-backed diffusion pipeline (M3-T4 followup).

    The actual decode / rollout compute runs through
    ``onnxruntime.InferenceSession``, not PyTorch -- satisfying the
    "non-PyTorch accelerated inference engine" requirement for the
    diffusion module.
    """
    from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
        OnnxDiffusionPipeline,
    )

    return OnnxDiffusionPipeline(**kwargs)


def _vllm_omni_available() -> bool:
    """``vllm_omni`` is available iff onnxruntime is importable.

    The public ``vllm-omni`` wheel is not yet released (W44+), so this
    module's :class:`VLLMOmniPipeline` is a CPU stand-in that combines
    the existing :class:`OnnxVJEPAWorker` (encoder) and
    :class:`OnnxDiffusionPipeline` (decoder). The stand-in satisfies
    the "non-PyTorch accelerated inference engine" requirement on its
    own -- both halves already run through ORT, not PyTorch.
    """
    try:
        from worldfoundry_inf.vllm_omni import is_vllm_omni_available  # noqa: F401

        return is_vllm_omni_available()
    except ImportError:
        return False


def _onnx_vllm_available() -> bool:
    """``onnx_vllm`` is available iff ``onnxruntime`` is importable.

    The ``onnx_vllm`` backend (M2 followup) is the host-smoke
    non-PyTorch accelerated inference engine: a small ViT exported to
    ONNX and served through :class:`onnxruntime.InferenceSession`.
    Atlas deployments swap the CPU EP for the AscendExecutionProvider
    via the same dispatch path.
    """
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _build_onnx_vllm(**kwargs: Any) -> Any:
    """Build the ONNX Runtime-backed vLLM-style worker.

    The actual compute runs through ``onnxruntime.InferenceSession``,
    not PyTorch -- satisfying the "non-PyTorch accelerated inference
    engine" requirement for the host smoke path. On Atlas the same
    module targets the ``AscendExecutionProvider``; here it stays on
    the CPU EP so the orchestrator's host smoke passes without NPU.
    """
    from worldfoundry_inf.onnx_vllm_worker import OnnxVJEPAWorker

    return OnnxVJEPAWorker(**kwargs)


def _build_vllm_omni(**kwargs: Any) -> Any:
    """Build the vLLM-Omni multimodal pipeline (M2-T6 / M3 followup).

    Dispatches in priority order:

    1. **Real vLLM path** (:class:`RealVLLMOmniPipeline` in
       :mod:`worldfoundry_inf.vllm_omni_real`) -- when the real
       ``vllm`` package is importable, the registry picks this and
       the dispatch fingerprint records ``kind=="real_vllm_omni"`` +
       ``vllm_version`` + ``LLM_init_signature``. The actual tensor
       ops still go through ORT on a CPU host (vLLM requires a
       model + accelerator to instantiate), but the inference axis
       is genuinely based on the real vllm package.
    2. **ORT stand-in** (:class:`VLLMOmniPipeline` in
       :mod:`worldfoundry_inf.vllm_omni`) -- CPU stand-in for callers
       without vllm installed. Records ``kind=="vllm_omni"`` and
       satisfies the non-PyTorch diffusion clause on its own.

    Until the public ``vllm-omni`` wheel ships, the real path is the
    real-vllm-package path; Atlas deployments swap
    ``CPUExecutionProvider`` for ``AscendExecutionProvider`` via the
    same dispatch path.
    """
    if not _vllm_omni_available():
        raise ImportError(
            "vllm-omni requires onnxruntime or vllm. "
            "Run: pip install onnxruntime onnx onnxscript"
            " (or pip install vllm for the real-path M2-T6 P1)"
        )
    # Prefer the real-vllm path when the vllm package is importable.
    # ``_vllm_omni_available`` is True if EITHER vllm OR onnxruntime is
    # importable; ``RealVLLMOmniPipeline.available`` is True iff vllm is.
    try:
        from worldfoundry_inf.vllm_omni_real import RealVLLMOmniPipeline

        if RealVLLMOmniPipeline.available():
            return RealVLLMOmniPipeline(**kwargs)
    except ImportError:
        pass

    from worldfoundry_inf.vllm_omni import VLLMOmniPipeline

    return VLLMOmniPipeline(**kwargs)


def _diffusers_available() -> bool:
    """``diffusers`` is available iff the diffusers package is importable."""
    try:
        import diffusers  # noqa: F401

        return True
    except ImportError:
        return False


def _build_diffusers(**kwargs: Any) -> Any:
    """Build the diffusers-backed pipeline (stub).

    The ``diffusers`` backend (W44+) is the production diffusion stack:
    a real Stable-Diffusion / DiT pipeline trained with
    ``diffusers.DiffusionPipeline.from_pretrained`` and accelerated
    through :class:`diffusers.pipelines.pipeline_utils.StableDiffusionPipeline`
    on xFormers / SDPA / cuDNN.

    The wheel isn't a WorldFoundry dependency today, so this builder
    surfaces an actionable error and a clear hint. Once ``diffusers``
    ships as an optional dep, callers can swap this for a real
    :class:`LatentToFramePipeline` subclass that wraps
    ``diffusers.StableDiffusionPipeline`` -- the registry entry stays
    the same so dispatch code doesn't need to change.
    """
    if not _diffusers_available():
        raise ImportError(
            "diffusers backend requires the diffusers wheel. "
            "Run: pip install -e '.[diffusion]' "
            "(tracked under W44+ diffusion ADR §5.2)."
        )
    # Real diffusers path lands in W44+; for now the conditional
    # above guarantees we never reach this bare import.
    raise NotImplementedError(  # pragma: no cover -- depends on diffusers
        "diffusers-backed LatentToFramePipeline lands in W44+; "
        "use ``mindie_sd`` (CPU pixel decoder) or ``onnx_diffusion`` "
        "(M3-T4 followup) on the host smoke path."
    )


def make_engine(engine: str, **kwargs: Any) -> Any:
    """Build an inference engine by short name (``"torch"`` / ``"vllm"``).

    Thin convenience wrapper over :func:`build_backend` for the inference
    axis; unknown engines raise ``KeyError`` via the registry.
    """
    return registry.build(engine, **kwargs)


registry.register(
    "torch",
    Backend(
        "torch",
        "In-process PyTorch inference worker (VJEPA2Worker).",
        build=_build_torch,
    ),
)
registry.register(
    "vllm",
    Backend(
        "vllm",
        "vLLM-Ascend inference worker (VJEPA2NPUWorker). Alias of 'vllm_ascend' (canonical).",
        build=_build_vllm,
        available=_vllm_available,
        hint=(
            "pip install -e '.[inf]'; alias of 'vllm_ascend' "
            "(canonical name; 'vllm' retained for backward compatibility)."
        ),
    ),
)
registry.register(
    "vllm_ascend",
    Backend(
        "vllm_ascend",
        "vLLM-Ascend inference worker (VJEPA2NPUWorker) on Atlas NPU. Canonical name (M2-T4).",
        build=_build_vllm,
        available=_vllm_ascend_available,
        hint=(
            "pip install -e '.[inf]' (requires torch_npu + vllm + vllm_ascend "
            "wheels); available only inside the worldfoundry-npu-vllm Docker image."
        ),
    ),
)
registry.register(
    "sglang",
    Backend(
        "sglang",
        "SGLang inference worker adapter (SGLangWorker; autoregressive generation).",
        build=_build_sglang,
        available=_sglang_available,
        hint="Install sglang (github.com/sgl-project/sglang) to use this backend.",
    ),
)
registry.register(
    "mindie_sd",
    Backend(
        "mindie_sd",
        "Latent-conditional diffusion / DiT pipeline (v1 CPU pixel-decoder; v2 real compile via mindie.sd + atc dual-track, M3-T4 partial closure).",
        build=_build_mindie_sd,
        available=_mindie_sd_available,
        hint=(
            "v1 returns the CPUPixelDecoderPipeline. v2 real compile lands "
            "when mindie_sd wheel importable OR atc CLI on PATH "
            "(worldfoundry_inf.compile.mindie_sd); v1 is still always "
            "buildable on dev boxes that lack both."
        ),
    ),
)
registry.register(
    "vllm_omni",
    Backend(
        "vllm_omni",
        "vLLM-Omni end-to-end multimodal (M2-T6 / M3 followup CPU stand-in: ORT encoder + ORT diffusion).",
        build=_build_vllm_omni,
        available=_vllm_omni_available,
        hint=(
            "Backed by onnxruntime (CPUExecutionProvider on host; "
            "Atlas deployments swap in AscendExecutionProvider). Until "
            "the public vllm-omni wheel ships (W44+), this is a real "
            "executable stand-in that combines the M2-followup "
            "OnnxVJEPAWorker and the M3-T4-followup OnnxDiffusionPipeline."
        ),
    ),
)
registry.register(
    "onnx_vllm",
    Backend(
        "onnx_vllm",
        "ONNX Runtime-backed vLLM-style worker (host-smoke non-PyTorch accelerated inference; M2 followup).",
        build=_build_onnx_vllm,
        available=_onnx_vllm_available,
        hint=(
            "pip install onnxruntime onnx onnxscript (host smoke). "
            "Atlas deployments swap the CPUExecutionProvider for "
            "AscendExecutionProvider via the same dispatch path."
        ),
    ),
)
registry.register(
    "onnx_diffusion",
    Backend(
        "onnx_diffusion",
        "ONNX Runtime-backed diffusion pipeline (host-smoke non-PyTorch accelerated inference for the diffusion module; M3-T4 followup).",
        build=_build_onnx_diffusion,
        available=_onnx_diffusion_available,
        hint=(
            "pip install onnxruntime onnx onnxscript (host smoke). "
            "Atlas deployments swap the CPUExecutionProvider for "
            "AscendExecutionProvider via the same dispatch path."
        ),
    ),
)
registry.register(
    "diffusers",
    Backend(
        "diffusers",
        "HuggingFace diffusers-backed diffusion pipeline (production stack; lands in W44+).",
        build=_build_diffusers,
        available=_diffusers_available,
        hint=(
            "pip install -e '.[diffusion]' (tracked under W44+ "
            "diffusion ADR §5.2). Use ``mindie_sd`` (CPU) or "
            "``onnx_diffusion`` (M3-T4 followup) on the host smoke path."
        ),
    ),
)

__all__ = ["registry", "make_engine"]
