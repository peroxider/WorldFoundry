"""Inference-engine axis — which runtime serves world-model inference.

Backends:
* ``torch``  — in-process PyTorch worker (``VJEPA2Worker``; no extra deps).
* ``vllm``   — vLLM-Ascend worker (``VJEPA2NPUWorker``; needs ``.[inf]``).
* ``sglang`` — SGLang worker adapter (``SGLangWorker``; autoregressive
  generation surface, not the V-JEPA ``encode/predict/plan`` surface).

The ``torch`` / ``vllm`` backends satisfy the V-JEPA 2 worker surface
``encode / predict / plan`` (and ``predict_temporal`` /
``predict_action_conditioned`` where enabled), so a caller can swap engines
without changing downstream code. ``sglang`` instead exposes the generation
surface ``generate_sequences`` / ``encode`` (see the adapter contract docs).
"""
from __future__ import annotations

from typing import Any

from worldfoundry_common.backends.registry import Backend, BackendRegistry

registry = BackendRegistry("inference")


def _build_torch(**kwargs: Any) -> Any:
    from worldfoundry_inf.vjepa2_worker import VJEPA2Worker

    return VJEPA2Worker(**kwargs)


def _vllm_available() -> bool:
    try:
        import worldfoundry_inf.vllm_ascend_worker  # noqa: F401

        return True
    except ImportError:
        return False


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
        "vLLM-Ascend inference worker (VJEPA2NPUWorker).",
        build=_build_vllm,
        available=_vllm_available,
        hint="pip install -e '.[inf]'",
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

__all__ = ["registry", "make_engine"]
