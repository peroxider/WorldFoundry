"""SGLang inference worker adapter.

SGLang is an autoregressive generation engine (LLM-style); unlike vLLM it does
not expose a Hardware-Pluggable worker, and it does not serve non-autoregressive
V-JEPA models natively. Its honest WorldFoundry role is a generation/rollout
engine, so :class:`SGLangWorker` implements the
:class:`~worldfoundry_common.adapters.base.RolloutWorker` surface
(``generate_sequences`` / ``update_weights`` / ``shutdown``) plus ``encode``,
backed by the offline ``sglang.srt.entrypoints.engine.Engine``.

Configuration mirrors ``sglang.srt.server_args.ServerArgs`` fields
(``tp_size``, ``dtype``, ``mem_fraction_static``, ``load_format``, …).

``sglang`` is not a declared WorldFoundry dependency yet; it is imported
lazily so this module imports cleanly when sglang is absent.
"""
from __future__ import annotations

from typing import Any

from worldfoundry_common.npu_env import is_npu_available


def _require_sglang(component: str) -> Any:
    """Import the sglang package, raising an actionable error if absent."""
    try:
        import sglang  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            f"sglang is required to build the {component}. "
            "Install it from https://github.com/sgl-project/sglang first."
        ) from exc
    return sglang


class SGLangWorker:
    """Adapter wrapping ``sglang.srt.entrypoints.engine.Engine``.

    On Ascend NPU the offline engine must be told to run on ``npu`` with the
    ``ascend`` attention backend — the device is ``npu`` (not ``ascend``); the
    ``ascend`` value only selects the attention backend. Per the SGLang Ascend
    docs this stack needs torch 2.8.0 + sglang v0.5.10 + triton-ascend (a
    separate environment from the vllm-ascend baseline, which pins torch 2.9.0).

    Args:
        model_path: HF model path for the engine.
        tp_size: tensor-parallel size.
        dtype: model dtype (e.g. ``"bfloat16"``).
        skip_tokenizer_init: pass token IDs directly when True (RL training).
        device: ``"npu"`` when running on Ascend (default when NPU detected),
            else ``"cuda"``/``"cpu"``.
        attention_backend: ``"ascend"`` on NPU, else ``"triton"``/``"flashinfer"``.
        **engine_kwargs: remaining ``ServerArgs`` fields forwarded to the
            engine (``mem_fraction_static``, ``load_format``, …).
    """

    def __init__(
        self,
        model_path: str,
        *,
        tp_size: int = 1,
        dtype: str = "bfloat16",
        skip_tokenizer_init: bool = False,
        device: str | None = None,
        attention_backend: str | None = None,
        **engine_kwargs: Any,
    ) -> None:
        self._model_path = model_path
        self._tp_size = int(tp_size)
        self._dtype = dtype
        self._skip_tokenizer_init = bool(skip_tokenizer_init)
        self._device = device or ("npu" if is_npu_available() else "cuda")
        self._attention_backend = attention_backend or (
            "ascend" if is_npu_available() else "triton"
        )
        self._engine_kwargs: dict[str, Any] = dict(engine_kwargs)
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        """Lazily build the offline sglang engine from ``ServerArgs``."""
        if self._engine is None:
            _require_sglang("sglang worker")
            from sglang.srt.entrypoints.engine import Engine
            from sglang.srt.server_args import ServerArgs

            server_args = ServerArgs(
                model_path=self._model_path,
                tp_size=self._tp_size,
                dtype=self._dtype,
                skip_tokenizer_init=self._skip_tokenizer_init,
                device=self._device,
                attention_backend=self._attention_backend,
                **self._engine_kwargs,
            )
            # Pass the ServerArgs object directly (Engine(**asdict(...)) would
            # recursively convert nested dataclasses into dicts and break the
            # ServerArgs(**kwargs) type checks).
            self._engine = Engine(server_args=server_args)
        return self._engine

    def generate_sequences(self, prompts: Any, **kwargs: Any) -> Any:
        """Generate completions for ``prompts`` (list of str or token IDs)."""
        engine = self._ensure_engine()
        return engine.async_generate(prompt=prompts, **kwargs)

    def encode(self, prompts: Any, **kwargs: Any) -> Any:
        """Encode ``prompts`` to embeddings via the engine."""
        engine = self._ensure_engine()
        return engine.async_encode(prompt=prompts, **kwargs)

    def update_weights(self, state_dict: dict[str, Any]) -> None:
        """No-op unless the engine exposes weight sync.

        RLinf wires online weight sync through a patched
        ``TokenizerManager.sync_hf_weight``; here we surface the hook so a
        rollout loop can call it uniformly, and raise if unsupported.
        """
        engine = self._engine
        if engine is not None and hasattr(engine.tokenizer_manager, "sync_hf_weight"):
            engine.tokenizer_manager.sync_hf_weight(state_dict)
            return
        raise NotImplementedError(
            "SGLang weight sync requires RLinf's TokenizerManager patch; "
            "the plain sglang engine does not expose sync_hf_weight."
        )

    def shutdown(self) -> None:
        """Release the engine."""
        engine = self._engine
        self._engine = None
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()


__all__ = ["SGLangWorker"]
