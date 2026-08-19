"""Atlas NPU-aware ORT execution-provider selection (M2 / M3 followup).

Purpose
-------
The user's stated goal is:

    "持续执行开发直到世界模型训练基于RLinf执行，推理基于vllm(或者
    vllm-omni)，世界模型的扩散模块也采用加速的推理引擎(非pytorch)"

The ORT-backed inference engine (:class:`OnnxVJEPAWorker`) and diffusion
engine (:class:`OnnxDiffusionPipeline`) already satisfy the
"non-PyTorch accelerated inference engine" requirement on the host
smoke path -- they run through ``onnxruntime.InferenceSession``, not
PyTorch eager. To deploy them on Atlas the ORT session needs to be
constructed with the ``AscendExecutionProvider`` (cann-ep), not the
host's ``CPUExecutionProvider``. This module is the single source of
truth that decides which providers list to pass when callers don't
override it explicitly:

* Atlas box with the ``onnxruntime-ascend`` wheel registered ->
  ``["AscendExecutionProvider", "CPUExecutionProvider"]``.
* Anything else (host smoke, CI, dev box) ->
  ``["CPUExecutionProvider"]``.

So Atlas deployments swap the providers without changing call sites,
and the existing workers' ``provider_summary()`` already reports the
resolved provider list, which makes the swap visible in CI logs and
the W25 bench runs.

GateGuard fact-force declaration
-------------------------------
* Module path: :mod:`worldfoundry_inf.workers.onnx_npu_provider` (new file).
* Importers (stdlib + in-tree only):
    - ``worldfoundry_common.npu_env.is_npu_available`` (existing helper).
  No new third-party dependencies.
* Public API (additive, non-breaking):
    - :func:`is_ascend_provider_available` -> ``bool``
    - :func:`default_providers` -> ``list[str]``
* Data schemas (return types):
    - :func:`is_ascend_provider_available` returns ``True`` iff the
      installed ``onnxruntime`` wheel advertises
      ``"AscendExecutionProvider"`` in
      :func:`onnxruntime.get_available_providers`.
    - :func:`default_providers` returns a non-empty ``list[str]``;
      when ``requested`` is provided, the same list is returned.
* Safety: no destructive operations, no modifications to existing
  files, pure additive helper.

Public API
----------
* :func:`is_ascend_provider_available` -- probe for the Ascend EP.
* :func:`default_providers` -- select the right provider list.
"""
from __future__ import annotations

from typing import Iterable


def is_ascend_provider_available() -> bool:
    """Return True iff the installed ``onnxruntime`` advertises ``AscendExecutionProvider``.

    The probe imports ``onnxruntime`` (which is already a transitive
    dependency of :class:`~worldfoundry_inf.workers.onnx_vllm_worker.OnnxVJEPAWorker`
    and :class:`~worldfoundry_inf.diffusion.backends.onnx_diffusion.OnnxDiffusionPipeline`)
    and asks for its registered execution providers. On host smoke
    boxes (no Atlas NPU wheel) the function returns ``False``; on Atlas
    boxes with the ``onnxruntime-ascend`` package installed it returns
    ``True``.

    A genuine ``ImportError`` (the wheel itself is missing) is
    reported as ``False`` so callers can use this as a guard without
    try/except. A wheel that is present but does not advertise the
    Ascend EP also returns ``False`` -- the caller's session will fall
    back to ``CPUExecutionProvider`` via :func:`default_providers`.
    """
    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        return False
    try:
        providers = ort.get_available_providers()
    except Exception:  # pragma: no cover -- defensive
        return False
    return "AscendExecutionProvider" in providers


def default_providers(
    requested: Iterable[str] | None = None,
) -> list[str]:
    """Select the ORT execution providers list for a new session.

    Resolution order:
        1. If ``requested`` is provided (non-empty), it is returned
           verbatim. This lets callers pin ``CPUExecutionProvider`` for
           tests or pin a non-default provider for benchmarks.
        2. Otherwise, if both :func:`worldfoundry_common.npu_env.is_npu_available`
           and :func:`is_ascend_provider_available` are True, return
           ``["AscendExecutionProvider", "CPUExecutionProvider"]`` --
           the ORT session registers the Ascend EP first and falls
           back to the CPU EP for sub-graphs the Ascend EP cannot
           run.
        3. Otherwise return ``["CPUExecutionProvider"]`` -- the host
           smoke path stays reproducible across CPU-only CI.

    Args:
        requested: explicit providers list. When ``None`` or empty,
            the function auto-selects.

    Returns:
        A non-empty list of ORT execution provider names. Callers
        pass this directly to :class:`onnxruntime.InferenceSession`.
    """
    if requested is not None:
        explicit = list(requested)
        if explicit:
            return explicit
    # Lazy import to avoid pulling torch_npu into a CPU-only host's
    # import graph (mirrors worldfoundry_common.npu_env's own guard).
    from worldfoundry_common.npu_env import is_npu_available

    if is_npu_available() and is_ascend_provider_available():
        return ["AscendExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


__all__ = [
    "is_ascend_provider_available",
    "default_providers",
]