"""Adapter contracts between external frameworks and WorldFoundry.

A *runner* owns a training loop; a *worker* is a single role (rollout /
inference) executed inside that loop. These protocols are the seams the
external-framework adapters (``verl``, ``rlinf``, ``sglang``) satisfy so the
pluggable backend layer can swap a framework without changing downstream code.

Two worker surfaces exist because the frameworks genuinely differ:

* :class:`InferenceWorker` — the V-JEPA surface (``encode`` / ``predict`` /
  ``plan``) served by the native ``torch`` and ``vllm`` engines.
* :class:`RolloutWorker` — the autoregressive generation surface
  (``generate_sequences``) served by ``sglang`` and by verl's rollout backend.

All adapters import their third-party dependency lazily inside their methods,
so importing this package (or any adapter) never requires verl / rlinf /
sglang to be installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InferenceWorker(Protocol):
    """Contract every V-JEPA inference-engine worker must satisfy.

    Mirrors :class:`worldfoundry_inf.workers.vjepa2_worker.VJEPA2Worker` (and its
    ``vllm`` sibling) so a caller can swap ``torch`` / ``vllm`` / any future
    V-JEPA engine without changing downstream code.
    """

    def encode(self, *args: Any, **kwargs: Any) -> Any: ...
    def predict(self, *args: Any, **kwargs: Any) -> Any: ...
    def plan(self, *args: Any, **kwargs: Any) -> Any: ...
    def close(self) -> None: ...


@runtime_checkable
class RolloutWorker(Protocol):
    """Contract every autoregressive generation/rollout worker must satisfy.

    Mirrors verl's ``RolloutBase.generate_sequences`` and SGLang's offline
    engine, so a rollout loop can drive either backend uniformly.
    """

    def generate_sequences(self, prompts: Any, **kwargs: Any) -> Any: ...
    def update_weights(self, state_dict: dict[str, Any]) -> None: ...
    def shutdown(self) -> None: ...


class TrainingRunner(ABC):
    """Contract every training-framework runner must satisfy.

    A runner wraps an external framework's entrypoint (e.g. verl's
    ``verl.interface.Main`` or RLinf's ``EmbodiedRunner``) behind one
    ``run(config) -> metrics`` surface. Implementing subclasses keep the
    third-party import inside :meth:`run`.
    """

    @abstractmethod
    def run(self, config: dict, **kwargs: Any) -> dict[str, Any]:
        """Run the training loop described by ``config``; return summary metrics.

        ``config`` is a plain dict (YAML-able). Adapters translate it into the
        framework's native config object (OmegaConf / Hydra) internally.
        """
        raise NotImplementedError
