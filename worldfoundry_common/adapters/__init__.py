"""External-framework adapters for the WorldFoundry pluggable layer.

Each module wraps one external framework's runner/worker behind the contracts
in :mod:`worldfoundry_common.adapters.base`, and imports the framework lazily
so selecting/importing an adapter never requires its third-party dependency:

* :mod:`worldfoundry_common.adapters.verl`   — ``VerlRunner`` / ``VerlRolloutWorker``
* :mod:`worldfoundry_common.adapters.rlinf`  — ``RLinfRunner`` / ``RLinfEnvWorker``
* :mod:`worldfoundry_common.adapters.sglang` — ``SGLangWorker``
"""
from __future__ import annotations

from worldfoundry_common.adapters.base import (
    InferenceWorker,
    RolloutWorker,
    TrainingRunner,
)
from worldfoundry_common.adapters.rlinf import RLinfEnvWorker, RLinfRunner
from worldfoundry_common.adapters.sglang import SGLangWorker
from worldfoundry_common.adapters.verl import (
    VerlRolloutWorker,
    VerlRunner,
    build_npu_config,
)

__all__ = [
    "InferenceWorker",
    "RolloutWorker",
    "TrainingRunner",
    "RLinfRunner",
    "RLinfEnvWorker",
    "SGLangWorker",
    "VerlRunner",
    "VerlRolloutWorker",
    "build_npu_config",
]
