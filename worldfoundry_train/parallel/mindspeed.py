"""MMStrategy — optional MindSpeed-MM parallel strategy (not in NPU baseline).

Community-doc reality (github.com/Ascend/MindSpeed-MM) that shapes this stub:

* **Not required for NPU FSDP.** verl's ``FSDPWorker`` runs native
  ``torch.distributed.fsdp`` on Ascend via torch_npu + HCCL directly; MindSpeed
  is a Megatron-Core-based (TP/PP/DP + FSDP2) accelerator, not a prerequisite
  for FSDP.
* **Version conflict with the baseline.** ``mindspeed-mm`` hard-locks
  ``torch==2.7.1``, which conflicts with the project's locked ``torch 2.9.0 /
  torch_npu 2.9.0`` (vllm-ascend baseline). MindSpeed is therefore an OPTIONAL
  enhancement kept lazy and out of the default path.

Implementing this class means wiring Megatron-Core + ``mindspeed.megatron_adaptor``
(``initialize_megatron`` → ``get_model``/``get_optimizer`` → ``train``) in a
separate environment, or running the MindSpeed-MM prebuilt multimodal suite via
its own ``examples/*/train_*.sh`` scripts — neither of which belongs in the
baseline's torch-2.9.0 env.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from worldfoundry_train.parallel.strategy import DistributedStrategy

_MM_STUB_MSG = (
    "MMStrategy not implemented. MindSpeed-MM is an optional enhancement that "
    "hard-locks torch==2.7.1 (conflicts with the torch 2.9.0 baseline) and is a "
    "Megatron-based prebuilt-model suite rather than a generic NPU FSDP strategy. "
    "Use the native torch FSDP baseline instead; see specs/docs for MindSpeed "
    "integration guidance."
)


class MMStrategy(DistributedStrategy):
    """Optional stub — MindSpeed-MM is not part of the torch 2.9.0 NPU baseline."""

    def wrap(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError(_MM_STUB_MSG)

    def backward(self, loss: torch.Tensor) -> None:
        raise NotImplementedError(_MM_STUB_MSG)

    def step(self, optimizer: Optimizer) -> None:
        raise NotImplementedError(_MM_STUB_MSG)