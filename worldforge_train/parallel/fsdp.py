"""FSDP launcher (Phase-2 W9: full FSDP strategy impl).

Per Global Constraint G3 + spec §3.3:
- FSDP is ONLY used for full-finetune or long-seq training.
- LoRA training defaults to DDP (1.2B model + LoRA fits per-card).

Implements true FSDP wrapping (torch.distributed.fsdp.FullyShardedDataParallel)
on multi-rank, with configurable sharding strategy, mixed precision, activation
checkpointing and checkpoint helpers. Single-rank fallback preserves existing
CPU-test behavior.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Optimizer

from worldforge_common.npu_env import is_npu_available
from worldforge_train.parallel.strategy import DistributedStrategy


def setup_fsdp() -> tuple[int, int, int]:
    """Initialize torch.distributed for FSDP.

    Mirrors setup_ddp() but is explicit about the FSDP code path. On Ascend NPU
    the HCCL backend is selected; on CPU-only hosts gloo is used for unit tests.

    Returns:
        (world_size, local_rank, global_rank)
    """
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if is_npu_available():
            backend = "hccl"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(local_rank)

    return world_size, local_rank, global_rank


def cleanup_fsdp() -> None:
    """Destroy the process group created by setup_fsdp()."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _default_auto_wrap_policy() -> Callable[[nn.Module, int, int], bool] | None:
    """Return a size-based auto-wrap policy if transformer policy is unavailable.

    V-JEPA 2 is a ViT; we wrap each Transformer encoder layer so that FSDP
    sharding happens at layer granularity. If the model has no obvious layer
    list, fall back to a 100M-parameter size threshold.
    """
    try:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        # V-JEPA 2 HF module names contain "encoder.layer.*" blocks.
        return transformer_auto_wrap_policy(
            transformer_layer_cls={nn.TransformerEncoderLayer},
        )
    except Exception:  # pragma: no cover -- defensive import path
        pass

    try:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        return size_based_auto_wrap_policy(min_num_params=100_000_000)
    except Exception:  # pragma: no cover
        return None


class FSDPStrategy(DistributedStrategy):
    """FSDP-based strategy with full ZeRO-3 / Hybrid-Shard support.

    On multi-rank the model is wrapped by
    ``torch.distributed.fsdp.FullyShardedDataParallel``. On single-rank the
    model is returned unchanged so CPU unit tests can exercise the dispatch
    layer without a distributed backend.

    Args:
        auto_wrap_policy: policy passed to FSDP. If None, a default ViT layer
            policy is used.
        mixed_precision: None, "bf16" or "fp16". Defaults to None (full).
        sharding_strategy: "FULL_SHARD" (ZeRO-3), "SHARD_GRAD_OP" (ZeRO-2),
            "HYBRID_SHARD" or "_HYBRID_SHARD_ZERO2".
        activation_checkpointing: if True, wrap FSDP sub-modules with
            torch.utils.checkpoint to trade compute for memory.
        backward_prefetch: "BACKWARD_PRE" or "BACKWARD_POST".
        cpu_offload: if True, offload FSDP params to CPU.
    """

    def __init__(
        self,
        auto_wrap_policy: Callable[[nn.Module, int, int], bool] | None = None,
        mixed_precision: str | None = None,
        sharding_strategy: str = "FULL_SHARD",
        activation_checkpointing: bool = False,
        backward_prefetch: str = "BACKWARD_PRE",
        cpu_offload: bool = False,
    ) -> None:
        self.auto_wrap_policy = auto_wrap_policy
        self.mixed_precision = mixed_precision
        self.sharding_strategy = sharding_strategy
        self.activation_checkpointing = activation_checkpointing
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self._wrapped: nn.Module | None = None

    def wrap(self, model: nn.Module) -> nn.Module:
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            CPUOffload,
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
        )

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            self._wrapped = model
            return model

        policy = self.auto_wrap_policy or _default_auto_wrap_policy()

        sharding_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
            "_HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        }
        if self.sharding_strategy not in sharding_map:
            raise ValueError(
                f"unknown sharding_strategy={self.sharding_strategy!r}; "
                f"expected one of {list(sharding_map)}"
            )

        mp: MixedPrecision | None = None
        if self.mixed_precision == "bf16":
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.mixed_precision == "fp16":
            mp = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32,
            )
        elif self.mixed_precision is not None:
            raise ValueError(
                f"unknown mixed_precision={self.mixed_precision!r}; "
                f"expected None, 'bf16' or 'fp16'"
            )

        kwargs: dict[str, Any] = {
            "use_orig_params": True,
            "sharding_strategy": sharding_map[self.sharding_strategy],
        }
        if policy is not None:
            kwargs["auto_wrap_policy"] = policy
        if mp is not None:
            kwargs["mixed_precision"] = mp
        if self.backward_prefetch is not None:
            kwargs["backward_prefetch"] = BackwardPrefetch[self.backward_prefetch]
        if self.cpu_offload:
            kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        wrapped = FullyShardedDataParallel(model, **kwargs)
        self._wrapped = wrapped

        if self.activation_checkpointing:
            self._apply_activation_checkpointing(wrapped)

        return wrapped

    def _apply_activation_checkpointing(self, model: nn.Module) -> None:
        """Apply gradient checkpointing to transformer blocks inside FSDP units.

        Uses PyTorch's ``apply_activation_checkpointing`` helper which is
        designed to work with FSDP-wrapped models. The heuristic targets
        submodules containing both attention and MLP/FC components, i.e.
        ViT-style transformer encoder layers.

        If the checkpoint helper is unavailable, this is a no-op rather than
        raising.
        """
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
                checkpoint_wrapper,
                CheckpointImpl,
            )
        except Exception:  # pragma: no cover -- defensive
            return

        def _check_fn(submodule: nn.Module) -> bool:
            names = [name.lower() for name, _ in submodule.named_modules()]
            has_attn = any("attention" in n or "attn" in n for n in names)
            # HuggingFace uses "mlp"/"fc"; PyTorch native TransformerEncoderLayer
            # uses "linear1"/"linear2" for the FFN.
            has_mlp = any(
                "mlp" in n or "fc" in n or "feedforward" in n
                or "linear1" in n or "linear2" in n
                for n in names
            )
            return has_attn and has_mlp

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=_check_fn,
        )

    def backward(self, loss: "torch.Tensor") -> None:
        loss.backward()

    def step(self, optimizer: Optimizer) -> None:
        # When the model is truly FSDP-wrapped, FSDP handles gradient reduction
        # internally via reduce-scatter. Manual all_reduce would be incorrect
        # and would destroy the sharded gradient layout. We preserve the manual
        # all_reduce only for the single-rank / non-FSDP test path to stay
        # equivalent to the Phase-1 trainer baseline.
        if self._wrapped is None or not self._is_fsdp_module(self._wrapped):
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                for param_group in optimizer.param_groups:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= dist.get_world_size()

        optimizer.step()
        optimizer.zero_grad()

    @staticmethod
    def _is_fsdp_module(module: nn.Module) -> bool:
        from torch.distributed.fsdp import FullyShardedDataParallel
        return isinstance(module, FullyShardedDataParallel)