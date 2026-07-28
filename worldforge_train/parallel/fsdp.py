"""FSDP launcher (Phase-1: stub returning same shape as DDP; Phase-2 W9: full impl).

Per Global Constraint G3 + spec §3.3:
- FSDP is ONLY used for full-finetune or long-seq training.
- LoRA training defaults to DDP (1.2B model + LoRA fits per-card).

Phase-1 ships a minimal stub so `train.py --parallel fsdp` does not crash;
Phase-2 W9 implements Hybrid Shard ZeRO-3 + Activation Checkpointing.
"""
from __future__ import annotations

import warnings


def setup_fsdp() -> tuple[int, int, int]:
    """Phase-1 stub: behaves like DDP but warns that full FSDP is deferred to W9."""
    warnings.warn(
        "FSDP path is a Phase-1 stub. Hybrid Shard ZeRO-3 lands in Phase-2 W9. "
        "Use --parallel ddp for LoRA training now.",
        UserWarning,
        stacklevel=2,
    )
    from worldforge_train.parallel.ddp import setup_ddp
    return setup_ddp()


def cleanup_fsdp() -> None:
    from worldforge_train.parallel.ddp import cleanup_ddp
    cleanup_ddp()
