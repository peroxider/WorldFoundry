"""Runtime discovery of LoRA-targetable module names in a model."""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def discover_lora_targets(
    model: nn.Module,
    candidates: Iterable[str],
    suffix_only: bool = True,
) -> list[str]:
    """Return fully-qualified module names in `model` whose leaf name ends with
    one of the candidate suffixes.

    Args:
        model: a torch.nn.Module (e.g. VJEPA2Model).
        candidates: candidate leaf names like ["q_proj", "mlp.fc1"].
        suffix_only: if True (default), match by suffix only (so "mlp.fc1"
            matches "blocks.0.mlp.fc1"). If False, require exact match on the
            leaf name.
    """
    candidates = list(candidates)
    matches: list[str] = []
    for name, _module in model.named_modules():
        leaf = name.split(".")[-1] if name else ""
        if not leaf:
            continue
        for cand in candidates:
            cand_leaf = cand.split(".")[-1]
            if suffix_only and leaf == cand_leaf:
                matches.append(name)
                break
            if not suffix_only and leaf == cand_leaf:
                matches.append(name)
                break
    return matches