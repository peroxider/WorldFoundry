"""Runtime discovery of LoRA-targetable module names in a model."""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def discover_lora_targets(
    model: nn.Module,
    candidates: Iterable[str],
    suffix_only: bool = True,
) -> list[str]:
    """Return fully-qualified module names in `model` matching `candidates`.

    Args:
        model: a torch.nn.Module (e.g. VJEPA2Model).
        candidates: candidate names. Each may be a leaf name (e.g. "q_proj") or
            a dotted suffix (e.g. "mlp.fc1") or a fully-qualified path
            (e.g. "blocks.0.attn.q_proj").
        suffix_only: if True (default), match by leaf name only — "q_proj"
            matches any module whose leaf is "q_proj", and "mlp.fc1" matches
            any module whose leaf is "fc1" (the last segment of the candidate).
            If False, require an exact full-path match — the module's fully-
            qualified name must equal the candidate exactly, so "mlp.fc1"
            matches no module while "blocks.0.attn.q_proj" matches only that
            exact path.
    """
    candidates = list(candidates)
    matches: list[str] = []
    for name, _module in model.named_modules():
        if not name:
            continue
        for cand in candidates:
            if suffix_only:
                leaf = name.split(".")[-1]
                cand_leaf = cand.split(".")[-1]
                if leaf == cand_leaf:
                    matches.append(name)
                    break
            else:
                if name == cand:
                    matches.append(name)
                    break
    return matches
