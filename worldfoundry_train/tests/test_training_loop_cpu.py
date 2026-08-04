"""CPU e2e: real tiny V-JEPA 2 + real LoRA + 3-step training loop.

NOT NPU-gated -- runs by default. Validates:
1. ``lora_wrapper.build_lora_model`` wraps real VJEPA2Model and returns
   trainable_params > 0 (G4 sanity check)
2. ``FSDPStrategy.wrap`` returns the model unchanged in single-rank mode
3. ``run_training_loop`` runs 3 real forward+backward+step passes,
   returns 3 finite losses, and weight deltas are non-zero
"""
from __future__ import annotations

import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

from worldforge_train.model.lora_wrapper import build_lora_model
from worldforge_train.parallel.fsdp import FSDPStrategy
from worldforge_train.training_loop import run_training_loop


def _build_real_tiny_vjepa2() -> VJEPA2Model:
    cfg = VJEPA2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=224,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        frames_per_clip=1,
        drop_path_rate=0.0,
    )
    return VJEPA2Model(cfg).eval()


def _synthetic_batch_iter(n: int):
    for _ in range(n):
        yield {
            "pixel_values_videos": torch.randn(1, 1, 3, 224, 224),
            # bool_masked_pos shape must match last_hidden_state (B, num_patches, hidden)
            # -> mask is (B, num_patches). 224/16=14 -> 14*14=196 patches.
            # Mark half the patches as masked so mse_loss has non-empty input
            # (all-False mask -> empty indexed tensor -> NaN loss).
            "bool_masked_pos": (torch.rand(1, 196) < 0.5),
        }


def test_real_vjepa2_lora_three_step_training_loop_on_cpu():
    torch.manual_seed(42)
    base = _build_real_tiny_vjepa2()

    lora_result = build_lora_model(base, r=4, alpha=8)
    model = lora_result.peft_model
    assert lora_result.trainable_params > 0
    assert lora_result.target_modules

    strategy = FSDPStrategy()
    wrapped = strategy.wrap(model)
    assert wrapped is model

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    losses = run_training_loop(
        model, _synthetic_batch_iter(n=3), num_steps=3,
        strategy=strategy, optimizer=optimizer,
    )

    assert len(losses) == 3
    for v in losses:
        assert v == v
        assert v >= 0.0

    changed = 0
    for n, p in model.named_parameters():
        if p.requires_grad and not torch.equal(initial[n], p.detach()):
            changed += 1
    assert changed > 0