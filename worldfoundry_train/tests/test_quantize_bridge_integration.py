"""CPU e2e: tiny VJEPA2Model -> quantize (fake) -> bridge -> run_training 1 step.

Mirrors worldforge_inf/tests/test_quant_integration_cpu.py pattern with
one extra step (LoRA + 1-step training). No NPU required.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import transformers

from worldforge_train.parallel.fsdp import FSDPStrategy
from worldforge_train.quantize_bridge import maybe_quantize_first
from worldforge_train.trainer import run_training

import worldforge_train.trainer as _trainer_mod


def _make_tiny_vjepa2_and_save(tmp_path: Path) -> Path:
    cfg = transformers.VJEPA2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        image_size=196, patch_size=14, tubelet_size=1, in_chans=3,
        frames_per_clip=1, drop_path_rate=0.0,
    )
    model = transformers.VJEPA2Model(cfg)
    out = tmp_path / "tiny"
    model.save_pretrained(out)
    return out


def _make_batch():
    """(B=1, T=1, C=3, H=W=196) frames + (B=1, T*Hg*Wg) bool mask matching tiny cfg."""
    Hg, Wg = 14, 14
    return {
        "pixel_values_videos": torch.zeros(1, 1, 3, 196, 196),
        "bool_masked_pos": torch.ones(1, Hg * Wg, dtype=torch.bool),
    }


def test_bridge_to_train_end_to_end(tmp_path, monkeypatch):
    src = _make_tiny_vjepa2_and_save(tmp_path)
    out_dir = tmp_path / "quant"
    cfg = {
        "model": {"hf_repo": str(src)},
        "quant": {
            "type": "fake",
            "source_dir": str(src),
            "output_dir": str(out_dir),
            "calibration_dataset": "videomix22m-512",
        },
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-4, "log_every_n": 1},
    }
    cfg = maybe_quantize_first(cfg)
    assert cfg["model"]["hf_repo"] == str(out_dir)
    assert (out_dir / "quant_meta.json").is_file()

    # Stub out _build_dataset — the trainer calls it before our batch_fn is used,
    # but we provide batch_fn so the dataset is never iterated. The stub returns
    # a minimal duck-typed object with len() > 0. monkeypatch (not direct
    # assignment) so the real _build_dataset is restored after this test;
    # test_train_cli.py relies on it raising FileNotFoundError.
    class _StubDS:
        def __len__(self):
            return 1
    monkeypatch.setattr(_trainer_mod, "_build_dataset", lambda _cfg: _StubDS())

    base_model = transformers.AutoModel.from_pretrained(cfg["model"]["hf_repo"])
    losses = run_training(
        cfg, FSDPStrategy(), num_steps=1, base_model=base_model,
        batch_fn=lambda _i: _make_batch(),
    )
    assert len(losses) == 1
    assert losses[0] == pytest.approx(losses[0])  # finite; not NaN/Inf
    assert 0.0 <= losses[0] < 1e6