"""CLI-level smoke: train.py main() accepts a YAML with quant: block.

Mock everything below the bridge so we only exercise the cfg-load +
bridge + setup_ddp wiring. Verifies that adding quant: doesn't break
the existing CLI flow.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from worldforge_train import train as train_mod


def test_main_accepts_quant_block(tmp_path, monkeypatch):
    cfg = {
        "model": {"hf_repo": "fake/hf_repo"},
        "quant": {
            "type": "fake",
            "source_dir": "/nonexistent",
            "output_dir": "/tmp/out",
        },
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-4, "log_every_n": 1},
        "data": {
            "manifest": str(tmp_path / "m.csv"),
            "mask_tubelet_size": [1, 14, 14],
            "mask_ratio": 0.5,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", str(cfg_path), "--steps", "0", "--parallel", "ddp",
    ])
    seen = {}
    monkeypatch.setattr(
        "worldforge_train.quantize_bridge.maybe_quantize_first",
        lambda c: (seen.update({"called_with": c}) or c),
    )
    monkeypatch.setattr(
        "worldforge_train.parallel.ddp.setup_ddp", lambda: (1, 0, 0),
    )
    monkeypatch.setattr(
        "worldforge_train.train.run_ddp_training", lambda *a, **kw: [],
    )
    rc = train_mod.main()
    assert rc == 0
    assert "quant" in seen["called_with"]
    assert seen["called_with"]["quant"]["type"] == "fake"


def test_main_without_quant_block_still_runs(tmp_path, monkeypatch):
    cfg = {
        "model": {"hf_repo": "fake/hf_repo"},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-4, "log_every_n": 1},
        "data": {
            "manifest": str(tmp_path / "m.csv"),
            "mask_tubelet_size": [1, 14, 14],
            "mask_ratio": 0.5,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", str(cfg_path), "--steps", "0", "--parallel", "ddp",
    ])
    monkeypatch.setattr(
        "worldforge_train.parallel.ddp.setup_ddp", lambda: (1, 0, 0),
    )
    monkeypatch.setattr(
        "worldforge_train.train.run_ddp_training", lambda *a, **kw: [],
    )
    rc = train_mod.main()
    assert rc == 0