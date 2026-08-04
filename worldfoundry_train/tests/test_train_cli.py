"""CLI smoke for worldforge_train.train (W14).

Validates the CLI surface change (``--steps`` flag, FSDP no longer raises)
without modifying any existing test. Phase-1 contract preserved
(``test_train_help_runs`` continues to pass).
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from worldforge_train.train import parse_args, run_fsdp_training


def test_cli_help_lists_parallel_and_steps():
    r = subprocess.run(
        [sys.executable, "-m", "worldforge_train.train", "--help"],
        capture_output=True, text=True,
        cwd="/mnt/c/WorkSpace/WorldForge",
    )
    assert r.returncode == 0
    assert "--parallel" in r.stdout
    assert "--steps" in r.stdout


def test_cli_help_advertises_ddp_and_fsdp_choices():
    r = subprocess.run(
        [sys.executable, "-m", "worldforge_train.train", "--help"],
        capture_output=True, text=True,
        cwd="/mnt/c/WorkSpace/WorldForge",
    )
    assert "ddp" in r.stdout and "fsdp" in r.stdout


def test_parse_args_accepts_steps_override():
    args = parse_args(["--config", "x.yaml", "--steps", "5", "--parallel", "fsdp"])
    assert args.steps == 5
    assert args.parallel == "fsdp"
    assert args.strict_versions is False


def test_parse_args_defaults_steps_to_zero():
    args = parse_args(["--config", "x.yaml"])
    assert args.steps == 0


def test_cli_missing_manifest_exits_nonzero_with_explicit_error():
    r = subprocess.run(
        [sys.executable, "-m", "worldforge_train.train",
         "--config", "/nonexistent/manifest.yaml",
         "--steps", "1"],
        capture_output=True, text=True,
        cwd="/mnt/c/WorkSpace/WorldForge",
    )
    assert r.returncode != 0
    assert "Manifest not found" in r.stderr or "FileNotFoundError" in r.stderr


def test_cli_rejects_unknown_parallel_value():
    r = subprocess.run(
        [sys.executable, "-m", "worldforge_train.train",
         "--config", "x.yaml", "--parallel", "bogus"],
        capture_output=True, text=True,
        cwd="/mnt/c/WorkSpace/WorldForge",
    )
    assert r.returncode == 2  # argparse default for invalid choice


def test_run_fsdp_training_no_longer_raises_not_implemented():
    # Pass cfg with nonexistent manifest; trainer will raise FileNotFoundError
    # (from Vjepa2Dataset.__init__), NOT NotImplementedError.
    cfg = {"data": {"manifest": "/nonexistent", "mask_tubelet_size": [1, 14, 14],
                    "mask_ratio": 0.5},
           "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
           "training": {"per_device_batch": 2, "epochs": 1, "lr": 1e-4,
                        "log_every_n": 1}}
    with pytest.raises((FileNotFoundError, ValueError)) as exc_info:
        run_fsdp_training(cfg, local_rank=0, global_rank=0, world_size=1, num_steps=1)
    assert "NotImplementedError" not in type(exc_info.value).__name__


# --- W16 Checkpoint CLI tests ---

def test_cli_help_includes_checkpoint_flags():
    r = subprocess.run(
        [sys.executable, "-m", "worldforge_train.train", "--help"],
        capture_output=True, text=True,
        cwd="/mnt/c/WorkSpace/WorldForge",
    )
    assert r.returncode == 0
    assert "--checkpoint-dir" in r.stdout
    assert "--resume-from" in r.stdout
    assert "--save-every-n" in r.stdout


def test_parse_args_accepts_checkpoint_flags():
    args = parse_args([
        "--config", "x.yaml",
        "--checkpoint-dir", "/tmp/ckpts",
        "--resume-from", "/tmp/ckpts/step_5.pt",
        "--save-every-n", "10",
    ])
    assert str(args.checkpoint_dir) == "/tmp/ckpts"
    assert str(args.resume_from) == "/tmp/ckpts/step_5.pt"
    assert args.save_every_n == 10


def test_parse_args_defaults_checkpoint_flags_to_none():
    args = parse_args(["--config", "x.yaml"])
    assert args.checkpoint_dir is None
    assert args.resume_from is None
    assert args.save_every_n is None
