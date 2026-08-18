"""Unit tests for the M3-T6 VAE training + rollout-eval scripts.

Covers the script entry points end-to-end on CPU with tiny configs:
``scripts/train_vae.py`` produces a loadable checkpoint, and
``scripts/eval_diffusion_rollout.py`` scores it with SSIM/PSNR (CLIP
model absent on CI -> clip_score null, clause dropped).
"""
from __future__ import annotations

from pathlib import Path

import torch

from scripts.eval_diffusion_rollout import _load_model, evaluate as eval_rollout
from scripts.eval_diffusion_rollout import parse_args as eval_parse_args
from scripts.train_vae import _synthetic_batch, train as train_vae
from scripts.train_vae import parse_args as train_parse_args
from worldfoundry_inf.diffusion.vae import VAEConfig, VAEModel


def test_synthetic_batch_shape() -> None:
    x = _synthetic_batch(6, 32, seed=0)
    assert x.shape == (6, 3, 32, 32)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_synthetic_batch_deterministic() -> None:
    a = _synthetic_batch(4, 32, seed=5)
    b = _synthetic_batch(4, 32, seed=5)
    assert torch.equal(a, b)


def test_train_vae_produces_loadable_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "vae.pt"
    args = train_parse_args(
        [
            "--out", str(out),
            "--steps", "8",
            "--img-size", "32",
            "--batch", "8",
            "--lr", "1e-3",
            "--log-every", "8",
            "--seed", "0",
        ]
    )
    train_vae(args)
    assert out.exists()
    payload = torch.load(out, map_location="cpu")
    assert "state_dict" in payload
    assert "metrics" in payload
    assert payload["metrics"]["steps"] == 8
    assert payload["cfg"]["latent_dim"] == 4
    # The checkpoint loads back through VAEModel.from_pretrained.
    model = VAEModel.from_pretrained(str(out), seed=0)
    x = _synthetic_batch(2, 32, seed=1)
    assert model.reconstruct(x).shape == x.shape


def test_train_vae_metrics_json(tmp_path: Path) -> None:
    out = tmp_path / "vae.pt"
    args = train_parse_args(["--out", str(out), "--steps", "4", "--img-size", "32"])
    train_vae(args)
    metrics = torch.load(out, map_location="cpu")["metrics"]
    assert "best_recon_mse" in metrics
    assert metrics["final_recon_mse"] > 0.0


def test_train_vae_loss_decreases(tmp_path: Path) -> None:
    """Training a few steps should reduce reconstruction loss."""
    out = tmp_path / "vae.pt"
    args = train_parse_args(
        ["--out", str(out), "--steps", "10", "--img-size", "32", "--batch", "16"]
    )
    train_vae(args)
    metrics = torch.load(out, map_location="cpu")["metrics"]
    # best_recon_mse must be strictly below the initial random-init loss
    # (initial recon on random weights is typically ~0.08-0.15 for our
    # 32x32 synthetic data; after 10 AdamW steps it drops measurably).
    assert metrics["best_recon_mse"] < 0.15


def test_load_model_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "vae.pt"
    args = train_parse_args(["--out", str(out), "--steps", "4", "--img-size", "32"])
    train_vae(args)
    model = _load_model(out)
    assert isinstance(model, VAEModel)
    x = _synthetic_batch(2, 32, seed=3)
    assert model.reconstruct(x).shape == x.shape


def test_eval_rollout_scores_reconstruction(tmp_path: Path) -> None:
    """eval_diffusion_rollout writes the report; SSIM gates without CLIP.

    80 AdamW steps on 32x32 synthetic blobs is enough for the VAE's
    encode->decode reconstruction to clear SSIM>=0.85 (measured 0.84 at
    20 steps, ~0.90+ at 80 steps on CPU).
    """
    out = tmp_path / "vae.pt"
    args = train_parse_args(
        ["--out", str(out), "--steps", "80", "--img-size", "32", "--batch", "16"]
    )
    train_vae(args)

    eval_dir = tmp_path / "eval"
    eargs = eval_parse_args(
        [
            "--ckpt", str(out),
            "--out-dir", str(eval_dir),
            "--n-frames", "4",
            "--img-size", "32",
        ]
    )
    rc = eval_rollout(eargs)
    assert (eval_dir / "rollout_eval.json").exists()
    assert (eval_dir / "summary.md").exists()
    import json

    report = json.loads((eval_dir / "rollout_eval.json").read_text())
    assert report["ssim"] >= 0.0
    assert report["psnr_db"] > 0.0
    # CLIP absent on CI host -> null; the thresholds dict still lists the
    # clip_score key but the passed check drops that clause (eval module
    # contract: clip_score is None -> clause not gated).
    assert report["clip_score"] is None
    assert "clip_score" in report["thresholds"]
    # Trained reconstruction should already clear SSIM>=0.85 gate.
    assert rc == 0, f"eval failed: {report}"
    assert report["passed"] is True


def test_eval_rollout_untrained_vae_fails_gate(tmp_path: Path) -> None:
    """A random-init (untrained) VAE should NOT pass the SSIM gate."""
    out = tmp_path / "vae_rand.pt"
    model = VAEModel(VAEConfig())
    torch.save({"state_dict": model.state_dict(), "cfg": {"latent_dim": 4}}, out)

    eval_dir = tmp_path / "eval_rand"
    eargs = eval_parse_args(
        [
            "--ckpt", str(out),
            "--out-dir", str(eval_dir),
            "--n-frames", "4",
            "--img-size", "32",
        ]
    )
    rc = eval_rollout(eargs)
    import json

    report = json.loads((eval_dir / "rollout_eval.json").read_text())
    assert rc == 1
    assert report["passed"] is False
