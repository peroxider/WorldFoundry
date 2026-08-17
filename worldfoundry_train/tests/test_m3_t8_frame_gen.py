"""M3-T8: 5-stage orchestrator --frame-gen-backend wiring tests.

Locks down:
- argparse accepts ``--frame-gen-backend {none, mindie_sd}``; rejects others
- Default ``none`` preserves all pre-M3-T8 behavior (no-op frame_gen block)
- ``frame_gen_phase(cfg, "none", device)`` returns a structured no-op dict
  with ``frame_gen_dispatched.dispatched == False``
- ``frame_gen_phase(cfg, "mindie_sd", device)`` dispatches via
  ``make_engine("mindie_sd")``, writes
  ``frame_gen_dispatched.engine_type == "CPUPixelDecoderPipeline"``,
  computes M3-T6 rollout metrics, and times the phase
- Summary JSON includes ``frame_gen``, ``frame_gen_dispatched``,
  ``rss_mb_per_phase.after_frame_gen``, and
  ``config_serializable.frame_gen_backend``
- ``provider_summary()`` exists on the LatentToFramePipeline ABC and
  the CPU pixel-decoder override surfaces class identity + shape kwargs
- The end-to-end run with ``--frame-gen-backend none`` still passes
  (zero regression on existing 5-stage pipeline)

The M3-T8 dispatch error path is covered via monkeypatching
``make_engine`` to raise ImportError so the test stays host-portable.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# provider_summary ABC + override
# --------------------------------------------------------------------------- #
def test_provider_summary_default_on_abc():
    """The ABC must expose a default provider_summary() that doesn't raise."""
    from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline

    # Build a minimal stub subclass that satisfies the abstract methods
    # so we can call the inherited default.
    class _Stub(LatentToFramePipeline):
        name = "stub"

        def encode_latents(self, video):  # pragma: no cover - not invoked
            return video

        def decode_latents(self, latent, grid):  # pragma: no cover
            return latent

        def rollout(self, past_video, actions, past_frames, future_frames):
            return past_video[:, :future_frames]

    summary = _Stub().provider_summary()
    assert summary["engine"] == "_Stub"
    assert summary["kind"] == "unknown"
    assert summary["available"] is True


def test_cpu_pixel_decoder_provider_summary_includes_shape_kwargs():
    """CPU pipeline override must surface class identity + shape kwargs."""
    from worldfoundry_inf.diffusion.backends.cpu import (
        CPUPixelDecoderPipeline,
    )

    p = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14, seed=7)
    s = p.provider_summary()
    assert s["engine"] == "cpu_pixel_decoder"
    assert s["kind"] == "cpu_pixel_decoder"
    assert s["available"] is True
    extras = s["extras"]
    assert extras["class"] == "CPUPixelDecoderPipeline"
    assert extras["encoder_dim"] == 64
    assert extras["img_size"] == 196
    assert extras["patch_size"] == 14
    assert extras["seed"] == 7


# --------------------------------------------------------------------------- #
# frame_gen_phase() function
# --------------------------------------------------------------------------- #
def test_frame_gen_phase_none_is_structured_noop():
    """backend='none' returns a no-op dict with dispatched=False."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from render_train_rl_cpu import V2Config, frame_gen_phase

    cfg = V2Config(out_dir=Path("/tmp/does_not_matter"), num_clips=1, frames_per_clip=2)
    out = frame_gen_phase(cfg, "none", "cpu")
    assert out["backend"] == "none"
    assert out["elapsed_seconds"] == 0.0
    assert out["frame_metrics"] is None
    assert out["n_rollout_frames"] == 0
    fg = out["frame_gen_dispatched"]
    assert fg["backend"] == "none"
    assert fg["dispatched"] is False
    assert fg["engine_type"] is None
    assert fg["kind"] is None
    assert fg["compile_path"] is None
    assert fg["dispatch_error"] is None


def test_frame_gen_phase_unknown_backend_records_dispatch_error():
    """A programmer-typo'd backend records a structured error (no crash)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from render_train_rl_cpu import V2Config, frame_gen_phase

    cfg = V2Config(out_dir=Path("/tmp/x"), num_clips=1, frames_per_clip=2)
    out = frame_gen_phase(cfg, "not_a_backend", "cpu")
    assert out["frame_gen_dispatched"]["backend"] == "not_a_backend"
    assert "Unknown frame_gen_backend" in (
        out["frame_gen_dispatched"]["dispatch_error"] or ""
    )


def test_frame_gen_phase_mindie_sd_dispatches_cpu_pipeline(monkeypatch):
    """backend='mindie_sd' builds the CPU pipeline, records fingerprint,
    runs rollout, and computes M3-T6 metrics."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from render_train_rl_cpu import V2Config, frame_gen_phase

    # Build a small cfg; ``frame_gen_phase`` constructs the CPU
    # pipeline with ``img_size=cfg.height, patch_size=cfg.height//8``
    # so the test must use heights that are multiples of 8. past_video
    # of shape (1, 4, 3, 32, 32) covers past_frames=2 + future_frames=2
    # at the orchestrator's 32x32 default.
    cfg = V2Config(
        out_dir=Path("/tmp/x"), num_clips=1, frames_per_clip=4,
        height=32, width=32,
    )
    past_video = torch.zeros(1, 4, 3, 32, 32, dtype=torch.float32)

    out = frame_gen_phase(
        cfg, "mindie_sd", "cpu",
        predicted_video=past_video, past_frames=2, future_frames=2,
    )
    assert out["backend"] == "mindie_sd"
    assert out["elapsed_seconds"] > 0.0
    assert out["n_rollout_frames"] == 2
    fg = out["frame_gen_dispatched"]
    assert fg["backend"] == "mindie_sd"
    assert fg["dispatched"] is True
    assert fg["engine_type"] == "CPUPixelDecoderPipeline"
    assert fg["kind"] == "cpu_pixel_decoder"
    assert fg["dispatch_error"] is None
    # M3-T6 metrics populated.
    fm = out["frame_metrics"]
    assert fm is not None
    assert "ssim" in fm
    assert "psnr_db" in fm
    assert "n_frames" in fm
    assert fm["n_frames"] == 2


def test_frame_gen_phase_mindie_sd_records_dispatch_error_on_make_engine_failure(
    monkeypatch,
):
    """If make_engine raises, fingerprint records structured error (no raise)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from render_train_rl_cpu import V2Config, frame_gen_phase

    def _boom(_backend, **kwargs):
        raise ImportError("simulated missing wheel")

    monkeypatch.setattr(
        "worldfoundry_common.backends.inference.make_engine", _boom,
    )
    cfg = V2Config(out_dir=Path("/tmp/x"), num_clips=1, frames_per_clip=2)
    out = frame_gen_phase(cfg, "mindie_sd", "cpu")
    fg = out["frame_gen_dispatched"]
    assert fg["backend"] == "mindie_sd"
    assert fg["dispatched"] is False
    assert "simulated missing wheel" in (fg["dispatch_error"] or "")


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def test_argparse_accepts_frame_gen_backend_choices():
    """--frame-gen-backend accepts {none, mindie_sd}; rejects other values."""
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "render_train_rl_cpu.py"),
            "--frame-gen-backend", "mindie_sd",
            "--help",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    # --help exits 0; the choice list must show {none,mindie_sd}
    assert proc.returncode == 0
    assert "{none,mindie_sd}" in proc.stdout


def test_argparse_rejects_unknown_frame_gen_backend():
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "render_train_rl_cpu.py"),
            "--frame-gen-backend", "bogus",
            "--help",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    # argparse rejects the value with exit code 2 and writes to stderr
    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr


# --------------------------------------------------------------------------- #
# End-to-end summary JSON
# --------------------------------------------------------------------------- #
def test_end_to_end_default_none_writes_noop_fingerprint(tmp_path: Path) -> None:
    """Default --frame-gen-backend=none must produce a dispatched=False fingerprint."""
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "render_train_rl_cpu.py"),
            "--out-dir", str(tmp_path), "--num-clips", "1", "--frames-per-clip", "2",
            "--height", "32", "--width", "32", "--num-steps", "2",
            "--ppo-env-steps", "4", "--num-outer-iters", "1",
            "--num-train-per-iter", "2", "--num-ppo-epochs", "1",
            # default --frame-gen-backend=none (explicit for clarity)
            "--frame-gen-backend", "none",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    summary = json.loads((tmp_path / "render_train_rl.json").read_text())

    # New top-level keys present.
    assert "frame_gen" in summary
    assert "frame_gen_dispatched" in summary

    # No-op frame_gen block.
    fg = summary["frame_gen"]
    assert fg["backend"] == "none"
    assert fg["elapsed_seconds"] == 0.0
    assert fg["frame_metrics"] is None
    assert fg["n_rollout_frames"] == 0

    # No-op dispatched fingerprint.
    fgd = summary["frame_gen_dispatched"]
    assert fgd["backend"] == "none"
    assert fgd["dispatched"] is False
    assert fgd["engine_type"] is None
    assert fgd["dispatch_error"] is None

    # config_serializable mirrors the CLI choice.
    assert summary["config_serializable"]["frame_gen_backend"] == "none"

    # RSS slot present even when the phase is a no-op.
    assert "after_frame_gen" in summary["rss_mb_per_phase"]


def test_end_to_end_mindie_sd_dispatches_cpu_pipeline(tmp_path: Path) -> None:
    """--frame-gen-backend=mindie_sd dispatches CPUPixelDecoderPipeline end-to-end."""
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "render_train_rl_cpu.py"),
            "--out-dir", str(tmp_path), "--num-clips", "1", "--frames-per-clip", "4",
            "--height", "32", "--width", "32", "--num-steps", "2",
            "--ppo-env-steps", "4", "--num-outer-iters", "1",
            "--num-train-per-iter", "2", "--num-ppo-epochs", "1",
            "--frame-gen-backend", "mindie_sd",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    summary = json.loads((tmp_path / "render_train_rl.json").read_text())

    fg = summary["frame_gen"]
    assert fg["backend"] == "mindie_sd"
    assert fg["elapsed_seconds"] > 0.0
    assert fg["n_rollout_frames"] > 0
    fm = fg["frame_metrics"]
    assert fm is not None
    assert "ssim" in fm
    assert "psnr_db" in fm

    fgd = summary["frame_gen_dispatched"]
    assert fgd["backend"] == "mindie_sd"
    assert fgd["dispatched"] is True
    assert fgd["engine_type"] == "CPUPixelDecoderPipeline"
    assert fgd["kind"] == "cpu_pixel_decoder"
    assert fgd["dispatch_error"] is None

    assert summary["config_serializable"]["frame_gen_backend"] == "mindie_sd"
    assert "after_frame_gen" in summary["rss_mb_per_phase"]
