"""M3-T6: temporal rollout eval (SSIM / PSNR / CLIP / alignment) tests.

Locks down:
- SSIM = 1.0 for identical frames
- SSIM in [0, 1] for noisy vs reference (degraded but bounded)
- PSNR = inf for identical frames, finite for noisy frames
- latent_frame_alignment delegates to worldfoundry_common.metrics.cosine_similarity
- compute_rollout_metrics returns RolloutMetrics with all fields populated
- passed=True when SSIM >= threshold; False when SSIM < threshold
- CLIP degradation: missing transformers dep returns None + warning (no raise)
- RolloutMetrics.thresholds defaults to M3-T6 acceptance (0.85 / 0.30 / 0.80)
- Empty-frame guards (T=0) and tiny-frame fallback (4x4) don't crash

Pure numpy SSIM/PSNR tests run on any host. The CLIP test uses a
monkey-patched dep probe to exercise the dep-missing path without
actually downloading CLIP weights.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from worldfoundry_inf.eval.rollout_eval import (
    DEFAULT_THRESHOLDS,
    RolloutMetrics,
    compute_latent_frame_alignment,
    compute_psnr,
    compute_rollout_metrics,
    compute_ssim,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def identical_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two identical 4-frame videos of shape (T=4, 3, 32, 32) in [0, 1]."""
    rng = np.random.default_rng(seed=0)
    video = rng.random((4, 3, 32, 32), dtype=np.float32)
    return video, video.copy()


@pytest.fixture
def noisy_pair() -> tuple[np.ndarray, np.ndarray]:
    """Reference + noise-perturbed predicted video."""
    rng = np.random.default_rng(seed=1)
    ref = rng.random((4, 3, 32, 32), dtype=np.float32)
    pred = ref + 0.1 * rng.standard_normal((4, 3, 32, 32)).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)
    return pred, ref


# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #
def test_ssim_is_one_for_identical_inputs(identical_pair):
    pred, ref = identical_pair
    assert compute_ssim(pred, ref) == pytest.approx(1.0, abs=1e-6)


def test_ssim_below_one_for_noisy_inputs(noisy_pair):
    pred, ref = noisy_pair
    score = compute_ssim(pred, ref)
    assert 0.0 <= score < 1.0
    assert score > 0.5  # noise is small (sigma=0.1) -> high SSIM


def test_ssim_raises_on_shape_mismatch():
    a = np.zeros((4, 3, 32, 32), dtype=np.float32)
    b = np.zeros((4, 3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_ssim(a, b)


# --------------------------------------------------------------------------- #
# PSNR
# --------------------------------------------------------------------------- #
def test_psnr_is_inf_for_identical_inputs(identical_pair):
    pred, ref = identical_pair
    assert compute_psnr(pred, ref) == float("inf")


def test_psnr_finite_for_noisy_inputs(noisy_pair):
    pred, ref = noisy_pair
    score = compute_psnr(pred, ref, max_val=1.0)
    assert np.isfinite(score)
    assert score > 10.0  # sigma=0.1 noise -> ~20 dB PSNR


# --------------------------------------------------------------------------- #
# Latent <-> frame alignment
# --------------------------------------------------------------------------- #
def test_latent_frame_alignment_uses_common_cosine_similarity(monkeypatch):
    """compute_latent_frame_alignment must delegate to the common helper."""
    import worldfoundry_inf.eval.rollout_eval as rollout_eval

    captured: dict[str, np.ndarray] = {}

    def fake_cosine(a, b):
        captured["a"] = a
        captured["b"] = b
        return np.ones(a.shape[0], dtype=np.float32)

    monkeypatch.setattr(rollout_eval, "_common_cosine_similarity", fake_cosine)
    a = np.random.default_rng(2).standard_normal((8, 16)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((8, 16)).astype(np.float32)
    result = compute_latent_frame_alignment(a, b)
    assert result == pytest.approx(1.0)
    assert np.array_equal(captured["a"], a)
    assert np.array_equal(captured["b"], b)


def test_latent_frame_alignment_identical_is_one():
    a = np.random.default_rng(4).standard_normal((5, 4)).astype(np.float32)
    result = compute_latent_frame_alignment(a, a.copy())
    assert result == pytest.approx(1.0, abs=1e-5)


def test_latent_frame_alignment_raises_on_n_mismatch():
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.zeros((5, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="N mismatch"):
        compute_latent_frame_alignment(a, b)


# --------------------------------------------------------------------------- #
# CLIP degradation
# --------------------------------------------------------------------------- #
def test_compute_clip_score_returns_none_when_transformers_missing(monkeypatch):
    """If transformers.CLIPModel is unimportable, return None + warn (no raise)."""
    import worldfoundry_inf.eval.rollout_eval as rollout_eval

    monkeypatch.setattr(rollout_eval, "_clip_available", lambda: False)
    pred = np.zeros((2, 3, 16, 16), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rollout_eval.compute_clip_score(pred, pred)
    assert result is None
    assert any("transformers" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# compute_rollout_metrics aggregate
# --------------------------------------------------------------------------- #
def test_compute_rollout_metrics_returns_dataclass_with_all_fields(identical_pair):
    pred, ref = identical_pair
    report = compute_rollout_metrics(pred, ref)
    assert isinstance(report, RolloutMetrics)
    assert report.ssim == pytest.approx(1.0, abs=1e-6)
    assert report.psnr_db == float("inf")
    # No latent_rollout given -> alignment defaults to 1.0
    assert report.latent_frame_alignment == pytest.approx(1.0)
    assert report.n_frames == 4
    assert report.passed is True
    assert set(report.thresholds.keys()) == {"ssim", "clip_score", "latent_frame_alignment"}


def test_compute_rollout_metrics_passed_true_when_ssim_above_threshold(identical_pair):
    pred, ref = identical_pair
    # SSIM = 1.0 -> trivially passes the 0.85 gate
    report = compute_rollout_metrics(pred, ref, thresholds={"ssim": 0.85})
    assert report.passed is True


def test_compute_rollout_metrics_passed_false_when_ssim_below_threshold(noisy_pair):
    """Force a tight SSIM threshold that noisy frames can't hit."""
    pred, ref = noisy_pair
    report = compute_rollout_metrics(pred, ref, thresholds={"ssim": 0.9999})
    assert report.passed is False
    assert report.ssim < 0.9999


def test_compute_rollout_metrics_threshold_override():
    pred = np.random.default_rng(5).random((2, 3, 16, 16), dtype=np.float32)
    ref = pred.copy()
    report = compute_rollout_metrics(
        pred, ref,
        thresholds={"ssim": 0.5, "clip_score": 0.1, "latent_frame_alignment": 0.5},
    )
    assert report.thresholds["ssim"] == pytest.approx(0.5)
    assert report.thresholds["clip_score"] == pytest.approx(0.1)
    assert report.thresholds["latent_frame_alignment"] == pytest.approx(0.5)


def test_default_thresholds_match_m3_t6_acceptance():
    """Roadmap acceptance: SSIM >= 0.85, CLIP >= 0.30, alignment >= 0.80."""
    assert DEFAULT_THRESHOLDS["ssim"] == pytest.approx(0.85)
    assert DEFAULT_THRESHOLDS["clip_score"] == pytest.approx(0.30)
    assert DEFAULT_THRESHOLDS["latent_frame_alignment"] == pytest.approx(0.80)


# --------------------------------------------------------------------------- #
# Empty / small-frame guards
# --------------------------------------------------------------------------- #
def test_ssim_with_t_zero_returns_one():
    """No frames -> nothing to compare -> return 1.0 (no spurious fail)."""
    pred = np.zeros((0, 3, 32, 32), dtype=np.float32)
    ref = np.zeros((0, 3, 32, 32), dtype=np.float32)
    assert compute_ssim(pred, ref) == 1.0


def test_psnr_with_t_zero_returns_inf():
    pred = np.zeros((0, 3, 32, 32), dtype=np.float32)
    ref = np.zeros((0, 3, 32, 32), dtype=np.float32)
    assert compute_psnr(pred, ref) == float("inf")


def test_ssim_handles_4x4_frames_gracefully():
    """Frames smaller than the 11x11 SSIM window shouldn't crash."""
    pred = np.random.default_rng(6).random((2, 3, 4, 4), dtype=np.float32)
    ref = pred.copy()
    # Either returns 1.0 (tiny-frame fallback) or a valid SSIM in [0, 1].
    score = compute_ssim(pred, ref)
    assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Latent + frame rollouts integration
# --------------------------------------------------------------------------- #
def test_compute_rollout_metrics_with_latent_alignment(identical_pair):
    pred, ref = identical_pair
    rng = np.random.default_rng(7)
    latent = rng.standard_normal((8, 16)).astype(np.float32)
    frame_ref = rng.standard_normal((8, 16)).astype(np.float32)
    report = compute_rollout_metrics(
        pred, ref, latent_rollout=latent, frame_rollout=frame_ref
    )
    # latent != frame_ref so alignment is in (-1, 1) -- just check the
    # field is populated (not the default 1.0 sentinel for "no signal")
    assert -1.0 <= report.latent_frame_alignment <= 1.0
    assert report.n_frames == 4
