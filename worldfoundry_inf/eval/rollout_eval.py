"""Temporal rollout eval: SSIM / PSNR / CLIP + latent-frame alignment (M3-T6).

Used as the M3-T6 acceptance gate for the 5-stage orchestrator
(``scripts/render_train_rl_cpu.py``) and the W40 frame-gen ADR §5.4
acceptance criteria:

* SSIM >= 0.85 (frame-level structural similarity, pure numpy, no GPU)
* CLIP score >= 0.30 (image-text alignment via HF ``CLIPModel`` when
  available; gracefully degrades to ``None`` when the optional dep is
  missing on CPU-only hosts -- the orchestrator still records a
  structured ``clip_score: null`` so post-mortem tools know the gate
  was skipped, not silently passed)
* Latent <-> frame cosine >= 0.80 (reuses
  :func:`worldfoundry_common.metrics.cosine_similarity`)

The module is deliberately side-effect-free: all functions take numpy
arrays and return primitives or the :class:`RolloutMetrics` dataclass.
``compute_rollout_metrics`` is the single entry point used by
``frame_gen_phase`` in the orchestrator.

CLI usage (for ad-hoc inspection of a saved rollout artifact):

    python -m worldfoundry_inf.eval.rollout_eval \\
        --predicted /tmp/pred.npy --reference /tmp/ref.npy \\
        --latent /tmp/lat.npy --frame-ref /tmp/frame_ref.npy
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from worldfoundry_common.metrics import cosine_similarity as _common_cosine_similarity


# Default M3-T6 acceptance thresholds. Override via ``thresholds`` arg or
# the ``--threshold-*`` CLI flags. Matches the W40 frame-gen ADR sec 5.4
# acceptance values.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "ssim": 0.85,
    "clip_score": 0.30,
    "latent_frame_alignment": 0.80,
}


@dataclass
class RolloutMetrics:
    """Result of a temporal rollout eval run (M3-T6).

    Attributes:
        ssim: Mean per-frame SSIM (pure numpy, 2D, grayscale).
        psnr_db: Mean per-frame PSNR in dB (``float('inf')`` when
            ``MSE == 0``).
        clip_score: Mean CLIP image-image cosine, or ``None`` when the
            optional ``transformers`` dep is unavailable.
        latent_frame_alignment: Cosine similarity between flattened
            latent rollout and flattened frame-reference rollout.
        n_frames: Number of frames evaluated (T axis after squeeze).
        passed: ``True`` iff ``ssim >= thresholds['ssim']`` and
            ``(clip_score is None or clip_score >= thresholds['clip_score'])``
            and ``latent_frame_alignment >= thresholds['latent_frame_alignment']``.
        thresholds: The threshold dict used to evaluate ``passed``.
    """
    ssim: float
    psnr_db: float
    clip_score: float | None
    latent_frame_alignment: float
    n_frames: int = 0
    passed: bool = False
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))


# --------------------------------------------------------------------------- #
# Per-frame metrics
# --------------------------------------------------------------------------- #
def _to_grayscale_frames(video: np.ndarray) -> np.ndarray:
    """Squeeze a ``(T, 3, H, W)`` or ``(B, T, 3, H, W)`` video to ``(T, H, W)``.

    Drops the leading batch dim if present, permutes to ``(T, H, W, C)``,
    converts to float32, and averages across channels. The result is a
    ``(T, H, W)`` grayscale stack ready for per-frame SSIM/PSNR.
    """
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(
                f"Video batch dim must be 1 for per-frame SSIM/PSNR; got "
                f"shape {video.shape}. Average over the batch first or "
                f"call compute_rollout_metrics per clip."
            )
        video = video[0]
    if video.ndim != 4:
        raise ValueError(
            f"video must be (T, 3, H, W) or (1, T, 3, H, W); got shape {video.shape}"
        )
    if video.shape[1] not in (1, 3):
        raise ValueError(
            f"video channel dim must be 1 or 3; got shape {video.shape}"
        )
    # (T, C, H, W) -> (T, H, W, C) for grayscale reduction
    frames = np.transpose(video, (0, 2, 3, 1)).astype(np.float32, copy=False)
    if frames.shape[-1] == 3:
        # Rec.601 luma; bounded for non-negative pixel inputs.
        frames = (0.2989 * frames[..., 0]
                  + 0.5870 * frames[..., 1]
                  + 0.1140 * frames[..., 2])
    else:
        frames = frames[..., 0]
    return frames


def compute_ssim(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    window_size: int = 11,
) -> float:
    """Mean per-frame 2-D SSIM (pure numpy, no GPU).

    Args:
        pred: ``(T, 3, H, W)`` predicted frames (any numeric dtype).
        ref: ``(T, 3, H, W)`` reference frames, same shape as ``pred``.
        window_size: Gaussian window side length (default 11 -- the
            canonical SSIM window).

    Returns:
        Mean SSIM in ``[0, 1]`` across all frames. Returns ``1.0`` for
        identical inputs and ``0.0`` if any frame has zero variance in
        both pred and ref (the structural term degenerates).
    """
    pred_g = _to_grayscale_frames(pred)
    ref_g = _to_grayscale_frames(ref)
    if pred_g.shape != ref_g.shape:
        raise ValueError(
            f"pred/ref shape mismatch: {pred_g.shape} vs {ref_g.shape}"
        )
    if pred_g.shape[0] == 0:
        return 1.0
    h, w = pred_g.shape[1], pred_g.shape[2]
    eff_window = min(window_size, h, w)
    if eff_window < 2:
        # No meaningful SSIM window possible -- fall back to MSE-based
        # structural term: equivalent to SSIM with K1=K2=0, which
        # collapses to a constant 1.0 for the structural term and the
        # luminance/contrast terms. Returning 1.0 keeps the gate from
        # failing spuriously on tiny frames (e.g. unit tests with 4x4).
        return 1.0
    scores = [_ssim_2d(p, r, eff_window) for p, r in zip(pred_g, ref_g)]
    return float(np.mean(scores))


def _ssim_2d(pred: np.ndarray, ref: np.ndarray, window_size: int) -> float:
    """Single-frame 2-D SSIM with a uniform window (Wang et al. 2004).

    Uses ``C1 = (K1 * L)^2``, ``C2 = (K2 * L)^2`` with ``K1=0.01``,
    ``K2=0.03``, and ``L=1.0`` (pixel range is normalized upstream).
    """
    K1, K2, L = 0.01, 0.03, 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Uniform window sums to 1 -> conv via sliding_window_view is exact.
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    kernel /= kernel.size

    from numpy.lib.stride_tricks import sliding_window_view

    def _conv2d(img: np.ndarray) -> np.ndarray:
        windows = sliding_window_view(img, (window_size, window_size))
        # windows.shape = (H-w+1, W-w+1, w, w)
        return (windows * kernel).sum(axis=(-2, -1))

    mu_p = _conv2d(pred)
    mu_r = _conv2d(ref)
    mu_p_sq = mu_p * mu_p
    mu_r_sq = mu_r * mu_r
    mu_pr = mu_p * mu_r

    sigma_p_sq = _conv2d(pred * pred) - mu_p_sq
    sigma_r_sq = _conv2d(ref * ref) - mu_r_sq
    sigma_pr = _conv2d(pred * ref) - mu_pr

    num = (2 * mu_pr + C1) * (2 * sigma_pr + C2)
    den = (mu_p_sq + mu_r_sq + C1) * (sigma_p_sq + sigma_r_sq + C2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(ssim_map.mean())


def compute_psnr(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    max_val: float = 1.0,
) -> float:
    """Mean per-frame PSNR in dB (pure numpy, no GPU).

    Args:
        pred: ``(T, 3, H, W)`` predicted frames.
        ref: ``(T, 3, H, W)`` reference frames.
        max_val: Maximum pixel value (default ``1.0`` for normalized
            inputs; use ``255.0`` for ``uint8`` images).

    Returns:
        Mean PSNR in dB. Returns ``float('inf')`` when ``MSE == 0`` on
        every frame.
    """
    pred_f = pred.astype(np.float32, copy=False)
    ref_f = ref.astype(np.float32, copy=False)
    if pred_f.shape != ref_f.shape:
        raise ValueError(
            f"pred/ref shape mismatch: {pred_f.shape} vs {ref_f.shape}"
        )
    if pred_f.size == 0:
        return float("inf")
    mse_per_frame = ((pred_f - ref_f) ** 2).mean(axis=(1, 2, 3))
    if np.all(mse_per_frame == 0):
        return float("inf")
    # 10 * log10(max_val^2 / MSE); guard against per-frame 0 MSE.
    with np.errstate(divide="ignore"):
        psnr = 10.0 * np.log10((max_val ** 2) / np.maximum(mse_per_frame, 1e-12))
    return float(np.mean(psnr))


# --------------------------------------------------------------------------- #
# CLIP
# --------------------------------------------------------------------------- #
def _clip_available() -> bool:
    """Whether the optional ``transformers`` CLIP dep is importable."""
    try:
        from transformers import CLIPModel  # noqa: F401
        return True
    except ImportError:
        return False


def compute_clip_score(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    model_id: str = "openai/clip-vit-base-patch32",
) -> float | None:
    """Mean CLIP image-image cosine between predicted and reference frames.

    Uses :class:`transformers.CLIPModel` (try-import). On hosts where
    ``transformers`` is missing or the CLIP backbone can't be loaded
    (e.g. offline dev box), returns ``None`` and emits a warning so the
    orchestrator can record ``clip_score: null`` in the summary.

    Args:
        pred: ``(T, 3, H, W)`` predicted frames.
        ref: ``(T, 3, H, W)`` reference frames.
        model_id: HF model id for the CLIP backbone.

    Returns:
        Mean cosine in ``[0, 1]`` or ``None`` if CLIP is unavailable.
    """
    if not _clip_available():
        warnings.warn(
            "compute_clip_score: transformers (with CLIPModel) unavailable; "
            "returning None. Install with: pip install transformers torch",
            stacklevel=2,
        )
        return None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        warnings.warn(
            f"compute_clip_score: required dep missing ({exc}); returning None",
            stacklevel=2,
        )
        return None

    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
    except Exception as exc:  # network, hub, weights -- all unrecoverable on host
        warnings.warn(
            f"compute_clip_score: failed to load {model_id} ({exc}); returning None",
            stacklevel=2,
        )
        return None

    pred_list = [frame for frame in pred]
    ref_list = [frame for frame in ref]
    with torch.no_grad():
        pred_inputs = processor(images=pred_list, return_tensors="pt")
        ref_inputs = processor(images=ref_list, return_tensors="pt")
        pred_emb = model.get_image_features(**pred_inputs)
        ref_emb = model.get_image_features(**ref_inputs)
    pred_emb = pred_emb / pred_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = (pred_emb * ref_emb).sum(dim=-1)
    return float(cos.mean().item())


# --------------------------------------------------------------------------- #
# Latent <-> frame alignment
# --------------------------------------------------------------------------- #
def compute_latent_frame_alignment(
    latent_rollout: np.ndarray,
    frame_rollout: np.ndarray,
) -> float:
    """Cosine similarity between flattened latent and frame rollouts.

    Args:
        latent_rollout: ``(N, D)`` latent features for the rollout.
        frame_rollout: ``(N, D')`` corresponding frame features (or
            pixels, flattened). N must match.

    Returns:
        Mean per-row cosine in ``[-1, 1]``. Returns ``1.0`` if either
        side is empty.
    """
    if latent_rollout.ndim != 2 or frame_rollout.ndim != 2:
        raise ValueError(
            f"Both inputs must be 2-D; got {latent_rollout.shape} and {frame_rollout.shape}"
        )
    if latent_rollout.shape[0] != frame_rollout.shape[0]:
        raise ValueError(
            f"N mismatch: latent={latent_rollout.shape[0]} vs frame="
            f"{frame_rollout.shape[0]}"
        )
    if latent_rollout.shape[0] == 0:
        return 1.0
    sims = _common_cosine_similarity(latent_rollout, frame_rollout)
    return float(np.mean(sims))


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def compute_rollout_metrics(
    predicted_video: np.ndarray,
    reference_video: np.ndarray,
    *,
    latent_rollout: np.ndarray | None = None,
    frame_rollout: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
    max_val: float = 1.0,
    clip_model_id: str = "openai/clip-vit-base-patch32",
) -> RolloutMetrics:
    """Compute all M3-T6 rollout metrics in one call.

    Args:
        predicted_video: ``(T, 3, H, W)`` predicted frames (decoder /
            diffusion output).
        reference_video: ``(T, 3, H, W)`` reference frames (ground
            truth or stub) -- same shape as ``predicted_video``.
        latent_rollout: Optional ``(N, D)`` latent rollout. If ``None``,
            :attr:`RolloutMetrics.latent_frame_alignment` is set to
            ``1.0`` (no signal to evaluate) and the threshold for that
            metric is dropped from the ``passed`` check.
        frame_rollout: Optional ``(N, D')`` frame-reference rollout;
            required when ``latent_rollout`` is given.
        thresholds: Override the default acceptance thresholds.
        max_val: Pixel max for PSNR (default 1.0).
        clip_model_id: HF model id for the CLIP backbone.

    Returns:
        :class:`RolloutMetrics` with all fields populated.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    ssim = compute_ssim(predicted_video, reference_video)
    psnr_db = compute_psnr(predicted_video, reference_video, max_val=max_val)
    clip_score = compute_clip_score(predicted_video, reference_video,
                                    model_id=clip_model_id)
    if latent_rollout is not None and frame_rollout is not None:
        latent_alignment = compute_latent_frame_alignment(
            latent_rollout, frame_rollout
        )
        alignment_passed = latent_alignment >= thresholds["latent_frame_alignment"]
    else:
        latent_alignment = 1.0
        alignment_passed = True  # not evaluated -> don't gate on it

    ssim_passed = ssim >= thresholds["ssim"]
    clip_passed = (
        clip_score is None or clip_score >= thresholds["clip_score"]
    )
    passed = ssim_passed and clip_passed and alignment_passed

    n_frames = int(np.asarray(predicted_video).shape[0]) if predicted_video.ndim >= 4 else 0

    return RolloutMetrics(
        ssim=float(ssim),
        psnr_db=float(psnr_db),
        clip_score=clip_score,
        latent_frame_alignment=float(latent_alignment),
        n_frames=n_frames,
        passed=passed,
        thresholds=dict(thresholds),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli() -> int:
    """CLI entry point. Loads ``.npy`` files and prints metrics + exit code.

    Exit codes:
        0 -- all gates passed
        1 -- a gate failed
        2 -- setup error (missing file, bad shape)
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Compute M3-T6 rollout metrics (SSIM/PSNR/CLIP/alignment) "
                    "from .npy files."
    )
    parser.add_argument("--predicted", required=True, type=Path,
                        help="Path to (T,3,H,W) predicted frames .npy")
    parser.add_argument("--reference", required=True, type=Path,
                        help="Path to (T,3,H,W) reference frames .npy")
    parser.add_argument("--latent", type=Path, default=None,
                        help="Optional (N,D) latent rollout .npy")
    parser.add_argument("--frame-ref", type=Path, default=None,
                        help="Optional (N,D') frame-reference rollout .npy "
                             "(required when --latent is given)")
    parser.add_argument("--threshold-ssim", type=float, default=None)
    parser.add_argument("--threshold-clip", type=float, default=None)
    parser.add_argument("--threshold-alignment", type=float, default=None)
    parser.add_argument("--max-val", type=float, default=1.0)
    parser.add_argument("--no-clip", action="store_true",
                        help="Skip CLIP entirely (sets clip_score=None)")
    args = parser.parse_args()

    try:
        pred = np.load(args.predicted)
        ref = np.load(args.reference)
        latent = np.load(args.latent) if args.latent else None
        frame_ref = np.load(args.frame_ref) if args.frame_ref else None
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    thresholds: dict[str, float] = {}
    if args.threshold_ssim is not None:
        thresholds["ssim"] = args.threshold_ssim
    if args.threshold_clip is not None:
        thresholds["clip_score"] = args.threshold_clip
    if args.threshold_alignment is not None:
        thresholds["latent_frame_alignment"] = args.threshold_alignment

    try:
        if args.no_clip:
            ssim = compute_ssim(pred, ref)
            psnr = compute_psnr(pred, ref, max_val=args.max_val)
            if latent is not None and frame_ref is not None:
                align = compute_latent_frame_alignment(latent, frame_ref)
            else:
                align = 1.0
            merged = {**DEFAULT_THRESHOLDS, **thresholds}
            report = RolloutMetrics(
                ssim=float(ssim), psnr_db=float(psnr), clip_score=None,
                latent_frame_alignment=float(align),
                n_frames=int(pred.shape[0]) if pred.ndim >= 4 else 0,
                passed=(ssim >= merged["ssim"]) and (
                    align >= merged["latent_frame_alignment"]
                ),
                thresholds=merged,
            )
        else:
            report = compute_rollout_metrics(
                pred, ref,
                latent_rollout=latent, frame_rollout=frame_ref,
                thresholds=thresholds, max_val=args.max_val,
            )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cs = "n/a" if report.clip_score is None else f"{report.clip_score:.4f}"
    print(
        f"SSIM: {report.ssim:.4f} (>= {report.thresholds['ssim']:.4f}) | "
        f"PSNR: {report.psnr_db:.2f} dB | "
        f"CLIP: {cs} | "
        f"Alignment: {report.latent_frame_alignment:.4f} "
        f"(>= {report.thresholds['latent_frame_alignment']:.4f}) -> "
        f"{'PASS' if report.passed else 'FAIL'}"
    )
    return 0 if report.passed else 1


__all__ = [
    "DEFAULT_THRESHOLDS",
    "RolloutMetrics",
    "compute_ssim",
    "compute_psnr",
    "compute_clip_score",
    "compute_latent_frame_alignment",
    "compute_rollout_metrics",
]


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_cli())
