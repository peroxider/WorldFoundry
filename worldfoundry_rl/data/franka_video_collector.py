"""Collect RGB-action transition clips from a FrankaKitchen-like env (W2.16).

This module is the FrankaKitchen counterpart of
:mod:`worldfoundry_train.data.synthetic_video` + :mod:`worldfoundry_train.data.synthetic_manifest`.
Rather than synthesize physics, it drives a real MuJoCo env (e.g. via
:class:`worldfoundry_rl.env_franka_kitchen.FrankaKitchenEnv`) for
``num_clips * frames_per_clip`` steps, captures the rendered RGB frames
into cv2 ``.mp4`` files + a CSV manifest compatible with
:class:`worldfoundry_train.data.dataset.Vjepa2Dataset`, and returns the
in-memory ``(frames, actions)`` arrays so the caller can build an
aligned :class:`worldfoundry_rl.data.franka_dataset.FrankaTransitionDataset`
without re-decoding the videos.

Why an in-memory array + on-disk manifest: ``Vjepa2Dataset`` consumes
the .mp4 + manifest to stay schema-aligned with W22 W22.7 manifest
contract, while the orchestrator wants the **same** frames to train on
without re-encoding. Returning both keeps the data path linear: write
once, read once via manifest, plus a direct in-memory view for the
train phase.

Public surface
--------------

- :func:`collect_franka_video_clips` — drives the env, writes the
  manifest, returns the aligned arrays.

Install/runtime: requires ``dm_control`` + ``mujoco`` + ``opencv-python``
+ ``numpy``. The first two are only needed when this module is actually
called (deferred import inside the helper).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np


_LOGGER = logging.getLogger(__name__)

# Action sampling constants. ``ACTION_NOISE_STD`` keeps actions
# non-zero so the world-model isn't trained on the v2.13 "oracle"
# degenerate case (zero-action -> identical latent -> no predictor
# gradient). 0.1 was chosen empirically — large enough to escape the
# degenerate fixed point, small enough that dm_control's per-step
# reward stays in the same magnitude as zero-action.
_ACTION_NOISE_STD: float = 0.1
_DEFAULT_FPS: float = 10.0


def _open_writer(path: Path, fps: float, height: int, width: int) -> Any:
    """Open a cv2 VideoWriter with the mp4v fourcc.

    Mirrors :func:`worldfoundry_train.data.synthetic_video._open_writer`
    so the manifest is indistinguishable in codec / fps from the
    synthetic path.
    """
    import cv2  # local import keeps this module import-light when unused

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    return writer


def _sample_action(rng: np.random.Generator, action_dim: int) -> np.ndarray:
    """Return one action vector with a small zero-mean jitter.

    We start from a zero action (Franka at rest) and add ~N(0, 0.1²)
    noise so the predictor is forced to learn that actions are not
    degenerate.
    """
    return rng.normal(0.0, _ACTION_NOISE_STD, size=(action_dim,)).astype(np.float32)


def collect_franka_video_clips(
    env_factory: Callable[[], Any],
    *,
    num_clips: int,
    frames_per_clip: int,
    height: int,
    width: int,
    seed: int = 42,
    fps: float = _DEFAULT_FPS,
    out_dir: Path | None = None,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Drive ``env_factory`` for ``num_clips * frames_per_clip`` steps.

    Args:
        env_factory: zero-arg callable returning a Franka-shaped env
            exposing ``reset()`` and ``step(action)`` and
            ``last_frame() -> (H, W, 3) uint8``. The same env instance
            is reused across all clips (no factory re-call per clip);
            this matches dm_control's per-task single-instance model
            and avoids leaking GL contexts.
        num_clips: how many clips to write.
        frames_per_clip: frames per clip (must match the encoder's
            expected time dim).
        height/width: render resolution; must be >= 8 and divisible by
            the encoder's patch_size (32x32 default matches
            ``TinyEncoderConfig(image_size=32, patch_size=8)``).
        seed: master RNG seed for action sampling; same seed produces
            byte-identical .mp4s given identical env dynamics.
        fps: frames-per-second written into the mp4v container.
        out_dir: where to write the .mp4s and manifest. Defaults to
            ``<cwd>/franka_video_collector``; the orchestrator overrides
            this with ``cfg.out_dir / "videos"`` so artifacts land
            under the per-run bench directory.

    Returns:
        ``(manifest_path, frames, actions)`` where
        ``frames`` has shape ``(num_clips, frames_per_clip, H, W, 3) uint8``
        and ``actions`` has shape
        ``(num_clips, frames_per_clip, action_dim) float32``.

        The action stored at ``actions[c, t]`` is the action executed
        at env step ``t`` of clip ``c`` (i.e. the action that produced
        ``frames[c, t]``).
    """
    if num_clips < 1:
        raise ValueError(f"num_clips must be >= 1; got {num_clips}")
    if frames_per_clip < 2:
        raise ValueError(f"frames_per_clip must be >= 2; got {frames_per_clip}")
    if height < 8 or width < 8:
        raise ValueError(f"height/width must be >= 8; got {height}x{width}")

    if out_dir is None:
        out_dir = Path.cwd() / "franka_video_collector"
    out_dir = Path(out_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    env = env_factory()
    try:
        try:
            obs, _info = env.reset(seed=int(seed))
        except TypeError:
            obs, _info = env.reset()
    except Exception:
        _LOGGER.warning(
            "env.reset() failed during collect_franka_video_clips; "
            "aborting before manifest write", exc_info=True,
        )
        raise
    action_dim = int(getattr(env, "action_space").shape[0])

    rng = np.random.default_rng(int(seed))

    frames = np.zeros((num_clips, frames_per_clip, height, width, 3), dtype=np.uint8)
    actions = np.zeros((num_clips, frames_per_clip, action_dim), dtype=np.float32)

    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_path", "num_frames", "height", "width"])
        for c in range(num_clips):
            video_path = videos_dir / f"clip_{c:04d}.mp4"
            video_writer = _open_writer(video_path, fps=fps, height=height, width=width)
            try:
                try:
                    env.reset(seed=int(seed) + 1000 * c)
                except TypeError:
                    env.reset()
                for t in range(frames_per_clip):
                    action = _sample_action(rng, action_dim)
                    _obs, _reward, _term, _trunc, _info = env.step(action)
                    frame = env.last_frame()
                    if frame.shape != (height, width, 3):
                        import cv2  # noqa: PLC0415
                        frame = cv2.resize(frame, (width, height))
                    frames[c, t] = frame
                    actions[c, t] = action
                    video_writer.write(frame)
            finally:
                video_writer.release()
            writer.writerow(
                [str(video_path.resolve()), frames_per_clip, height, width],
            )
    _LOGGER.info(
        "collect_franka_video_clips: wrote %d clips (T=%d, %dx%d) -> %s",
        num_clips, frames_per_clip, height, width, manifest_path,
    )
    return manifest_path, frames, actions


__all__ = ["collect_franka_video_clips"]