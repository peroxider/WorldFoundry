"""Synthetic physics video generator for the CPU render-train-inference demo.

Produces deterministic short ``.mp4`` clips from procedural physics so the
rest of the pipeline has real video files to consume — no network, no
large dataset download. The motion is simple enough to learn in <100
optimizer steps but rich enough that the JEPA cannot solve it from a
single frame (i.e. the model must build a temporal representation).

Two generators ship:

- :func:`write_bouncing_ball_video` — a 2D ball that bounces inside the
  frame under gravity with elastic walls and a faint radial trail.
- :func:`write_rotating_stripes_video` — concentric sine-modulated
  stripes that rotate at a fixed angular velocity, giving the encoder a
  strong shape / motion signal.

Both use OpenCV's :class:`cv2.VideoWriter` (mp4v fourcc) and are
byte-deterministic given the same seed — the orchestrator + tests rely
on this for reproducibility.

Why not imageio/PyAV? OpenCV ships in the base image on this box and is
fast enough for 8-frame clips. Imageio is not installed; PyAV is heavier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def _open_writer(path: Path, fps: float, height: int, width: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    return writer


def write_bouncing_ball_video(
    path: Path,
    *,
    num_frames: int = 8,
    height: int = 64,
    width: int = 64,
    fps: float = 10.0,
    seed: int = 0,
    ball_radius: int = 4,
    gravity: float = 0.5,
    elastic: float = 0.92,
) -> Path:
    """Write a deterministic bouncing-ball clip to ``path`` and return the path.

    The ball starts at a seed-dependent position with seed-dependent
    velocity. Walls reflect with elasticity < 1 so motion eventually
    decays — over ``num_frames`` frames this is invisible, but it
    prevents pathological infinite-bounce drifts across clips.

    Args:
        path: output ``.mp4`` path (parent dirs created).
        num_frames: number of frames per clip.
        height/width: spatial size in pixels (square recommended).
        fps: video frame rate.
        seed: RNG seed; same seed -> byte-identical clip.
        ball_radius: ball radius in pixels.
        gravity: downward acceleration in pixels/frame**2.
        elastic: coefficient of restitution on wall collisions (0..1).
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2; got {num_frames}")
    if height < 8 or width < 8:
        raise ValueError(f"height/width must be >= 8; got {height}x{width}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(ball_radius + 4, width - ball_radius - 4))
    y = float(rng.uniform(ball_radius + 4, height * 0.5))
    vx = float(rng.uniform(-2.5, 2.5))
    vy = float(rng.uniform(0.0, 1.5))
    hue_start = float(rng.uniform(0, 180))

    writer = _open_writer(path, fps, height, width)
    try:
        for t in range(num_frames):
            frame = np.full((height, width, 3), 24, dtype=np.uint8)
            vy += gravity
            x += vx
            y += vy
            if x - ball_radius < 0:
                x = float(ball_radius)
                vx = -vx * elastic
            elif x + ball_radius >= width:
                x = float(width - ball_radius - 1)
                vx = -vx * elastic
            if y - ball_radius < 0:
                y = float(ball_radius)
                vy = -vy * elastic
            elif y + ball_radius >= height:
                y = float(height - ball_radius - 1)
                vy = -vy * elastic
            cx, cy = int(round(x)), int(round(y))
            hue = int((hue_start + t * 6) % 180)
            hsv = np.full((1, 1, 3), (hue, 220, 255), dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(3)
            color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
            cv2.circle(frame, (cx, cy), ball_radius, color, thickness=-1, lineType=cv2.LINE_AA)
            if t > 0:
                trail = np.full_like(frame, 32, dtype=np.uint8)
                mask = (frame == 24).all(axis=-1)
                frame[mask] = cv2.addWeighted(frame, 0.85, trail, 0.15, 0)[mask]
            writer.write(frame)
    finally:
        writer.release()
        if not path.is_file():
            raise RuntimeError(f"video not written: {path}")
    return path


def write_rotating_stripes_video(
    path: Path,
    *,
    num_frames: int = 8,
    height: int = 64,
    width: int = 64,
    fps: float = 10.0,
    seed: int = 0,
    angular_velocity: float = 0.35,
) -> Path:
    """Write a deterministic concentric-rotating-stripes clip.

    Each frame is a sin-modulated radial pattern; the phase rotates at
    ``angular_velocity`` rad/frame. Hue is also seed-dependent so
    different clips have different colours. The encoder can predict the
    next frame from the rotation rate alone.

    Args:
        path: output ``.mp4`` path (parent dirs created).
        num_frames/height/width/fps/seed: see :func:`write_bouncing_ball_video`.
        angular_velocity: radians per frame.
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2; got {num_frames}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    phase_offset = float(rng.uniform(0, 2 * np.pi))
    hue = int(rng.uniform(0, 180))

    ys = np.arange(height, dtype=np.float32) - height / 2.0
    xs = np.arange(width, dtype=np.float32) - width / 2.0
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)

    writer = _open_writer(path, fps, height, width)
    try:
        for t in range(num_frames):
            phase = phase_offset + angular_velocity * t
            intensity = 0.5 + 0.5 * np.sin(0.4 * r - phase)
            v = (intensity * 255).astype(np.uint8)
            hsv = np.full((height, width, 3), (hue, 200, 255), dtype=np.uint8)
            hsv[..., 2] = v
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            writer.write(frame)
    finally:
        writer.release()
        if not path.is_file():
            raise RuntimeError(f"video not written: {path}")
    return path


def write_synthetic_clip(
    out_dir: Path,
    clip_idx: int,
    *,
    num_frames: int,
    height: int,
    width: int,
    seed: int,
) -> Path:
    """Write clip ``clip_idx`` using the routing function above.

    Even indices use bouncing ball; odd indices use rotating stripes.
    The per-clip seed is ``seed * 1000 + clip_idx`` so different seeds
    produce different motion trajectories without collisions.
    """
    writer: Callable[..., Path]
    if clip_idx % 2 == 0:
        writer = write_bouncing_ball_video
    else:
        writer = write_rotating_stripes_video
    path = out_dir / f"clip_{clip_idx:04d}.mp4"
    return writer(
        path,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed * 1000 + clip_idx,
    )


__all__ = [
    "write_bouncing_ball_video",
    "write_rotating_stripes_video",
    "write_synthetic_clip",
]