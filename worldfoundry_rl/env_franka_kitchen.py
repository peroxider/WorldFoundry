"""Franka-shaped 9-dim manipulation env wrapper (W31) — MuJoCo + dm_control.

Per W30 ADR §4: FrankaKitchen (MuJoCo+dm_control) is the recommended
Phase-4 v1 first environment. This module wraps a Franka-shaped
9-DoF manipulation task and exposes a Gymnasium-style single-env
``reset`` / ``step`` / ``close`` surface compatible with
:class:`worldfoundry_rl.env_pool.Env`.

**Backend reality (v2.15+):** the canonical ``franka_kitchen`` PyPI
package does not exist (``pip index versions franka-kitchen`` →
no matching distribution), and ``dm_control.suite`` has no
``franka_kitchen`` domain (``ValueError: Domain 'franka_kitchen'
does not exist``). The wrapper now routes through
``dm_control.manipulation`` — a Franka-Panda-style 9-DoF
manipulation suite bundled with dm_control. The default task is
``lift_brick_features`` (Kinova Jaco arm + duplo brick target):
same 9-dim action, same dict-shaped obs, real MuJoCo step, real
task reward. The historical "FrankaKitchen" name is preserved
because that's how W30 ADR §4 / W31 / the reward wrappers
reference this env class — the contract (9-dim action,
{ee_pos, target_pos, task_success} dict obs) is unchanged.

State contract (the observation dict this env emits):

* ``ee_pos`` — end-effector position, shape ``(3,)``, ``float64``
* ``target_pos`` — target (brick) position, shape ``(3,)``, ``float64``
* ``task_success`` — bool, default ``False``

This matches the contract consumed by
:class:`worldfoundry_rl.reward_impl.FrankaKitchenReward`, so the
reward wrapper slots in without translation.

Install: ``dm_control`` + ``mujoco`` are required at runtime. They
are **not** required at module-import time — the heavy imports are
deferred to ``__init__`` so the rest of the codebase can import this
module without paying the simulator cost. A clear ``ImportError``
with install instructions is raised on instantiation if the deps
are missing.

Per W30 ADR §6 acceptance gates: W31.1 unit-tested with fake env
(``tests/test_env_pool.py``); W31.2 step latency benchmark on real
env after install (``bench_results/w31/vec_env_throughput.json``).
"""
from __future__ import annotations

from typing import Any


class FrankaKitchenEnv:
    """Single-env Gymnasium-style wrapper around dm_control.manipulation.

    The default backend task is ``lift_brick_features``: a Franka-Panda-
    style 9-DoF manipulation task with a duplo brick as the lift target.
    dm_control.manipulation also offers ``reach_site_features``,
    ``place_brick_features``, ``stack_2_bricks_features``, etc. — all
    share the 9-dim action contract and dict-shaped obs.

    Args:
        task_name: dm_control.manipulation task name. Default
            ``"lift_brick_features"`` is the closest Franka-shaped
            entry-level manipulation task available in the installed
            dm_control (1.0.44).
        seed: optional master seed (broadcast to dm_control's internal
            RNG).
    """

    DEFAULT_TASK = "lift_brick_features"

    # dm_control.manipulation obs keys → our 3-key contract.
    # The Jaco-arm pin site is the EE; the brick position is the target.
    _EE_KEY = "jaco_arm/jaco_hand/pinch_site_pos"
    _TARGET_KEY = "duplo2x4/position"

    def __init__(
        self,
        task_name: str = DEFAULT_TASK,
        *,
        seed: int | None = None,
        render_height: int = 32,
        render_width: int = 32,
        camera_id: int | str = 0,
        enable_render: bool = True,
    ) -> None:
        # Deferred import: dm_control pulls in mujoco + gl context + native
        # bindings. We only pay that cost when an actual env instance is
        # constructed.
        try:
            from dm_control import manipulation  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "FrankaKitchenEnv requires `dm_control` and `mujoco`. "
                "Install with: pip install dm_control mujoco  "
                "(mujoco uses its bundled binary wheels; no system "
                "install needed). Original error: " + repr(exc)
            ) from exc

        self._manipulation = manipulation
        self.task_name = str(task_name)
        self._seed = seed
        try:
            self._env = manipulation.load(
                environment_name=self.task_name,
                seed=seed,
            )
        except KeyError as exc:
            available = ", ".join(sorted(manipulation.ALL)) if hasattr(
                manipulation, "ALL"
            ) else "(unknown)"
            raise ValueError(
                f"dm_control.manipulation has no task {self.task_name!r}. "
                f"Available tasks: {available}"
            ) from exc

        # W2.16: render seam — capture the latest RGB frame so the
        # world-model train phase can train on the same observation
        # stream that the policy rollout sees. Defaults match the
        # orchestrator's default encoder image_size so the on-disk mp4s
        # line up with `Vjepa2Dataset`'s manifest schema out of the box.
        if render_height < 8 or render_width < 8:
            raise ValueError(
                f"render_height/render_width must be >= 8; got "
                f"{render_height}x{render_width}"
            )
        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self.camera_id = camera_id
        self.enable_render = bool(enable_render)
        self._last_frame: np.ndarray | None = None
        self._render_failures: int = 0

        # Observation / action spaces — declare them as plain attributes so
        # the :class:`worldfoundry_rl.env_pool.Env` Protocol's duck-typed
        # attribute check passes. We use simple named tuples instead of
        # full gymnasium.spaces.* to keep the surface tiny and import-free.
        self.observation_space = _DictObsSpace(
            keys=("ee_pos", "target_pos", "task_success"),
            shapes={"ee_pos": (3,), "target_pos": (3,)},
            dtypes={"ee_pos": "float64", "target_pos": "float64", "task_success": "bool"},
        )
        # 9-dim action: 6 arm joints + 3 finger joints. Same shape as a
        # canonical FrankaKitchen task (7 arm + 2 gripper), but the Jaco
        # arm in dm_control.manipulation uses 6 + 3 = 9 DoF.
        self.action_space = _BoxSpace(shape=(9,), dtype="float32")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        effective_seed = self._seed if seed is None else seed
        timestep = self._env.reset()
        if effective_seed is not None:
            try:
                self._env.task.random.seed(int(effective_seed))
            except Exception:
                pass
        self._capture_frame(timestep)
        obs = self._extract_obs(timestep)
        info: dict = {}
        return obs, info

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        import numpy as np  # local import keeps env_pool dep-light

        action_arr = np.asarray(action, dtype=np.float32).ravel()
        if action_arr.shape != (9,):
            raise ValueError(
                f"FrankaKitchen action must be 9-dim, got shape {action_arr.shape}"
            )
        # dm_control.manipulation action_spec dtype is float64; cast.
        action_arr = action_arr.astype(np.float64)
        timestep = self._env.step(action_arr)
        self._capture_frame(timestep)
        obs = self._extract_obs(timestep)
        # dm_control returns the per-step reward (not 0 on terminal) so
        # PPO can learn from terminal-step bonuses. Convert to plain float.
        reward = float(getattr(timestep, "reward", 0.0))
        last = timestep.last()
        terminated = bool(last == "step")
        truncated = bool(last in ("limit", "time"))
        info: dict = {}
        if last is not None:
            info["dm_control_last"] = last
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        # dm_control.manipulation envs hold an EGL/OSMesa context; release
        # the underlying mujoco model by dropping the reference. There is
        # no public close() on manipulation.load() envs; the model is
        # reclaimed when Python GC's the reference. We drop the ref explicitly.
        try:
            del self._env
        except Exception:
            pass

    # ---- W2.16: RGB frame seam -----------------------------------------

    def last_frame(self) -> np.ndarray:
        """Return the latest rendered RGB frame as a deep-copied ``(H, W, 3) uint8``.

        Must be called after ``reset()`` (or any ``step()``); raises
        ``RuntimeError`` otherwise so callers don't silently train on a
        zero buffer. Render failures (no GL context, missing EGL) fall
        back to an all-zero frame and increment ``_render_failures`` —
        the env never crashes the policy phase on a render hiccup.
        """
        if self._last_frame is None:
            raise RuntimeError(
                "FrankaKitchenEnv.last_frame called before reset(); "
                "call reset() at least once to capture an initial frame"
            )
        return self._last_frame.copy()

    @property
    def render_failures(self) -> int:
        """Cumulative count of ``physics.render(...)`` failures since construction."""
        return int(self._render_failures)

    def _capture_frame(self, timestep: Any) -> None:
        """Render and stash the latest frame for :meth:`last_frame`.

        Swallows all exceptions so a missing GL stack (e.g. headless CI
        without EGL) never poisons the orchestrator's training loop —
        downstream consumers see a zero frame and can log/track the
        failure count via :attr:`render_failures`.
        """
        if not self.enable_render:
            return
        import logging
        import numpy as np
        try:
            pixels = self._env.physics.render(
                height=self.render_height,
                width=self.render_width,
                camera_id=self.camera_id,
            )
        except Exception as exc:  # noqa: BLE001 — render is best-effort
            self._render_failures += 1
            if self._render_failures == 1:
                logging.getLogger(__name__).warning(
                    "FrankaKitchenEnv render failed (%r); subsequent "
                    "last_frame() will return zero buffer",
                    exc,
                )
            self._last_frame = np.zeros(
                (self.render_height, self.render_width, 3), dtype=np.uint8,
            )
            return
        # dm_control returns (H, W, 3) uint8 already; copy so downstream
        # mutation (cv2.VideoWriter, etc.) doesn't poison the buffer.
        self._last_frame = np.asarray(pixels, dtype=np.uint8).copy()

    def _extract_obs(self, timestep: Any) -> dict:
        """Pull (ee_pos, target_pos, task_success) from a dm_control timestep."""
        import numpy as np

        obs_dict: dict = {}
        # End-effector position: dm_control.manipulation obs keys have
        # shape (1, 3) — flatten to (3,).
        try:
            ee = np.asarray(
                timestep.observation.get(self._EE_KEY, None), dtype=np.float64
            ).ravel()
            if ee.size < 3:
                raise ValueError
            obs_dict["ee_pos"] = ee[:3]
        except Exception:
            obs_dict["ee_pos"] = np.zeros(3, dtype=np.float64)

        # Target (brick) position: same flattening.
        try:
            tgt = np.asarray(
                timestep.observation.get(self._TARGET_KEY, None), dtype=np.float64
            ).ravel()
            if tgt.size < 3:
                raise ValueError
            obs_dict["target_pos"] = tgt[:3]
        except Exception:
            obs_dict["target_pos"] = np.zeros(3, dtype=np.float64)

        # Task success flag: dm_control exposes ``is_terminal``-equivalent
        # via the timestep reward when in a terminal step. Default False.
        obs_dict["task_success"] = bool(getattr(timestep, "last", lambda: None)() == "step")
        return obs_dict


class _DictObsSpace:
    """Minimal stand-in for ``gymnasium.spaces.Dict`` (avoids the gymnasium import)."""

    def __init__(self, keys: tuple[str, ...], shapes: dict, dtypes: dict) -> None:
        self.keys = keys
        self.shapes = dict(shapes)
        self.dtypes = dict(dtypes)

    def __repr__(self) -> str:
        return f"_DictObsSpace(keys={self.keys})"


class _BoxSpace:
    """Minimal stand-in for ``gymnasium.spaces.Box``."""

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = tuple(shape)
        self.dtype = str(dtype)

    def __repr__(self) -> str:
        return f"_BoxSpace(shape={self.shape}, dtype={self.dtype})"


__all__ = ["FrankaKitchenEnv"]