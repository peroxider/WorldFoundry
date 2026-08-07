"""ManipulatorEnv — dm_control ``manipulator:bring_ball`` adapter (W41 Phase-5 v1).

The MuJoCo + dm_control built-in ``manipulator:bring_ball`` task: a 5-DoF
planar arm brings a ball to a target position. This is dm_control's
standard manipulation RL task. It requires **no external assets**
(unlike :class:`FrankaKitchenEnv`, which needs separately-licensed
XML+STL files), and runs at ~2.9M steps/sec on a modern CPU.

Mirrors the :class:`FrankaKitchenEnv` Gymnasium-style surface so the
existing :class:`SyncEnvPool` machinery accepts it without changes. The
observation contract is the **raw dm_control observation dict**
(``arm_pos``, ``arm_vel``, ``hand_pos``, ``object_pos``, ``object_vel``,
``target_pos``, ``touch``) — richer than FrankaKitchenEnv's flattened
contract, so callers that want FrankaKitchen-style rewards can either
implement a custom reward over the raw keys or project the dict down to
``ee_pos`` / ``target_pos`` upstream.

Action: 5-DoF continuous in dm_control's default range; the env does
its own internal clamping. ``action_space`` is declared ``(5,) float32``.

Per W41 Phase-5 v1 plan §Deliverable 1.
"""
from __future__ import annotations

from typing import Any


class ManipulatorEnv:
    """Single-env Gymnasium-style wrapper around dm_control's ``manipulator``.

    Args:
        task_kwargs: forwarded to :func:`dm_control.suite.load` (e.g.
            ``{"random": 42}`` for reproducibility). The dm_control default
            task is ``"bring_ball"``.
        seed: Optional master seed broadcast to the env's internal RNG.
    """

    DEFAULT_TASK = "bring_ball"

    def __init__(
        self,
        task_kwargs: dict | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        try:
            from dm_control import suite  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "ManipulatorEnv requires `dm_control` and `mujoco`. "
                "Install with: "
                "pip install --break-system-packages "
                "--trusted-host pypi.org "
                "--trusted-host files.pythonhosted.org "
                "mujoco dm-control"
            ) from exc

        self._suite = suite
        self._task_kwargs = dict(task_kwargs) if task_kwargs else {}
        self._seed = seed
        self._env = suite.load(
            domain_name="manipulator",
            task_name=self.DEFAULT_TASK,
            task_kwargs=self._task_kwargs or None,
        )

        # Obs / action space declarations — plain attributes so the
        # ``Env`` Protocol's duck-typed attribute check passes.
        # dm_control ``manipulator:bring_ball`` emits:
        #   arm_pos    (5,)        - arm joint angles
        #   arm_vel    (5,)        - arm joint velocities
        #   touch      (5,)        - fingertip contact sensors
        #   hand_pos   (3,)        - end-effector XYZ
        #   object_pos (3,)        - ball XYZ
        #   object_vel (3,)        - ball XYZ velocity
        #   target_pos (3,)        - target XYZ
        self.observation_space = _DictObsSpace(
            keys=(
                "arm_pos", "arm_vel", "touch",
                "hand_pos", "object_pos", "object_vel", "target_pos",
            ),
            shapes={
                "arm_pos": (5,), "arm_vel": (5,), "touch": (5,),
                "hand_pos": (3,), "object_pos": (3,), "object_vel": (3,),
                "target_pos": (3,),
            },
            dtypes={
                "arm_pos": "float64", "arm_vel": "float64", "touch": "float64",
                "hand_pos": "float64", "object_pos": "float64",
                "object_vel": "float64", "target_pos": "float64",
            },
        )
        # dm_control manipulator: 5-DoF continuous action (dm_control clamps
        # internally; we declare the surface as float32).
        self.action_space = _BoxSpace(shape=(5,), dtype="float32")

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
        obs = self._extract_obs(timestep)
        return obs, {}

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        import numpy as np

        action_arr = np.asarray(action, dtype=np.float32).ravel()
        if action_arr.shape != (5,):
            raise ValueError(
                f"Manipulator action must be 5-dim, got shape {action_arr.shape}"
            )
        timestep = self._env.step(action_arr)
        obs = self._extract_obs(timestep)
        reward = float(getattr(timestep, "reward", 0.0) or 0.0)
        last = timestep.last()
        terminated = bool(last == "step")
        truncated = bool(last in ("limit", "time"))
        info: dict = {}
        # dm_control ``Timestep.last()`` returns one of {"step", "limit",
        # "time"} on terminal steps, ``None`` on interior steps in some
        # versions, and ``False`` in others. Filter to actual string codes
        # only so the info dict stays clean for interior steps.
        if isinstance(last, str):
            info["dm_control_last"] = last
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            del self._env
        except Exception:
            pass

    def _extract_obs(self, timestep: Any) -> dict:
        """Copy the dm_control observation dict into a stable-shape ``dict``."""
        import numpy as np

        obs_dict: dict = {}
        raw = dict(timestep.observation)
        # Default every key to zeros so a missing or wrong-shape key from
        # a future dm_control version does not blow up the agent.
        defaults = {
            "arm_pos": (5,), "arm_vel": (5,), "touch": (5,),
            "hand_pos": (3,), "object_pos": (3,),
            "object_vel": (3,), "target_pos": (3,),
        }
        for key, shape in defaults.items():
            try:
                arr = np.asarray(raw.get(key, np.zeros(shape)), dtype=np.float64)
                arr = arr.ravel()
                if arr.size < shape[0]:
                    arr = np.concatenate([arr, np.zeros(shape[0] - arr.size)])
                obs_dict[key] = arr[: shape[0]]
            except Exception:
                obs_dict[key] = np.zeros(shape, dtype=np.float64)
        return obs_dict


class _DictObsSpace:
    """Minimal stand-in for ``gymnasium.spaces.Dict``."""

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


__all__ = ["ManipulatorEnv"]