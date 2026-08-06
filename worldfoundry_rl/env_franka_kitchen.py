"""FrankaKitchen env wrapper (W31) — MuJoCo + dm_control, CPU-only first-env.

Per W30 ADR §4: FrankaKitchen (MuJoCo+dm_control) is the recommended
Phase-4 v1 first environment. This module wraps ``dm_control.suite``
FrankaKitchen tasks and exposes a Gymnasium-style single-env
``reset`` / ``step`` / ``close`` surface compatible with
:class:`worldfoundry_rl.env_pool.Env`.

State contract (the observation dict this env emits):

* ``ee_pos`` — end-effector position, shape ``(3,)``, ``float64``
* ``target_pos`` — target position, shape ``(3,)``, ``float64``
* ``task_success`` — bool, default ``False``

This matches the contract consumed by :class:`worldfoundry_rl.reward_impl.FrankaKitchenReward`,
so the reward wrapper slots in without translation.

Install: ``dm_control`` + ``mujoco`` are required at runtime. They are
**not** required at module-import time — the heavy imports are deferred
to ``__init__`` so the rest of the codebase can import this module
without paying the simulator cost. A clear ``ImportError`` with install
instructions is raised on instantiation if the deps are missing.

Per W30 ADR §6 acceptance gates: W31.1 unit-tested with fake env
(``tests/test_env_pool.py``); W31.2 step latency benchmark on real env
after install (``bench_results/w31/vec_env_throughput.json``).
"""
from __future__ import annotations

from typing import Any


class FrankaKitchenEnv:
    """Single-env Gymnasium-style wrapper around dm_control's FrankaKitchen.

    Args:
        task_name: dm_control task name. Default ``"microwave_kettle"``
            matches W30 ADR §4 (entry-level FrankaKitchen manipulation).
        seed: optional master seed (broadcast to dm_control's internal
            RNG).
    """

    DEFAULT_TASK = "microwave_kettle"

    def __init__(
        self,
        task_name: str = DEFAULT_TASK,
        *,
        seed: int | None = None,
    ) -> None:
        # Deferred import: dm_control pulls in mujoco + gl context + native
        # bindings. We only pay that cost when an actual FrankaKitchen
        # instance is constructed.
        try:
            from dm_control import suite  # type: ignore[import-not-found]
            from dm_control.suite.wrappers import pixels  # noqa: F401  (importable smoke)
        except ImportError as exc:
            raise ImportError(
                "FrankaKitchenEnv requires `dm_control` and `mujoco`. "
                "Install with: pip install dm_control mujoco  "
                "(mujoco uses its bundled binary wheels; no system "
                "install needed). Original error: " + repr(exc)
            ) from exc

        self._suite = suite
        self.task_name = str(task_name)
        self._seed = seed
        self._env = suite.load(
            domain_name="franka_kitchen",
            task_name=self.task_name,
            task_kwargs={"random": seed} if seed is not None else None,
        )

        # Observation / action spaces — declare them as plain attributes so
        # the :class:`worldfoundry_rl.env_pool.Env` Protocol's duck-typed
        # attribute check passes. We use simple named tuples instead of
        # full gymnasium.spaces.* to keep the surface tiny and import-free.
        self.observation_space = _DictObsSpace(
            keys=("ee_pos", "target_pos", "task_success"),
            shapes={"ee_pos": (3,), "target_pos": (3,)},
            dtypes={"ee_pos": "float64", "target_pos": "float64", "task_success": "bool"},
        )
        # dm_control FrankaKitchen has 9-dim action: 7 arm joints + 2
        # gripper (the env splits 9 DoF into arm + gripper internally).
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
            # dm_control doesn't re-seed on reset() — apply via internal
            # random_state for repeatability.
            try:
                self._env.task.random.seed(int(effective_seed))
            except Exception:
                pass
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
        timestep = self._env.step(action_arr)
        obs = self._extract_obs(timestep)
        reward = float(getattr(timestep, "reward", 0.0)) if timestep.last() is False else 0.0
        # dm_control Timestep: ``last()`` returns "step"/"limit"/"time"/None
        # depending on which termination triggered. Map "step"/"limit"/"time"
        # to terminated/truncated; default both False.
        last = timestep.last()
        terminated = bool(last == "step")  # task success / failure cutoff
        truncated = bool(last in ("limit", "time"))
        info: dict = {}
        if timestep.last() is not None:
            info["dm_control_last"] = last
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        # dm_control suite envs hold an EGL/OSMesa context; release the
        # underlying mujoco model via suite's internal cleanup. There is
        # no public close() on suite.load() envs; the model is reclaimed
        # when Python GC's the reference. We drop the ref explicitly.
        try:
            del self._env
        except Exception:
            pass

    def _extract_obs(self, timestep: Any) -> dict:
        """Pull (ee_pos, target_pos, task_success) from a dm_control timestep."""
        import numpy as np

        obs_dict: dict = {}
        # End-effector position: dm_control exposes a sites API; the
        # FrankaKitchen tasks expose "end_effector" under physics.named.data.
        try:
            ee = np.asarray(
                timestep.observation.get("end_effector_pos", None), dtype=np.float64
            ).ravel()
            if ee.size < 3:
                raise ValueError
            obs_dict["ee_pos"] = ee[:3]
        except Exception:
            obs_dict["ee_pos"] = np.zeros(3, dtype=np.float64)

        # Target position: dm_control FrankaKitchen tasks publish the
        # per-task target under ``target_pos`` (microwave_kettle: kettle
        # spout position). Fall back to a zero vector if the task layout
        # omits it (older dm_control versions).
        try:
            tgt = np.asarray(
                timestep.observation.get("target_pos", None), dtype=np.float64
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