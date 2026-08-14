"""MockEnv — CPU-only synthetic env for the RL loop (no simulator dep).

A drop-in for the ``Env`` structural Protocol in :mod:`worldfoundry_rl.env_pool`
when no external simulator (MuJoCo / SAPIEN / DROID) is available. It is a
flat-observation, continuous-action "goal is the origin" task whose reward
penalises both state magnitude and action magnitude.

This is the canonical ``sim_env`` backend ``mock`` artifact: registered in
``worldfoundry_common/backends/sim_env.py`` and addressable by the picklable
spec ``"worldfoundry_rl.env_mock:MockEnv"``, so it works with both
:class:`~worldfoundry_rl.env_pool.SyncEnvPool` and
:class:`~worldfoundry_rl.env_pool.AsyncEnvPool`.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class _BoxSpace:
    """Minimal continuous box space (shape + high/low), enough for the pool."""

    def __init__(self, shape: tuple[int, ...], low: float = -1.0, high: float = 1.0):
        self.shape = tuple(shape)
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class MockEnv:
    """Simple continuous-control task: drive the state toward the origin.

    Args:
        obs_dim: state dimension (default 8).
        action_dim: action dimension (default 4).
        seed: master seed.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 4,
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(int(seed))
        self._state: npt.NDArray[np.float32] | None = None
        self.observation_space = _BoxSpace((self.obs_dim,))
        self.action_space = _BoxSpace((self.action_dim,))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._state = (self._rng.standard_normal(self.obs_dim) * 0.1).astype(
            np.float32
        )
        return self._state.copy(), {}

    def step(
        self,
        action: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float32).ravel()[: self.action_dim]
        pad = np.zeros(max(0, self.obs_dim - self.action_dim), dtype=np.float32)
        full_a = np.concatenate([a, pad])[: self.obs_dim]
        noise = (self._rng.standard_normal(self.obs_dim) * 0.05).astype(np.float32)
        self._state = (0.95 * self._state + 0.3 * full_a + noise).astype(np.float32)
        reward = float(-np.dot(self._state, self._state) - 0.1 * np.dot(a, a))
        return self._state.copy(), reward, False, False, {}

    def close(self) -> None:
        return None


__all__ = ["MockEnv"]
