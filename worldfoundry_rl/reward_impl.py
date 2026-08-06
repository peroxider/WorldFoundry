"""Phase-4 task-specific reward implementations (MuJoCo baseline + intrinsic).

These satisfy the :class:`worldfoundry_rl.reward.RewardFn` protocol
(structurally — no inheritance required). Each implementation documents
its state contract: which keys it expects inside the ``next_state``
dict it is given.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class FrankaKitchenReward:
    """Shaped reward for FrankaKitchen manipulation (MuJoCo + dm_control).

    Sum of:
      * distance penalty: ``-distance_coef * ||ee_pos - target_pos||^2``
      * action regularization: ``-action_coef * ||action||^2``
      * per-step penalty: ``-step_penalty`` (constant)
      * one-time success bonus at the first ``task_success == True``

    State contract for ``next_state`` (dict):
      * ``ee_pos`` — end-effector position, shape ``(3,)``
      * ``target_pos`` — target position, shape ``(3,)``
      * ``task_success`` — bool, default ``False``

    Action contract: any array-like with non-zero length (raveled to
    1-D for the L2 norm). Accepts numpy arrays and torch tensors.

    Mirrors the standard FrankaKitchen reward composition used in
    DeepMind / RLinf references (distance + action L2 + step penalty +
    success bonus) so Phase-4 v1 numbers are reproducible against
    dm_control's published baselines (±5% per W30.4 gate).
    """

    def __init__(
        self,
        distance_coef: float = 1.0,
        action_coef: float = 0.01,
        step_penalty: float = 0.0,
        success_bonus: float = 1.0,
    ) -> None:
        if distance_coef < 0:
            raise ValueError(f"distance_coef must be >= 0, got {distance_coef}")
        if action_coef < 0:
            raise ValueError(f"action_coef must be >= 0, got {action_coef}")
        if step_penalty < 0:
            raise ValueError(f"step_penalty must be >= 0, got {step_penalty}")
        if success_bonus < 0:
            raise ValueError(f"success_bonus must be >= 0, got {success_bonus}")
        self.distance_coef = float(distance_coef)
        self.action_coef = float(action_coef)
        self.step_penalty = float(step_penalty)
        self.success_bonus = float(success_bonus)
        self._has_succeeded = False

    def __call__(self, state: object, action: object, next_state: object) -> float:
        ee = np.asarray(next_state["ee_pos"], dtype=np.float64).ravel()
        target = np.asarray(next_state["target_pos"], dtype=np.float64).ravel()
        diff = ee - target
        dist_sq = float(diff @ diff)

        act_arr = np.asarray(action, dtype=np.float64).ravel()
        act_sq = float(act_arr @ act_arr)

        success_now = bool(next_state.get("task_success", False))
        bonus = 0.0
        if success_now and not self._has_succeeded:
            bonus = self.success_bonus
        self._has_succeeded = self._has_succeeded or success_now

        return (
            -self.distance_coef * dist_sq
            -self.action_coef * act_sq
            -self.step_penalty
            + bonus
        )

    def reset(self) -> None:
        self._has_succeeded = False


class LatentDistanceReward:
    """Intrinsic reward from world-model latent distance.

    ``reward = -beta * ||encoder(state) - encoder(next_state)||^2``

    Useful as a curiosity bonus / exploration signal: rewards the agent
    for moving between visually distinct states (i.e., states whose
    V-JEPA 2 encoder embeddings differ). The encoder is any callable
    mapping ``array-like -> array-like`` (HF VJEPA 2 encoder adapter,
    a tiny CNN, or a feature extractor wrapper).

    The encoder is held by reference; do not mutate externally without
    resetting, since latent changes invalidate the squared-distance
    semantics.
    """

    def __init__(
        self,
        encoder: Callable[[object], np.ndarray],
        beta: float = 0.1,
    ) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.encoder = encoder
        self.beta = float(beta)

    def __call__(self, state: object, action: object, next_state: object) -> float:
        s = np.asarray(self.encoder(state), dtype=np.float64).ravel()
        ns = np.asarray(self.encoder(next_state), dtype=np.float64).ravel()
        diff = s - ns
        dist_sq = float(diff @ diff)
        return -self.beta * dist_sq

    def reset(self) -> None:
        pass


__all__ = ["FrankaKitchenReward", "LatentDistanceReward"]