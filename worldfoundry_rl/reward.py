"""Reward model interface for Phase-4 RL training (canonical W36).

Closes roadmap G9 ("no reward interface"). Defines:

* :class:`RewardFn` — runtime-checkable Protocol that all task / intrinsic
  reward implementations satisfy. The contract is two-method:
  ``__call__(state, action, next_state) -> float`` plus
  ``reset() -> None`` so that episode-level state (e.g. one-time
  success bonuses) can be cleared between episodes.
* :func:`ensure_reward_fn` — type guard that raises ``TypeError`` if its
  argument does not satisfy the protocol (used by trainers to fail-fast
  on misconfigured configs).
* :func:`compose` — weighted-sum composite that returns a new
  ``RewardFn`` whose call forwards to all sub-rewards and whose reset
  fans out to every component.

Concrete task-specific rewards live in :mod:`worldfoundry_rl.reward_impl`
(FrankaKitchen manipulation reward + a latent-distance intrinsic). New
MuJoCo / SAPIEN tasks added in Phase-4 v2 should subclass-free implement
the protocol (no ABC required; structural typing via ``Protocol`` is
the contract).

Per spec: unblocked by W30 (MuJoCo baseline); required by W34 (PPO/GRPO
losses) and W37 (RL training integration).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardFn(Protocol):
    """Callable reward with episode-scoped reset.

    Implementations MUST:
      * be deterministic given the same triple ``(state, action,
        next_state)`` for the duration of one episode;
      * be safe to call with tensors, arrays, or dicts in
        ``state`` / ``next_state`` (the protocol does not pin the
        container type — task-specific rewards interpret it);
      * expose ``reset()`` for episode boundaries so any internal
        one-shot flags are cleared.
    """

    def __call__(self, state: object, action: object, next_state: object) -> float: ...

    def reset(self) -> None: ...


def ensure_reward_fn(obj: object) -> RewardFn:
    """Type guard: raises ``TypeError`` if ``obj`` is not a ``RewardFn``."""
    if not isinstance(obj, RewardFn):
        raise TypeError(
            f"{type(obj).__name__} does not satisfy worldfoundry_rl.reward.RewardFn: "
            "missing __call__ or reset method"
        )
    return obj


def compose(*rewards: RewardFn, weights: list[float] | None = None) -> RewardFn:
    """Weighted-sum composite reward.

    Returns a fresh ``RewardFn`` whose call is
    ``sum(weight_i * reward_i(state, action, next_state))`` and whose
    reset fans out to every component.

    Args:
        rewards: one or more reward callables. Must be non-empty.
        weights: per-reward weight list; defaults to all-ones.
            Length must match ``len(rewards)`` if provided.
    """
    if not rewards:
        raise ValueError("compose() requires at least one reward")
    if weights is None:
        weights = [1.0] * len(rewards)
    if len(weights) != len(rewards):
        raise ValueError(
            f"weights length ({len(weights)}) must match rewards ({len(rewards)})"
        )

    class _Composite:
        def __init__(self) -> None:
            self._components = list(zip(rewards, weights))

        def __call__(self, state: object, action: object, next_state: object) -> float:
            return float(
                sum(w * r(state, action, next_state) for r, w in self._components)
            )

        def reset(self) -> None:
            for r, _ in self._components:
                r.reset()

    return _Composite()


__all__ = ["RewardFn", "ensure_reward_fn", "compose"]