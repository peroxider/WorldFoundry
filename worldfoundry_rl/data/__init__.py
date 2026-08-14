"""Data utilities for ``worldfoundry_rl`` (W2.16+).

Currently houses the FrankaKitchen video collector and transition dataset
that feed real RGB + action transitions into the world-model train phase.
Synthetic data lives in ``worldfoundry_train.data.synthetic_*`` and is
imported by the orchestrator's ``--data-source synthetic`` path; this
package is the franka_kitchen counterpart.
"""
from __future__ import annotations

__all__: list[str] = []