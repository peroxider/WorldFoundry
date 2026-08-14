"""Decoupled RTI recipe: render -> train -> inference -> policy (RLinf-shape).

This package turns the monolithic ``scripts/render_train_rl_cpu.py``
orchestrator into a *stage-based, backend-swappable* recipe:

* Each pipeline step is a :class:`Stage` with a stable ``run(ctx)`` contract.
* The execution device is resolved once through
  :func:`worldfoundry_train.device.resolve_device` (``WORLDFORGE_DEVICE``
  env / NPU auto-detect), so the *same* stages run on ``cpu`` today and
  ``npu`` tomorrow without changing pipeline code.
* A :class:`BackendRegistry` maps a backend name -> ordered stage list.
  The ``cpu`` backend is registered here; an ``npu`` backend can register
  its own (possibly accelerator-tuned) stages without touching the CPU path.

The phase *implementations* are the already-proven functions in
``scripts/render_train_rl_cpu.py`` (loaded once via a single import seam),
so this package adds orchestration/decoupling, not a duplicate of the
training math.
"""
from worldfoundry_train.recipe.config import RecipeConfig
from worldfoundry_train.recipe.pipeline import run_recipe
from worldfoundry_train.recipe.backends import (
    get_backend_stages,
    list_backends,
    register_backend,
)

__all__ = [
    "RecipeConfig",
    "run_recipe",
    "get_backend_stages",
    "list_backends",
    "register_backend",
]
