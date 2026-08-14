"""Recipe pipeline: resolve device, build stages, run in order, emit summary.

The pipeline is intentionally thin and backend-agnostic — it does not
know whether a stage is CPU or NPU. It:

1. Resolves the device once (via ``RecipeConfig.resolve_device``).
2. Looks up the stage map for the configured backend.
3. Instantiates only the stages listed in ``cfg.stages`` (respecting order).
4. Runs each stage with a shared :class:`RecipeContext`.
5. Writes a JSON summary next to the output directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from worldfoundry_train.recipe.backends import get_backend_stages
from worldfoundry_train.recipe.config import RecipeConfig
from worldfoundry_train.recipe.stages import RecipeContext


def run_recipe(cfg: RecipeConfig, *, write_summary: bool = True) -> dict:
    """Execute the configured recipe and return the aggregate summary.

    Args:
        cfg: the recipe configuration for this run.
        write_summary: if True, write ``<out_dir>/recipe.json``.

    Returns:
        A dict: ``{"config": {...}, "device": str, "stages": {name: metrics}}``.
    """
    device = cfg.resolve_device()
    stage_map = get_backend_stages(cfg.backend)
    ctx = RecipeContext(cfg=cfg, device=device)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for stage_name in cfg.stages:
        if stage_name not in stage_map:
            raise KeyError(
                f"stage {stage_name!r} not available on backend {cfg.backend!r}; "
                f"available: {sorted(stage_map)}"
            )
        stage = stage_map[stage_name]()
        stage.run(ctx)

    summary = {
        "backend": cfg.backend,
        "device": device,
        "config": _config_serializable(cfg),
        "stages": ctx.results,
    }
    if write_summary:
        summary_path = cfg.out_dir / "recipe.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"[recipe] summary: {summary_path}")
    return summary


def _config_serializable(cfg: RecipeConfig) -> dict:
    out = {}
    for field_name, value in cfg.__dict__.items():
        if field_name == "stages":
            out[field_name] = list(value)
        else:
            out[field_name] = value
    return out


__all__ = ["run_recipe"]
