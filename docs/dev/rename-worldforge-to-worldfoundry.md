# Rename: WorldForge → WorldFoundry

> **Date:** 2026-08-04
> **Scope:** Phase-1 / Phase-2 transition window (before any external publish)
> **Commit:** see `git log --grep='rename worldforge'`

## Motivation

The project is positioned as an Ascend NPU **world-model 渲训推 (render / train / infer) framework** that orchestrates open-source components (vllm-ascend, mindspeed, verl / RLinf, msmodelslim). The old name `WorldForge` carried two limitations:

1. **"Forge"** suggests a single-point act of crafting. The project is an industrial pipeline + multi-component orchestrator — closer to **Foundry** than **Forge**.
2. **No semantic room for 渲 (frame rendering / Phase-5 W40+) or 实机运行 (real-machine execution, post-Phase-5) in the brand.

`WorldFoundry` retains `World` (world-model focus) and uses `Foundry` (industrial-scale, pipeline-native) to capture the long-term positioning.

## Scope of rename

- 4 Python packages: `worldforge_{common,train,inf,rl}` → `worldfoundry_{common,train,inf,rl}`
- Docker image tag: `worldforge-base:*` → `worldfoundry-base:*`
- vLLM plugin entry-point: `worldforge_inf` → `worldfoundry_inf`
- Description in `pyproject.toml` updated to reflect "渲训推 + 实机后续 + 开源编排" long-term positioning.

## What is NOT renamed

- Git history (immutable; old name persists in `git log`)
- `.superpowers/sdd/` task reports (immutable history)
- Build artifacts (`.egg-info/`, `dist/`, `.venv/`, `.pytest_cache/`, `.ruff_cache/`)
- `bench_results/`, `models/`

## Migration (for any external user, if published before)

Not applicable at rename time — project unpublished. If/when published, this section will document:
- PyPI: `pip install worldfoundry` (not `worldforge`)
- Docker: `docker pull worldfoundry-base:phase1`
- vLLM plugin entry-point import name: `worldfoundry_inf`
