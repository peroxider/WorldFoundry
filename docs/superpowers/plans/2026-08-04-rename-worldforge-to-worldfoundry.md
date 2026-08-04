# Rename: WorldForge → WorldFoundry — Execution Report

> **Date:** 2026-08-04
> **Status:** Executed (10 commits on branch `rename/worldforge-to-worldfoundry`)
> **Companion docs:**
> - `docs/dev/rename-worldforge-to-worldfoundry.md` — motivation + scope summary
> - `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md` — long-term positioning
> - `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` — RL framework decision context

This file replaces the original implementation plan. The original plan (~1098 lines, T1–T14, G1–G12) was **corrupted in its initial commit** when T8's `sed s/WorldForge/WorldFoundry/g` pass over-rewrote the brand name inside the rename plan itself, leaving the file in a self-referential broken state (`WorldFoundry → WorldFoundry`, `worldfoundry → worldfoundry`). Because no clean version ever existed in git history, the file has been replaced with this completion report. The original plan was the working document for execution; this report is the durable record of what was actually done.

## 1. Original plan structure (preserved for historical reference)

The rename plan was organized as 14 tasks (T1–T14) governed by 12 global constraints (G1–G12):

- **T1 — docs/dev/ rename note** (`docs/dev/rename-worldforge-to-worldfoundry.md`): commit `3d46586`
- **T2 — pyproject.toml** (name + description + entry-points): commit `f611143`
- **T3 — Python package directory rename** (`worldforge_{common,train,inf,rl}` → `worldfoundry_{common,train,inf,rl}`): commit `dab5c27`
- **T4 — Imports + docstrings sweep**: commit `9c9abd0`
- **T5 — Docker image tag rename** (`docker/worldforge-base` → `docker/worldfoundry-base`): commit `ad5fbdb`
- **T6 — Shell + python scripts**: commit `db8e218`
- **T7 — YAML configs**: commit `4b014e3`
- **T8 — Docs sweep** (specs/plans/dev/user/roadmap): commit `02a4c6f`
- **T9 — Spike files sweep** (`spikes/w35-3-rlinf/` historical archive): commit `1ba6b85`
- **T10 — README rename**: commit `b597a9a`
- **T11 — Post-rename cleanup** (metrics, pip install hints, escaped regex in tests): commit `a7bf6fa`
- **T12 — Spike preservation** (locked to "do" path per user instruction: rename `worldforge` references inside spike files, keep spike as W35.3 historical archive): executed inside T9
- **T13 — Adversarial verification** (33-check protocol via subagent): PASS — see §4
- **T14 — Push / PR strategy**: pending — see §6

## 2. Global constraints (G1–G12) — summary

- **G1 — Scope:** code, docs, configs, scripts, Docker tags, entry-points — everything in the source tree.
- **G2 — Preserve:** git history (immutable), `.superpowers/sdd/` task reports (immutable history), build artifacts (`.egg-info/`, `dist/`, `.venv/`, `.pytest_cache/`, `.ruff_cache/`), `bench_results/`, `models/`.
- **G3 — Historical doc handling:** docs dated **before 2026-08-03** are immutable historical records and retain `WorldForge`. Docs dated **on/after 2026-08-03** (forward-looking) and the new rename doc apply `WorldFoundry`.
- **G4 — pyproject.toml:** `name = "worldfoundry"`, description rewritten to capture 渲训推+实机+开源编排 long-term positioning, vLLM plugin entry-point renamed.
- **G5 — Docker:** image tag `worldforge-base:*` → `worldfoundry-base:*` (matches Phase-1 `phase1` tag and future `phase2`/`phase3`/`phase4` tags).
- **G6 — Python packages:** 4 directories renamed; in-tree tests import via package paths after rename.
- **G7 — Verification:** pytest 312 passed before/after rename; ruff baseline (134 pre-existing errors) unchanged.
- **G8 — Spike preservation:** `spikes/w35-3-rlinf/` retained as W35.3 decision historical archive; spike-internal references to `worldforge` renamed per T12 "do" path.
- **G9 — Backward compatibility:** none (project unpublished; this is the rename window).
- **G10 — Migration docs:** PyPI/Docker/vLLM entry-point migration notes live in `docs/dev/rename-worldforge-to-worldfoundry.md`.
- **G11 — No force-push, no history rewrite:** `git filter-repo` rejected — git history keeps both names.
- **G12 — Adversarial verification gate:** T13 must PASS before T14 (push/PR).

## 3. Commit log (chronological)

```
3d46586 docs: add rename note for worldforge → worldfoundry                              (T1)
f611143 build(pyproject): rename worldforge → worldfoundry in name/description/entry-points  (T2)
dab5c27 refactor(packages): rename worldforge_{common,train,inf,rl} → worldfoundry_{common,train,inf,rl}  (T3)
9c9abd0 refactor(python): sweep all imports + docstrings from worldforge → worldfoundry    (T4)
ad5fbdb build(docker): rename docker/worldforge-base → docker/worldfoundry-base (image tag phase1)  (T5)
db8e218 chore(scripts): rename worldforge → worldfoundry in shell + python scripts        (T6)
4b014e3 chore(configs): rename worldforge → worldfoundry in YAML                          (T7)
02a4c6f docs: rename worldforge → worldfoundry across specs/plans/dev/user/roadmap         (T8)
1ba6b85 chore(spikes): rename worldforge → worldfoundry in w35-3-rlinf spike (historical archive)  (T9, T12)
b597a9a docs(readme): rename worldforge → worldfoundry                                    (T10)
a7bf6fa chore: post-rename cleanup (metrics, pip install hints, escaped regex in tests)    (T11)
```

10 commits total, all on branch `rename/worldforge-to-worldfoundry`. Final branch head: `a7bf6fa`.

## 4. Verification

T13 ran an adversarial verification subagent (33-check protocol). Outcome:

- **PASS** — pytest 312 tests pass (baseline preserved), ruff 134 errors unchanged (pre-existing, unrelated to rename), 4 packages import correctly, entry-points resolve, Docker tag renamed in Dockerfile + README, scripts/YAML/configs free of `worldforge` brand references (except in G2-preserved historical contexts).
- **FAIL** in initial verification pass: T8's over-aggressive `sed` rewrote the brand name inside 12 historical docs (dated 2026-07-27 through 2026-07-30) and inside the rename plan itself. Per G3, historical docs dated before 2026-08-03 must retain `WorldForge`. Root cause: T8's sed pass did not exclude historical plans/specs. Recovery: see §5.

## 5. Corruption recovery (post-T13)

The T8 sed pass corrupted:
- 5 historical plans (`docs/superpowers/plans/2026-07-27` through `2026-07-30`) — brand rewritten inside body
- 5 historical specs (`docs/superpowers/specs/2026-07-27`, `2026-07-28`, `2026-07-29`, `W9_EVAL_REPORT`, `W11_W12_QUANT_REPORT`) — brand rewritten inside body
- 1 rename note (`docs/dev/rename-worldforge-to-worldfoundry.md`) — 3 lines corrupted (`WorldForge` → `WorldFoundry` and `worldforge_inf` → `worldfoundry_inf`)
- 1 rename plan (this file) — title and rationale lines corrupted; because this file was created on the branch with no master counterpart, the corruption was present from the initial commit

**Recovery actions:**
- 11 historical files (5 plans + 5 specs + rename note) restored from `master` via `git checkout master -- <files>` — they now correctly retain `WorldForge` branding
- The 7 forward-looking files (post-2026-08-03) keep `WorldFoundry` as intended
- This file (the rename plan) is replaced wholesale with the present completion report

## 6. Forward-looking docs that retain `WorldFoundry` (verified post-recovery)

- `docs/dev/environment.md`, `docs/dev/runbook.md`, `docs/user/quickstart.md`
- `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md`
- `docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md`
- `docs/superpowers/specs/2026-08-04-w22-7-ab-experiment-report.md`
- `docs/superpowers/specs/2026-08-04-w35-3-rlinf-custom-policy-spike-plan.md`
- `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`

## 7. T14 — Push / PR strategy (pending user decision)

Open questions for the user before T14:

1. **Push target:** `origin rename/worldforge-to-worldfoundry` (branch only), or create PR against `main` (the published branch per repo conventions)?
2. **Merge strategy:** squash-merge (10 commits → 1), or preserve commit history (10 commits preserved, more granular attribution)?
3. **PR description:** generate from this report, or write fresh?
4. **Post-merge branch cleanup:** delete `rename/worldforge-to-worldfoundry` immediately, or retain for N days?

These are user-policy decisions and were not made during execution. T14 will be executed only after explicit user instruction.