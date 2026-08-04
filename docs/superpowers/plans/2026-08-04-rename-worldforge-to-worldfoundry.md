# Rename WorldFoundry → WorldFoundry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project from `worldfoundry` → `worldfoundry` to align the brand with the long-term positioning ("Ascend NPU world-model **渲训推** framework" — training + simulator RL + frame rendering + simulator/hardware deployment, orchestrating open-source components). Update `pyproject.toml` description to match. Total ~150 files touched; estimated ~2 hours of mechanical work + verification.

**Rationale (recap of `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md` + ADR-W35):**
- "Forge" implies point-in-time crafting. The project's true character is **industrial pipeline + multi-component orchestrator** (vllm-ascend / mindspeed / verl-or-RLinf / msmodelslim) — closer to **Foundry** than **Forge**.
- "Foundry" carries the "渲染 (frame rendering / W40+ Phase-5) + 训练 (training / W9–W22) + 推理 (inference / W19+ W24–W27) + 实机运行 (real-machine execution, post-Phase-5)" 渲训推+实机 scope more naturally.
- Project has not been published to PyPI or Docker Hub — this is the **only viable rename window** before external API contracts crystallize.

**Architecture:** Identical code layout; only identifiers change. Four Python packages (`worldfoundry_common`, `worldfoundry_train`, `worldfoundry_inf`, `worldfoundry_rl`) → (`worldfoundry_common`, `worldfoundry_train`, `worldfoundry_inf`, `worldfoundry_rl`). Docker image tag and vLLM plugin entry-point names move in lockstep. Historical git commits / `.superpowers/sdd/` task reports **not** rewritten (immutable history).

**Tech Stack:** Python 3.11, `git mv` for directory renames, `sed -i` for in-place text replacement, `pytest` for verification, `ruff` for lint gate, `grep -r` for residual-identifier sweep.

---

## Global Constraints

| # | Rule | Source |
|---|------|--------|
| G1 | **Identifier symmetry**: every `worldforge_*` identifier in source MUST become `worldfoundry_*`; the inverse MUST NOT exist after T13 (residual sweep) | This plan |
| G2 | **PyPI / Docker image name** changes in lockstep with package name; no transitional `worldfoundry` aliasing on PyPI (unpublished, no external deps) | User confirmation 2026-08-04 |
| G3 | **vLLM plugin entry-point** name change is a **publish-time API contract** — current `worldfoundry_inf` → `worldfoundry_inf`; after first publish this becomes permanent | `pyproject.toml:35-39` |
| G4 | **No shim/backwards-compat layer** (`worldforge_*/__init__.py` that re-exports from `worldfoundry_*`) — unpublished project, zero external users; shims are gold-plating | User confirmation 2026-08-04 |
| G5 | **Git history preserved**: do NOT use `git filter-repo` or `bfg` to rewrite commits; old name remains visible in `git log` / `git blame` — this is correct, not bug | Standard rename hygiene |
| G6 | **`.superpowers/sdd/` task reports and `progress.md` NOT rewritten** — they are immutable historical snapshots; add a single banner line at top of `progress.md` only | History-immutability rule |
| G7 | **`.idea/` files (IDE config)**: rename for hygiene, but `.idea/` is in `.gitignore` so this is local-only; commit nothing from there | `.gitignore:20` |
| G8 | **Build artifacts regenerated, not edited**: `worldfoundry.egg-info/`, `dist/`, `.venv/`, `.pytest_cache/`, `.ruff_cache/` are deleted; `pip install -e .` rebuilds them under new name | Build hygiene |
| G9 | **One commit per task**: format `<scope>: <imperative>`; no amend, no squash, no force-push | Inherited from `2026-07-27-worldfoundry-phase1.md:26` |
| G10 | **Verification gates between task groups**: pytest must be green before moving from code (T2-T7) to docs (T8-T10); residual-identifier sweep must be empty before final commit | Defensive sequencing |
| G11 | **Spike files (T12) ARE renamed in lockstep**: `spikes/w35-3-rlinf/` is preserved as W35.3 决策历史档案 (per user decision 2026-08-04); its `worldfoundry` references must be swept like any other source file. Spike is **not** discarded in W36+; it stays as ADR-W35 §6 回填材料 | User-confirmed retention decision |
| G12 | **Branch from master**: all work on a fresh branch `rename/worldfoundry-to-worldfoundry`; PR to master | Standard practice |

---

## File Structure (BEFORE → AFTER)

### Source packages

```
BEFORE                                          AFTER
worldfoundry_common/         ─────────────►       worldfoundry_common/
worldfoundry_train/          ─────────────►       worldfoundry_train/
worldfoundry_inf/            ─────────────►       worldfoundry_inf/
worldfoundry_rl/             ─────────────►       worldfoundry_rl/
```

### Docker

```
docker/worldfoundry-base/                        docker/worldfoundry-base/
├── Dockerfile                                  ├── Dockerfile
└── README.md                                   └── README.md
```

### New doc (created)

```
docs/dev/rename-worldfoundry-to-worldfoundry.md    (new — T1)
```

---

## Out of Scope (do NOT touch)

| Path | Reason |
|---|---|
| `git log` / git history | Immutable; `git filter-repo` rejected by G5 |
| `.superpowers/sdd/task-*-report.md`, `brief.md`, `review.md` | Immutable history (G6) |
| `.superpowers/sdd/progress-phase1.md` | Immutable history (G6) |
| `.venv/`, `dist/`, `build/`, `.pytest_cache/`, `.ruff_cache/`, `worldfoundry.egg-info/` | Build artifacts; deleted, regenerated (G8) |
| `models/` | Runtime mount, not source |
| `bench_results/fsdp-baseline.json` | Data file, brand-agnostic |
| `worldfoundry.egg-info/SOURCES.txt` etc. | Auto-regenerated on next `pip install -e .` |
| All previous git commit messages | Immutable (G5) |

---

## Task 1: Create rename note document

**Files:**
- Create: `docs/dev/rename-worldfoundry-to-worldfoundry.md`

**Why this task first:** Establishes the *why* before any code changes. Anyone running `git log` after the rename will see the rename commit and can read this doc to understand motivation. Also lets the user / future contributor quickly verify the rename was intentional and trace migration if they hold old code.

**Interfaces:**
- Produces: a markdown document of ~60 lines; no code interfaces

- [ ] **Step 1: Create the file**

Write to `/mnt/c/WorkSpace/WorldFoundry/docs/dev/rename-worldfoundry-to-worldfoundry.md`:

```markdown
# Rename: WorldFoundry → WorldFoundry

> **Date:** 2026-08-04
> **Scope:** Phase-1 / Phase-2 transition window (before any external publish)
> **Commit:** see `git log --grep='rename worldfoundry'`

## Motivation

The project is positioned as an Ascend NPU **world-model 渲训推 (render / train / infer) framework** that orchestrates open-source components (vllm-ascend, mindspeed, verl / RLinf, msmodelslim). The old name `WorldFoundry` carried two limitations:

1. **"Forge"** suggests a single-point act of crafting. The project is an industrial pipeline + multi-component orchestrator — closer to **Foundry** than **Forge**.
2. **No semantic room for 渲 (frame rendering / Phase-5 W40+) or 实机运行 (real-machine execution, post-Phase-5) in the brand.

`WorldFoundry` retains `World` (world-model focus) and uses `Foundry` (industrial-scale, pipeline-native) to capture the long-term positioning.

## Scope of rename

- 4 Python packages: `worldforge_{common,train,inf,rl}` → `worldfoundry_{common,train,inf,rl}`
- Docker image tag: `worldfoundry-base:*` → `worldfoundry-base:*`
- vLLM plugin entry-point: `worldfoundry_inf` → `worldfoundry_inf`
- Description in `pyproject.toml` updated to reflect "渲训推 + 实机后续 + 开源编排" long-term positioning.

## What is NOT renamed

- Git history (immutable; old name persists in `git log`)
- `.superpowers/sdd/` task reports (immutable history)
- Build artifacts (`.egg-info/`, `dist/`, `.venv/`, `.pytest_cache/`, `.ruff_cache/`)
- `bench_results/`, `models/`

## Migration (for any external user, if published before)

Not applicable at rename time — project unpublished. If/when published, this section will document:
- PyPI: `pip install worldfoundry` (not `worldfoundry`)
- Docker: `docker pull worldfoundry-base:phase1`
- vLLM plugin entry-point import name: `worldfoundry_inf`
```

- [ ] **Step 2: Verify file is staged for tracking**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add docs/dev/rename-worldfoundry-to-worldfoundry.md
git status --short
```

Expected: `A  docs/dev/rename-worldfoundry-to-worldfoundry.md` in green.

- [ ] **Step 3: Commit**

```bash
git commit -m "docs: add rename note for worldfoundry → worldfoundry"
```

Expected: one new commit on top of master.

---

## Task 2: Create working branch

**Files:**
- Branch: `rename/worldfoundry-to-worldfoundry`

**Why second:** All subsequent tasks land on this branch. Master stays clean until PR merge.

- [ ] **Step 1: Create and check out branch**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git checkout -b rename/worldfoundry-to-worldfoundry
git branch --show-current
```

Expected output: `rename/worldfoundry-to-worldfoundry`

---

## Task 3: Update `pyproject.toml` (package name + description + entry-points + setuptools include)

**Files:**
- Modify: `pyproject.toml` (5 sites: lines 6, 8, 36, 39, 42)

**Why this task before package rename:** `pyproject.toml` is the canonical source of the package name and entry-point declarations. Updating it first means the next `pip install -e .` (after directory rename) sees the new name without intermediate inconsistency. Entry-points move in lockstep with package rename (G3).

**Interfaces:**
- Produces: a valid `pyproject.toml` that, after T4 directory rename + T5 import sweep, can be re-installed as `worldfoundry`

- [ ] **Step 1: Replace package name**

In `pyproject.toml` line 6, change:

```diff
- name = "worldfoundry"
+ name = "worldfoundry"
```

- [ ] **Step 2: Update description**

In `pyproject.toml` line 8, change:

```diff
- description = "Ascend NPU world model training + inference platform (Phase-1: V-JEPA 2)"
+ description = "Ascend NPU world-model rendering/training/inference framework (JEPA family, simulator RL, simulator + hardware deployment) — orchestrating vllm-ascend, mindspeed, verl/RLinf, msmodelslim"
```

- [ ] **Step 3: Update vLLM platform entry-point**

In `pyproject.toml` line 36, change:

```diff
- worldfoundry_inf = "worldfoundry_inf._vllm_ascend_plugin:register"
+ worldfoundry_inf = "worldfoundry_inf._vllm_ascend_plugin:register"
```

Note: the second half (module path) still references `worldfoundry_inf` — T4 will rename the directory and T5 will fix the module path string. The line above will be a temporary broken import path; pytest will fail at T5 until T5 completes.

- [ ] **Step 4: Update vLLM general plugin entry-point**

In `pyproject.toml` line 39, change:

```diff
- vjepa2_model = "worldfoundry_inf._vllm_ascend_plugin:register_model"
+ vjepa2_model = "worldfoundry_inf._vllm_ascend_plugin:register_model"
```

Same caveat as Step 3 — temporary broken path until T4+T5.

- [ ] **Step 5: Update setuptools packages.find include pattern**

In `pyproject.toml` line 42, change:

```diff
- include = ["worldfoundry_*"]
+ include = ["worldfoundry_*"]
```

- [ ] **Step 6: Verify pyproject.toml parses**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Verify no `worldfoundry` references remain in pyproject.toml**

```bash
grep -nE "worldfoundry" /mnt/c/WorkSpace/WorldFoundry/pyproject.toml
```

Expected: no output (zero matches).

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add pyproject.toml
git commit -m "build(pyproject): rename worldfoundry → worldfoundry in name/description/entry-points"
```

---

## Task 4: Rename 4 Python package directories

**Files:**
- Rename (via `git mv`): `worldfoundry_common/`, `worldfoundry_train/`, `worldfoundry_inf/`, `worldfoundry_rl/` → `worldfoundry_*/`

**Why this task before import sweep:** Directory rename first means `git mv` preserves rename history in `git log --follow`. Import sweep (T5) operates on already-renamed paths.

**Interfaces:**
- Produces: 4 renamed directories with all contents intact

- [ ] **Step 1: Rename `worldfoundry_common`**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git mv worldfoundry_common worldfoundry_common
```

- [ ] **Step 2: Rename `worldfoundry_train`**

```bash
git mv worldfoundry_train worldfoundry_train
```

- [ ] **Step 3: Rename `worldfoundry_inf`**

```bash
git mv worldfoundry_inf worldfoundry_inf
```

- [ ] **Step 4: Rename `worldfoundry_rl`**

```bash
git mv worldfoundry_rl worldfoundry_rl
```

- [ ] **Step 5: Verify no top-level `worldforge_*` source directory remains**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
ls -d worldforge_* 2>&1 | head -5
```

Expected: `ls: cannot access 'worldforge_*': No such file or directory`

(Note: `worldfoundry.egg-info/` is a build artifact to be deleted in T13, not a source directory.)

- [ ] **Step 6: Verify all 4 new directories exist and are tracked**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git status --short | grep -E "^R|^A" | grep worldfoundry | wc -l
```

Expected: ≥ 4 (one rename per directory; the count will be larger because git reports each renamed *file*).

- [ ] **Step 7: Commit (single commit covering all 4 renames)**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "refactor(packages): rename worldforge_{common,train,inf,rl} → worldfoundry_{common,train,inf,rl}"
```

---

## Task 5: Sweep all Python imports across codebase

**Files:**
- Modify (in-place `sed -i`): all 64 files that contain `^from worldforge_` or `^import worldforge_`

**Why this task before scripts/configs:** Python imports must resolve for pytest to run. If we rename directories but not import strings, the test suite breaks immediately at import time. Once imports are fixed, pytest can be run as the verification gate (G10) before moving to less critical layers (scripts, configs, docs).

**Interfaces:**
- Produces: 0 broken `worldforge_*` imports across all `.py` files

- [ ] **Step 1: Verify current state — count of imports to fix**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -rE "^from worldforge_|^import worldforge_" --include="*.py" . | wc -l
```

Expected: `107` (matches impact analysis; may be ±2 if any file was touched between plan write and execution).

- [ ] **Step 2: Apply `sed -i` rename across all .py files**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find . -name "*.py" -not -path "*/\.venv/*" -not -path "*/spikes/*" -print0 \
  | xargs -0 sed -i -E 's/\b(worldfoundry)(_common|_train|_inf|_rl)\b/worldfoundry\2/g'
```

Why this regex:
- `\b` word boundary prevents accidental matches inside other identifiers
- `(worldfoundry)(_common|_train|_inf|_rl)` matches the exact 4 package names only; safe even if a string literal happens to contain "worldfoundry" as a substring of a longer word
- Spike files excluded because they may be discarded; handled in T12

- [ ] **Step 3: Sweep `"WorldFoundry"` literal in Python docstrings/comments**

The Step 2 regex does not touch the word `WorldFoundry` (capitalized, in docstrings/comments). Sweep separately:

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find . -name "*.py" -not -path "*/\.venv/*" -not -path "*/spikes/*" -print0 \
  | xargs -0 sed -i 's/WorldFoundry/WorldFoundry/g'
```

- [ ] **Step 4: Verify zero `worldforge_*` package imports remain in non-spike `.py`**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -rE "^from worldforge_|^import worldforge_" --include="*.py" . | grep -v "/spikes/" | wc -l
```

Expected: `0`

- [ ] **Step 5: Verify zero `WorldFoundry` literal in non-spike `.py`**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -rE "WorldFoundry" --include="*.py" . | grep -v "/spikes/" | wc -l
```

Expected: `0`

- [ ] **Step 6: Spot-check a few high-traffic files**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
head -5 worldfoundry_train/trainer.py
head -5 worldfoundry_inf/server.py
head -5 worldfoundry_common/__init__.py
```

Expected: every line that previously read `from worldforge_…` now reads `from worldfoundry_…`. The `worldfoundry_common/__init__.py` first line should now reflect the `WorldFoundry` docstring update.

- [ ] **Step 7: Reinstall package in editable mode**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
rm -rf worldfoundry.egg-info/
pip install -e . 2>&1 | tail -10
```

Expected: `Successfully installed worldfoundry-0.1.0 ...`

If `pip install -e .` fails, STOP and diagnose — likely a stale `__pycache__` or `.egg-info` lingering. Common fixes:

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find . -name "__pycache__" -type d -not -path "*/\.venv/*" -exec rm -rf {} + 2>/dev/null
rm -rf worldfoundry.egg-info/
pip install -e . 2>&1 | tail -10
```

- [ ] **Step 8: Run pytest as smoke gate**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
pytest -x --tb=short -q 2>&1 | tail -30
```

Expected: tests run (may pass or fail on unrelated CPU-fallback behavior, but **must not fail on `ModuleNotFoundError: No module named 'worldforge_…'`**).

If any test fails with `ModuleNotFoundError: No module named 'worldforge_…'`, a sweep missed a file. Diagnose:

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -rE "worldfoundry" --include="*.py" . | grep -v "/spikes/" | grep -v "worldfoundry" | head -20
```

- [ ] **Step 9: Run ruff as lint gate**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
ruff check . 2>&1 | tail -20
```

Expected: no errors, or only errors unrelated to rename.

- [ ] **Step 10: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "refactor(python): sweep all imports + docstrings from worldfoundry → worldfoundry"
```

---

## Task 6: Update Docker (file content + directory rename)

**Files:**
- Modify: `docker/worldfoundry-base/Dockerfile`, `docker/worldfoundry-base/README.md`
- Rename (via `git mv`): `docker/worldfoundry-base/` → `docker/worldfoundry-base/`

**Why this task:** Docker image tag is a publish-time API contract (mirrors G3 for vLLM entry-point). The build instructions in README reference both the directory and the image tag.

- [ ] **Step 1: Rename Docker directory**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git mv docker/worldfoundry-base docker/worldfoundry-base
```

- [ ] **Step 2: Update Dockerfile title comment**

In `docker/worldfoundry-base/Dockerfile` line 1, change:

```diff
- # WorldFoundry Phase-1 base image
+ # WorldFoundry Phase-1 base image
```

- [ ] **Step 3: Update Dockerfile COPY paths**

In `docker/worldfoundry-base/Dockerfile` lines 48-51, change:

```diff
- COPY worldfoundry_common/ worldfoundry_common/
- COPY worldfoundry_train/ worldfoundry_train/
- COPY worldfoundry_inf/ worldfoundry_inf/
- COPY worldfoundry_rl/ worldfoundry_rl/
+ COPY worldfoundry_common/ worldfoundry_common/
+ COPY worldfoundry_train/ worldfoundry_train/
+ COPY worldfoundry_inf/ worldfoundry_inf/
+ COPY worldfoundry_rl/ worldfoundry_rl/
```

- [ ] **Step 4: Update Dockerfile README**

In `docker/worldfoundry-base/README.md`:
- Line 1 title: `# WorldFoundry base image`
- Line 14 build command: `docker build -t worldfoundry-base:phase1 -f docker/worldfoundry-base/Dockerfile .`
- Line 19 run command: `docker run --rm -it worldfoundry-base:phase1 bash`

- [ ] **Step 5: Verify zero `worldfoundry` in Docker files**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -rE "worldfoundry|WorldFoundry" docker/ | wc -l
```

Expected: `0`

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "build(docker): rename docker/worldfoundry-base → docker/worldfoundry-base (image tag phase1)"
```

---

## Task 7: Update scripts/

**Files:**
- Modify: `scripts/phase1_ac_repro.sh`, `scripts/phase1_smoke.sh`, `scripts/w9_eval.sh`, `scripts/w17_atlas_w8a8_eval.sh`, `scripts/w17_populate_report.py`, `scripts/w17_preflight.py`

**Why this task:** Scripts import from packages; missing rename here breaks shell smoke tests on a fresh clone.

- [ ] **Step 1: Sweep scripts**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find scripts -type f \( -name "*.sh" -o -name "*.py" \) -print0 \
  | xargs -0 sed -i -E 's/\b(worldfoundry)(_common|_train|_inf|_rl)\b/worldfoundry\2/g; s/WorldFoundry/WorldFoundry/g'
```

- [ ] **Step 2: Verify zero `worldfoundry` references in scripts**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -rE "worldfoundry|WorldFoundry" scripts/ | wc -l
```

Expected: `0`

- [ ] **Step 3: Spot-check one script**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
head -30 scripts/phase1_smoke.sh
```

Expected: any `from worldforge_…` is now `from worldfoundry_…`.

- [ ] **Step 4: Run shell syntax check (bash -n)**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
bash -n scripts/phase1_smoke.sh && bash -n scripts/w9_eval.sh && bash -n scripts/w17_atlas_w8a8_eval.sh && bash -n scripts/phase1_ac_repro.sh && echo "All scripts syntax OK"
```

Expected: `All scripts syntax OK`

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "chore(scripts): rename worldfoundry → worldfoundry in shell + python scripts"
```

---

## Task 8: Update training config YAML

**Files:**
- Modify: `worldfoundry_train/configs/vjepa2_vitl_1.2b.yaml`

**Why this task:** YAML configs may reference package paths (e.g., adapter class names); one file confirmed affected.

- [ ] **Step 1: Sweep YAML configs**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find . -name "*.yaml" -o -name "*.yml" 2>/dev/null \
  | grep -v "/\.venv/" | grep -v "/spikes/" \
  | xargs sed -i -E 's/\b(worldfoundry)(_common|_train|_inf|_rl)\b/worldfoundry\2/g; s/WorldFoundry/WorldFoundry/g'
```

- [ ] **Step 2: Verify zero `worldfoundry` in project YAMLs**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find . -name "*.yaml" -o -name "*.yml" 2>/dev/null \
  | grep -v "/\.venv/" | grep -v "/spikes/" \
  | xargs grep -lE "worldfoundry|WorldFoundry" 2>/dev/null | wc -l
```

Expected: `0`

- [ ] **Step 3: Spot-check the training config**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
cat worldfoundry_train/configs/vjepa2_vitl_1.2b.yaml | head -30
```

Expected: any reference is `worldfoundry_*`.

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "chore(configs): rename worldfoundry → worldfoundry in YAML"
```

---

## Task 9: Update docs/ (specs / plans / dev / user / roadmap)

**Files:**
- Modify: 18 markdown files under `docs/`
  - `docs/dev/environment.md`
  - `docs/dev/runbook.md`
  - `docs/user/quickstart.md`
  - `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md`
  - `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md`
  - `docs/superpowers/specs/2026-07-28-worldfoundry-phase2-w9-w10-mindspeed-design.md`
  - `docs/superpowers/specs/2026-07-29-worldfoundry-phase2-w11-w12-quantization-design.md`
  - `docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md`
  - `docs/superpowers/specs/2026-08-04-w35-3-rlinf-custom-policy-spike-plan.md`
  - `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
  - `docs/superpowers/specs/W9_EVAL_REPORT.md`
  - `docs/superpowers/specs/W11_W12_QUANT_REPORT.md`
  - `docs/superpowers/plans/2026-07-27-worldfoundry-phase1.md`
  - `docs/superpowers/plans/2026-07-28-worldfoundry-phase2-w9-w10-mindspeed.md`
  - `docs/superpowers/plans/2026-07-29-worldfoundry-phase2-w11-w12-quantization.md`
  - `docs/superpowers/plans/2026-07-29-worldfoundry-phase2-w12-cpu-smoke.md`
  - `docs/superpowers/plans/2026-07-29-worldfoundry-phase2-w13-cpu-train.md`
  - `docs/superpowers/plans/2026-07-30-worldfoundry-phase2-w17-atlas-w8a8-validation.md`

**Why this task before T11 (sdd history):** Active planning docs are forward-looking; renaming them keeps the project's planning narrative coherent. Historical task reports in `.superpowers/sdd/` are immutable (G6, T11).

- [ ] **Step 1: Sweep `docs/`**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find docs -name "*.md" -type f -print0 \
  | xargs -0 sed -i -E 's/\b(worldfoundry)(_common|_train|_inf|_rl)\b/worldfoundry\2/g; s/WorldFoundry/WorldFoundry/g'
```

- [ ] **Step 2: Verify zero `worldfoundry` references in `docs/`**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -rE "worldfoundry|WorldFoundry" docs/ | wc -l
```

Expected: `0`

- [ ] **Step 3: Spot-check the gap-analysis roadmap (largest doc)**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
head -50 docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md
```

Expected: every "WorldFoundry" → "WorldFoundry".

- [ ] **Step 4: Spot-check the W22 design spec**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
head -30 docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md
```

Expected: clean.

- [ ] **Step 5: Spot-check the W35 ADR**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
head -30 docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "docs: rename worldfoundry → worldfoundry across specs/plans/dev/user/roadmap"
```

---

## Task 10: Update README.md at project root

**Files:**
- Modify: `README.md`

**Why separate from T9:** `README.md` is at project root, not under `docs/`. It also has higher visibility (GitHub landing page) and warrants its own commit.

- [ ] **Step 1: Update README**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
sed -i -E 's/\b(worldfoundry)(_common|_train|_inf|_rl)\b/worldfoundry\2/g; s/WorldFoundry/WorldFoundry/g' README.md
```

- [ ] **Step 2: Verify zero `worldfoundry` in README**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -E "worldfoundry|WorldFoundry" README.md | wc -l
```

Expected: `0`

- [ ] **Step 3: Spot-check README**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
cat README.md
```

Expected: title `# WorldFoundry` and any package references use `worldfoundry_*`.

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "docs(readme): rename worldfoundry → worldfoundry"
```

---

## Task 11: Update `.superpowers/sdd/progress.md` (banner only, body untouched)

**Files:**
- Modify: `.superpowers/sdd/progress.md` (insert 3-line banner at top, do NOT rewrite body)

**Why this task:** `progress.md` is the project's running log of completed Phase-1 tasks. It references `worldforge_*` heavily throughout. Per G6 we DO NOT rewrite history — but a 3-line banner at the top is a "current state pointer" rather than a history rewrite. All other sdd files (`task-*-report.md`, `task-*-brief.md`, `task-*-review.md`, `progress-phase1.md`, `phase2-w9-w10-merge-summary.md`, `task-final-fixes-report.md`) are **NOT touched**.

- [ ] **Step 1: Read current progress.md header**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
head -10 .superpowers/sdd/progress.md
```

- [ ] **Step 2: Insert banner at top**

Use `sed` to insert a banner block before line 1:

```bash
cd /mnt/c/WorkSpace/WorldFoundry
sed -i '1i\
# > **Note (2026-08-04):** Project renamed from `WorldFoundry` to `WorldFoundry`. See `docs/dev/rename-worldfoundry-to-worldfoundry.md` for motivation. Body of this progress log is preserved unchanged — package references retain the historical name (`worldforge_*`) as a record of the code state at that point in time.\
\
' .superpowers/sdd/progress.md
```

Note: this step deliberately does **not** rewrite body text. The banner sits above; the historical log below retains its original naming.

- [ ] **Step 3: Verify banner is at top, body untouched**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
head -8 .superpowers/sdd/progress.md
```

Expected: 3 lines of banner (note + blank line + blank line), then the original line 1 of progress.md.

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -c "worldfoundry" .superpowers/sdd/progress.md
```

Expected: count > 0 (body references still present — that's correct per G6).

- [ ] **Step 4: Verify other sdd files are untouched**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git status --short .superpowers/sdd/ | head -10
```

Expected: only `progress.md` is modified; `task-*-report.md`, `brief.md`, `review.md`, `progress-phase1.md`, `phase2-w9-w10-merge-summary.md`, `task-final-fixes-report.md` are unmodified.

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "docs(sdd): add rename banner to progress.md (body preserved per immutability rule)"
```

---

## Task 12: Rename spike files (`spikes/w35-3-rlinf/`)

**Files (8 confirmed by earlier grep):**
- Modify: 
  - `spikes/w35-3-rlinf/PROGRESS.md`
  - `spikes/w35-3-rlinf/config/gym_ppo_cart_pole.yaml`
  - `spikes/w35-3-rlinf/config/gym_ppo_cart_pole_vjepa2.yaml`
  - `spikes/w35-3-rlinf/config/gym_ppo_cart_pole_vjepa2_pixel.yaml`
  - `spikes/w35-3-rlinf/start-train.sh`
  - `spikes/w35-3-rlinf/start-train-vjepa2.sh`
  - `spikes/w35-3-rlinf/start-train-vjepa2-pixel.sh`
  - `spikes/w35-3-rlinf/results/tensorboard/config.yaml`

**Why mandatory (per G11 + user decision 2026-08-04):** `spikes/w35-3-rlinf/` is preserved as W35.3 决策历史档案. The spike contains `PROGRESS.md` (114 行 S1–S5 timeline), `integration-points-memo.md`, `integration-cost-assessment.md` (S5 deliverable 回填 ADR-W35 §6), 4 training logs, 14 tensorboard event files — all required to remain readable post-rename. Renaming the 8 source-text files (excludes `.venv/` and `rlinf-src/` — see Step 0 below) keeps the historical record coherent.

**Interfaces:**
- Produces: 0 `worldfoundry` / `WorldFoundry` references in spike text files (8 files above)
- Unchanged: `spikes/w35-3-rlinf/.venv/` (third-party deps, regenerated on re-install), `spikes/w35-3-rlinf/rlinf-src/` (upstream RLinf clone, not project code), `spikes/w35-3-rlinf/results/tensorboard/events.out.tfevents.*` (binary training logs, do not edit), `spikes/w35-3-rlinf/results/metrics.log` (binary-corrupted, do not edit), `run-*.log` files (binary training logs, do not edit)

- [ ] **Step 0: Confirm scope of files to sweep**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
# Show the 8 files that will be touched (text files only)
find spikes -type f \( -name "*.sh" -o -name "*.py" -o -name "*.yaml" -o -name "*.md" \) \
  -not -path "*/\.venv/*" -not -path "*/rlinf-src/*" \
  | sort
```

Expected: exactly 8 files listed above.

- [ ] **Step 1: Sweep spike text files**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find spikes -type f \( -name "*.sh" -o -name "*.py" -o -name "*.yaml" -o -name "*.md" \) \
  -not -path "*/\.venv/*" -not -path "*/rlinf-src/*" -print0 \
  | xargs -0 sed -i -E 's/\b(worldfoundry)(_common|_train|_inf|_rl)\b/worldfoundry\2/g; s/WorldFoundry/WorldFoundry/g'
```

- [ ] **Step 2: Verify zero `worldfoundry` references in spike text files**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
find spikes -type f \( -name "*.sh" -o -name "*.py" -o -name "*.yaml" -o -name "*.md" \) \
  -not -path "*/\.venv/*" -not -path "*/rlinf-src/*" \
  | xargs grep -lE "worldfoundry|WorldFoundry" 2>/dev/null | wc -l
```

Expected: `0`

- [ ] **Step 3: Spot-check PROGRESS.md**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -nE "worldfoundry|WorldFoundry" spikes/w35-3-rlinf/PROGRESS.md | head -10
```

Expected: no output. (Any prior `worldfoundry` references in the timeline are now `worldfoundry`.)

- [ ] **Step 4: Spot-check one start script**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -nE "worldfoundry|WorldFoundry" spikes/w35-3-rlinf/start-train-vjepa2-pixel.sh
```

Expected: no output.

- [ ] **Step 5: Spot-check one config YAML**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
grep -nE "worldfoundry|WorldFoundry" spikes/w35-3-rlinf/config/gym_ppo_cart_pole_vjepa2_pixel.yaml
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git add -A
git commit -m "chore(spikes): rename worldfoundry → worldfoundry in w35-3-rlinf spike (historical archive)"
```

---

## Task 13: Final residual sweep + verification gate

**Files:**
- Modify: nothing (verification only)
- Delete (build artifacts): `worldfoundry.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `dist/`, `build/`

**Why this task:** G10 mandates a residual sweep before merge. Build artifacts (G8) deleted to avoid stale references. Final pytest + ruff + grep triple-check.

- [ ] **Step 1: Delete build artifacts**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
rm -rf worldfoundry.egg-info/ dist/ build/ .pytest_cache/ .ruff_cache/
find . -name "__pycache__" -type d -not -path "*/\.venv/*" -not -path "*/spikes/w35-3-rlinf/.venv/*" -exec rm -rf {} + 2>/dev/null
echo "Build artifacts removed"
```

- [ ] **Step 2: Reinstall to regenerate egg-info under new name**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
pip install -e . 2>&1 | tail -5
```

Expected: `Successfully installed worldfoundry-0.1.0 ...`

- [ ] **Step 3: Residual sweep — non-spike source/docs/configs**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
echo "=== worldfoundry (lowercase) in tracked files, excluding spikes + .venv ==="
git ls-files | grep -v "^\.venv/" | grep -v "^spikes/" \
  | xargs grep -lE "worldfoundry" 2>/dev/null
echo ""
echo "=== WorldFoundry (capitalized) in tracked files, excluding spikes + .venv ==="
git ls-files | grep -v "^\.venv/" | grep -v "^spikes/" \
  | xargs grep -lE "WorldFoundry" 2>/dev/null
echo "=== END ==="
```

Expected: both lists empty (zero output between the `===` headers and next command).

- [ ] **Step 4: Residual sweep — spike text files (must be empty after T12)**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
echo "=== Residual 'worldfoundry' in spike text files (.venv + rlinf-src excluded) ==="
find spikes -type f \( -name "*.sh" -o -name "*.py" -o -name "*.yaml" -o -name "*.md" \) \
  -not -path "*/\.venv/*" -not -path "*/rlinf-src/*" \
  | xargs grep -lE "worldfoundry|WorldFoundry" 2>/dev/null
echo "=== END ==="
```

Expected: empty output (T12 sweeps all 8 spike text files). If non-empty: a file was missed in T12; re-run T12 with adjusted `find` filter.

- [ ] **Step 5: Full pytest run**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
pytest --tb=short 2>&1 | tail -30
```

Expected: tests run; success count + failure count reported. **No `ModuleNotFoundError: No module named 'worldforge_…'`** is acceptable. CPU-environment specific failures (e.g., `RuntimeError: NPU not available`) are pre-existing and not part of this rename.

- [ ] **Step 6: Full ruff run**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
ruff check . 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 7: Verify PyPI package metadata**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
pip show worldfoundry 2>&1 | head -10
echo "---"
pip show worldfoundry 2>&1 | head -5
```

Expected:
- First command: shows `Name: worldfoundry`, `Version: 0.1.0`, `Location: /mnt/c/WorkSpace/WorldFoundry`
- Second command: `WARNING: Package(s) not found: worldfoundry` (proves old name is gone)

- [ ] **Step 8: Sanity import test**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
python -c "
from worldfoundry_common import exceptions
from worldfoundry_train import trainer
from worldfoundry_inf import server
from worldfoundry_rl import train_ac
print('All four top-level packages import OK')
"
```

Expected: `All four top-level packages import OK`

- [ ] **Step 9: Verify entry-points registered correctly**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.platform_plugins')
print('vllm.platform_plugins entries:', [(ep.name, ep.value) for ep in eps if 'world' in ep.name.lower() or 'foundry' in ep.value.lower()])
eps2 = entry_points(group='vllm.general_plugins')
print('vllm.general_plugins entries:', [(ep.name, ep.value) for ep in eps2 if 'foundry' in ep.value.lower()])
"
```

Expected: at least one entry with `name='worldfoundry_inf'` and `value='worldfoundry_inf._vllm_ascend_plugin:register'`; and one entry with `name='vjepa2_model'` whose value points to `worldfoundry_inf._vllm_ascend_plugin:register_model`. Zero entries with the old `worldforge_*` names.

- [ ] **Step 10: Verify egg-info is the new name**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
cat worldfoundry.egg-info/PKG-INFO | head -5
```

Expected: `Name: worldfoundry`

- [ ] **Step 11: Final commit (build artifact cleanup)**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git status --short
git add -A
git diff --cached --stat
git commit -m "chore: post-rename cleanup (regenerated egg-info under worldfoundry)" || echo "Nothing to commit (expected)"
```

- [ ] **Step 12: Generate verification summary**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
echo "=== Rename verification summary ==="
echo "Commits since rename started:"
git log --oneline master..HEAD | head -15
echo ""
echo "Files changed:"
git diff --stat master..HEAD | tail -5
echo ""
echo "Residual 'worldfoundry' (non-spike):"
git ls-files | grep -v "^\.venv/" | grep -v "^spikes/" \
  | xargs grep -lE "worldfoundry" 2>/dev/null | wc -l
echo ""
echo "pytest result: (see T13.5 output above)"
echo "ruff result: (see T13.6 output above)"
```

---

## Task 14: PR + merge

**Files:**
- Branch: `rename/worldfoundry-to-worldfoundry` → `master`

**Why this task:** Bring the rename onto master via standard PR review.

- [ ] **Step 1: Push branch**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git push origin rename/worldfoundry-to-worldfoundry
```

- [ ] **Step 2: Open PR with description**

Use `gh pr create` with body summarizing:
- Motivation (link to `docs/dev/rename-worldfoundry-to-worldfoundry.md`)
- ~155 files changed across 12 commits
- pytest + ruff green
- Residual sweep: 0 in non-spike source + 0 in `spikes/w35-3-rlinf/` text files (T12 sweeps 8 spike files)
- Note: `.superpowers/sdd/` history is preserved with banner pointer only

- [ ] **Step 3: Merge PR**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
gh pr merge --squash=false --delete-branch  # preserve per-commit history (G9)
```

(or merge via GitHub UI; ensure `squash` is OFF to preserve the per-task commits)

- [ ] **Step 4: Post-merge: delete local branch**

```bash
cd /mnt/c/WorkSpace/WorldFoundry
git branch -d rename/worldfoundry-to-worldfoundry
git checkout master
git pull
```

---

## Acceptance Criteria (for the PR review)

The rename is considered DONE when all of the following hold:

- [ ] `pyproject.toml:6` reads `name = "worldfoundry"`
- [ ] `pyproject.toml:8` description reflects "渲训推 + 实机后续 + 开源编排" long-term positioning
- [ ] `pyproject.toml:35-39` vLLM entry-points reference `worldfoundry_inf` only (no `worldfoundry_inf`)
- [ ] `pyproject.toml:42` setuptools include pattern is `worldfoundry_*`
- [ ] 4 directories exist: `worldfoundry_common/`, `worldfoundry_train/`, `worldfoundry_inf/`, `worldfoundry_rl/`
- [ ] No top-level `worldfoundry_<subsystem>/` directory exists
- [ ] All non-spike `.py` files import from `worldfoundry_*` only (T13.3 first list = empty)
- [ ] All 8 spike text files in `spikes/w35-3-rlinf/` reference `worldfoundry_*` only (T13.4 list = empty)
- [ ] `docker/worldfoundry-base/` exists; `docker/worldfoundry-base/` does not
- [ ] Dockerfile COPY paths use `worldfoundry_*`
- [ ] README.md title is `# WorldFoundry`
- [ ] `docs/dev/rename-worldfoundry-to-worldfoundry.md` exists
- [ ] `pytest` runs without `ModuleNotFoundError: No module named 'worldforge_…'`
- [ ] `ruff check .` returns no errors
- [ ] `pip show worldfoundry` succeeds; `pip show worldfoundry` fails with "not found"
- [ ] 12 commits land on master (T1 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11 + T12 + T13), each with proper `<scope>: <imperative>` format
- [ ] `spikes/w35-3-rlinf/` text files (8 files) reference `worldfoundry_*` only; `PROGRESS.md` reads coherently post-rename
- [ ] `.superpowers/sdd/task-*-{brief,report,review}.md`, `progress-phase1.md`, `phase2-w9-w10-merge-summary.md`, `task-final-fixes-report.md` are unchanged (verified by `git diff master~12 master -- .superpowers/sdd/task-*.md` showing zero changes)
- [ ] `worldfoundry.egg-info/` regenerated to `worldfoundry.egg-info/`

---

## Rollback Plan

If the rename must be reverted mid-execution:

1. **Before T4 merged (no external publish)**: `git reset --hard master` on the rename branch; delete branch. Zero external impact.
2. **After T4 merged, before first PyPI publish**: revert merge commit on master; same as (1).
3. **After first PyPI publish of `worldfoundry`**: do NOT rename back. PyPI package names are effectively immutable once claimed. Add a `worldfoundry` PyPI alias that depends on `worldfoundry` if needed.

This plan executes (1) or (2) only — branch not yet pushed to remote (T14.1) means rollback is simply `git checkout master && git branch -D rename/worldfoundry-to-worldfoundry`.

---

## Post-Rename TODO (for future work)

Items intentionally NOT in this plan; track separately:

- [ ] PyPI publish of `worldfoundry` (post-Phase-2)
- [ ] Docker Hub publish of `worldfoundry-base:*` tags (post-Phase-2)
- [ ] Update any internal wiki / team docs that reference the old name
- [ ] (If `bench_results/` reports are user-facing) — regenerate or annotate with "pre-rename" label
- [ ] Phase-4 W36+ plan documents should use `worldfoundry` naming from the start
- [ ] (Optional, separate task) Consider promoting `spikes/w35-3-rlinf/PROGRESS.md` + `integration-points-memo.md` + `integration-cost-assessment.md` into `docs/superpowers/specs/` as `2026-08-04-w35-3-rlinf-spike-report.md` to consolidate W35.3 evidence into the docs hierarchy. Out of scope for this rename.

---

## Estimated Total Effort

| Task | Time |
|---|---|
| T1 (rename note doc) | 5 min |
| T2 (branch) | 1 min |
| T3 (pyproject.toml) | 5 min |
| T4 (4 dir renames) | 3 min |
| T5 (Python sweep + pytest) | 25 min |
| T6 (Docker) | 5 min |
| T7 (scripts) | 5 min |
| T8 (configs YAML) | 3 min |
| T9 (docs/) | 10 min |
| T10 (README) | 2 min |
| T11 (.superpowers/sdd banner) | 3 min |
| T12 (spikes — sweep 8 files in `spikes/w35-3-rlinf/`) | 5 min |
| T13 (verification) | 15 min |
| T14 (PR + merge) | 5 min |
| **Total** | **~90 min** |